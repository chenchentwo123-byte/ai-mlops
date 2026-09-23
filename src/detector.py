"""Grounding DINO wrapper: load once, detect many images from a text prompt.

Each box is assigned **one** class by scoring the token span of that class,
instead of decoding every token above the text threshold (which concatenates
the whole caption onto every box when many labels are packed together).
"""

from __future__ import annotations

import os
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable, Sequence

import numpy as np
import torch
from PIL import Image

from .config import ROOT, resolve_path
from .prompts import to_caption


class _WeightLock:
    """One process at a time may mmap model.safetensors. Concurrent open corrupts the header."""

    def __init__(self, model_dir: Path, timeout: float = 600) -> None:
        self.path = Path(model_dir) / ".weights.lock"
        self.timeout = timeout
        self._fh = None

    def __enter__(self):
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self._fh = open(self.path, "a+b")
        if self._fh.tell() == 0:
            self._fh.write(b"\0")
            self._fh.flush()
        deadline = time.time() + self.timeout
        while True:
            try:
                self._lock()
                break
            except OSError:
                if time.time() > deadline:
                    raise TimeoutError(f"等待权重锁超时：{self.path}")
                time.sleep(0.5)
        return self

    def _lock(self) -> None:
        self._fh.seek(0)
        if os.name == "nt":
            import msvcrt

            msvcrt.locking(self._fh.fileno(), msvcrt.LK_NBLCK, 1)
        else:
            import fcntl

            fcntl.flock(self._fh.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)

    def _unlock(self) -> None:
        self._fh.seek(0)
        if os.name == "nt":
            import msvcrt

            msvcrt.locking(self._fh.fileno(), msvcrt.LK_UNLCK, 1)
        else:
            import fcntl

            fcntl.flock(self._fh.fileno(), fcntl.LOCK_UN)

    def __exit__(self, *exc) -> None:
        if self._fh is None:
            return
        try:
            self._unlock()
        except Exception:
            pass
        try:
            self._fh.close()
        except Exception:
            pass
        self._fh = None


# Tiny 官方 safetensors 完整大小。拷坏时 header 还能读、张量对不上，会报 incomplete metadata。
TINY_SAFETENSORS_BYTES = 689_359_096


def _check_safetensors(model_dir: Path) -> None:
    path = Path(model_dir) / "model.safetensors"
    if not path.is_file():
        raise FileNotFoundError(f"缺少权重文件：{path}")
    size = path.stat().st_size
    if size != TINY_SAFETENSORS_BYTES:
        raise RuntimeError(
            f"权重文件不完整：{path}\n"
            f"实际 {size} 字节，完整应为 {TINY_SAFETENSORS_BYTES} 字节"
            f"（差 {TINY_SAFETENSORS_BYTES - size}）。\n"
            "请从本机重新拷贝 models/grounding-dino-tiny/model.safetensors，不要断点续传拼文件。"
        )
    with path.open("rb") as fh:
        header_len = int.from_bytes(fh.read(8), "little")
    if header_len <= 0 or header_len > size - 8:
        raise RuntimeError(
            f"权重头损坏：{path} header_len={header_len} file={size}。请重新拷贝完整文件。"
        )


@dataclass
class Detection:
    label: str
    score: float
    xyxy: tuple[int, int, int, int]  # pixel coords, x1 y1 x2 y2

    def as_xywh(self) -> tuple[int, int, int, int]:
        x1, y1, x2, y2 = self.xyxy
        return x1, y1, x2 - x1, y2 - y1

    def as_yolo(self, width: int, height: int) -> tuple[float, float, float, float]:
        x1, y1, x2, y2 = self.xyxy
        cx = ((x1 + x2) / 2.0) / width
        cy = ((y1 + y2) / 2.0) / height
        bw = (x2 - x1) / width
        bh = (y2 - y1) / height
        return cx, cy, bw, bh

    def to_dict(self) -> dict[str, Any]:
        return {
            "label": self.label,
            "score": round(float(self.score), 4),
            "bbox_xyxy": list(self.xyxy),
        }


def list_gpus() -> list[dict[str, Any]]:
    """Visible CUDA devices: ``[{index, name, memory_gb}, ...]``."""
    if not torch.cuda.is_available():
        return []
    out = []
    for i in range(torch.cuda.device_count()):
        props = torch.cuda.get_device_properties(i)
        out.append(
            {
                "index": i,
                "name": props.name,
                "memory_gb": round(props.total_memory / (1024**3), 1),
            }
        )
    return out


def device_choices() -> list[str]:
    """Dropdown values: auto, cpu, cuda:0 (Name 8.0GB), ..."""
    choices = ["auto", "cpu"]
    for gpu in list_gpus():
        choices.append(f"cuda:{gpu['index']}  ({gpu['name']}, {gpu['memory_gb']}GB)")
    return choices


def _parse_gpu_id(value: Any) -> int | None:
    if value is None or value == "":
        return None
    try:
        return int(value)
    except (TypeError, ValueError):
        return None


def _resolve_device(name: str, gpu_id: int | None = None) -> torch.device:
    """``auto`` / ``cpu`` / ``cuda`` / ``cuda:0``. ``gpu_id`` overrides the index."""
    raw = (name or "auto").strip()
    # UI labels look like ``cuda:0  (RTX 5060, 8.0GB)``
    token = raw.split()[0].lower() if raw else "auto"
    if token in ("auto",):
        if not torch.cuda.is_available():
            return torch.device("cpu")
        idx = 0 if gpu_id is None else gpu_id
        n = torch.cuda.device_count()
        if idx < 0 or idx >= n:
            raise ValueError(f"gpu_id={idx} 超出范围，当前可见 GPU 0..{n - 1}")
        return torch.device(f"cuda:{idx}")
    if token in ("cpu",):
        return torch.device("cpu")
    if token in ("cuda", "gpu"):
        idx = 0 if gpu_id is None else gpu_id
        if not torch.cuda.is_available():
            raise RuntimeError("配置了 CUDA，但当前进程看不到 GPU（检查驱动 / CUDA_VISIBLE_DEVICES）")
        n = torch.cuda.device_count()
        if idx < 0 or idx >= n:
            raise ValueError(f"gpu_id={idx} 超出范围，当前可见 GPU 0..{n - 1}")
        return torch.device(f"cuda:{idx}")
    if token.startswith("cuda:"):
        idx = int(token.split(":", 1)[1])
        if not torch.cuda.is_available():
            raise RuntimeError("配置了 CUDA，但当前进程看不到 GPU")
        n = torch.cuda.device_count()
        if idx < 0 or idx >= n:
            raise ValueError(f"device={token} 超出范围，当前可见 GPU 0..{n - 1}")
        return torch.device(f"cuda:{idx}")
    return torch.device(token)


def _resolve_dtype(name: str, device: torch.device) -> torch.dtype:
    name = (name or "auto").lower()
    if name == "float16":
        return torch.float16
    if name == "float32":
        return torch.float32
    if device.type == "cuda":
        return torch.float16
    return torch.float32


def _box_iou(a: np.ndarray, b: np.ndarray) -> float:
    x1 = max(a[0], b[0])
    y1 = max(a[1], b[1])
    x2 = min(a[2], b[2])
    y2 = min(a[3], b[3])
    inter = max(0.0, x2 - x1) * max(0.0, y2 - y1)
    if inter <= 0:
        return 0.0
    area_a = max(0.0, a[2] - a[0]) * max(0.0, a[3] - a[1])
    area_b = max(0.0, b[2] - b[0]) * max(0.0, b[3] - b[1])
    union = area_a + area_b - inter
    return inter / union if union > 0 else 0.0


def nms(detections: list[Detection], iou_threshold: float, class_aware: bool = True) -> list[Detection]:
    """Keep higher-score boxes when IoU exceeds the threshold.

    ``class_aware=True`` only suppresses boxes of the same label (person and
    face may overlap). ``False`` suppresses across classes (near-duplicate boxes).
    """
    if iou_threshold <= 0 or len(detections) <= 1:
        return detections

    def _nms_group(group: list[Detection]) -> list[Detection]:
        group = sorted(group, key=lambda d: d.score, reverse=True)
        selected: list[Detection] = []
        for det in group:
            box = np.array(det.xyxy, dtype=np.float32)
            if any(_box_iou(box, np.array(s.xyxy, dtype=np.float32)) > iou_threshold for s in selected):
                continue
            selected.append(det)
        return selected

    if not class_aware:
        kept = _nms_group(detections)
        kept.sort(key=lambda d: d.score, reverse=True)
        return kept

    kept: list[Detection] = []
    by_label: dict[str, list[Detection]] = {}
    for det in detections:
        by_label.setdefault(det.label, []).append(det)
    for group in by_label.values():
        kept.extend(_nms_group(group))
    kept.sort(key=lambda d: d.score, reverse=True)
    return kept


def _find_subseq(haystack: Sequence[int], needle: Sequence[int], start: int = 0) -> int | None:
    if not needle:
        return None
    n = len(needle)
    limit = len(haystack) - n + 1
    for i in range(start, max(start, limit)):
        if tuple(haystack[i : i + n]) == tuple(needle):
            return i
    return None


def class_token_spans(tokenizer, input_ids: Sequence[int], classes: Sequence[str]) -> list[tuple[int, int] | None]:
    """Locate each class phrase inside the tokenized caption.

    Search left-to-right so earlier classes claim their own tokens
    (``tv`` will not steal tokens from a later ``tvcabinet``).
    """
    ids = [int(x) for x in input_ids]
    spans: list[tuple[int, int] | None] = []
    cursor = 0
    for cls in classes:
        token_ids = tokenizer.encode(cls, add_special_tokens=False)
        start = _find_subseq(ids, token_ids, cursor)
        if start is None:
            # BERT sometimes keeps a leading space as its own piece after "."
            token_ids = tokenizer.encode(" " + cls, add_special_tokens=False)
            start = _find_subseq(ids, token_ids, cursor)
        if start is None:
            spans.append(None)
            continue
        end = start + len(token_ids)
        spans.append((start, end))
        cursor = end
    return spans


def _clip_box(box, width: int, height: int) -> tuple[int, int, int, int]:
    if hasattr(box, "tolist"):
        box = box.tolist()
    x1, y1, x2, y2 = [float(v) for v in box]
    x1 = int(round(max(0, min(width - 1, x1))))
    y1 = int(round(max(0, min(height - 1, y1))))
    x2 = int(round(max(0, min(width, x2))))
    y2 = int(round(max(0, min(height, y2))))
    return x1, y1, x2, y2


def _cxcywh_to_xyxy(box, width: int, height: int) -> tuple[int, int, int, int]:
    if hasattr(box, "tolist"):
        box = box.tolist()
    cx, cy, bw, bh = [float(v) for v in box]
    x1 = (cx - bw / 2.0) * width
    y1 = (cy - bh / 2.0) * height
    x2 = (cx + bw / 2.0) * width
    y2 = (cy + bh / 2.0) * height
    return _clip_box((x1, y1, x2, y2), width, height)


class GroundingDINODetector:
    """Lazy-loaded Grounding DINO detector."""

    def __init__(
        self,
        model_id: str = "models/grounding-dino-tiny",
        device: str = "auto",
        gpu_id: int | None = None,
        dtype: str = "auto",
        chunk_size: int = 10,
        local_files_only: bool = True,
    ) -> None:
        self.model_id = model_id
        self.device_name = device
        self.gpu_id = _parse_gpu_id(gpu_id)
        self.device = _resolve_device(device, self.gpu_id)
        self.dtype = _resolve_dtype(dtype, self.device)
        self.chunk_size = max(1, int(chunk_size))
        self.local_files_only = local_files_only
        self.processor = None
        self.model = None

    def _model_path(self) -> str:
        raw = Path(self.model_id)
        if raw.exists():
            return str(raw.resolve())
        resolved = resolve_path(self.model_id, ROOT)
        if resolved.exists():
            return str(resolved)
        raise FileNotFoundError(
            f"本地模型不存在: {self.model_id}（已解析 {resolved}）。"
            "请把 grounding-dino-tiny 放到 models/grounding-dino-tiny ，不要走 Hugging Face 下载。"
        )

    def load(self) -> None:
        if self.model is not None:
            return
        from transformers import AutoModelForZeroShotObjectDetection, AutoProcessor

        path = self._model_path()
        _check_safetensors(path)
        kwargs = {"local_files_only": True} if self.local_files_only else {}
        model_kwargs = dict(kwargs)
        if self.dtype == torch.float16:
            model_kwargs["dtype"] = torch.float16
        print(f"加载权重（排队）：{path}", flush=True)
        with _WeightLock(Path(path)):
            print(f"加载权重（开始）：{path}", flush=True)
            self.processor = AutoProcessor.from_pretrained(path, **kwargs)
            try:
                self.model = AutoModelForZeroShotObjectDetection.from_pretrained(
                    path, **model_kwargs
                )
            except TypeError:
                model_kwargs.pop("dtype", None)
                if self.dtype == torch.float16:
                    model_kwargs["torch_dtype"] = torch.float16
                self.model = AutoModelForZeroShotObjectDetection.from_pretrained(
                    path, **model_kwargs
                )
        self.model.to(device=self.device, dtype=self.dtype)
        self.model.eval()
        if self.device.type == "cuda":
            torch.backends.cudnn.benchmark = True
        self.model_id = path

    def detect(
        self,
        image: Image.Image,
        caption: str,
        classes: Iterable[str] | None = None,
        box_threshold: float = 0.35,
        text_threshold: float = 0.25,
        nms_iou: float = 0.5,
        max_detections: int = 100,
        chunk_size: int | None = None,
    ) -> list[Detection]:
        self.load()
        if image.mode != "RGB":
            image = image.convert("RGB")

        class_list = [c.strip().lower().strip(".") for c in (classes or []) if c and str(c).strip()]
        if not class_list:
            from .prompts import parse_classes

            class_list = parse_classes(caption)
        if not class_list:
            return []

        size = chunk_size or self.chunk_size
        detections: list[Detection] = []
        if len(class_list) > size:
            for i in range(0, len(class_list), size):
                chunk = class_list[i : i + size]
                detections.extend(
                    self._detect_chunk(image, chunk, box_threshold, text_threshold)
                )
        else:
            detections = self._detect_chunk(image, class_list, box_threshold, text_threshold)

        detections = nms(detections, nms_iou, class_aware=True)
        # 不同 chunk / 近义类别可能对同一物体出两个几乎重合的框，只留分数更高的。
        detections = nms(detections, 0.90, class_aware=False)
        detections.sort(key=lambda d: d.score, reverse=True)
        if max_detections > 0:
            detections = detections[:max_detections]
        return detections

    def _detect_chunk(
        self,
        image: Image.Image,
        classes: list[str],
        box_threshold: float,
        text_threshold: float,
    ) -> list[Detection]:
        assert self.processor is not None and self.model is not None
        caption = to_caption(classes)
        encoded = self.processor(images=image, text=caption, return_tensors="pt")
        if hasattr(encoded, "to"):
            encoded = encoded.to(self.device)
        tensors = {
            k: v.to(self.device) if hasattr(v, "to") else v
            for k, v in encoded.items()
        }

        use_autocast = self.device.type == "cuda" and self.dtype == torch.float16
        with torch.inference_mode():
            if use_autocast:
                with torch.autocast(device_type="cuda", dtype=torch.float16):
                    outputs = self.model(**tensors)
            else:
                outputs = self.model(**tensors)

        return self._boxes_from_outputs(
            outputs,
            tensors["input_ids"][0],
            image.size,
            classes,
            box_threshold,
            text_threshold,
        )

    def _boxes_from_outputs(
        self,
        outputs,
        input_ids,
        image_size: tuple[int, int],
        classes: list[str],
        box_threshold: float,
        text_threshold: float,
    ) -> list[Detection]:
        assert self.processor is not None
        tokenizer = self.processor.tokenizer
        width, height = image_size

        ids = input_ids.detach().cpu().tolist()
        spans = class_token_spans(tokenizer, ids, classes)
        valid = [(i, span) for i, span in enumerate(spans) if span is not None]
        if not valid:
            return []

        token_probs = outputs.logits[0].detach().sigmoid()
        pred_boxes = outputs.pred_boxes[0].detach()
        seq_len = min(len(ids), token_probs.shape[-1])
        probs = token_probs[:, :seq_len].float().cpu()
        boxes = pred_boxes.float().cpu().numpy()

        score_cols = []
        class_idx = []
        for i, (start, end) in valid:
            start = max(0, min(seq_len - 1, start))
            end = max(start + 1, min(seq_len, end))
            score_cols.append(probs[:, start:end].max(dim=1).values)
            class_idx.append(i)
        scores = torch.stack(score_cols, dim=1)
        best_score, local_idx = scores.max(dim=1)
        thresh = max(float(text_threshold), float(box_threshold))
        keep = (best_score >= thresh).nonzero(as_tuple=False).flatten().tolist()
        best_score_list = best_score.tolist()
        local_idx_list = local_idx.tolist()

        detections: list[Detection] = []
        for q in keep:
            xyxy = _cxcywh_to_xyxy(boxes[q], width, height)
            if xyxy[2] <= xyxy[0] or xyxy[3] <= xyxy[1]:
                continue
            detections.append(
                Detection(
                    label=classes[class_idx[local_idx_list[q]]],
                    score=float(best_score_list[q]),
                    xyxy=xyxy,
                )
            )
        return detections
