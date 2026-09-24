"""YOLOE visual-prompt detector. Returns the same Detection list as Grounding DINO."""

from __future__ import annotations

import re
import tempfile
import threading
from dataclasses import dataclass
from pathlib import Path
from typing import Sequence

import numpy as np
from PIL import Image

from .config import ROOT, resolve_path
from .detector import Detection, _resolve_device, nms

MODEL_OPTIONS = {
    "s": {"label": "S · 速度优先", "filename": "yoloe-11s-seg.pt"},
    "m": {"label": "M · 平衡", "filename": "yoloe-11m-seg.pt"},
    "l": {"label": "L · 精度优先", "filename": "yoloe-11l-seg.pt"},
    "x": {"label": "X · 最高精度", "filename": "yoloe-26x-seg.pt"},
}

_SPLIT_RE = re.compile(r"[,;\n|]+")


def parse_visual_classes(text: str) -> list[str]:
    """Split class names. Preserve case; unique by casefold; order kept."""
    if not text:
        return []
    seen: set[str] = set()
    classes: list[str] = []
    for raw in _SPLIT_RE.split(text):
        name = re.sub(r"\s+", " ", raw).strip()
        if not name:
            continue
        key = name.casefold()
        if key in seen:
            continue
        seen.add(key)
        classes.append(name)
    return classes


def visual_caption(classes: Sequence[str]) -> str:
    names = [str(c).strip() for c in classes if str(c).strip()]
    return "visual: " + ", ".join(names) if names else "visual:"


def normalize_model_key(model_key: str | None) -> str:
    key = (model_key or "s").strip().lower().split()[0]
    if key in MODEL_OPTIONS:
        return key
    raise ValueError(f"未知 YOLOE 规格：{model_key}（可选 s / m / l / x）")


def weights_filename(model_key: str) -> str:
    return MODEL_OPTIONS[normalize_model_key(model_key)]["filename"]


def resolve_weights(model_key: str = "s", weights_dir: str | Path | None = None) -> Path:
    directory = resolve_path(weights_dir or "models/yoloe", ROOT)
    return directory / weights_filename(model_key)


@dataclass
class VisualPrompt:
    image: Image.Image
    xyxy: tuple[int, int, int, int]
    class_id: int


class YOLOEVisualDetector:
    """Load one local YOLOE weight file; inject visual class embeddings; detect."""

    def __init__(
        self,
        weights: str | Path,
        device: str = "auto",
        gpu_id: int | None = None,
        imgsz: int = 960,
    ) -> None:
        self.weights = Path(weights)
        self.device_name = device
        self.gpu_id = gpu_id
        self.imgsz = int(imgsz)
        self.model = None
        self.device = None
        self.class_names: list[str] = []
        self._lock = threading.Lock()

    def load(self) -> None:
        if not self.weights.is_file():
            raise FileNotFoundError(
                f"缺少 YOLOE 权重：{self.weights}\n"
                "请把对应 .pt 放到 models/yoloe/（例如 yoloe-11s-seg.pt）。\n"
                "权重来自 Ultralytics YOLOE 发布文件，文件名不要改。"
            )
        try:
            from ultralytics import YOLOE
        except ImportError as exc:
            raise RuntimeError("未安装 ultralytics。请执行：pip install ultralytics==8.4.128") from exc

        self.device = _resolve_device(self.device_name, self.gpu_id)
        self.model = YOLOE(str(self.weights))
        self.model.to(self.device)
        self.model.overrides["device"] = str(self.device)

    def set_visual_classes(
        self,
        class_names: Sequence[str],
        prompts: Sequence[VisualPrompt],
        strategy: str = "semantic",
    ) -> None:
        if self.model is None:
            self.load()
        names = [str(n).strip() for n in class_names if str(n).strip()]
        if not names:
            raise ValueError("请至少填写一个类别名。")
        if not prompts:
            raise ValueError("请至少画一个视觉提示框。")
        strategy = (strategy or "semantic").strip().lower()
        if strategy not in ("semantic", "multi"):
            raise ValueError("第一期只支持 semantic / multi。")
        for p in prompts:
            if p.class_id < 0 or p.class_id >= len(names):
                raise ValueError(f"视觉提示 class_id={p.class_id} 超出类别列表。")

        from ultralytics.models.yolo.yoloe import YOLOEVPSegPredictor
        import torch
        import torch.nn.functional as functional

        buckets: dict[str, list] = {name: [] for name in names}
        with self._lock, tempfile.TemporaryDirectory() as td:
            tmp = Path(td)
            for i, prompt in enumerate(prompts):
                image = prompt.image.convert("RGB")
                path = tmp / f"ref_{i:03d}.png"
                image.save(path)
                x1, y1, x2, y2 = _clip_xyxy(prompt.xyxy, image.size)
                vis = {
                    "bboxes": np.array([[x1, y1, x2, y2]], dtype=np.float32),
                    "cls": np.array([0], dtype=np.int32),
                }
                self.model.predict(
                    str(path),
                    refer_image=str(path),
                    visual_prompts=vis,
                    predictor=YOLOEVPSegPredictor,
                    conf=0.01,
                    imgsz=self.imgsz,
                    verbose=False,
                )
                pe = self.model.model.pe.detach().clone()
                buckets[names[prompt.class_id]].append(pe)

            missing = [name for name, samples in buckets.items() if not samples]
            if missing:
                raise ValueError("这些类别还没有画框：" + ", ".join(missing))

            if strategy == "multi":
                model_names: list[str] = []
                pieces = []
                for name in names:
                    for sample in buckets[name]:
                        model_names.append(name)
                        pieces.append(functional.normalize(sample, dim=-1))
                embedding = torch.cat(pieces, dim=1)
                self.model.set_classes(model_names, embedding)
            else:
                embedding = torch.cat(
                    [
                        functional.normalize(torch.stack(buckets[name]).mean(0), dim=-1)
                        for name in names
                    ],
                    dim=1,
                )
                self.model.set_classes(list(names), embedding)
            self.model.predictor = None
            self.class_names = list(names)

    def detect(
        self,
        image: Image.Image,
        *,
        confidence: float = 0.10,
        nms_iou: float = 0.50,
        max_detections: int = 100,
        manuals: Sequence[Detection] | None = None,
    ) -> list[Detection]:
        if self.model is None:
            raise RuntimeError("YOLOE 尚未加载。")
        if not self.class_names:
            raise RuntimeError("还没有注入视觉类别，请先画提示框。")
        image = image.convert("RGB") if isinstance(image, Image.Image) else Image.fromarray(image).convert("RGB")
        arr = np.array(image)
        with self._lock:
            results = self.model.predict(
                arr,
                conf=float(confidence),
                imgsz=self.imgsz,
                verbose=False,
            )
        detections = _results_to_detections(results, self.class_names)
        if manuals:
            detections = merge_manual_boxes(detections, manuals, iou_threshold=0.5)
        detections = nms(detections, float(nms_iou), class_aware=True)
        detections = nms(detections, 0.90, class_aware=False)
        return detections[: max(1, int(max_detections))]


def _clip_xyxy(xyxy: Sequence[float], size: tuple[int, int]) -> tuple[int, int, int, int]:
    w, h = size
    x1, y1, x2, y2 = (int(v) for v in xyxy)
    if x2 < x1:
        x1, x2 = x2, x1
    if y2 < y1:
        y1, y2 = y2, y1
    x1 = min(max(0, x1), max(0, w - 1))
    y1 = min(max(0, y1), max(0, h - 1))
    x2 = min(max(x1 + 1, x2), w)
    y2 = min(max(y1 + 1, y2), h)
    return x1, y1, x2, y2


def _results_to_detections(results, class_names: Sequence[str]) -> list[Detection]:
    out: list[Detection] = []
    if not results:
        return out
    result = results[0]
    boxes = getattr(result, "boxes", None)
    if boxes is None or len(boxes) == 0:
        return out
    xyxy = boxes.xyxy.detach().cpu().tolist()
    confs = boxes.conf.detach().cpu().tolist()
    clss = boxes.cls.detach().cpu().tolist()
    for box, score, cls in zip(xyxy, confs, clss):
        idx = int(cls)
        if idx < 0 or idx >= len(class_names):
            continue
        x1, y1, x2, y2 = (int(round(v)) for v in box)
        out.append(Detection(label=str(class_names[idx]), score=float(score), xyxy=(x1, y1, x2, y2)))
    out.sort(key=lambda d: d.score, reverse=True)
    return out


def merge_manual_boxes(
    detections: Sequence[Detection],
    manuals: Sequence[Detection],
    iou_threshold: float = 0.5,
) -> list[Detection]:
    """Drop predicted boxes that overlap a same-class manual box, then append manuals."""
    from .detector import _box_iou

    kept: list[Detection] = []
    for det in detections:
        box = np.array(det.xyxy, dtype=np.float32)
        drop = False
        for man in manuals:
            if man.label != det.label:
                continue
            if _box_iou(box, np.array(man.xyxy, dtype=np.float32)) >= iou_threshold:
                drop = True
                break
        if not drop:
            kept.append(det)
    kept.extend(manuals)
    kept.sort(key=lambda d: d.score, reverse=True)
    return kept
