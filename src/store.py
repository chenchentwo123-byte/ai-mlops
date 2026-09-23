"""Canonical annotation store: one JSON per image under data/annotations."""

from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Sequence

from PIL import Image

from .detector import Detection

ANN_SUFFIX = ".json"


def now_iso() -> str:
    return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


def detections_from_record(record: dict) -> list[Detection]:
    out: list[Detection] = []
    for item in record.get("detections") or []:
        box = item.get("bbox_xyxy") or item.get("xyxy") or [0, 0, 0, 0]
        out.append(
            Detection(
                label=str(item.get("label", "object")),
                score=float(item.get("score", 0.0)),
                xyxy=(int(box[0]), int(box[1]), int(box[2]), int(box[3])),
            )
        )
    return out


def build_record(
    image_path: Path,
    image: Image.Image,
    detections: Sequence[Detection],
    classes: Sequence[str],
    prompt: str,
    *,
    model_id: str = "",
    box_threshold: float = 0.35,
    text_threshold: float = 0.25,
    nms_iou: float = 0.5,
) -> dict[str, Any]:
    w, h = image.size
    return {
        "image": image_path.name,
        "image_path": str(image_path),
        "width": w,
        "height": h,
        "prompt": prompt,
        "classes": list(classes),
        "created_at": now_iso(),
        "model": model_id,
        "thresholds": {
            "box": box_threshold,
            "text": text_threshold,
            "nms": nms_iou,
        },
        "count": len(detections),
        "detections": [d.to_dict() for d in detections],
    }


def annotation_path(annotations_dir: Path, image_path: Path) -> Path:
    return Path(annotations_dir) / f"{image_path.stem}{ANN_SUFFIX}"


def save_record(record: dict, annotations_dir: Path, stem: str | None = None) -> Path:
    annotations_dir = Path(annotations_dir)
    annotations_dir.mkdir(parents=True, exist_ok=True)
    name = stem or Path(record.get("image", "image")).stem
    path = annotations_dir / f"{name}{ANN_SUFFIX}"
    path.write_text(json.dumps(record, ensure_ascii=False, indent=2), encoding="utf-8")
    return path


def load_record(path: Path) -> dict:
    with Path(path).open("r", encoding="utf-8") as fh:
        return json.load(fh)


def list_records(annotations_dir: Path) -> list[Path]:
    root = Path(annotations_dir)
    if not root.exists():
        return []
    files = [p for p in root.iterdir() if p.is_file() and p.suffix.lower() == ANN_SUFFIX]
    files.sort(key=lambda p: p.name.lower())
    return files
