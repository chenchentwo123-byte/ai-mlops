"""Delete stale images, canonical annotations, previews, and exports."""

from __future__ import annotations

import time
from pathlib import Path
from typing import Iterable

KEEP_NAMES = {".gitkeep", ".gitkeep.txt"}
IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".webp", ".tif", ".tiff"}
ANN_EXTS = {".json", ".jsonl", ".txt", ".yaml", ".yml"}


def _iter_files(root: Path) -> Iterable[Path]:
    if not root.exists():
        return []
    if root.is_file():
        return [root]
    return [p for p in root.rglob("*") if p.is_file()]


def _is_stale(path: Path, cutoff: float) -> bool:
    if path.name in KEEP_NAMES:
        return False
    try:
        return path.stat().st_mtime < cutoff
    except OSError:
        return False


def purge_dir(root: Path, cutoff: float, dry_run: bool) -> list[str]:
    removed: list[str] = []
    for path in _iter_files(Path(root)):
        if not _is_stale(path, cutoff):
            continue
        removed.append(str(path))
        if not dry_run:
            try:
                path.unlink()
            except OSError:
                continue
    return removed


def cleanup(
    *,
    images_dir: Path,
    annotations_dir: Path,
    previews_dir: Path,
    exports_dir: Path | None = None,
    keep_days: float = 7,
    include_exports: bool = True,
    dry_run: bool = False,
) -> dict:
    keep_days = max(0.0, float(keep_days))
    cutoff = time.time() - keep_days * 86400
    removed: dict[str, list[str]] = {
        "images": purge_dir(Path(images_dir), cutoff, dry_run),
        "annotations": purge_dir(Path(annotations_dir), cutoff, dry_run),
        "previews": purge_dir(Path(previews_dir), cutoff, dry_run),
        "exports": [],
    }
    if include_exports and exports_dir is not None:
        removed["exports"] = purge_dir(Path(exports_dir), cutoff, dry_run)
    total = sum(len(v) for v in removed.values())
    return {
        "keep_days": keep_days,
        "cutoff_epoch": cutoff,
        "dry_run": dry_run,
        "removed_count": total,
        "removed": removed,
    }
