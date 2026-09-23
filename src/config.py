from __future__ import annotations

from pathlib import Path
from typing import Any

import yaml

ROOT = Path(__file__).resolve().parent.parent
DEFAULT_CONFIG = ROOT / "config.yaml"


def load_config(path: str | Path | None = None) -> dict[str, Any]:
    cfg_path = Path(path) if path else DEFAULT_CONFIG
    if not cfg_path.exists():
        return {}
    with cfg_path.open("r", encoding="utf-8") as fh:
        data = yaml.safe_load(fh) or {}
    return data


def resolve_path(value: str | Path, root: Path | None = None) -> Path:
    p = Path(value)
    if p.is_absolute():
        return p
    return (root or ROOT) / p


def data_dirs(cfg: dict[str, Any] | None = None, root: Path | None = None) -> dict[str, Path]:
    cfg = cfg if cfg is not None else load_config()
    paths = cfg.get("paths", {})
    base = root or ROOT
    return {
        "images": resolve_path(paths.get("images", "data/images"), base),
        "annotations": resolve_path(paths.get("annotations", "data/annotations"), base),
        "previews": resolve_path(paths.get("previews", "data/previews"), base),
        "exports": resolve_path(paths.get("exports", "data/exports"), base),
    }
