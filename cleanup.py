"""CLI: purge stale images / annotations / previews / exports."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.cleanup import cleanup
from src.config import data_dirs, load_config


def main() -> int:
    cfg = load_config()
    clean_cfg = cfg.get("cleanup", {})
    parser = argparse.ArgumentParser(description="清理过期图片和标注")
    parser.add_argument("--keep-days", type=float, default=float(clean_cfg.get("keep_days", 7)))
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--no-exports", action="store_true", help="不清理 data/exports")
    args = parser.parse_args()
    dirs = data_dirs(cfg)
    result = cleanup(
        images_dir=dirs["images"],
        annotations_dir=dirs["annotations"],
        previews_dir=dirs["previews"],
        exports_dir=dirs["exports"],
        keep_days=args.keep_days,
        include_exports=not args.no_exports,
        dry_run=args.dry_run,
    )
    print(json.dumps(result, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
