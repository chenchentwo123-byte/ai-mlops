"""CLI: batch pre-annotate a folder of images with Grounding DINO."""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

from PIL import Image

ROOT = Path(__file__).resolve().parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.cleanup import cleanup
from src.config import data_dirs, load_config, resolve_path
from src.detector import GroundingDINODetector, list_gpus
from src.exporters import export_from_store, list_images
from src.prompts import normalize_prompt
from src.store import build_record, save_record
from src.visualize import draw_detections


def parse_args() -> argparse.Namespace:
    cfg = load_config()
    model_cfg = cfg.get("model", {})
    inf_cfg = cfg.get("inference", {})
    path_cfg = cfg.get("paths", {})
    exp_cfg = cfg.get("export", {})
    clean_cfg = cfg.get("cleanup", {})

    parser = argparse.ArgumentParser(
        description="Grounding DINO 提示词预标注（规范 JSON 落盘，导出时再转格式）"
    )
    parser.add_argument("--images", default=path_cfg.get("images", "data/images"), help="输入图片目录")
    parser.add_argument("--prompt", default="", help='类别提示词，如 "person. face. sofa."')
    parser.add_argument(
        "--annotations",
        default=path_cfg.get("annotations", "data/annotations"),
        help="规范标注目录（每图一个 JSON）",
    )
    parser.add_argument("--previews", default=path_cfg.get("previews", "data/previews"), help="可视化输出目录")
    parser.add_argument("--exports", default=path_cfg.get("exports", "data/exports"), help="导出目录")
    parser.add_argument(
        "--format",
        default=None,
        choices=["yolo", "coco", "labelme", "all"],
        help="推理后立刻导出该格式；省略则只写规范 JSON",
    )
    parser.add_argument("--export-only", action="store_true", help="不推理，只把已有规范标注转成 --format")
    parser.add_argument("--model", default=model_cfg.get("id", "models/grounding-dino-tiny"))
    parser.add_argument(
        "--device",
        default=model_cfg.get("device", "auto"),
        help="auto | cpu | cuda | cuda:0 | cuda:1",
    )
    parser.add_argument(
        "--gpu-id",
        type=int,
        default=int(model_cfg.get("gpu_id", 0) or 0),
        help="多卡时指定 GPU 编号，配合 --device auto/cuda",
    )
    parser.add_argument("--box-threshold", type=float, default=float(inf_cfg.get("box_threshold", 0.35)))
    parser.add_argument("--text-threshold", type=float, default=float(inf_cfg.get("text_threshold", 0.25)))
    parser.add_argument("--nms-iou", type=float, default=float(inf_cfg.get("nms_iou", 0.5)))
    parser.add_argument("--max-detections", type=int, default=int(inf_cfg.get("max_detections", 100)))
    parser.add_argument("--chunk-size", type=int, default=int(inf_cfg.get("chunk_size", 8)))
    parser.add_argument("--no-preview", action="store_true", help="不保存可视化图")
    parser.add_argument("--cleanup", action="store_true", help="按 keep_days 清理过期文件")
    parser.add_argument("--keep-days", type=float, default=float(clean_cfg.get("keep_days", 7)))
    parser.add_argument("--dry-run", action="store_true", help="清理时只列出将删除的文件")
    parser.add_argument("--config", default=str(ROOT / "config.yaml"))
    return parser.parse_args()


def _do_cleanup(args, dirs) -> None:
    result = cleanup(
        images_dir=dirs["images"],
        annotations_dir=dirs["annotations"],
        previews_dir=dirs["previews"],
        exports_dir=dirs["exports"],
        keep_days=args.keep_days,
        include_exports=True,
        dry_run=args.dry_run,
    )
    verb = "将删除" if args.dry_run else "已删除"
    print(f"清理（保留 {result['keep_days']} 天，{verb} {result['removed_count']} 个文件）")
    for kind, files in result["removed"].items():
        if files:
            print(f"  {kind}: {len(files)}")


def main() -> int:
    args = parse_args()
    cfg = load_config(args.config) if args.config else load_config()
    dirs = data_dirs(cfg)
    # CLI flags override config paths.
    dirs["images"] = resolve_path(args.images)
    dirs["annotations"] = resolve_path(args.annotations)
    dirs["previews"] = resolve_path(args.previews)
    dirs["exports"] = resolve_path(args.exports)

    if args.cleanup:
        _do_cleanup(args, dirs)
        if args.export_only is False and not args.prompt:
            return 0

    if args.export_only:
        fmt = args.format or "yolo"
        result = export_from_store(dirs["annotations"], dirs["exports"], fmt, dirs["images"])
        print(f"从规范标注导出 {result['count']} 张 → {fmt}")
        print(f"类别: {result['classes']}")
        print(f"目录: {dirs['exports']}")
        return 0 if result["count"] else 2

    caption, classes = normalize_prompt(args.prompt)
    if not classes:
        print("错误：提示词为空。例如 --prompt \"cat, dog, person\"")
        return 2

    images = list_images(dirs["images"])
    if not images:
        print(f"错误：在 {dirs['images']} 找不到图片（jpg/png/bmp/webp/tif）")
        return 2

    print(f"模型: {args.model}")
    gpus = list_gpus()
    if gpus:
        print("GPU:")
        for g in gpus:
            print(f"  cuda:{g['index']}  {g['name']}  {g['memory_gb']}GB")
    else:
        print("GPU: 无")
    print(f"提示词: {caption}")
    print(f"类别 ({len(classes)}): {classes}")
    print(f"图片: {len(images)} 张  @ {dirs['images']}")
    print(f"规范标注: {dirs['annotations']}")

    detector = GroundingDINODetector(
        model_id=args.model,
        device=args.device,
        gpu_id=args.gpu_id,
        chunk_size=args.chunk_size,
        local_files_only=bool(cfg.get("model", {}).get("local_files_only", True)),
    )
    t0 = time.perf_counter()
    detector.load()
    print(f"模型加载完成，用时 {time.perf_counter() - t0:.1f}s，设备 {detector.device}")

    total_boxes = 0
    t_infer = time.perf_counter()

    for i, image_path in enumerate(images, start=1):
        image = Image.open(image_path).convert("RGB")
        detections = detector.detect(
            image,
            caption=caption,
            classes=classes,
            box_threshold=args.box_threshold,
            text_threshold=args.text_threshold,
            nms_iou=args.nms_iou,
            max_detections=args.max_detections,
        )
        total_boxes += len(detections)
        record = build_record(
            image_path,
            image,
            detections,
            classes,
            caption,
            model_id=args.model,
            box_threshold=args.box_threshold,
            text_threshold=args.text_threshold,
            nms_iou=args.nms_iou,
        )
        save_record(record, dirs["annotations"], image_path.stem)
        if not args.no_preview:
            vis = draw_detections(image, detections)
            dirs["previews"].mkdir(parents=True, exist_ok=True)
            vis.save(dirs["previews"] / f"{image_path.stem}_vis.jpg", quality=92)
        print(f"[{i}/{len(images)}] {image_path.name}: {len(detections)} boxes")

    if args.format:
        result = export_from_store(dirs["annotations"], dirs["exports"], args.format, dirs["images"])
        print(f"已导出 {result['count']} 张 → {args.format} @ {dirs['exports']}")

    elapsed = time.perf_counter() - t_infer
    print(
        f"完成：{len(images)} 张图，{total_boxes} 个框，"
        f"推理 {elapsed:.1f}s（{elapsed / max(len(images), 1):.2f}s/张）"
    )
    print(f"规范标注: {dirs['annotations']}")
    if not args.no_preview:
        print(f"可视化: {dirs['previews']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
