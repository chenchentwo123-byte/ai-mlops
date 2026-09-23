"""One Grounding DINO replica on a GPU. Pulls image tasks from Redis."""

from __future__ import annotations

import argparse
import os
import socket
import sys
import time
from pathlib import Path

from PIL import Image

ROOT = Path(__file__).resolve().parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.config import data_dirs, load_config
from src.detector import GroundingDINODetector
from src.prompts import to_caption
from src.queue import complete_task, job_status, mark_job_running, pop_task, redis_enabled
from src.store import build_record, save_record
from src.visualize import draw_detections
from src import db as pg


def parse_args() -> argparse.Namespace:
    cfg = load_config()
    model_cfg = cfg.get("model", {})
    parser = argparse.ArgumentParser(description="Grounding DINO Redis worker（同一张卡上的一份模型）")
    parser.add_argument("--gpu-id", type=int, default=int(model_cfg.get("gpu_id", 0) or 0))
    parser.add_argument("--replica-id", type=int, default=0)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--config", default=str(ROOT / "config.yaml"))
    return parser.parse_args()


def _job_dirs(job: dict | None, fallback: dict[str, Path]) -> tuple[Path, Path]:
    if job and job.get("annotations_dir"):
        ann = Path(job["annotations_dir"])
        prev = Path(job.get("previews_dir") or (ann.parent / "previews"))
        ann.mkdir(parents=True, exist_ok=True)
        prev.mkdir(parents=True, exist_ok=True)
        return ann, prev
    return fallback["annotations"], fallback["previews"]


def main() -> int:
    args = parse_args()
    cfg = load_config(args.config)
    if not redis_enabled(cfg):
        print("redis.enabled 为 false，worker 退出。在 config.yaml 打开 redis 后再启动。")
        return 2

    dirs = data_dirs(cfg)
    model_cfg = cfg.get("model", {})
    inf_cfg = cfg.get("inference", {})
    name = socket.gethostname()
    tag = f"gpu={args.gpu_id} r{args.replica_id}"
    worker_id = f"{name}:{args.gpu_id}:r{args.replica_id}:{os.getpid()}"
    use_pg = pg.postgres_enabled(cfg)
    print(
        f"[worker {name} {tag}] 加载 {model_cfg.get('id')} → cuda:{args.gpu_id}",
        flush=True,
    )
    detector = GroundingDINODetector(
        model_id=model_cfg.get("id", "models/grounding-dino-tiny"),
        device=args.device,
        gpu_id=args.gpu_id,
        dtype=str(model_cfg.get("dtype", "auto")),
        chunk_size=int(inf_cfg.get("chunk_size", 32)),
        local_files_only=bool(model_cfg.get("local_files_only", True)),
    )
    t0 = time.perf_counter()
    detector.load()
    print(
        f"[worker {tag}] 就绪 device={detector.device} "
        f"{time.perf_counter() - t0:.1f}s，等待 Redis 任务…",
        flush=True,
    )

    idle = 0
    while True:
        try:
            task = pop_task(timeout=5, cfg=cfg)
        except Exception as exc:
            idle += 1
            if idle <= 3 or idle % 12 == 0:
                print(f"[worker {tag}] Redis 读取失败: {exc}", flush=True)
            time.sleep(2)
            continue
        if not task:
            continue
        idle = 0
        job_id = task.get("job_id") or ""
        image_path = Path(task.get("image_path") or "")
        print(f"[worker {tag}] {job_id} {image_path.name}", flush=True)
        pg_job = None
        try:
            mark_job_running(job_id, cfg=cfg)
            if use_pg:
                try:
                    pg_job = pg.get_job(job_id, cfg)
                    if pg_job and pg_job.get("status") == "cancelled":
                        print(f"[worker {tag}] 跳过已取消任务 {job_id}", flush=True)
                        continue
                    claimed = pg.claim_image(job_id, str(image_path), worker_id, cfg)
                    if not claimed:
                        print(f"[worker {tag}] 未抢到租约，跳过 {image_path.name}", flush=True)
                        continue
                    pg_job = pg.get_job(job_id, cfg) or pg_job
                except Exception as exc:
                    print(f"[worker {tag}] Postgres 更新失败（继续 Redis）：{exc}", flush=True)
            redis_job = job_status(job_id, cfg=cfg)
            job = pg_job or redis_job
            if not image_path.exists():
                raise FileNotFoundError(str(image_path))
            image = Image.open(image_path).convert("RGB")
            classes = job.get("classes") or []
            if isinstance(classes, str):
                import json

                classes = json.loads(classes)
            caption = job.get("prompt") or to_caption(classes)
            if use_pg:
                try:
                    pg.heartbeat_image(job_id, str(image_path), worker_id, cfg)
                except Exception:
                    pass
            t_inf = time.perf_counter()
            detections = detector.detect(
                image,
                caption=caption,
                classes=classes,
                box_threshold=float(job.get("box_threshold") or 0.35),
                text_threshold=float(job.get("text_threshold") or 0.25),
                nms_iou=float(job.get("nms_iou") or 0.5),
            )
            infer_s = time.perf_counter() - t_inf
            if use_pg:
                try:
                    if pg.job_is_cancelled(job_id, cfg):
                        print(f"[worker {tag}] 推理后发现已取消，丢弃 {image_path.name}", flush=True)
                        continue
                    pg.heartbeat_image(job_id, str(image_path), worker_id, cfg)
                except Exception:
                    pass
            record = build_record(
                image_path,
                image,
                detections,
                classes,
                caption,
                model_id=str(detector.model_id),
                box_threshold=float(job.get("box_threshold") or 0.35),
                text_threshold=float(job.get("text_threshold") or 0.25),
                nms_iou=float(job.get("nms_iou") or 0.5),
            )
            ann_dir, prev_dir = _job_dirs(pg_job, dirs)
            ann_path = save_record(record, ann_dir, image_path.stem)
            preview_path = None
            if job.get("save_preview"):
                prev_dir.mkdir(parents=True, exist_ok=True)
                vis = draw_detections(image, detections)
                preview_path = prev_dir / f"{image_path.stem}_vis.jpg"
                vis.save(preview_path, quality=92)
            complete_task(job_id, str(image_path), ok=True, cfg=cfg)
            if use_pg:
                try:
                    pg.complete_image(
                        job_id,
                        str(image_path),
                        ok=True,
                        box_count=len(detections),
                        ann_path=str(ann_path),
                        preview_path=str(preview_path) if preview_path else None,
                        cfg=cfg,
                    )
                except Exception as exc:
                    print(f"[worker {tag}] Postgres 完成失败：{exc}", flush=True)
            print(
                f"[worker {tag}] 完成 {image_path.name}  "
                f"{len(detections)} 框  {infer_s:.2f}s",
                flush=True,
            )
        except Exception as exc:
            print(f"[worker {tag}] 失败 {image_path}: {exc}", flush=True)
            try:
                complete_task(job_id, str(image_path), ok=False, error=str(exc), cfg=cfg)
            except Exception:
                pass
            if use_pg:
                try:
                    pg.complete_image(
                        job_id, str(image_path), ok=False, error=str(exc), cfg=cfg
                    )
                except Exception:
                    pass
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
