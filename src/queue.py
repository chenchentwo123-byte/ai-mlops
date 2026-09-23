"""Redis job queue for same-GPU Grounding DINO replicas."""

from __future__ import annotations

import json
import time
import uuid
from typing import Any

from .config import load_config

TASK_TTL = 7 * 24 * 3600


class RedisDisabled(RuntimeError):
    pass


def redis_cfg(cfg: dict | None = None) -> dict[str, Any]:
    cfg = cfg if cfg is not None else load_config()
    return dict(cfg.get("redis") or {})


def redis_enabled(cfg: dict | None = None) -> bool:
    return bool(redis_cfg(cfg).get("enabled"))


def connect(cfg: dict | None = None):
    rcfg = redis_cfg(cfg)
    if not rcfg.get("enabled"):
        raise RedisDisabled("config.yaml 里 redis.enabled 为 false")
    try:
        import redis
    except ImportError as exc:
        raise RedisDisabled("未安装 redis：pip install redis") from exc
    connect_s = float(rcfg.get("socket_timeout", 5) or 5)
    client = redis.Redis(
        host=str(rcfg.get("host", "127.0.0.1")),
        port=int(rcfg.get("port", 6379)),
        db=int(rcfg.get("db", 0)),
        password=rcfg.get("password") or None,
        socket_connect_timeout=connect_s,
        # BRPOP 自己有超时；socket_timeout 不能 ≤ BRPOP 秒数，否则空队列会被当成读失败。
        socket_timeout=max(30.0, connect_s + 25.0),
        decode_responses=True,
    )
    client.ping()
    return client


def prefix(cfg: dict | None = None) -> str:
    return str(redis_cfg(cfg).get("prefix") or "gdino").rstrip(":")


def _keys(cfg: dict | None = None) -> dict[str, str]:
    p = prefix(cfg)
    return {
        "queue": f"{p}:queue",
        "job": f"{p}:job:",
        "jobs": f"{p}:jobs",
    }


def new_job_id() -> str:
    return time.strftime("%Y%m%d_%H%M%S") + "_" + uuid.uuid4().hex[:8]


def enqueue_job(
    *,
    image_paths: list[str],
    prompt: str,
    classes: list[str],
    box_threshold: float,
    text_threshold: float,
    nms_iou: float,
    save_preview: bool,
    job_id: str | None = None,
    cfg: dict | None = None,
) -> dict[str, Any]:
    client = connect(cfg)
    keys = _keys(cfg)
    job_id = job_id or new_job_id()
    job_key = keys["job"] + job_id
    payload = {
        "id": job_id,
        "status": "queued",
        "prompt": prompt,
        "classes": json.dumps(classes, ensure_ascii=False),
        "box_threshold": str(box_threshold),
        "text_threshold": str(text_threshold),
        "nms_iou": str(nms_iou),
        "save_preview": "1" if save_preview else "0",
        "total": str(len(image_paths)),
        "done": "0",
        "failed": "0",
        "created_at": str(int(time.time())),
        "last_error": "",
        "last_image": "",
        "images": json.dumps(image_paths, ensure_ascii=False),
    }
    pipe = client.pipeline()
    pipe.hset(job_key, mapping=payload)
    pipe.expire(job_key, TASK_TTL)
    pipe.lpush(keys["jobs"], job_id)
    pipe.ltrim(keys["jobs"], 0, 199)
    for path in image_paths:
        task = json.dumps({"job_id": job_id, "image_path": path}, ensure_ascii=False)
        pipe.rpush(keys["queue"], task)
    pipe.execute()
    return job_status(job_id, cfg=cfg)


def queued_pairs(cfg: dict | None = None) -> set[tuple[str, str]]:
    client = connect(cfg)
    items = client.lrange(_keys(cfg)["queue"], 0, -1)
    out: set[tuple[str, str]] = set()
    for raw in items:
        try:
            task = json.loads(raw)
        except (TypeError, json.JSONDecodeError):
            continue
        out.add((str(task.get("job_id") or ""), str(task.get("image_path") or "")))
    return out


def enqueue_remaining(
    *,
    job_id: str,
    image_paths: list[str],
    prompt: str,
    classes: list[str] | str,
    box_threshold: float,
    text_threshold: float,
    nms_iou: float,
    save_preview: bool,
    total: int,
    done: int,
    failed: int,
    cfg: dict | None = None,
) -> int:
    """Re-queue leftover images without resetting Redis done/failed counters."""
    if not image_paths:
        return 0
    if isinstance(classes, str):
        try:
            classes = json.loads(classes)
        except json.JSONDecodeError:
            classes = [classes]
    client = connect(cfg)
    keys = _keys(cfg)
    existing = queued_pairs(cfg)
    to_add = [p for p in image_paths if (job_id, p) not in existing]
    job_key = keys["job"] + job_id
    payload = {
        "id": job_id,
        "status": "queued",
        "prompt": prompt,
        "classes": json.dumps(list(classes or []), ensure_ascii=False),
        "box_threshold": str(box_threshold),
        "text_threshold": str(text_threshold),
        "nms_iou": str(nms_iou),
        "save_preview": "1" if save_preview else "0",
        "total": str(int(total)),
        "done": str(int(done)),
        "failed": str(int(failed)),
        "last_error": "",
        "last_image": "",
    }
    pipe = client.pipeline()
    pipe.hset(job_key, mapping=payload)
    pipe.expire(job_key, TASK_TTL)
    for path in to_add:
        task = json.dumps({"job_id": job_id, "image_path": path}, ensure_ascii=False)
        pipe.rpush(keys["queue"], task)
    pipe.execute()
    return len(to_add)


def drop_job_from_queue(job_id: str, cfg: dict | None = None) -> int:
    """Remove remaining Redis tasks for a cancelled job. In-flight BRPOP is skipped by PG lease."""
    if not job_id:
        return 0
    client = connect(cfg)
    keys = _keys(cfg)
    queue = keys["queue"]
    items = client.lrange(queue, 0, -1)
    dropped = 0
    pipe = client.pipeline()
    for raw in items:
        try:
            task = json.loads(raw)
        except (TypeError, json.JSONDecodeError):
            continue
        if str(task.get("job_id") or "") == job_id:
            pipe.lrem(queue, 1, raw)
            dropped += 1
    job_key = keys["job"] + job_id
    pipe.hset(job_key, "status", "cancelled")
    pipe.execute()
    return dropped


def requeue_job_paths(job: dict, image_paths: list[str], cfg: dict | None = None) -> int:
    if not job or not image_paths:
        return 0
    classes = job.get("classes") or []
    if isinstance(classes, str):
        try:
            classes = json.loads(classes)
        except json.JSONDecodeError:
            classes = []
    return enqueue_remaining(
        job_id=job["id"],
        image_paths=image_paths,
        prompt=job.get("prompt") or "",
        classes=classes,
        box_threshold=float(job.get("box_threshold") or 0.35),
        text_threshold=float(job.get("text_threshold") or 0.25),
        nms_iou=float(job.get("nms_iou") or 0.5),
        save_preview=bool(job.get("save_preview", True)),
        total=int(job.get("total") or 0),
        done=int(job.get("done") or 0),
        failed=int(job.get("failed") or 0),
        cfg=cfg,
    )


def resume_unfinished(cfg: dict | None = None) -> str:
    """After restart: PG leftover queued/running images go back to Redis. Done stays done."""
    from pathlib import Path

    from . import db as pg

    if not pg.postgres_enabled(cfg):
        return "Postgres 未启用，跳过续跑"
    if not redis_enabled(cfg):
        return "Redis 未启用，跳过续跑"
    try:
        connect(cfg).ping()
    except Exception as exc:
        return f"Redis 不可用，无法续跑：{exc}"

    try:
        reclaimed = pg.reclaim_interrupted(cfg)
        jobs = pg.list_unfinished_jobs(cfg)
    except Exception as exc:
        return f"读取未完成任务失败：{exc}"

    if not jobs and reclaimed == 0:
        return "没有未完成的任务需要续跑"

    queued_n = 0
    missing_n = 0
    job_n = 0
    for job in jobs:
        job_id = job["id"]
        leftover = pg.list_unfinished_images(job_id, cfg)
        keep: list[str] = []
        for row in leftover:
            path = row.get("image_path") or ""
            if path and Path(path).is_file():
                keep.append(path)
            elif path:
                try:
                    pg.complete_image(
                        job_id,
                        path,
                        ok=False,
                        error="重启后续跑时文件已不存在",
                        cfg=cfg,
                    )
                except Exception:
                    pass
                missing_n += 1
        if not keep:
            continue
        fresh = pg.get_job(job_id, cfg) or job
        classes = fresh.get("classes") or []
        if isinstance(classes, str):
            try:
                classes = json.loads(classes)
            except json.JSONDecodeError:
                classes = []
        added = enqueue_remaining(
            job_id=job_id,
            image_paths=keep,
            prompt=fresh.get("prompt") or "",
            classes=classes,
            box_threshold=float(fresh.get("box_threshold") or 0.35),
            text_threshold=float(fresh.get("text_threshold") or 0.25),
            nms_iou=float(fresh.get("nms_iou") or 0.5),
            save_preview=bool(fresh.get("save_preview", True)),
            total=int(fresh.get("total") or 0),
            done=int(fresh.get("done") or 0),
            failed=int(fresh.get("failed") or 0),
            cfg=cfg,
        )
        queued_n += added
        job_n += 1
        try:
            pg.mark_job_running(job_id, cfg)
        except Exception:
            pass

    parts = [f"续跑：{job_n} 个任务重新入队 {queued_n} 张"]
    if reclaimed:
        parts.append(f"中断中 {reclaimed} 张改回排队")
    if missing_n:
        parts.append(f"{missing_n} 张文件缺失记失败")
    if queued_n == 0 and missing_n == 0 and reclaimed == 0:
        return "没有未完成的任务需要续跑"
    return "，".join(parts)


def pop_task(timeout: int = 5, cfg: dict | None = None) -> dict[str, Any] | None:
    client = connect(cfg)
    keys = _keys(cfg)
    try:
        item = client.brpop(keys["queue"], timeout=timeout)
    except Exception as exc:
        name = type(exc).__name__
        if "Timeout" in name or "timeout" in str(exc).lower():
            return None
        raise
    if not item:
        return None
    _, raw = item
    return json.loads(raw)


def mark_job_running(job_id: str, cfg: dict | None = None) -> None:
    client = connect(cfg)
    job_key = _keys(cfg)["job"] + job_id
    current = client.hget(job_key, "status")
    if current in ("queued", None, ""):
        client.hset(job_key, "status", "running")


def complete_task(
    job_id: str,
    image_path: str,
    ok: bool,
    error: str = "",
    cfg: dict | None = None,
) -> dict[str, Any]:
    client = connect(cfg)
    job_key = _keys(cfg)["job"] + job_id
    field = "done" if ok else "failed"
    pipe = client.pipeline()
    pipe.hincrby(job_key, field, 1)
    pipe.hset(job_key, mapping={"last_image": image_path, "last_error": error[:500] if error else ""})
    pipe.hgetall(job_key)
    result = pipe.execute()
    data = result[-1] or {}
    total = int(data.get("total") or 0)
    done = int(data.get("done") or 0)
    failed = int(data.get("failed") or 0)
    if total and done + failed >= total:
        client.hset(job_key, "status", "done" if failed == 0 else "done_with_errors")
        data["status"] = "done" if failed == 0 else "done_with_errors"
    elif data.get("status") == "queued":
        client.hset(job_key, "status", "running")
        data["status"] = "running"
    return _normalize(data)


def job_status(job_id: str, cfg: dict | None = None) -> dict[str, Any]:
    client = connect(cfg)
    data = client.hgetall(_keys(cfg)["job"] + job_id) or {}
    return _normalize(data)


def queue_length(cfg: dict | None = None) -> int:
    client = connect(cfg)
    return int(client.llen(_keys(cfg)["queue"]))


def ping(cfg: dict | None = None) -> str:
    client = connect(cfg)
    client.ping()
    qlen = queue_length(cfg)
    return f"Redis OK  {redis_cfg(cfg).get('host')}:{redis_cfg(cfg).get('port')}  queue={qlen}"


def _normalize(data: dict[str, Any]) -> dict[str, Any]:
    if not data:
        return {"id": "", "status": "missing", "total": 0, "done": 0, "failed": 0}
    classes_raw = data.get("classes") or "[]"
    try:
        classes = json.loads(classes_raw) if isinstance(classes_raw, str) else classes_raw
    except json.JSONDecodeError:
        classes = []
    total = int(data.get("total") or 0)
    done = int(data.get("done") or 0)
    failed = int(data.get("failed") or 0)
    return {
        "id": data.get("id") or "",
        "status": data.get("status") or "unknown",
        "prompt": data.get("prompt") or "",
        "classes": classes,
        "box_threshold": float(data.get("box_threshold") or 0.35),
        "text_threshold": float(data.get("text_threshold") or 0.25),
        "nms_iou": float(data.get("nms_iou") or 0.5),
        "save_preview": data.get("save_preview") == "1",
        "total": total,
        "done": done,
        "failed": failed,
        "created_at": data.get("created_at") or "",
        "last_error": data.get("last_error") or "",
        "last_image": data.get("last_image") or "",
        "images": _parse_images(data.get("images")),
        "progress": (done + failed) / total if total else 0.0,
    }


def _parse_images(raw: Any) -> list[str]:
    if not raw:
        return []
    if isinstance(raw, list):
        return [str(p) for p in raw]
    try:
        parsed = json.loads(raw)
    except json.JSONDecodeError:
        return []
    if isinstance(parsed, list):
        return [str(p) for p in parsed]
    return []
