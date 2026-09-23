"""Launch N Grounding DINO replicas on one GPU (config workers.replicas)."""

from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.config import load_config
from src.queue import ping, redis_enabled, resume_unfinished
from src.spawn import replica_count, spawn_replicas, stop_replicas, worker_gpu_id


def main() -> int:
    cfg = load_config()
    if not redis_enabled(cfg):
        print("config.yaml 里 redis.enabled 不是 true，无法启动 worker。")
        return 2
    try:
        print(ping(cfg))
    except Exception as exc:
        print(f"Redis 连不上：{exc}")
        return 2

    try:
        print(resume_unfinished(cfg), flush=True)
    except Exception as exc:
        print(f"续跑未完成任务失败：{exc}", flush=True)

    gpu_id = worker_gpu_id(cfg)
    replicas = replica_count(cfg)
    procs = spawn_replicas(cfg)
    print(f"全部 {replicas} 份 replica 已拉起（cuda:{gpu_id}），Ctrl+C 结束。", flush=True)
    try:
        for p in procs:
            p.wait()
    except KeyboardInterrupt:
        print("正在停止 worker…", flush=True)
        stop_replicas(procs)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
