"""Spawn N Grounding DINO replicas on one GPU."""

from __future__ import annotations

import subprocess
import sys
import time
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parent.parent


def worker_gpu_id(cfg: dict[str, Any]) -> int:
    workers_cfg = cfg.get("workers") or {}
    model_cfg = cfg.get("model") or {}
    return int(workers_cfg.get("gpu_id", model_cfg.get("gpu_id", 0)) or 0)


def replica_count(cfg: dict[str, Any]) -> int:
    workers_cfg = cfg.get("workers") or {}
    return max(1, int(workers_cfg.get("replicas", 1) or 1))


def spawn_replicas(
    cfg: dict[str, Any],
    *,
    stagger: float = 2.5,
) -> list[subprocess.Popen]:
    gpu_id = worker_gpu_id(cfg)
    replicas = replica_count(cfg)
    python = sys.executable
    worker = str(ROOT / "worker.py")
    procs: list[subprocess.Popen] = []
    print(f"在 cuda:{gpu_id} 上启动 {replicas} 份模型", flush=True)
    for i in range(replicas):
        p = subprocess.Popen(
            [
                python,
                worker,
                "--gpu-id",
                str(gpu_id),
                "--replica-id",
                str(i),
                "--device",
                "cuda",
            ],
            cwd=str(ROOT),
        )
        procs.append(p)
        print(f"  replica {i}  pid={p.pid}  cuda:{gpu_id}", flush=True)
        if i + 1 < replicas and stagger > 0:
            time.sleep(stagger)
    return procs


def stop_replicas(procs: list[subprocess.Popen], timeout: float = 8.0) -> None:
    for p in procs:
        if p.poll() is None:
            p.terminate()
    deadline = time.time() + timeout
    for p in procs:
        remaining = max(0.1, deadline - time.time())
        try:
            p.wait(timeout=remaining)
        except subprocess.TimeoutExpired:
            p.kill()
