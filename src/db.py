"""PostgreSQL: users, jobs, per-image progress. Images stay on disk."""

from __future__ import annotations

import hashlib
import re
import secrets
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from .config import ROOT, load_config, resolve_path

PBKDF2_ROUNDS = 200_000
USERNAME_RE = re.compile(r"^[a-zA-Z0-9_]{3,32}$")


class PostgresDisabled(RuntimeError):
    pass


class AuthError(RuntimeError):
    pass


def postgres_cfg(cfg: dict | None = None) -> dict[str, Any]:
    cfg = cfg if cfg is not None else load_config()
    return dict(cfg.get("postgres") or {})


def postgres_enabled(cfg: dict | None = None) -> bool:
    return bool(postgres_cfg(cfg).get("enabled"))


def connect(cfg: dict | None = None):
    pcfg = postgres_cfg(cfg)
    if not pcfg.get("enabled"):
        raise PostgresDisabled("config.yaml 里 postgres.enabled 为 false")
    try:
        import psycopg
        from psycopg.rows import dict_row
    except ImportError as exc:
        raise PostgresDisabled("未安装 psycopg：pip install 'psycopg[binary]'") from exc
    kwargs: dict[str, Any] = {
        "host": str(pcfg.get("host", "127.0.0.1")),
        "port": int(pcfg.get("port", 5432)),
        "dbname": str(pcfg.get("db", "gdino")),
        "user": str(pcfg.get("user", "gdino")),
        "autocommit": True,
        "row_factory": dict_row,
    }
    password = pcfg.get("password")
    if password not in (None, ""):
        kwargs["password"] = str(password)
    try:
        return psycopg.connect(**kwargs)
    except Exception as exc:
        raise PostgresDisabled(f"PostgreSQL 连不上：{exc}") from exc


def hash_password(password: str) -> str:
    salt = secrets.token_hex(16)
    dk = hashlib.pbkdf2_hmac(
        "sha256", password.encode("utf-8"), salt.encode("utf-8"), PBKDF2_ROUNDS
    )
    return f"pbkdf2_sha256${PBKDF2_ROUNDS}${salt}${dk.hex()}"


def verify_password(password: str, stored: str) -> bool:
    try:
        _algo, rounds, salt, digest = stored.split("$", 3)
        dk = hashlib.pbkdf2_hmac(
            "sha256", password.encode("utf-8"), salt.encode("utf-8"), int(rounds)
        )
        return secrets.compare_digest(dk.hex(), digest)
    except Exception:
        return False


def _safe_username(name: str) -> str:
    name = (name or "").strip()
    if not USERNAME_RE.match(name):
        raise AuthError("用户名须为 3–32 位字母、数字或下划线")
    return name


SCHEMA = """
CREATE TABLE IF NOT EXISTS users (
    id SERIAL PRIMARY KEY,
    username TEXT UNIQUE NOT NULL,
    password_hash TEXT NOT NULL,
    role TEXT NOT NULL DEFAULT 'user',
    disabled BOOLEAN NOT NULL DEFAULT FALSE,
    created_at TIMESTAMPTZ NOT NULL DEFAULT NOW()
);
CREATE TABLE IF NOT EXISTS jobs (
    id TEXT PRIMARY KEY,
    user_id INTEGER NOT NULL REFERENCES users(id) ON DELETE CASCADE,
    status TEXT NOT NULL DEFAULT 'queued',
    prompt TEXT NOT NULL DEFAULT '',
    classes JSONB NOT NULL DEFAULT '[]'::jsonb,
    box_threshold DOUBLE PRECISION,
    text_threshold DOUBLE PRECISION,
    nms_iou DOUBLE PRECISION,
    save_preview BOOLEAN NOT NULL DEFAULT TRUE,
    model TEXT,
    images_dir TEXT,
    annotations_dir TEXT,
    previews_dir TEXT,
    exports_dir TEXT,
    total INTEGER NOT NULL DEFAULT 0,
    done INTEGER NOT NULL DEFAULT 0,
    failed INTEGER NOT NULL DEFAULT 0,
    last_error TEXT,
    last_image TEXT,
    created_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    updated_at TIMESTAMPTZ NOT NULL DEFAULT NOW()
);
CREATE TABLE IF NOT EXISTS job_images (
    id SERIAL PRIMARY KEY,
    job_id TEXT NOT NULL REFERENCES jobs(id) ON DELETE CASCADE,
    orig_name TEXT,
    image_path TEXT NOT NULL,
    ann_path TEXT,
    preview_path TEXT,
    status TEXT NOT NULL DEFAULT 'queued',
    error TEXT,
    box_count INTEGER NOT NULL DEFAULT 0,
    finished_at TIMESTAMPTZ,
    locked_by TEXT,
    locked_at TIMESTAMPTZ,
    heartbeat_at TIMESTAMPTZ
);
CREATE INDEX IF NOT EXISTS job_images_job_id_idx ON job_images (job_id);
CREATE UNIQUE INDEX IF NOT EXISTS job_images_job_path_uidx ON job_images (job_id, image_path);
CREATE TABLE IF NOT EXISTS exports (
    id SERIAL PRIMARY KEY,
    job_id TEXT NOT NULL REFERENCES jobs(id) ON DELETE CASCADE,
    format TEXT NOT NULL,
    path TEXT NOT NULL,
    created_at TIMESTAMPTZ NOT NULL DEFAULT NOW()
);
"""

MIGRATIONS = [
    "ALTER TABLE users ADD COLUMN IF NOT EXISTS disabled BOOLEAN NOT NULL DEFAULT FALSE",
    "ALTER TABLE job_images ADD COLUMN IF NOT EXISTS locked_by TEXT",
    "ALTER TABLE job_images ADD COLUMN IF NOT EXISTS locked_at TIMESTAMPTZ",
    "ALTER TABLE job_images ADD COLUMN IF NOT EXISTS heartbeat_at TIMESTAMPTZ",
    "CREATE UNIQUE INDEX IF NOT EXISTS job_images_job_path_uidx ON job_images (job_id, image_path)",
]


def lease_seconds(cfg: dict | None = None) -> int:
    cfg = cfg if cfg is not None else load_config()
    return max(30, int((cfg.get("workers") or {}).get("lease_seconds") or 180))


def init_db(cfg: dict | None = None) -> str:
    cfg = cfg if cfg is not None else load_config()
    with connect(cfg) as conn:
        conn.execute(SCHEMA)
        for sql in MIGRATIONS:
            conn.execute(sql)
        pcfg = postgres_cfg(cfg)
        admin_name = _safe_username(str(pcfg.get("admin_username") or "admin"))
        admin_pass = str(pcfg.get("admin_password") or "CHANGE_ME")
        row = conn.execute(
            "SELECT id FROM users WHERE username = %s", (admin_name,)
        ).fetchone()
        if row:
            return f"PostgreSQL OK，管理员账号已存在：{admin_name}"
        conn.execute(
            "INSERT INTO users (username, password_hash, role) VALUES (%s, %s, 'admin')",
            (admin_name, hash_password(admin_pass)),
        )
        return f"PostgreSQL OK，已创建默认管理员：{admin_name} / {admin_pass}（请尽快改密）"


def ping(cfg: dict | None = None) -> str:
    pcfg = postgres_cfg(cfg)
    with connect(cfg) as conn:
        n = conn.execute("SELECT COUNT(*) AS n FROM users").fetchone()["n"]
        j = conn.execute("SELECT COUNT(*) AS n FROM jobs").fetchone()["n"]
    return (
        f"PostgreSQL OK  {pcfg.get('host')}:{pcfg.get('port')}/{pcfg.get('db')}  "
        f"users={n} jobs={j}"
    )


def get_user_by_name(username: str, cfg: dict | None = None) -> dict | None:
    with connect(cfg) as conn:
        return conn.execute(
            """
            SELECT id, username, password_hash, role, disabled, created_at
            FROM users WHERE username = %s
            """,
            (username.strip(),),
        ).fetchone()


def login(username: str, password: str, cfg: dict | None = None) -> dict[str, Any]:
    user = get_user_by_name(username, cfg)
    if not user or not verify_password(password, user["password_hash"]):
        raise AuthError("用户名或密码错误")
    if user.get("disabled"):
        raise AuthError("账号已停用")
    return public_user(user)


def public_user(user: dict) -> dict[str, Any]:
    return {
        "id": int(user["id"]),
        "username": user["username"],
        "role": user["role"],
        "disabled": bool(user.get("disabled")),
    }


def is_admin(user: dict | None) -> bool:
    return bool(user) and user.get("role") == "admin"


def create_user(
    actor: dict | None,
    username: str,
    password: str,
    role: str = "user",
    cfg: dict | None = None,
) -> dict[str, Any]:
    if not is_admin(actor):
        raise AuthError("只有管理员可以创建用户")
    username = _safe_username(username)
    if not password or len(password) < 8:
        raise AuthError("密码至少 8 位")
    role = (role or "user").strip().lower()
    if role not in ("user", "admin"):
        raise AuthError("角色只能是 user 或 admin")
    with connect(cfg) as conn:
        exists = conn.execute(
            "SELECT id FROM users WHERE username = %s", (username,)
        ).fetchone()
        if exists:
            raise AuthError(f"用户已存在：{username}")
        row = conn.execute(
            """
            INSERT INTO users (username, password_hash, role)
            VALUES (%s, %s, %s)
            RETURNING id, username, role, created_at
            """,
            (username, hash_password(password), role),
        ).fetchone()
    return dict(row)


def list_users(actor: dict | None, cfg: dict | None = None) -> list[dict]:
    if not is_admin(actor):
        raise AuthError("只有管理员可以查看用户列表")
    with connect(cfg) as conn:
        rows = conn.execute(
            """
            SELECT u.id, u.username, u.role, u.disabled, u.created_at,
                   COUNT(j.id) AS job_count
            FROM users u
            LEFT JOIN jobs j ON j.user_id = u.id
            GROUP BY u.id
            ORDER BY u.id
            """
        ).fetchall()
    return [dict(r) for r in rows]


def change_password(
    user: dict | None,
    old_password: str,
    new_password: str,
    cfg: dict | None = None,
) -> None:
    if not user or not user.get("id"):
        raise AuthError("请先登录")
    if not new_password or len(new_password) < 8:
        raise AuthError("新密码至少 8 位")
    row = None
    with connect(cfg) as conn:
        row = conn.execute(
            "SELECT id, password_hash FROM users WHERE id = %s",
            (int(user["id"]),),
        ).fetchone()
    if not row or not verify_password(old_password or "", row["password_hash"]):
        raise AuthError("原密码错误")
    with connect(cfg) as conn:
        conn.execute(
            "UPDATE users SET password_hash = %s WHERE id = %s",
            (hash_password(new_password), int(user["id"])),
        )


def set_user_disabled(
    actor: dict | None,
    username: str,
    disabled: bool,
    cfg: dict | None = None,
) -> dict[str, Any]:
    if not is_admin(actor):
        raise AuthError("只有管理员可以停用/启用用户")
    username = _safe_username(username)
    if actor and actor.get("username") == username and disabled:
        raise AuthError("不能停用当前登录的管理员")
    with connect(cfg) as conn:
        row = conn.execute(
            """
            UPDATE users SET disabled = %s WHERE username = %s
            RETURNING id, username, role, disabled, created_at
            """,
            (bool(disabled), username),
        ).fetchone()
    if not row:
        raise AuthError(f"用户不存在：{username}")
    return dict(row)


def reset_password(
    actor: dict | None,
    username: str,
    new_password: str,
    cfg: dict | None = None,
) -> None:
    if not is_admin(actor):
        raise AuthError("只有管理员可以重置密码")
    if not new_password or len(new_password) < 8:
        raise AuthError("新密码至少 8 位")
    username = _safe_username(username)
    with connect(cfg) as conn:
        cur = conn.execute(
            "UPDATE users SET password_hash = %s WHERE username = %s",
            (hash_password(new_password), username),
        )
        if not cur.rowcount:
            raise AuthError(f"用户不存在：{username}")


def job_root(username: str, job_id: str, cfg: dict | None = None) -> Path:
    cfg = cfg if cfg is not None else load_config()
    base = resolve_path((cfg.get("paths") or {}).get("users", "data/users"), ROOT)
    return base / username / "jobs" / job_id


def job_dirs_for(username: str, job_id: str, cfg: dict | None = None) -> dict[str, Path]:
    root = job_root(username, job_id, cfg)
    dirs = {
        "root": root,
        "images": root / "images",
        "annotations": root / "annotations",
        "previews": root / "previews",
        "exports": root / "exports",
    }
    for p in dirs.values():
        p.mkdir(parents=True, exist_ok=True)
    return dirs


def create_job(
    user: dict,
    *,
    job_id: str,
    prompt: str,
    classes: list[str],
    box_threshold: float,
    text_threshold: float,
    nms_iou: float,
    save_preview: bool,
    model: str,
    image_rows: list[dict[str, str]],
    cfg: dict | None = None,
) -> dict[str, Any]:
    if not user or not user.get("id"):
        raise AuthError("请先登录")
    dirs = job_dirs_for(user["username"], job_id, cfg)
    from psycopg.types.json import Json

    with connect(cfg) as conn:
        conn.execute(
            """
            INSERT INTO jobs (
                id, user_id, status, prompt, classes,
                box_threshold, text_threshold, nms_iou, save_preview, model,
                images_dir, annotations_dir, previews_dir, exports_dir,
                total, done, failed
            ) VALUES (
                %s, %s, 'queued', %s, %s,
                %s, %s, %s, %s, %s,
                %s, %s, %s, %s,
                %s, 0, 0
            )
            """,
            (
                job_id,
                int(user["id"]),
                prompt,
                Json(list(classes)),
                box_threshold,
                text_threshold,
                nms_iou,
                bool(save_preview),
                model,
                str(dirs["images"]),
                str(dirs["annotations"]),
                str(dirs["previews"]),
                str(dirs["exports"]),
                len(image_rows),
            ),
        )
        for row in image_rows:
            conn.execute(
                """
                INSERT INTO job_images (job_id, orig_name, image_path, status)
                VALUES (%s, %s, %s, 'queued')
                """,
                (job_id, row.get("orig_name") or "", row["image_path"]),
            )
    return get_job(job_id, cfg)


def get_job(job_id: str, cfg: dict | None = None) -> dict[str, Any] | None:
    if not job_id:
        return None
    with connect(cfg) as conn:
        row = conn.execute(
            """
            SELECT j.*, u.username
            FROM jobs j JOIN users u ON u.id = j.user_id
            WHERE j.id = %s
            """,
            (job_id,),
        ).fetchone()
    return dict(row) if row else None


def can_access_job(user: dict | None, job: dict | None) -> bool:
    if not user or not job:
        return False
    if is_admin(user):
        return True
    return int(job["user_id"]) == int(user["id"])


def list_jobs(user: dict | None, cfg: dict | None = None) -> list[dict]:
    if not user:
        return []
    with connect(cfg) as conn:
        if is_admin(user):
            rows = conn.execute(
                """
                SELECT j.id, j.status, j.total, j.done, j.failed, j.prompt,
                       j.created_at, j.updated_at, u.username
                FROM jobs j JOIN users u ON u.id = j.user_id
                ORDER BY j.created_at DESC
                LIMIT 200
                """
            ).fetchall()
        else:
            rows = conn.execute(
                """
                SELECT j.id, j.status, j.total, j.done, j.failed, j.prompt,
                       j.created_at, j.updated_at, u.username
                FROM jobs j JOIN users u ON u.id = j.user_id
                WHERE j.user_id = %s
                ORDER BY j.created_at DESC
                LIMIT 200
                """,
                (int(user["id"]),),
            ).fetchall()
    return [dict(r) for r in rows]


def list_job_images(job_id: str, cfg: dict | None = None) -> list[dict]:
    with connect(cfg) as conn:
        rows = conn.execute(
            """
            SELECT id, job_id, orig_name, image_path, ann_path, preview_path,
                   status, error, box_count, finished_at
            FROM job_images
            WHERE job_id = %s
            ORDER BY id
            """,
            (job_id,),
        ).fetchall()
    return [dict(r) for r in rows]


def mark_job_running(job_id: str, cfg: dict | None = None) -> None:
    with connect(cfg) as conn:
        conn.execute(
            """
            UPDATE jobs SET status = 'running', updated_at = NOW()
            WHERE id = %s AND status IN ('queued', 'running')
            """,
            (job_id,),
        )


def job_is_cancelled(job_id: str, cfg: dict | None = None) -> bool:
    job = get_job(job_id, cfg)
    return bool(job) and job.get("status") == "cancelled"


def claim_image(
    job_id: str,
    image_path: str,
    worker_id: str,
    cfg: dict | None = None,
) -> bool:
    """Take a lease. False if cancelled, already done, or held by a live worker."""
    ttl = lease_seconds(cfg)
    with connect(cfg) as conn:
        job = conn.execute("SELECT status FROM jobs WHERE id = %s", (job_id,)).fetchone()
        if not job or job["status"] == "cancelled":
            return False
        row = conn.execute(
            """
            UPDATE job_images
            SET status = 'running',
                locked_by = %s,
                locked_at = NOW(),
                heartbeat_at = NOW(),
                error = ''
            WHERE job_id = %s AND image_path = %s
              AND status IN ('queued', 'running')
              AND (
                    locked_by IS NULL
                    OR locked_by = %s
                    OR heartbeat_at IS NULL
                    OR heartbeat_at < NOW() - make_interval(secs => %s)
              )
            RETURNING id
            """,
            (worker_id, job_id, image_path, worker_id, ttl),
        ).fetchone()
        if not row:
            return False
        conn.execute(
            """
            UPDATE jobs SET status = 'running', updated_at = NOW()
            WHERE id = %s AND status IN ('queued', 'running')
            """,
            (job_id,),
        )
        return True


def heartbeat_image(
    job_id: str,
    image_path: str,
    worker_id: str,
    cfg: dict | None = None,
) -> None:
    with connect(cfg) as conn:
        conn.execute(
            """
            UPDATE job_images
            SET heartbeat_at = NOW()
            WHERE job_id = %s AND image_path = %s AND locked_by = %s AND status = 'running'
            """,
            (job_id, image_path, worker_id),
        )


def mark_image_running(job_id: str, image_path: str, cfg: dict | None = None) -> None:
    claim_image(job_id, image_path, "legacy", cfg)


def reclaim_interrupted(cfg: dict | None = None) -> int:
    """Only reclaim running images whose lease has expired."""
    ttl = lease_seconds(cfg)
    with connect(cfg) as conn:
        cur = conn.execute(
            """
            UPDATE job_images
            SET status = 'queued',
                error = 'lease expired, requeued',
                locked_by = NULL,
                locked_at = NULL,
                heartbeat_at = NULL
            WHERE status = 'running'
              AND job_id IN (SELECT id FROM jobs WHERE status IN ('queued', 'running'))
              AND (
                    heartbeat_at IS NULL
                    OR heartbeat_at < NOW() - make_interval(secs => %s)
              )
            """,
            (ttl,),
        )
        n = int(cur.rowcount or 0)
        conn.execute(
            """
            UPDATE jobs
            SET status = 'queued', updated_at = NOW()
            WHERE status = 'running'
              AND EXISTS (
                  SELECT 1 FROM job_images i
                  WHERE i.job_id = jobs.id AND i.status = 'queued'
              )
              AND NOT EXISTS (
                  SELECT 1 FROM job_images i
                  WHERE i.job_id = jobs.id AND i.status = 'running'
              )
            """
        )
        return n


def list_unfinished_jobs(cfg: dict | None = None) -> list[dict]:
    with connect(cfg) as conn:
        rows = conn.execute(
            """
            SELECT j.*, u.username
            FROM jobs j JOIN users u ON u.id = j.user_id
            WHERE j.status IN ('queued', 'running')
              AND EXISTS (
                SELECT 1 FROM job_images i
                WHERE i.job_id = j.id AND i.status IN ('queued', 'running')
            )
            ORDER BY j.created_at
            """
        ).fetchall()
    return [dict(r) for r in rows]


def list_unfinished_images(job_id: str, cfg: dict | None = None) -> list[dict]:
    """Queued only. Expired running rows are reclaimed to queued before resume."""
    with connect(cfg) as conn:
        rows = conn.execute(
            """
            SELECT id, job_id, orig_name, image_path, status, error
            FROM job_images
            WHERE job_id = %s AND status = 'queued'
            ORDER BY id
            """,
            (job_id,),
        ).fetchall()
    return [dict(r) for r in rows]


def _apply_job_counts(conn, job_id: str, last_image: str = "", last_error: str = "") -> None:
    counts = conn.execute(
        """
        SELECT
            COUNT(*) FILTER (WHERE status = 'done') AS done,
            COUNT(*) FILTER (WHERE status = 'failed') AS failed,
            COUNT(*) FILTER (WHERE status = 'cancelled') AS cancelled,
            COUNT(*) FILTER (WHERE status IN ('queued', 'running')) AS pending,
            COUNT(*) AS total
        FROM job_images WHERE job_id = %s
        """,
        (job_id,),
    ).fetchone()
    done = int(counts["done"] or 0)
    failed = int(counts["failed"] or 0)
    pending = int(counts["pending"] or 0)
    total = int(counts["total"] or 0)
    job = conn.execute("SELECT status FROM jobs WHERE id = %s", (job_id,)).fetchone()
    current = (job or {}).get("status") or ""
    running = conn.execute(
        "SELECT 1 FROM job_images WHERE job_id = %s AND status = 'running'",
        (job_id,),
    ).fetchone()
    if current == "cancelled":
        job_status = "cancelled"
    elif pending:
        job_status = "running" if running or done or failed else "queued"
    elif total and done + failed >= total:
        job_status = "done" if failed == 0 else "done_with_errors"
    else:
        job_status = current or "queued"
    conn.execute(
        """
        UPDATE jobs
        SET done = %s, failed = %s, total = %s, status = %s,
            last_image = COALESCE(NULLIF(%s, ''), last_image),
            last_error = COALESCE(%s, last_error),
            updated_at = NOW()
        WHERE id = %s
        """,
        (done, failed, total, job_status, last_image, last_error[:500] if last_error else "", job_id),
    )


def complete_image(
    job_id: str,
    image_path: str,
    *,
    ok: bool,
    error: str = "",
    box_count: int = 0,
    ann_path: str | None = None,
    preview_path: str | None = None,
    cfg: dict | None = None,
) -> dict[str, Any]:
    status = "done" if ok else "failed"
    with connect(cfg) as conn:
        job = conn.execute("SELECT status FROM jobs WHERE id = %s", (job_id,)).fetchone()
        if job and job["status"] == "cancelled":
            conn.execute(
                """
                UPDATE job_images
                SET status = 'cancelled',
                    error = COALESCE(NULLIF(%s, ''), error),
                    locked_by = NULL,
                    locked_at = NULL,
                    heartbeat_at = NULL,
                    finished_at = NOW()
                WHERE job_id = %s AND image_path = %s AND status IN ('queued', 'running')
                """,
                ((error or "")[:500], job_id, image_path),
            )
            return get_job(job_id, cfg) or {}
        conn.execute(
            """
            UPDATE job_images
            SET status = %s, error = %s, box_count = %s,
                ann_path = COALESCE(%s, ann_path),
                preview_path = COALESCE(%s, preview_path),
                finished_at = NOW(),
                locked_by = NULL,
                locked_at = NULL,
                heartbeat_at = NULL
            WHERE job_id = %s AND image_path = %s AND status <> 'cancelled'
            """,
            (status, (error or "")[:500], int(box_count), ann_path, preview_path, job_id, image_path),
        )
        _apply_job_counts(conn, job_id, image_path, error or "")
    return get_job(job_id, cfg) or {}


def cancel_job(user: dict | None, job_id: str, cfg: dict | None = None) -> dict[str, Any]:
    job = get_job(job_id, cfg)
    if not job:
        raise AuthError("任务不存在")
    if user and not can_access_job(user, job):
        raise AuthError("无权操作该任务")
    if job.get("status") in ("done", "done_with_errors", "cancelled"):
        return job
    with connect(cfg) as conn:
        conn.execute(
            """
            UPDATE job_images
            SET status = 'cancelled',
                error = 'cancelled',
                locked_by = NULL,
                locked_at = NULL,
                heartbeat_at = NULL,
                finished_at = NOW()
            WHERE job_id = %s AND status IN ('queued', 'running')
            """,
            (job_id,),
        )
        conn.execute(
            """
            UPDATE jobs
            SET status = 'cancelled', last_error = 'cancelled', updated_at = NOW()
            WHERE id = %s
            """,
            (job_id,),
        )
        _apply_job_counts(conn, job_id, last_error="cancelled")
    return get_job(job_id, cfg) or {}


def retry_failed(user: dict | None, job_id: str, cfg: dict | None = None) -> list[str]:
    job = get_job(job_id, cfg)
    if not job:
        raise AuthError("任务不存在")
    if user and not can_access_job(user, job):
        raise AuthError("无权操作该任务")
    with connect(cfg) as conn:
        rows = conn.execute(
            """
            UPDATE job_images
            SET status = 'queued',
                error = '',
                finished_at = NULL,
                locked_by = NULL,
                locked_at = NULL,
                heartbeat_at = NULL
            WHERE job_id = %s AND status = 'failed'
            RETURNING image_path
            """,
            (job_id,),
        ).fetchall()
        if job.get("status") == "cancelled":
            conn.execute(
                """
                UPDATE jobs SET status = 'queued', last_error = '', updated_at = NOW()
                WHERE id = %s
                """,
                (job_id,),
            )
        _apply_job_counts(conn, job_id)
    return [r["image_path"] for r in rows]


def done_annotation_paths(job_id: str, cfg: dict | None = None) -> list[str]:
    with connect(cfg) as conn:
        rows = conn.execute(
            """
            SELECT ann_path FROM job_images
            WHERE job_id = %s AND status = 'done' AND ann_path IS NOT NULL AND ann_path <> ''
            ORDER BY id
            """,
            (job_id,),
        ).fetchall()
    return [r["ann_path"] for r in rows if r.get("ann_path")]


def add_export(job_id: str, fmt: str, path: str, cfg: dict | None = None) -> None:
    with connect(cfg) as conn:
        conn.execute(
            "INSERT INTO exports (job_id, format, path) VALUES (%s, %s, %s)",
            (job_id, fmt, path),
        )


def format_job(job: dict | None) -> str:
    if not job or not job.get("id"):
        return "没有任务。"
    prompt = (job.get("prompt") or "").replace("\n", " ")
    if len(prompt) > 80:
        prompt = prompt[:77] + "..."
    created = job.get("created_at")
    if isinstance(created, datetime):
        created = created.astimezone(timezone.utc).strftime("%Y-%m-%d %H:%M")
    return (
        f"任务 {job['id']}\n"
        f"用户：{job.get('username') or '-'}  状态：{job.get('status')}\n"
        f"进度：{int(job.get('done') or 0) + int(job.get('failed') or 0)}/"
        f"{int(job.get('total') or 0)}"
        f"（完成 {job.get('done') or 0}，失败 {job.get('failed') or 0}）\n"
        f"未完成会在刷新或重启后续跑；已取消的不会再入队。\n"
        f"提示词：{prompt}\n"
        f"创建：{created or '-'}\n"
        f"最近图片：{job.get('last_image') or '-'}\n"
        f"最近错误：{job.get('last_error') or '-'}"
    )


def jobs_table(rows: list[dict]) -> list[list]:
    out = []
    for r in rows:
        created = r.get("created_at")
        if isinstance(created, datetime):
            created = created.strftime("%Y-%m-%d %H:%M")
        prompt = (r.get("prompt") or "").replace("\n", " ")
        if len(prompt) > 40:
            prompt = prompt[:37] + "..."
        out.append(
            [
                r.get("id"),
                r.get("username"),
                r.get("status"),
                f"{r.get('done')}/{r.get('total')}",
                r.get("failed"),
                prompt,
                created,
            ]
        )
    return out
