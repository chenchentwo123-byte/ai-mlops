"""Gradio UI: prompt-based Grounding DINO pre-annotation with live visualization."""

from __future__ import annotations

import atexit
import json
import shutil
import socket
import sys
import time
import zipfile
from pathlib import Path

import gradio as gr
from PIL import Image, ImageDraw

ROOT = Path(__file__).resolve().parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src import db as pg
from src.cleanup import cleanup
from src.config import data_dirs, load_config
from src.detector import Detection, GroundingDINODetector, device_choices, list_gpus
from src.exporters import IMAGE_EXTS, export_from_records, export_from_store, list_images
from src.prompts import normalize_prompt
from src.queue import (
    RedisDisabled,
    drop_job_from_queue,
    enqueue_job,
    job_status,
    new_job_id,
    ping as redis_ping,
    queue_length,
    redis_enabled,
    requeue_job_paths,
    resume_unfinished,
)
from src.spawn import replica_count, spawn_replicas, stop_replicas, worker_gpu_id
from src.store import build_record, detections_from_record, list_records, load_record, save_record
from src.visualize import draw_detections
from src.yoloe_detector import (
    MODEL_OPTIONS as YOLOE_MODEL_OPTIONS,
    VisualPrompt,
    YOLOEVisualDetector,
    normalize_model_key,
    parse_visual_classes,
    resolve_weights,
    visual_caption,
)

BROWSE_PAGE = 12
GALLERY_COLS = 6

APP_CSS = """
:root { --gd-radius: 14px; }
html, body {
  width: 100% !important;
  max-width: 100% !important;
  margin: 0 !important;
}
.gradio-container,
.gradio-container.app,
.gradio-container .contain,
.gradio-container .main,
.gradio-container .wrap,
.fillable {
  max-width: 100% !important;
  width: 100% !important;
}
.gradio-container {
  margin: 0 !important;
  padding: 12px 20px 28px !important;
}
footer { display: none !important; }
#app-title h1 { font-size: 1.55rem; letter-spacing: -0.02em; margin: 0; }
#app-title p { color: #64748b; margin: 4px 0 0; }
#top-card, #prompt-card, .gd-card {
  background: #fff;
  border: 1px solid #e2e8f0;
  border-radius: var(--gd-radius);
  padding: 12px 14px;
}
#login-row { align-items: end; }
/* Gradio 目录上传会把每个文件名拉成一长串；只留选择框，列表看右侧缩略图。 */
#upload-box {
  max-height: 150px !important;
  overflow: hidden !important;
}
#upload-box .file-preview,
#upload-box .file-preview-holder,
#upload-box [class*="file-preview"],
#upload-box [class*="FilePreview"],
#upload-box ul,
#upload-box tbody,
#upload-box .file {
  display: none !important;
  height: 0 !important;
  max-height: 0 !important;
  overflow: hidden !important;
  margin: 0 !important;
  padding: 0 !important;
}
#upload-gallery, #result-gallery { border-radius: var(--gd-radius); }
#vis-large img { border-radius: 12px; max-height: 72vh; object-fit: contain; }
#status-box textarea { font-family: ui-monospace, Consolas, monospace; font-size: 12.5px; }
#login-view {
  max-width: 420px;
  margin: 12vh auto 0;
  padding: 28px 28px 32px;
  background: #fff;
  border: 1px solid #e2e8f0;
  border-radius: 16px;
}
#login-view h1 { font-size: 1.45rem; margin: 0 0 6px; }
#workspace-bar { align-items: center; }
"""

CFG = load_config()
MODEL_CFG = CFG.get("model", {})
MODEL_ID = MODEL_CFG.get("id", "models/grounding-dino-tiny")
DEVICE = MODEL_CFG.get("device", "auto")
GPU_ID = MODEL_CFG.get("gpu_id", 0)
CHUNK_SIZE = int(CFG.get("inference", {}).get("chunk_size", 32))
DTYPE = str(MODEL_CFG.get("dtype", "auto"))
LOCAL_ONLY = bool(MODEL_CFG.get("local_files_only", True))
UI_CFG = CFG.get("ui", {})
DIRS = data_dirs(CFG)
CLEAN_CFG = CFG.get("cleanup", {})
YOLOE_CFG = CFG.get("yoloe") or {}
YOLOE_WEIGHTS_DIR = YOLOE_CFG.get("weights_dir", "models/yoloe")
YOLOE_DEFAULT = str(YOLOE_CFG.get("default", "s") or "s")
YOLOE_DEVICE = YOLOE_CFG.get("device", "auto")
YOLOE_GPU_ID = YOLOE_CFG.get("gpu_id", 0)
YOLOE_IMGSZ = int(YOLOE_CFG.get("imgsz", 960))
YOLOE_CONF = float(YOLOE_CFG.get("confidence", 0.10))
YOLOE_NMS = float(YOLOE_CFG.get("nms_iou", 0.50))

_detector: GroundingDINODetector | None = None
_status = "模型尚未加载"
_worker_procs: list = []
_yoloe: YOLOEVisualDetector | None = None
_yoloe_key: tuple | None = None
_yoloe_status = "YOLOE 尚未加载"


def host_ipv4() -> str:
    """Primary LAN IPv4 of this machine (not 127.0.0.1)."""
    try:
        sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        sock.settimeout(0.3)
        sock.connect(("8.8.8.8", 80))
        ip = sock.getsockname()[0]
        sock.close()
        if ip and not ip.startswith("127."):
            return ip
    except OSError:
        pass
    try:
        for info in socket.getaddrinfo(socket.gethostname(), None, socket.AF_INET):
            ip = info[4][0]
            if ip and not ip.startswith("127."):
                return ip
    except OSError:
        pass
    return "127.0.0.1"


def gpu_summary() -> str:
    gpus = list_gpus()
    if not gpus:
        return "未检测到 CUDA GPU，将使用 CPU。"
    lines = [f"可见 GPU {len(gpus)} 张："]
    for g in gpus:
        mark = " ← 默认" if int(GPU_ID or 0) == g["index"] else ""
        lines.append(f"  cuda:{g['index']}  {g['name']}  {g['memory_gb']}GB{mark}")
    return "\n".join(lines)


def _device_choice_for(device: str, gpu_id) -> str:
    choices = device_choices()
    configured = str(device or "auto").split()[0]
    if configured.startswith("cuda") and ":" not in configured:
        configured = f"cuda:{int(gpu_id or 0)}"
    for c in choices:
        if c.split()[0] == configured:
            return c
    return choices[0] if choices else "auto"


def _default_device_choice() -> str:
    return _device_choice_for(DEVICE, GPU_ID)


def _yoloe_device_choice() -> str:
    return _device_choice_for(YOLOE_DEVICE, YOLOE_GPU_ID)


def get_detector(device: str | None = None) -> GroundingDINODetector:
    global _detector, _status
    want = (device or DEVICE or "auto").split()[0]
    if _detector is not None:
        current = str(_detector.device)
        if want in ("auto", "cuda") and current.startswith("cuda"):
            return _detector
        if current == want or str(_detector.device_name).split()[0] == want:
            return _detector
        old = _detector
        _detector = None
        del old
        try:
            import torch

            if torch.cuda.is_available():
                torch.cuda.empty_cache()
        except Exception:
            pass
    _status = f"正在加载 {MODEL_ID} → {want} …"
    _detector = GroundingDINODetector(
        model_id=MODEL_ID,
        device=want,
        gpu_id=GPU_ID,
        dtype=DTYPE,
        chunk_size=CHUNK_SIZE,
        local_files_only=LOCAL_ONLY,
    )
    t0 = time.perf_counter()
    _detector.load()
    _status = (
        f"已加载 {MODEL_ID}  |  device={_detector.device}  |  "
        f"{time.perf_counter() - t0:.1f}s"
    )
    return _detector


def get_yoloe_detector(device: str | None = None, model_key: str | None = None) -> YOLOEVisualDetector:
    global _yoloe, _yoloe_key, _yoloe_status
    key = normalize_model_key(model_key or YOLOE_DEFAULT)
    want = (device or YOLOE_DEVICE or "auto").split()[0]
    gpu_id = None
    if want in ("auto", "cuda"):
        gpu_id = int(YOLOE_GPU_ID or 0)
    stamp = (key, want, gpu_id)
    if _yoloe is not None and _yoloe_key == stamp:
        return _yoloe
    if _yoloe is not None:
        old = _yoloe
        _yoloe = None
        _yoloe_key = None
        del old
        try:
            import torch

            if torch.cuda.is_available():
                torch.cuda.empty_cache()
        except Exception:
            pass
    weights = resolve_weights(key, YOLOE_WEIGHTS_DIR)
    _yoloe_status = f"正在加载 YOLOE {weights.name} → {want} …"
    det = YOLOEVisualDetector(
        weights=weights,
        device=want,
        gpu_id=gpu_id,
        imgsz=YOLOE_IMGSZ,
    )
    t0 = time.perf_counter()
    det.load()
    _yoloe = det
    _yoloe_key = stamp
    _yoloe_status = (
        f"已加载 YOLOE {weights.name}  |  device={det.device}  |  "
        f"{time.perf_counter() - t0:.1f}s"
    )
    return _yoloe


def _run_one(
    image: Image.Image,
    caption: str,
    classes: list[str],
    box_th: float,
    text_th: float,
    nms_iou: float,
    device: str | None = None,
):
    det = get_detector(device)
    detections = det.detect(
        image,
        caption=caption,
        classes=classes,
        box_threshold=box_th,
        text_threshold=text_th,
        nms_iou=nms_iou,
    )
    vis = draw_detections(image, detections)
    rows = [[d.label, f"{d.score:.3f}", *d.xyxy] for d in detections]
    counts: dict[str, int] = {}
    for d in detections:
        counts[d.label] = counts.get(d.label, 0) + 1
    summary = "，".join(f"{k}×{v}" for k, v in sorted(counts.items())) or "无检测"
    class_line = "解析类别：" + ", ".join(classes)
    return vis, rows, detections, f"{_status}\n{class_line}\n检测到 {len(detections)} 个目标：{summary}"


def _parsed_text(prompt: str) -> str:
    _, classes = normalize_prompt(prompt or "")
    if not classes:
        return "（未解析到类别）"
    return f"{len(classes)} 类：\n" + ", ".join(classes)


def _need_login(user) -> str | None:
    if not pg.postgres_enabled(CFG):
        return None
    if not user or not user.get("id"):
        return "请先登录。"
    return None


def _account_text(user) -> str:
    if not pg.postgres_enabled(CFG):
        return "PostgreSQL 未启用，无账号体系。"
    if not user:
        return "未登录。首次部署的管理员账号见服务器启动日志 / config.yaml（登录后请立刻改密）。"
    role = "管理员" if pg.is_admin(user) else "用户"
    extra = " · 已停用" if user.get("disabled") else ""
    return f"已登录：{user['username']}（{role}{extra}）"


def _user_rows(actor):
    users = pg.list_users(actor, CFG)
    rows = []
    for u in users:
        created = u.get("created_at")
        rows.append(
            [
                u["id"],
                u["username"],
                u["role"],
                "停用" if u.get("disabled") else "正常",
                u.get("job_count"),
                str(created),
            ]
        )
    return rows


def _login_fail(msg: str):
    return (
        None,
        _account_text(None),
        [],
        gr.update(choices=[], value=None),
        gr.update(visible=True),
        gr.update(visible=False),
        msg,
    )


def do_login(username, password):
    try:
        user = pg.login(username or "", password or "", CFG)
    except Exception as exc:
        return _login_fail(str(exc))
    jobs = pg.list_jobs(user, CFG)
    ids = [r["id"] for r in jobs]
    return (
        user,
        _account_text(user),
        pg.jobs_table(jobs),
        gr.update(choices=ids, value=ids[0] if ids else None),
        gr.update(visible=False),
        gr.update(visible=True),
        "",
    )


def do_logout():
    need_auth = pg.postgres_enabled(CFG)
    return (
        None,
        _account_text(None),
        [],
        gr.update(choices=[], value=None),
        gr.update(visible=need_auth),
        gr.update(visible=not need_auth),
        "",
    )


def do_create_user(actor, username, password, role):
    try:
        created = pg.create_user(actor, username or "", password or "", role or "user", CFG)
        return f"已创建 {created['username']}（{created['role']}）", _user_rows(actor)
    except Exception as exc:
        return str(exc), []


def refresh_users(actor):
    try:
        return _user_rows(actor)
    except Exception:
        return []


def do_change_password(user, old_password, new_password):
    err = _need_login(user)
    if err:
        return err
    try:
        pg.change_password(user, old_password or "", new_password or "", CFG)
        return "密码已更新。"
    except Exception as exc:
        return str(exc)


def do_disable_user(actor, username, disabled):
    try:
        row = pg.set_user_disabled(actor, username or "", bool(disabled), CFG)
        state = "停用" if row.get("disabled") else "启用"
        return f"已{state} {row['username']}", _user_rows(actor)
    except Exception as exc:
        return str(exc), []


def do_reset_password(actor, username, new_password):
    try:
        pg.reset_password(actor, username or "", new_password or "", CFG)
        return f"已重置 {username} 的密码", _user_rows(actor)
    except Exception as exc:
        return str(exc), []


def refresh_jobs(user):
    if _need_login(user):
        return [], gr.update(choices=[], value=None)
    jobs = pg.list_jobs(user, CFG)
    ids = [r["id"] for r in jobs]
    return pg.jobs_table(jobs), gr.update(choices=ids, value=ids[0] if ids else None)


def predict_single(image, prompt, box_th, text_th, nms_iou, save_ann, device, user):
    parsed = _parsed_text(prompt or "")
    err = _need_login(user)
    if err:
        return None, [], err, parsed
    if image is None:
        return None, [], "请先上传一张图片。", parsed
    caption, classes = normalize_prompt(prompt or "")
    if not classes:
        return image, [], "请填写提示词，例如：person. face. sofa. 或 person, helmet", parsed
    image = image.convert("RGB") if isinstance(image, Image.Image) else Image.fromarray(image).convert("RGB")
    vis, rows, detections, msg = _run_one(image, caption, classes, box_th, text_th, nms_iou, device)
    if save_ann:
        stamp = time.strftime("%Y%m%d_%H%M%S")
        if user and pg.postgres_enabled(CFG):
            job_id = new_job_id()
            dirs = pg.job_dirs_for(user["username"], job_id, CFG)
            img_path = dirs["images"] / f"ui_{stamp}.jpg"
            image.save(img_path, quality=92)
            record = build_record(
                img_path, image, detections, classes, caption,
                model_id=MODEL_ID, box_threshold=box_th, text_threshold=text_th, nms_iou=nms_iou,
            )
            ann_path = save_record(record, dirs["annotations"], img_path.stem)
            vis_path = dirs["previews"] / f"{img_path.stem}_vis.jpg"
            vis.save(vis_path, quality=92)
            pg.create_job(
                user,
                job_id=job_id,
                prompt=caption,
                classes=classes,
                box_threshold=box_th,
                text_threshold=text_th,
                nms_iou=nms_iou,
                save_preview=True,
                model=MODEL_ID,
                image_rows=[{"orig_name": img_path.name, "image_path": str(img_path)}],
                cfg=CFG,
            )
            pg.complete_image(
                job_id, str(img_path), ok=True, box_count=len(detections),
                ann_path=str(ann_path), preview_path=str(vis_path), cfg=CFG,
            )
            msg += f"\n已记入任务 {job_id}\n标注：{ann_path}"
        else:
            DIRS["images"].mkdir(parents=True, exist_ok=True)
            DIRS["annotations"].mkdir(parents=True, exist_ok=True)
            img_path = DIRS["images"] / f"ui_{stamp}.jpg"
            image.save(img_path, quality=92)
            record = build_record(
                img_path, image, detections, classes, caption,
                model_id=MODEL_ID, box_threshold=box_th, text_threshold=text_th, nms_iou=nms_iou,
            )
            save_record(record, DIRS["annotations"], img_path.stem)
            msg += f"\n已写入规范标注：{DIRS['annotations'] / (img_path.stem + '.json')}"
    return vis, rows, msg, parsed


def _file_src(item) -> Path | None:
    if item is None:
        return None
    if isinstance(item, dict):
        raw = item.get("path") or item.get("name") or ""
    elif hasattr(item, "name"):
        raw = item.name
    else:
        raw = item
    src = Path(str(raw))
    if src.suffix.lower() not in IMAGE_EXTS:
        return None
    return src


def list_upload_images(files) -> list[Path]:
    if not files:
        return []
    items = files if isinstance(files, list) else [files]
    out: list[Path] = []
    for item in items:
        src = _file_src(item)
        if src is not None and src.exists():
            out.append(src)
    return out


def _bar_html(pct, text: str) -> str:
    try:
        pct = max(0.0, min(100.0, float(pct or 0)))
    except (TypeError, ValueError):
        pct = 0.0
    return (
        "<div style='background:#e2e8f0;border-radius:999px;height:14px;overflow:hidden'>"
        f"<div style='width:{pct:.1f}%;height:14px;background:#2563eb'></div></div>"
        f"<div style='margin-top:6px;color:#475569;font-size:13px'>{text}</div>"
    )


def _progress_tuple(job: dict | None) -> tuple[float, str]:
    if not job or not job.get("id"):
        return 0.0, "没有任务"
    total = int(job.get("total") or 0)
    done = int(job.get("done") or 0)
    failed = int(job.get("failed") or 0)
    finished = done + failed
    pct = 100.0 * finished / total if total else 0.0
    status = job.get("status") or ""
    return pct, f"{finished}/{total}  {status}  完成 {done}  失败 {failed}"


def _job_progress_ui(job_id: str, user=None) -> tuple[float, str]:
    job_id = (job_id or "").strip()
    if not job_id:
        return 0.0, "等待提交"
    job = None
    if pg.postgres_enabled(CFG):
        try:
            job = pg.get_job(job_id, CFG)
            if job and user and not pg.can_access_job(user, job):
                return 0.0, "无权查看该任务"
        except Exception:
            job = None
    if job is None and redis_enabled(CFG):
        try:
            job = job_status(job_id, cfg=CFG)
        except Exception:
            job = None
    return _progress_tuple(job)


def _save_uploads(files, dest_dir: Path, progress=None) -> list[tuple[Path, str]]:
    """Copy browser-uploaded files into dest_dir. Returns (dest, original_name)."""
    dest_dir = Path(dest_dir)
    dest_dir.mkdir(parents=True, exist_ok=True)
    saved: list[tuple[Path, str]] = []
    srcs = list_upload_images(files)
    n = len(srcs)
    stamp = time.strftime("%Y%m%d_%H%M%S")
    for i, src in enumerate(srcs):
        dest = dest_dir / f"{stamp}_{i:04d}_{src.name}"
        shutil.copy2(src, dest)
        saved.append((dest, src.name))
        if progress is not None:
            try:
                progress((i + 1) / max(n, 1), desc=f"保存上传 {i + 1}/{n}")
            except Exception:
                pass
    return saved


def page_thumbs(paths: list[Path], page, page_size, *, labels: list[str] | None = None):
    total = len(paths)
    size = max(1, int(page_size or BROWSE_PAGE))
    n_pages = max(1, (total + size - 1) // size) if total else 1
    page = min(max(1, int(page or 1)), n_pages)
    start = (page - 1) * size
    chunk = paths[start : start + size]
    items = []
    for i, p in enumerate(chunk):
        cap = labels[start + i] if labels and start + i < len(labels) else p.name
        items.append((str(p), cap))
    info = f"第 {page}/{n_pages} 页 · 共 {total} 张 · 本页 {len(chunk)} 张"
    return items, page, info, [str(p) for p in paths]


def preview_uploads(files, page, page_size):
    paths = list_upload_images(files)
    items, page, info, stored = page_thumbs(paths, page, page_size)
    n = len(paths)
    if not paths:
        info = "还没有选择图片。选文件夹后这里会出缩略图。"
        bar = _bar_html(0, "尚未选择文件夹")
    else:
        info = f"待提交 {info}。确认提示词后点「开始预标注」。"
        bar = _bar_html(100, f"浏览器已传到服务器 {n} 张，提交后写入任务目录")
    return items, page, info, stored, bar


def preview_server_folder(folder, page, page_size):
    paths = _collect_server_images((folder or "").strip()) if (folder or "").strip() else []
    items, page, info, stored = page_thumbs(paths, page, page_size)
    n = len(paths)
    if not (folder or "").strip():
        info = "填写服务器目录后点「预览该目录」。"
        bar = _bar_html(0, "尚未填写服务器目录")
    elif not paths:
        info = f"目录不存在或没有图片：{folder}"
        bar = _bar_html(0, "没有图片")
    else:
        info = f"服务器目录 {info}。确认后点「从服务器目录入队」。"
        bar = _bar_html(100, f"已扫描 {n} 张，提交后入队（不拷贝原图）")
    return items, page, info, stored, bar


def _format_job(job: dict) -> str:
    if not job or not job.get("id"):
        return "没有任务。"
    return (
        f"任务 {job['id']}\n"
        f"状态：{job['status']}  {job['done'] + job['failed']}/{job['total']}"
        f"（完成 {job['done']}，失败 {job['failed']}）\n"
        f"队列等待：可点「刷新任务状态」\n"
        f"最近图片：{job.get('last_image') or '-'}\n"
        f"最近错误：{job.get('last_error') or '-'}\n"
        f"提示词：{job.get('prompt')}"
    )


def _empty_browse():
    return "", [], [], 1, "没有可浏览的图片", _bar_html(0, "没有任务")


def _job_view(status, job_id, stored, page, page_size, user=None):
    items, page, info, stored = page_gallery(job_id, stored, page, page_size)
    pct, text = _job_progress_ui(job_id, user)
    return status, stored, items, page, info, _bar_html(pct, text)


def _as_paths(stored) -> list[Path]:
    out: list[Path] = []
    for item in stored or []:
        p = Path(str(item))
        if p.exists():
            out.append(p)
    return out


def _stem_key(path: Path | str) -> str:
    stem = Path(str(path)).stem
    if stem.endswith("_vis"):
        return stem[:-4]
    return stem


def _preview_file(stem: str, job_id: str = "") -> Path:
    return _job_file_dirs(job_id)["previews"] / f"{_stem_key(stem)}_vis.jpg"


def _annotation_file(stem: str, job_id: str = "") -> Path:
    return _job_file_dirs(job_id)["annotations"] / f"{_stem_key(stem)}.json"


def _gallery_event_index(evt, page, page_size, columns=GALLERY_COLS) -> int | None:
    """Gallery SelectData.index 可能是 int，也可能是 (row, col)。"""
    if evt is None or getattr(evt, "index", None) is None:
        return None
    raw = evt.index
    size = max(1, int(page_size or BROWSE_PAGE))
    page = max(1, int(page or 1))
    cols = max(1, int(columns or GALLERY_COLS))
    if isinstance(raw, (list, tuple)):
        if len(raw) >= 2:
            local = int(raw[0]) * cols + int(raw[1])
        else:
            local = int(raw[0])
    else:
        local = int(raw)
    return (page - 1) * size + local


def _gallery_event_path(evt) -> Path | None:
    value = getattr(evt, "value", None) if evt is not None else None
    cand = None
    if isinstance(value, (list, tuple)) and value:
        cand = value[0]
        if isinstance(cand, (list, tuple)) and cand:
            cand = cand[0]
        if isinstance(cand, dict):
            cand = cand.get("image") or cand.get("name") or cand.get("path")
    elif isinstance(value, dict):
        cand = value.get("image") or value.get("name") or value.get("path")
    elif isinstance(value, str):
        cand = value
    if not cand:
        return None
    return Path(str(cand))


def _match_stored_path(clicked: Path | None, paths: list[Path]) -> Path | None:
    if clicked is None or not paths:
        return None
    key = _stem_key(clicked)
    clicked_s = str(clicked)
    for p in paths:
        if str(p) == clicked_s or p.name == clicked.name or _stem_key(p) == key:
            return p
    return None


def _paths_from_job(job_id: str) -> list[Path]:
    job_id = (job_id or "").strip()
    if not job_id:
        return []
    if pg.postgres_enabled(CFG):
        try:
            rows = pg.list_job_images(job_id, CFG)
            return _as_paths([r["image_path"] for r in rows])
        except Exception:
            pass
    if not redis_enabled(CFG):
        return []
    try:
        job = job_status(job_id, cfg=CFG)
    except Exception:
        return []
    return _as_paths(job.get("images") or [])


def _annotated_images() -> list[Path]:
    out: list[Path] = []
    for rec in list_records(DIRS["annotations"]):
        for ext in IMAGE_EXTS:
            cand = DIRS["images"] / f"{rec.stem}{ext}"
            if cand.exists():
                out.append(cand)
                break
    return out


def resolve_browse_paths(job_id: str, stored) -> list[Path]:
    paths = _as_paths(stored)
    if paths:
        return paths
    paths = _paths_from_job(job_id)
    if paths:
        return paths
    return _annotated_images()


def _job_file_dirs(job_id: str) -> dict[str, Path]:
    if job_id and pg.postgres_enabled(CFG):
        job = pg.get_job(job_id, CFG)
        if job and job.get("annotations_dir"):
            return {
                "images": Path(job["images_dir"]),
                "annotations": Path(job["annotations_dir"]),
                "previews": Path(job["previews_dir"]),
                "exports": Path(job.get("exports_dir") or Path(job["annotations_dir"]).parent / "exports"),
            }
    return DIRS


def _job_image_lookup(job_id: str) -> dict[str, dict]:
    """stem / basename → job_images row (ann_path, preview_path, image_path)."""
    out: dict[str, dict] = {}
    job_id = (job_id or "").strip()
    if not job_id or not pg.postgres_enabled(CFG):
        return out
    try:
        rows = pg.list_job_images(job_id, CFG)
    except Exception:
        return out
    for row in rows:
        img = Path(str(row.get("image_path") or ""))
        orig = str(row.get("orig_name") or "")
        for key in filter(None, [_stem_key(img), img.name, orig, Path(orig).stem if orig else ""]):
            out[key] = row
    return out


def page_gallery(job_id, stored, page, page_size):
    paths = resolve_browse_paths(job_id, stored)
    lookup = _job_image_lookup(job_id)
    total = len(paths)
    size = max(1, int(page_size or BROWSE_PAGE))
    n_pages = max(1, (total + size - 1) // size) if total else 1
    page = min(max(1, int(page or 1)), n_pages)
    start = (page - 1) * size
    chunk = paths[start : start + size]
    items = []
    for p in chunk:
        row = lookup.get(_stem_key(p)) or lookup.get(p.name) or {}
        prev = Path(row["preview_path"]) if row.get("preview_path") else _preview_file(p.stem, job_id)
        ann = Path(row["ann_path"]) if row.get("ann_path") else _annotation_file(p.stem, job_id)
        if prev.exists():
            items.append((str(prev), p.name))
        elif ann.exists():
            items.append((str(p), f"{p.name} · 点开画框"))
        else:
            items.append((str(p), f"{p.name} · 排队中"))
    info = f"第 {page}/{n_pages} 页 · 共 {total} 张 · 本页 {len(chunk)} 张。点缩略图看画框。"
    return items, page, info, [str(p) for p in paths]


def _open_vis(path: Path, job_id: str = ""):
    lookup = _job_image_lookup(job_id)
    row = lookup.get(_stem_key(path)) or lookup.get(path.name) or {}
    orig = Path(row["image_path"]) if row.get("image_path") else path
    if orig.stem.endswith("_vis") or not orig.exists():
        orig = path if path.exists() and not path.stem.endswith("_vis") else orig
    prev = Path(row["preview_path"]) if row.get("preview_path") else _preview_file(orig.stem or path.stem, job_id)
    ann = Path(row["ann_path"]) if row.get("ann_path") else _annotation_file(orig.stem or path.stem, job_id)
    if not prev.exists():
        prev = _preview_file(orig.stem or path.stem, job_id)
    if not ann.exists():
        ann = _annotation_file(orig.stem or path.stem, job_id)
    image = None
    src = orig if orig.exists() else path
    if src.exists() and not src.stem.endswith("_vis"):
        try:
            image = Image.open(src).convert("RGB")
        except Exception:
            image = None
    display_name = orig.name if orig.name else path.name
    if ann.exists():
        rec = load_record(ann)
        dets = detections_from_record(rec)
        vis = None
        if image is not None:
            vis = draw_detections(image, dets)
            try:
                prev.parent.mkdir(parents=True, exist_ok=True)
                vis.save(prev, quality=92)
            except Exception:
                pass
        elif prev.exists():
            vis = Image.open(prev).convert("RGB")
        rows = [[d.label, f"{d.score:.3f}", *d.xyxy] for d in dets]
        counts: dict[str, int] = {}
        for d in dets:
            counts[d.label] = counts.get(d.label, 0) + 1
        summary = "，".join(f"{k}×{v}" for k, v in sorted(counts.items())) or "无检测"
        return vis, rows, f"{display_name}\n{len(dets)} 个框：{summary}\n标注：{ann}"
    if prev.exists():
        return Image.open(prev).convert("RGB"), [], f"{display_name}\n已有预览，没有规范标注 JSON"
    if image is not None:
        return image, [], f"{display_name}\n还没有标注（任务可能还在排队）"
    if path.exists():
        try:
            return Image.open(path).convert("RGB"), [], f"{path.name}\n还没有标注（任务可能还在排队）"
        except Exception:
            pass
    return None, [], f"找不到文件：{path}"


def show_clicked(evt: gr.SelectData, job_id, stored, page, page_size):
    paths = resolve_browse_paths(job_id, stored)
    path = _match_stored_path(_gallery_event_path(evt), paths)
    if path is None:
        idx = _gallery_event_index(evt, page, page_size)
        if idx is None:
            return None, [], "点一张缩略图放大可视化。"
        if idx < 0 or idx >= len(paths):
            return None, [], "这页没有这张图，先刷新。"
        path = paths[idx]
    vis, rows, info = _open_vis(path, job_id)
    return vis, rows, info


def show_upload_clicked(evt: gr.SelectData, stored, page, page_size):
    paths = _as_paths(stored)
    path = _match_stored_path(_gallery_event_path(evt), paths)
    if path is None:
        idx = _gallery_event_index(evt, page, page_size)
        if idx is None:
            return None, "点一张待提交缩略图放大原图。"
        if idx < 0 or idx >= len(paths):
            return None, "这页没有这张图。"
        path = paths[idx]
    try:
        img = Image.open(path).convert("RGB")
    except Exception as exc:
        return None, f"打不开：{path.name}  {exc}"
    return img, f"{path.name}\n{img.size[0]}×{img.size[1]}"


def _submit_job(
    user,
    caption,
    classes,
    box_th,
    text_th,
    nms_iou,
    save_preview,
    image_rows,
    copy_note="",
    job_id: str | None = None,
):
    empty = _empty_browse()
    job_id = job_id or new_job_id()
    stored = [r["image_path"] for r in image_rows]
    pg_job = None
    if user and pg.postgres_enabled(CFG):
        pg_job = pg.create_job(
            user,
            job_id=job_id,
            prompt=caption,
            classes=classes,
            box_threshold=box_th,
            text_threshold=text_th,
            nms_iou=nms_iou,
            save_preview=bool(save_preview),
            model=MODEL_ID,
            image_rows=image_rows,
            cfg=CFG,
        )
    if not redis_enabled(CFG):
        return {
            "mode": "serial",
            "job_id": job_id,
            "pg_job": pg_job,
            "stored": stored,
        }
    try:
        job = enqueue_job(
            image_paths=stored,
            prompt=caption,
            classes=classes,
            box_threshold=box_th,
            text_threshold=text_th,
            nms_iou=nms_iou,
            save_preview=bool(save_preview),
            job_id=job_id,
            cfg=CFG,
        )
    except RedisDisabled as exc:
        return {"mode": "error", "msg": f"Redis 不可用：{exc}", "empty": empty}
    except Exception as exc:
        return {
            "mode": "error",
            "msg": f"入队失败：{exc}\n请确认 Redis 已启动。界面启动时应已按 replicas 拉起 worker。",
            "empty": empty,
        }
    n = replica_count(CFG)
    alive = sum(1 for p in _worker_procs if p.poll() is None)
    status_txt = pg.format_job(pg_job) if pg_job else _format_job(job)
    msg = (
        f"{copy_note}任务 {job_id} 已进 Redis / Postgres，共 {len(stored)} 张。\n"
        f"{status_txt}\n"
        f"同一张卡 cuda:{worker_gpu_id(CFG)} 上配置 {n} 份模型"
        f"（当前还活着 {alive} 份）。点「刷新」看进度；可取消或重试失败。"
    )
    items, page, info, stored = page_gallery(job_id, stored, 1, BROWSE_PAGE)
    pct, text = _job_progress_ui(job_id)
    return {
        "mode": "queued",
        "result": (msg, job_id, stored, items, page, info, _bar_html(pct, text)),
    }


def _run_serial_job(job_id, pg_job, saved, caption, classes, box_th, text_th, nms_iou, save_preview, device, progress):
    stored = [str(p) for p, _ in saved]
    det = get_detector(device)
    total = 0
    dirs = _job_file_dirs(job_id)
    dirs["annotations"].mkdir(parents=True, exist_ok=True)
    dirs["previews"].mkdir(parents=True, exist_ok=True)

    for path, _orig in progress.tqdm(saved, desc="预标注"):
        image = Image.open(path).convert("RGB")
        detections = det.detect(
            image,
            caption=caption,
            classes=classes,
            box_threshold=box_th,
            text_threshold=text_th,
            nms_iou=nms_iou,
        )
        total += len(detections)
        record = build_record(
            path, image, detections, classes, caption,
            model_id=MODEL_ID, box_threshold=box_th, text_threshold=text_th, nms_iou=nms_iou,
        )
        ann_path = save_record(record, dirs["annotations"], path.stem)
        vis = draw_detections(image, detections)
        preview_path = None
        if save_preview:
            preview_path = dirs["previews"] / f"{path.stem}_vis.jpg"
            vis.save(preview_path, quality=92)
        if pg_job:
            pg.complete_image(
                job_id, str(path), ok=True, box_count=len(detections),
                ann_path=str(ann_path),
                preview_path=str(preview_path) if preview_path else None,
                cfg=CFG,
            )

    msg = (
        f"{_status}\n"
        f"解析类别 ({len(classes)})：{', '.join(classes)}\n"
        f"完成 {len(saved)} 张，共 {total} 个框（本进程单卡，未走 Redis）。\n"
        f"任务 {job_id}\n"
        f"下面分页浏览，点缩略图看画框。"
    )
    items, page, info, stored = page_gallery(job_id, stored, 1, BROWSE_PAGE)
    pct, text = _job_progress_ui(job_id)
    return msg, job_id, stored, items, page, info, _bar_html(pct, text)


def _unpack_submit(result, saved, caption, classes, box_th, text_th, nms_iou, save_preview, device, progress):
    empty = _empty_browse()
    mode = result.get("mode")
    if mode == "queued":
        return result["result"]
    if mode == "error":
        return result["msg"], *result.get("empty", empty)
    return _run_serial_job(
        result["job_id"], result.get("pg_job"), saved, caption, classes,
        box_th, text_th, nms_iou, save_preview, device, progress,
    )


def predict_batch(files, prompt, box_th, text_th, nms_iou, save_preview, device, user, progress=gr.Progress()):
    empty = _empty_browse()
    err = _need_login(user)
    if err:
        return err, *empty
    caption, classes = normalize_prompt(prompt or "")
    if not classes:
        return "请填写提示词，例如：person, helmet, safety vest", *empty

    job_id = new_job_id()
    if user and pg.postgres_enabled(CFG):
        dest = pg.job_dirs_for(user["username"], job_id, CFG)["images"]
    else:
        dest = DIRS["images"]
    saved = _save_uploads(files, dest, progress)
    if not saved:
        return "请从本机选择文件夹 / 多张图片上传，或改用服务器目录入队。", *empty

    image_rows = [{"orig_name": orig, "image_path": str(p)} for p, orig in saved]
    result = _submit_job(
        user, caption, classes, box_th, text_th, nms_iou, save_preview, image_rows,
        copy_note=f"已上传 {len(saved)} 张，",
        job_id=job_id,
    )
    return _unpack_submit(result, saved, caption, classes, box_th, text_th, nms_iou, save_preview, device, progress)


def _collect_server_images(folder: str, limit: int = 20000) -> list[Path]:
    root = Path(folder or "").expanduser()
    if not root.is_dir():
        return []
    files = [p for p in root.rglob("*") if p.is_file() and p.suffix.lower() in IMAGE_EXTS]
    files.sort(key=lambda p: str(p).lower())
    return files[:limit]


def predict_server_dir(folder, prompt, box_th, text_th, nms_iou, save_preview, device, user, progress=gr.Progress()):
    empty = _empty_browse()
    err = _need_login(user)
    if err:
        return err, *empty
    caption, classes = normalize_prompt(prompt or "")
    if not classes:
        return "请填写提示词，例如：person, helmet, safety vest", *empty
    folder = (folder or "").strip()
    if not folder:
        return "请填写服务器上的图片目录。", *empty
    paths = _collect_server_images(folder)
    if not paths:
        return f"目录不存在或没有图片：{folder}", *empty
    image_rows = [{"orig_name": p.name, "image_path": str(p.resolve())} for p in paths]
    saved = [(p, p.name) for p in paths]
    result = _submit_job(
        user, caption, classes, box_th, text_th, nms_iou, save_preview, image_rows,
        copy_note=f"已从服务器目录收录 {len(paths)} 张（不拷贝原图），",
    )
    return _unpack_submit(result, saved, caption, classes, box_th, text_th, nms_iou, save_preview, device, progress)


def cancel_current_job(job_id, user, stored, page, page_size):
    err = _need_login(user)
    if err:
        return _job_view(err, job_id, stored, page, page_size, user)
    job_id = (job_id or "").strip()
    if not job_id:
        return _job_view("请先打开或提交一个任务。", "", stored, page, page_size, user)
    try:
        job = pg.cancel_job(user, job_id, CFG)
    except Exception as exc:
        return _job_view(str(exc), job_id, stored, page, page_size, user)
    dropped = 0
    if redis_enabled(CFG):
        try:
            dropped = drop_job_from_queue(job_id, CFG)
        except Exception:
            dropped = 0
    extra = f"，Redis 丢掉 {dropped} 条排队任务" if dropped else ""
    return _job_view(
        pg.format_job(job) + f"\n已取消，未完成的图不会再跑{extra}。",
        job_id, stored, page, page_size, user,
    )


def retry_failed_job(job_id, user, stored, page, page_size):
    err = _need_login(user)
    if err:
        return _job_view(err, job_id, stored, page, page_size, user)
    job_id = (job_id or "").strip()
    if not job_id:
        return _job_view("请先打开一个任务。", "", stored, page, page_size, user)
    try:
        paths = pg.retry_failed(user, job_id, CFG)
        job = pg.get_job(job_id, CFG) or {}
        added = 0
        if paths and redis_enabled(CFG):
            added = requeue_job_paths(job, paths, CFG)
    except Exception as exc:
        return _job_view(str(exc), job_id, stored, page, page_size, user)
    return _job_view(
        pg.format_job(job) + f"\n已把 {len(paths)} 张失败图改回排队，Redis 新入队 {added} 张。",
        job_id, stored, page, page_size, user,
    )


def refresh_job(job_id: str, stored, page, page_size, user=None):
    job_id = (job_id or "").strip()
    if not job_id:
        return _job_view("还没有任务 id。下面仍可浏览已有标注。", "", stored, page, page_size, user)
    extra = ""
    if redis_enabled(CFG):
        try:
            extra = f"\nRedis 队列长度：{queue_length(CFG)}"
        except Exception:
            extra = ""
    if pg.postgres_enabled(CFG):
        try:
            job = pg.get_job(job_id, CFG)
            if job and user and not pg.can_access_job(user, job):
                return _job_view("无权查看该任务。", job_id, stored, page, page_size, user)
            status = (pg.format_job(job) if job else "Postgres 里没有这个任务。") + extra
            return _job_view(status, job_id, stored, page, page_size, user)
        except Exception as exc:
            return _job_view(f"查询失败：{exc}", job_id, stored, page, page_size, user)
    if not redis_enabled(CFG):
        return _job_view("Redis 未启用。", job_id, stored, page, page_size, user)
    try:
        status = _format_job(job_status(job_id, cfg=CFG)) + extra
    except Exception as exc:
        status = f"查询失败：{exc}"
    return _job_view(status, job_id, stored, page, page_size, user)


def tick_job(job_id: str, stored, page, page_size, user=None):
    """Timer refresh: skip when no job is open so empty ticks don't wipe the gallery."""
    if not (job_id or "").strip():
        return gr.update(), stored, gr.update(), page, gr.update(), gr.update()
    return refresh_job(job_id, stored, page, page_size, user)


def open_selected_job(job_id, user, page_size):
    job_id = (job_id or "").strip()
    empty = _empty_browse()
    if not job_id:
        return "请选择一个任务。", *empty
    if pg.postgres_enabled(CFG):
        job = pg.get_job(job_id, CFG)
        if not job:
            return "任务不存在。", *empty
        if user and not pg.can_access_job(user, job):
            return "无权查看该任务。", *empty
        stored = [r["image_path"] for r in pg.list_job_images(job_id, CFG)]
        extra = ""
        if redis_enabled(CFG):
            try:
                extra = f"\nRedis 队列长度：{queue_length(CFG)}"
            except Exception:
                extra = ""
        status, stored, items, page, info, bar = _job_view(
            pg.format_job(job) + extra, job_id, stored, 1, page_size, user,
        )
        return status, job_id, stored, items, page, info, bar
    status, stored, items, page, info, bar = _job_view(
        f"打开任务 {job_id}", job_id, [], 1, page_size, user,
    )
    return status, job_id, stored, items, page, info, bar


def redis_line() -> str:
    n = replica_count(CFG)
    gpu = worker_gpu_id(CFG)
    if not redis_enabled(CFG):
        return "Redis：关闭（config.yaml redis.enabled=false，批量将在界面进程里单份模型串行跑）"
    try:
        ping_msg = redis_ping(CFG)
        alive = sum(1 for p in _worker_procs if p.poll() is None)
        return (
            f"{ping_msg}\n"
            f"worker：cuda:{gpu} × {n} 份（本进程已拉起 {alive} 份）"
        )
    except Exception as exc:
        return f"Redis：连不上（{exc}）。批量无法走多模型，请先启动 Redis 再重新 python app.py"


def start_replica_workers() -> str:
    """Pull up workers.replicas processes on workers.gpu_id. Called from app.py."""
    global _worker_procs
    if _worker_procs:
        return f"worker 已在跑：{len(_worker_procs)} 份"
    if not redis_enabled(CFG):
        return "redis.enabled=false，不启动多模型 worker"
    try:
        ping_msg = redis_ping(CFG)
        print(ping_msg, flush=True)
    except Exception as exc:
        return f"Redis 连不上，无法启动 {replica_count(CFG)} 份 worker：{exc}"
    try:
        print(resume_unfinished(CFG), flush=True)
    except Exception as exc:
        print(f"续跑未完成任务失败：{exc}", flush=True)
    _worker_procs = spawn_replicas(CFG)
    atexit.register(stop_replicas, _worker_procs)
    gpu = worker_gpu_id(CFG)
    n = replica_count(CFG)
    return f"已在 cuda:{gpu} 上拉起 {n} 份 worker（pid {[p.pid for p in _worker_procs]}）"


def _zip_written(written: list[str], zip_path: Path, root: Path) -> Path:
    zip_path.parent.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(zip_path, "w", zipfile.ZIP_DEFLATED) as zf:
        seen: set[str] = set()
        for w in written:
            p = Path(w)
            if not p.is_file():
                continue
            try:
                arc = p.relative_to(root).as_posix()
            except ValueError:
                arc = p.name
            if arc in seen:
                continue
            seen.add(arc)
            zf.write(p, arc)
    return zip_path


def do_export(fmt, user, job_id):
    err = _need_login(user)
    if err:
        return err, None
    job_id = (job_id or "").strip()
    if not job_id:
        return "请先选择或提交一个任务再导出。", None
    dirs = _job_file_dirs(job_id)
    records = []
    if pg.postgres_enabled(CFG):
        job = pg.get_job(job_id, CFG)
        if not job:
            return "任务不存在。", None
        if user and not pg.can_access_job(user, job):
            return "无权导出该任务。", None
        for ann in pg.done_annotation_paths(job_id, CFG):
            path = Path(ann)
            if path.is_file():
                try:
                    records.append(load_record(path))
                except Exception:
                    continue
        if not records:
            return (
                "该任务还没有已完成的规范标注可导出。"
                "（未跑完 / 失败的图不会进导出）",
                None,
            )
        result = export_from_records(records, dirs["exports"], fmt, dirs["images"])
    else:
        result = export_from_store(dirs["annotations"], dirs["exports"], fmt, dirs["images"])
    if result["count"] == 0:
        return f"规范标注目录是空的：{dirs['annotations']}", None
    if pg.postgres_enabled(CFG):
        try:
            pg.add_export(job_id, fmt, str(dirs["exports"]), CFG)
        except Exception:
            pass
    stamp = time.strftime("%Y%m%d_%H%M%S")
    zip_path = dirs["exports"] / "downloads" / f"{job_id}_{fmt}_{stamp}.zip"
    _zip_written(result.get("written") or [], zip_path, dirs["exports"])
    if not zip_path.is_file() or zip_path.stat().st_size == 0:
        return "导出文件打包失败。", None
    note = f"仅已完成 {result['count']} 张"
    msg = (
        f"{note} 已打包 → {fmt}\n"
        f"类别 ({len(result['classes'])})：{', '.join(result['classes'])}\n"
        f"点下方文件下载 {zip_path.name}"
    )
    return msg, str(zip_path)


def do_cleanup(keep_days, include_exports, dry_run, user=None):
    if pg.postgres_enabled(CFG) and not pg.is_admin(user):
        return "只有管理员可以清理全局过期文件。"
    result = cleanup(
        images_dir=DIRS["images"],
        annotations_dir=DIRS["annotations"],
        previews_dir=DIRS["previews"],
        exports_dir=DIRS["exports"],
        keep_days=float(keep_days),
        include_exports=bool(include_exports),
        dry_run=bool(dry_run),
    )
    verb = "将删除" if dry_run else "已删除"
    lines = [
        f"{verb} {result['removed_count']} 个文件（保留 {result['keep_days']} 天）",
        f"images: {len(result['removed']['images'])}",
        f"annotations: {len(result['removed']['annotations'])}",
        f"previews: {len(result['removed']['previews'])}",
        f"exports: {len(result['removed']['exports'])}",
    ]
    sample = []
    for files in result["removed"].values():
        sample.extend(files[:8])
    if sample:
        lines.append("示例：")
        lines.extend(sample[:12])
    return "\n".join(lines)


def _pil_rgb(image) -> Image.Image | None:
    if image is None:
        return None
    if isinstance(image, Image.Image):
        return image.convert("RGB")
    return Image.fromarray(image).convert("RGB")


def _select_xy(evt) -> tuple[int, int] | None:
    if evt is None:
        return None
    idx = getattr(evt, "index", None)
    if isinstance(idx, (list, tuple)) and len(idx) >= 2:
        return int(idx[0]), int(idx[1])
    value = getattr(evt, "value", None)
    if isinstance(value, dict):
        x = value.get("x")
        y = value.get("y")
        if x is not None and y is not None:
            return int(x), int(y)
    return None


def _draw_box_preview(image: Image.Image, xyxy, extra: list | None = None) -> Image.Image:
    vis = image.copy()
    draw = ImageDraw.Draw(vis)
    boxes = list(extra or [])
    if xyxy:
        boxes.append(xyxy)
    for box in boxes:
        x1, y1, x2, y2 = (int(v) for v in box)
        draw.rectangle([x1, y1, x2, y2], outline=(220, 38, 38), width=3)
    return vis


def _prompt_table(prompts: list | None):
    rows = []
    for i, item in enumerate(prompts or []):
        box = item.get("xyxy") or [0, 0, 0, 0]
        rows.append([i, item.get("label", ""), *box])
    return rows


def _set_ref_raw(image):
    img = _pil_rgb(image)
    return img, {"p1": None, "xyxy": None}, "上传后在图上点两次组框：先左上，再右下。"


def _click_visual_ref(evt: gr.SelectData, raw, click_state, prompts):
    empty_state = {"p1": None, "xyxy": None}
    img = _pil_rgb(raw)
    if img is None:
        return None, empty_state, "请先上传参考图，再点两次组成框。"
    xy = _select_xy(evt)
    if xy is None:
        return img, click_state or empty_state, "点参考图：第一次左上，第二次右下。"
    state = dict(click_state or empty_state)
    extra = [p.get("xyxy") for p in (prompts or []) if p.get("xyxy")]
    if not state.get("p1"):
        state = {"p1": list(xy), "xyxy": None}
        vis = _draw_box_preview(img, None, extra)
        return vis, state, f"已记下 ({xy[0]}, {xy[1]})，再点对角。"
    x1, y1 = state["p1"]
    x2, y2 = xy
    if x2 < x1:
        x1, x2 = x2, x1
    if y2 < y1:
        y1, y2 = y2, y1
    if x2 - x1 < 4 or y2 - y1 < 4:
        state = {"p1": None, "xyxy": None}
        return img, state, "框太小，请重新点两次。"
    xyxy = [int(x1), int(y1), int(x2), int(y2)]
    state = {"p1": None, "xyxy": xyxy}
    vis = _draw_box_preview(img, xyxy, extra)
    return vis, state, f"当前框 {xyxy[0]},{xyxy[1]} — {xyxy[2]},{xyxy[3]}，选类别后点「加入视觉提示」。"


def _class_choices(text: str):
    names = parse_visual_classes(text or "")
    value = names[0] if names else None
    return gr.update(choices=names, value=value)


def add_visual_prompt(raw, click_state, class_name, class_text, prompts):
    names = parse_visual_classes(class_text or "")
    img = _pil_rgb(raw)
    prompts = list(prompts or [])
    extra = [p.get("xyxy") for p in prompts if p.get("xyxy")]
    if img is None:
        return None, prompts, _prompt_table(prompts), click_state or {"p1": None, "xyxy": None}, "请先上传参考图。"
    if not names:
        return img, prompts, _prompt_table(prompts), click_state, "请填写类别名，例如：charging_nest, robot"
    label = (class_name or "").strip() or names[0]
    if label not in names:
        return img, prompts, _prompt_table(prompts), click_state, f"类别「{label}」不在列表里。"
    xyxy = (click_state or {}).get("xyxy")
    if not xyxy:
        return img, prompts, _prompt_table(prompts), click_state, "请在参考图上点两次组成框。"
    class_id = names.index(label)
    prompts.append(
        {
            "label": label,
            "class_id": class_id,
            "xyxy": [int(v) for v in xyxy],
            "image": img.copy(),
        }
    )
    extra.append(xyxy)
    vis = _draw_box_preview(img, None, extra)
    reset = {"p1": None, "xyxy": None}
    return vis, prompts, _prompt_table(prompts), reset, f"已加入 {label}  {xyxy}，共 {len(prompts)} 个提示。"


def clear_visual_prompts(raw):
    img = _pil_rgb(raw)
    return img, [], [], {"p1": None, "xyxy": None}, "已清空视觉提示。"


def _prompts_to_visual(prompts) -> list[VisualPrompt]:
    out: list[VisualPrompt] = []
    for item in prompts or []:
        image = item.get("image")
        if image is None:
            continue
        box = item.get("xyxy") or [0, 0, 0, 0]
        out.append(
            VisualPrompt(
                image=_pil_rgb(image),
                xyxy=(int(box[0]), int(box[1]), int(box[2]), int(box[3])),
                class_id=int(item.get("class_id") or 0),
            )
        )
    return out


def _manuals_for_image(image: Image.Image, names: list[str], prompts) -> list:
    manuals = []
    if image is None:
        return manuals
    target = image.tobytes()
    for item in prompts or []:
        ref = item.get("image")
        if ref is None or _pil_rgb(ref).tobytes() != target:
            continue
        box = item.get("xyxy") or [0, 0, 0, 0]
        idx = int(item.get("class_id") or 0)
        label = names[idx] if 0 <= idx < len(names) else item.get("label") or "object"
        manuals.append(Detection(label=label, score=1.0, xyxy=(int(box[0]), int(box[1]), int(box[2]), int(box[3]))))
    return manuals


def _save_visual_prompt_bundle(dest_dir: Path, names: list[str], prompts) -> None:
    dest_dir.mkdir(parents=True, exist_ok=True)
    payload = {"classes": names, "prompts": []}
    for i, item in enumerate(prompts or []):
        image = _pil_rgb(item.get("image"))
        filename = f"ref_{i:03d}.png"
        if image is not None:
            image.save(dest_dir / filename)
        payload["prompts"].append(
            {
                "image": filename,
                "xyxy": item.get("xyxy"),
                "class_id": int(item.get("class_id") or 0),
                "label": item.get("label"),
            }
        )
    (dest_dir / "visual_prompts.json").write_text(
        json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8"
    )


def _prepare_yoloe(class_text, prompts, model_key, device):
    names = parse_visual_classes(class_text or "")
    vis_prompts = _prompts_to_visual(prompts)
    if not names:
        raise ValueError("请填写类别名，例如：charging_nest, robot")
    if not vis_prompts:
        raise ValueError("请至少加入一个视觉提示框。")
    missing = [n for i, n in enumerate(names) if not any(p.class_id == i for p in vis_prompts)]
    if missing:
        raise ValueError("这些类别还没有画框：" + ", ".join(missing))
    det = get_yoloe_detector(device, model_key)
    det.set_visual_classes(names, vis_prompts, strategy="semantic")
    return det, names


def predict_visual_single(image, class_text, prompts, model_key, conf, nms_iou, save_ann, device, user):
    err = _need_login(user)
    if err:
        return None, [], err
    img = _pil_rgb(image)
    if img is None:
        return None, [], "请上传要检测的图片。"
    try:
        det, names = _prepare_yoloe(class_text, prompts, model_key, device)
        manuals = _manuals_for_image(img, names, prompts)
        detections = det.detect(img, confidence=float(conf), nms_iou=float(nms_iou), manuals=manuals)
    except Exception as exc:
        return img, [], str(exc)
    vis = draw_detections(img, detections)
    rows = [[d.label, f"{d.score:.3f}", *d.xyxy] for d in detections]
    caption = visual_caption(names)
    msg = f"{_yoloe_status}\n{caption}\n检测到 {len(detections)} 个目标。"
    if save_ann:
        stamp = time.strftime("%Y%m%d_%H%M%S")
        if user and pg.postgres_enabled(CFG):
            job_id = new_job_id()
            dirs = pg.job_dirs_for(user["username"], job_id, CFG)
            img_path = dirs["images"] / f"ui_{stamp}.jpg"
            img.save(img_path, quality=92)
            record = build_record(
                img_path, img, detections, names, caption,
                model_id=str(det.weights), box_threshold=float(conf), text_threshold=0.0, nms_iou=float(nms_iou),
            )
            ann_path = save_record(record, dirs["annotations"], img_path.stem)
            vis_path = dirs["previews"] / f"{img_path.stem}_vis.jpg"
            vis.save(vis_path, quality=92)
            _save_visual_prompt_bundle(dirs["root"] / "prompts", names, prompts)
            pg.create_job(
                user,
                job_id=job_id,
                prompt=caption,
                classes=names,
                box_threshold=float(conf),
                text_threshold=0.0,
                nms_iou=float(nms_iou),
                save_preview=True,
                model=str(det.weights),
                image_rows=[{"orig_name": img_path.name, "image_path": str(img_path)}],
                cfg=CFG,
            )
            pg.complete_image(
                job_id, str(img_path), ok=True, box_count=len(detections),
                ann_path=str(ann_path), preview_path=str(vis_path), cfg=CFG,
            )
            msg += f"\n已记入任务 {job_id}\n标注：{ann_path}"
        else:
            DIRS["images"].mkdir(parents=True, exist_ok=True)
            DIRS["annotations"].mkdir(parents=True, exist_ok=True)
            img_path = DIRS["images"] / f"ui_{stamp}.jpg"
            img.save(img_path, quality=92)
            record = build_record(
                img_path, img, detections, names, caption,
                model_id=str(det.weights), box_threshold=float(conf), text_threshold=0.0, nms_iou=float(nms_iou),
            )
            save_record(record, DIRS["annotations"], img_path.stem)
            msg += f"\n已写入规范标注：{DIRS['annotations'] / (img_path.stem + '.json')}"
    return vis, rows, msg


def _run_visual_serial_job(
    job_id, pg_job, saved, names, prompts, model_key, conf, nms_iou, save_preview, device, progress,
):
    stored = [str(p) for p, _ in saved]
    try:
        det, names = _prepare_yoloe(", ".join(names), prompts, model_key, device)
    except Exception as exc:
        return str(exc), job_id, stored, [], 1, str(exc), _bar_html(0, str(exc))
    caption = visual_caption(names)
    total = 0
    dirs = _job_file_dirs(job_id)
    dirs["annotations"].mkdir(parents=True, exist_ok=True)
    dirs["previews"].mkdir(parents=True, exist_ok=True)
    prompt_root = Path(dirs["annotations"]).parent / "prompts"
    _save_visual_prompt_bundle(prompt_root, names, prompts)

    for path, _orig in progress.tqdm(saved, desc="视觉预标注"):
        image = Image.open(path).convert("RGB")
        manuals = _manuals_for_image(image, names, prompts)
        detections = det.detect(image, confidence=float(conf), nms_iou=float(nms_iou), manuals=manuals)
        total += len(detections)
        record = build_record(
            path, image, detections, names, caption,
            model_id=str(det.weights), box_threshold=float(conf), text_threshold=0.0, nms_iou=float(nms_iou),
        )
        ann_path = save_record(record, dirs["annotations"], path.stem)
        vis = draw_detections(image, detections)
        preview_path = None
        if save_preview:
            preview_path = dirs["previews"] / f"{path.stem}_vis.jpg"
            vis.save(preview_path, quality=92)
        if pg_job:
            pg.complete_image(
                job_id, str(path), ok=True, box_count=len(detections),
                ann_path=str(ann_path),
                preview_path=str(preview_path) if preview_path else None,
                cfg=CFG,
            )

    msg = (
        f"{_yoloe_status}\n"
        f"{caption}\n"
        f"完成 {len(saved)} 张，共 {total} 个框（视觉提示，本进程串行，未走 Redis）。\n"
        f"任务 {job_id}"
    )
    items, page, info, stored = page_gallery(job_id, stored, 1, BROWSE_PAGE)
    pct, text = _job_progress_ui(job_id)
    return msg, job_id, stored, items, page, info, _bar_html(pct, text)


def _submit_visual_job(user, names, conf, nms_iou, save_preview, image_rows, model_id, job_id=None):
    job_id = job_id or new_job_id()
    caption = visual_caption(names)
    pg_job = None
    if user and pg.postgres_enabled(CFG):
        pg_job = pg.create_job(
            user,
            job_id=job_id,
            prompt=caption,
            classes=names,
            box_threshold=float(conf),
            text_threshold=0.0,
            nms_iou=float(nms_iou),
            save_preview=bool(save_preview),
            model=model_id,
            image_rows=image_rows,
            cfg=CFG,
        )
    return job_id, pg_job


def predict_visual_batch(
    files, class_text, prompts, model_key, conf, nms_iou, save_preview, device, user, progress=gr.Progress(),
):
    empty = _empty_browse()
    err = _need_login(user)
    if err:
        return err, *empty
    names = parse_visual_classes(class_text or "")
    if not names:
        return "请填写类别名，例如：charging_nest, robot", *empty
    if not _prompts_to_visual(prompts):
        return "请至少加入一个视觉提示框。", *empty
    job_id = new_job_id()
    if user and pg.postgres_enabled(CFG):
        dest = pg.job_dirs_for(user["username"], job_id, CFG)["images"]
    else:
        dest = DIRS["images"]
    saved = _save_uploads(files, dest, progress)
    if not saved:
        return "请从本机选择文件夹 / 多张图片上传，或改用服务器目录。", *empty
    image_rows = [{"orig_name": orig, "image_path": str(p)} for p, orig in saved]
    try:
        weights = str(resolve_weights(model_key or YOLOE_DEFAULT, YOLOE_WEIGHTS_DIR))
        job_id, pg_job = _submit_visual_job(
            user, names, conf, nms_iou, save_preview, image_rows, weights, job_id=job_id,
        )
    except Exception as exc:
        return str(exc), *empty
    return _run_visual_serial_job(
        job_id, pg_job, saved, names, prompts, model_key, conf, nms_iou, save_preview, device, progress,
    )


def predict_visual_server_dir(
    folder, class_text, prompts, model_key, conf, nms_iou, save_preview, device, user, progress=gr.Progress(),
):
    empty = _empty_browse()
    err = _need_login(user)
    if err:
        return err, *empty
    names = parse_visual_classes(class_text or "")
    if not names:
        return "请填写类别名，例如：charging_nest, robot", *empty
    if not _prompts_to_visual(prompts):
        return "请至少加入一个视觉提示框。", *empty
    folder = (folder or "").strip()
    if not folder:
        return "请填写服务器上的图片目录。", *empty
    paths = _collect_server_images(folder)
    if not paths:
        return f"目录不存在或没有图片：{folder}", *empty
    image_rows = [{"orig_name": p.name, "image_path": str(p.resolve())} for p in paths]
    saved = [(p, p.name) for p in paths]
    try:
        weights = str(resolve_weights(model_key or YOLOE_DEFAULT, YOLOE_WEIGHTS_DIR))
        job_id, pg_job = _submit_visual_job(user, names, conf, nms_iou, save_preview, image_rows, weights)
    except Exception as exc:
        return str(exc), *empty
    return _run_visual_serial_job(
        job_id, pg_job, saved, names, prompts, model_key, conf, nms_iou, save_preview, device, progress,
    )


def store_status(user=None, job_id="") -> str:
    job_id = (job_id or "").strip()
    dirs = _job_file_dirs(job_id) if job_id else DIRS
    n_img = len(list_images(dirs["images"]))
    n_ann = len(list_records(dirs["annotations"]))
    n_prev = len(list(dirs["previews"].glob("*"))) if dirs["previews"].exists() else 0
    who = ""
    if user:
        who = f"当前用户：{user.get('username')}\n"
    if job_id:
        who += f"当前任务：{job_id}\n"
    return (
        f"{who}"
        f"图片 {n_img}  |  规范标注 {n_ann}  |  预览 {n_prev}\n"
        f"images: {dirs['images']}\n"
        f"annotations: {dirs['annotations']}\n"
        f"previews: {dirs['previews']}\n"
        f"exports: {dirs['exports']}"
    )


def build_ui() -> gr.Blocks:
    inf = CFG.get("inference", {})
    need_auth = pg.postgres_enabled(CFG)
    with gr.Blocks(title="Grounding DINO 预标注") as demo:
        user_state = gr.State(None)
        upload_stored = gr.State([])
        stored_paths = gr.State([])

        with gr.Column(visible=need_auth, elem_id="login-view") as login_view:
            gr.Markdown("# Grounding DINO 预标注\n登录后进入标注工作台")
            login_user = gr.Textbox(label="用户名")
            login_pass = gr.Textbox(label="密码", type="password")
            login_btn = gr.Button("登录", variant="primary")
            login_error = gr.Textbox(label="提示", lines=2, interactive=False)
            gr.Markdown("首次部署的管理员账号见服务器启动日志 / config.yaml，登录后请立刻改密。")

        with gr.Column(visible=not need_auth) as workspace:
            with gr.Row(elem_id="workspace-bar"):
                account_box = gr.Textbox(
                    value=_account_text(None), label="账号", lines=1, interactive=False, scale=4,
                )
                logout_btn = gr.Button("退出登录", scale=1)
            with gr.Accordion("修改密码", open=False):
                with gr.Row():
                    old_pass = gr.Textbox(label="原密码", type="password")
                    new_pass = gr.Textbox(label="新密码（至少 8 位）", type="password")
                    change_pass_btn = gr.Button("保存新密码", variant="primary")
                pass_status = gr.Textbox(label="改密结果", lines=1, interactive=False)

            with gr.Group(elem_id="prompt-card"):
                with gr.Row():
                    prompt = gr.Textbox(
                        label="提示词",
                        placeholder="person. face. hand. sofa.   或   person, helmet, safety vest",
                        lines=2,
                        scale=4,
                    )
                    parsed = gr.Textbox(label="解析出的类别", lines=2, interactive=False, scale=2)
                with gr.Row():
                    device_dd = gr.Dropdown(
                        choices=device_choices(),
                        value=_default_device_choice(),
                        label="推理设备",
                        scale=2,
                    )
                    box_th = gr.Slider(0.05, 0.90, value=float(inf.get("box_threshold", 0.35)), step=0.01, label="box")
                    text_th = gr.Slider(0.05, 0.90, value=float(inf.get("text_threshold", 0.25)), step=0.01, label="text")
                    nms_iou = gr.Slider(0.10, 0.95, value=float(inf.get("nms_iou", 0.50)), step=0.05, label="NMS")
                with gr.Accordion("GPU / Redis", open=False):
                    gr.Textbox(value=gpu_summary(), label="GPU", lines=3, interactive=False)
                    gr.Textbox(value=redis_line(), label="Redis", lines=2, interactive=False)

            with gr.Tabs():
                with gr.Tab("任务历史"):
                    jobs_table = gr.Dataframe(
                        headers=["任务 id", "用户", "状态", "进度", "失败", "提示词", "创建时间"],
                        label="历史任务",
                        interactive=False,
                        wrap=True,
                    )
                    with gr.Row():
                        job_pick = gr.Dropdown(label="选择任务", choices=[], allow_custom_value=True, scale=3)
                        jobs_refresh = gr.Button("刷新历史", scale=1)
                        open_job_btn = gr.Button("打开到标注页", variant="primary", scale=1)
                    gr.Markdown("历史任务可打开查看可视化，也可直接导出已完成标注并下载 zip。")
                    hist_fmt = gr.Radio(
                        choices=["yolo", "coco", "labelme", "all"],
                        value=CFG.get("export", {}).get("format", "yolo"),
                        label="导出格式",
                    )
                    hist_export_btn = gr.Button("导出并下载", variant="primary")
                    hist_export_status = gr.Textbox(label="导出结果", lines=3)
                    hist_export_file = gr.File(label="下载", interactive=False)

                with gr.Tab("单张预览"):
                    with gr.Row():
                        inp = gr.Image(type="pil", label="原图", height=420)
                        out = gr.Image(type="pil", label="可视化", height=420)
                    table = gr.Dataframe(
                        headers=["label", "score", "x1", "y1", "x2", "y2"],
                        label="检测结果",
                        interactive=False,
                    )
                    save_ann = gr.Checkbox(value=False, label="写入规范标注（会新建 1 张图的任务，试阈值不要勾）")
                    status = gr.Textbox(label="状态", lines=3, elem_id="status-box")
                    btn = gr.Button("检测", variant="primary")
                    btn.click(
                        predict_single,
                        inputs=[inp, prompt, box_th, text_th, nms_iou, save_ann, device_dd, user_state],
                        outputs=[out, table, status, parsed],
                    )
                    prompt.change(_parsed_text, inputs=[prompt], outputs=[parsed])

                with gr.Tab("文件夹批量预标注"):
                    with gr.Row():
                        with gr.Column(scale=1, min_width=280):
                            uploads = gr.File(
                                label="本机文件夹",
                                file_count="directory",
                                type="filepath",
                                elem_id="upload-box",
                            )
                            server_dir = gr.Textbox(
                                label="或服务器目录（不拷贝）",
                                placeholder="/data/images/batch01",
                            )
                            save_preview = gr.Checkbox(value=True, label="保存可视化图")
                            with gr.Row():
                                preview_up_btn = gr.Button("刷新待提交缩略图")
                                preview_srv_btn = gr.Button("预览服务器目录")
                            with gr.Row():
                                batch_btn = gr.Button("开始预标注", variant="primary")
                                server_btn = gr.Button("从服务器入队")
                            job_id_box = gr.Textbox(label="任务 id", lines=1)
                            upload_bar = gr.HTML(value=_bar_html(0, "尚未选择文件夹"))
                            job_bar = gr.HTML(value=_bar_html(0, "等待提交"))
                            batch_status = gr.Textbox(label="进度明细", lines=6, elem_id="status-box")
                            with gr.Row():
                                refresh_btn = gr.Button("刷新进度")
                                cancel_btn = gr.Button("取消")
                                retry_btn = gr.Button("重试失败")
                        with gr.Column(scale=2):
                            gr.Markdown("**① 待提交图片** · 选文件夹后出缩略图，点一张放大原图")
                            with gr.Row():
                                up_prev = gr.Button("上一页")
                                up_next = gr.Button("下一页")
                                up_page = gr.Number(value=1, precision=0, label="页", minimum=1, scale=1)
                                page_size = gr.Number(value=BROWSE_PAGE, precision=0, label="每页", minimum=4, scale=1)
                            up_info = gr.Textbox(label="待提交分页", lines=1, interactive=False)
                            upload_gallery = gr.Gallery(
                                label="待提交缩略图",
                                columns=6,
                                height=280,
                                object_fit="contain",
                                allow_preview=False,
                                elem_id="upload-gallery",
                            )
                            gr.Markdown("**② 提交后可视化** · 有画框的显示预览，排队中显示原图；点缩略图放大")
                            with gr.Row():
                                prev_btn = gr.Button("上一页")
                                next_btn = gr.Button("下一页")
                                jump_btn = gr.Button("跳转")
                                browse_btn = gr.Button("浏览已有")
                                page_num = gr.Number(value=1, precision=0, label="页", minimum=1)
                            page_info = gr.Textbox(label="结果分页", lines=1, interactive=False)
                            gallery = gr.Gallery(
                                label="结果缩略图（点开放大可视化）",
                                columns=6,
                                height=280,
                                object_fit="contain",
                                allow_preview=False,
                                elem_id="result-gallery",
                            )

                    vis_img = gr.Image(type="pil", label="放大查看", height=560, elem_id="vis-large")
                    vis_info = gr.Textbox(label="该图信息", lines=2)
                    vis_table = gr.Dataframe(
                        headers=["label", "score", "x1", "y1", "x2", "y2"],
                        label="该图检测结果",
                        interactive=False,
                    )

                    def _turn_upload(files, folder, page, page_size, delta, use_server=False):
                        nxt = max(1, int(page or 1) + int(delta))
                        if use_server or (not list_upload_images(files) and (folder or "").strip()):
                            return preview_server_folder(folder, nxt, page_size)
                        return preview_uploads(files, nxt, page_size)

                    uploads.change(
                        preview_uploads,
                        inputs=[uploads, up_page, page_size],
                        outputs=[upload_gallery, up_page, up_info, upload_stored, upload_bar],
                    )
                    preview_up_btn.click(
                        preview_uploads,
                        inputs=[uploads, up_page, page_size],
                        outputs=[upload_gallery, up_page, up_info, upload_stored, upload_bar],
                    )
                    preview_srv_btn.click(
                        preview_server_folder,
                        inputs=[server_dir, up_page, page_size],
                        outputs=[upload_gallery, up_page, up_info, upload_stored, upload_bar],
                    )
                    up_prev.click(
                        lambda f, d, p, z: _turn_upload(f, d, p, z, -1),
                        inputs=[uploads, server_dir, up_page, page_size],
                        outputs=[upload_gallery, up_page, up_info, upload_stored, upload_bar],
                    )
                    up_next.click(
                        lambda f, d, p, z: _turn_upload(f, d, p, z, 1),
                        inputs=[uploads, server_dir, up_page, page_size],
                        outputs=[upload_gallery, up_page, up_info, upload_stored, upload_bar],
                    )
                    upload_gallery.select(
                        show_upload_clicked,
                        inputs=[upload_stored, up_page, page_size],
                        outputs=[vis_img, vis_info],
                    )

                    batch_btn.click(
                        predict_batch,
                        inputs=[uploads, prompt, box_th, text_th, nms_iou, save_preview, device_dd, user_state],
                        outputs=[batch_status, job_id_box, stored_paths, gallery, page_num, page_info, job_bar],
                    )
                    server_btn.click(
                        predict_server_dir,
                        inputs=[server_dir, prompt, box_th, text_th, nms_iou, save_preview, device_dd, user_state],
                        outputs=[batch_status, job_id_box, stored_paths, gallery, page_num, page_info, job_bar],
                    )
                    refresh_btn.click(
                        refresh_job,
                        inputs=[job_id_box, stored_paths, page_num, page_size, user_state],
                        outputs=[batch_status, stored_paths, gallery, page_num, page_info, job_bar],
                    )
                    cancel_btn.click(
                        cancel_current_job,
                        inputs=[job_id_box, user_state, stored_paths, page_num, page_size],
                        outputs=[batch_status, stored_paths, gallery, page_num, page_info, job_bar],
                    )
                    retry_btn.click(
                        retry_failed_job,
                        inputs=[job_id_box, user_state, stored_paths, page_num, page_size],
                        outputs=[batch_status, stored_paths, gallery, page_num, page_info, job_bar],
                    )

                    def _turn_page(job_id, stored, page, page_size, delta):
                        nxt = max(1, int(page or 1) + int(delta))
                        items, page, info, stored = page_gallery(job_id, stored, nxt, page_size)
                        return stored, items, page, info

                    prev_btn.click(
                        lambda j, s, p, z: _turn_page(j, s, p, z, -1),
                        inputs=[job_id_box, stored_paths, page_num, page_size],
                        outputs=[stored_paths, gallery, page_num, page_info],
                    )
                    next_btn.click(
                        lambda j, s, p, z: _turn_page(j, s, p, z, 1),
                        inputs=[job_id_box, stored_paths, page_num, page_size],
                        outputs=[stored_paths, gallery, page_num, page_info],
                    )
                    jump_btn.click(
                        lambda j, s, p, z: page_gallery(j, s, p, z),
                        inputs=[job_id_box, stored_paths, page_num, page_size],
                        outputs=[gallery, page_num, page_info, stored_paths],
                    )
                    browse_btn.click(
                        lambda j, s, z: page_gallery(j, s, 1, z),
                        inputs=[job_id_box, stored_paths, page_size],
                        outputs=[gallery, page_num, page_info, stored_paths],
                    )
                    gallery.select(
                        show_clicked,
                        inputs=[job_id_box, stored_paths, page_num, page_size],
                        outputs=[vis_img, vis_table, vis_info],
                    )

                with gr.Tab("视觉提示预标注"):
                    gr.Markdown(
                        "用参考图上的框当视觉提示（YOLOE），写出和文本预标注相同的规范 JSON。"
                        "批量在本进程串行，不进 Redis。权重放在 `models/yoloe/`。"
                    )
                    vp_prompt_state = gr.State([])
                    vp_click_state = gr.State({"p1": None, "xyxy": None})
                    vp_ref_raw = gr.State(None)
                    with gr.Row():
                        vp_classes = gr.Textbox(
                            label="类别（逗号分隔，保留大小写）",
                            placeholder="charging_nest, robot",
                            lines=1,
                            scale=3,
                        )
                        vp_class_pick = gr.Dropdown(label="当前框的类别", choices=[], scale=1)
                        vp_model = gr.Dropdown(
                            choices=[(f"{k} · {v['label']}", k) for k, v in YOLOE_MODEL_OPTIONS.items()],
                            value=YOLOE_DEFAULT if YOLOE_DEFAULT in YOLOE_MODEL_OPTIONS else "s",
                            label="YOLOE",
                            scale=1,
                        )
                    with gr.Row():
                        vp_device = gr.Dropdown(
                            choices=device_choices(),
                            value=_yoloe_device_choice(),
                            label="推理设备",
                            scale=2,
                        )
                        vp_conf = gr.Slider(0.01, 0.90, value=YOLOE_CONF, step=0.01, label="conf")
                        vp_nms = gr.Slider(0.10, 0.95, value=YOLOE_NMS, step=0.05, label="NMS")
                    with gr.Row():
                        with gr.Column(scale=1):
                            vp_ref = gr.Image(type="pil", label="参考图（点两次组框：左上 → 右下）", height=360)
                            vp_click_info = gr.Textbox(label="画框提示", lines=2, interactive=False)
                            with gr.Row():
                                vp_add_btn = gr.Button("加入视觉提示", variant="primary")
                                vp_clear_btn = gr.Button("清空提示")
                            vp_prompt_table = gr.Dataframe(
                                headers=["#", "label", "x1", "y1", "x2", "y2"],
                                label="已加入的视觉提示",
                                interactive=False,
                            )
                        with gr.Column(scale=1):
                            vp_target = gr.Image(type="pil", label="单张目标图", height=360)
                            vp_out = gr.Image(type="pil", label="可视化", height=360)
                    vp_table = gr.Dataframe(
                        headers=["label", "score", "x1", "y1", "x2", "y2"],
                        label="检测结果",
                        interactive=False,
                    )
                    vp_save_ann = gr.Checkbox(value=False, label="写入规范标注（会新建 1 张图的任务）")
                    vp_single_status = gr.Textbox(label="单张状态", lines=3)
                    vp_single_btn = gr.Button("检测当前图", variant="primary")
                    gr.Markdown("### 文件夹批量（串行，不进 Redis）")
                    with gr.Row():
                        vp_uploads = gr.File(
                            label="本机文件夹",
                            file_count="directory",
                            type="filepath",
                        )
                        vp_server_dir = gr.Textbox(label="或服务器目录（不拷贝）", placeholder="/data/images/batch01")
                    vp_save_preview = gr.Checkbox(value=True, label="保存可视化图")
                    with gr.Row():
                        vp_batch_btn = gr.Button("开始视觉预标注", variant="primary")
                        vp_server_btn = gr.Button("从服务器目录跑")
                    vp_batch_status = gr.Textbox(label="批量进度", lines=6)
                    vp_gallery = gr.Gallery(
                        label="结果缩略图",
                        columns=6,
                        height=220,
                        object_fit="contain",
                        allow_preview=False,
                    )

                    vp_classes.change(_class_choices, inputs=[vp_classes], outputs=[vp_class_pick])
                    vp_ref.upload(
                        _set_ref_raw,
                        inputs=[vp_ref],
                        outputs=[vp_ref_raw, vp_click_state, vp_click_info],
                    )
                    vp_ref.select(
                        _click_visual_ref,
                        inputs=[vp_ref_raw, vp_click_state, vp_prompt_state],
                        outputs=[vp_ref, vp_click_state, vp_click_info],
                    )
                    vp_add_btn.click(
                        add_visual_prompt,
                        inputs=[vp_ref_raw, vp_click_state, vp_class_pick, vp_classes, vp_prompt_state],
                        outputs=[vp_ref, vp_prompt_state, vp_prompt_table, vp_click_state, vp_click_info],
                    )
                    vp_clear_btn.click(
                        clear_visual_prompts,
                        inputs=[vp_ref_raw],
                        outputs=[vp_ref, vp_prompt_state, vp_prompt_table, vp_click_state, vp_click_info],
                    )
                    vp_single_btn.click(
                        predict_visual_single,
                        inputs=[
                            vp_target, vp_classes, vp_prompt_state, vp_model,
                            vp_conf, vp_nms, vp_save_ann, vp_device, user_state,
                        ],
                        outputs=[vp_out, vp_table, vp_single_status],
                    )
                    vp_batch_btn.click(
                        predict_visual_batch,
                        inputs=[
                            vp_uploads, vp_classes, vp_prompt_state, vp_model,
                            vp_conf, vp_nms, vp_save_preview, vp_device, user_state,
                        ],
                        outputs=[vp_batch_status, job_id_box, stored_paths, vp_gallery, page_num, page_info, job_bar],
                    )
                    vp_server_btn.click(
                        predict_visual_server_dir,
                        inputs=[
                            vp_server_dir, vp_classes, vp_prompt_state, vp_model,
                            vp_conf, vp_nms, vp_save_preview, vp_device, user_state,
                        ],
                        outputs=[vp_batch_status, job_id_box, stored_paths, vp_gallery, page_num, page_info, job_bar],
                    )

                with gr.Tab("导出 / 清理"):
                    store_box = gr.Textbox(label="当前存储", lines=6)
                    refresh = gr.Button("刷新存储状态")
                    refresh.click(store_status, inputs=[user_state, job_id_box], outputs=[store_box])
                    gr.Markdown("### 导出当前任务（只含已完成）")
                    fmt = gr.Radio(
                        choices=["yolo", "coco", "labelme", "all"],
                        value=CFG.get("export", {}).get("format", "yolo"),
                        label="格式",
                    )
                    export_btn = gr.Button("导出并下载", variant="primary")
                    export_status = gr.Textbox(label="导出结果", lines=5)
                    export_file = gr.File(label="下载 zip", interactive=False)
                    export_btn.click(
                        do_export,
                        inputs=[fmt, user_state, job_id_box],
                        outputs=[export_status, export_file],
                    )

                    with gr.Accordion("管理员：清理过期文件", open=False):
                        keep_days = gr.Number(
                            value=float(CLEAN_CFG.get("keep_days", 7)),
                            label="保留天数",
                            precision=1,
                        )
                        include_exports = gr.Checkbox(
                            value=bool(CLEAN_CFG.get("include_exports", True)),
                            label="同时清理 data/exports",
                        )
                        dry_run = gr.Checkbox(value=True, label="先预览（不真删）")
                        clean_btn = gr.Button("清理")
                        clean_status = gr.Textbox(label="清理结果", lines=8)
                        clean_btn.click(
                            do_cleanup,
                            inputs=[keep_days, include_exports, dry_run, user_state],
                            outputs=[clean_status],
                        )

                with gr.Tab("管理员"):
                    gr.Markdown("只有管理员可以创建 / 停用用户、重置密码。")
                    with gr.Row():
                        new_name = gr.Textbox(label="新用户名")
                        new_user_pass = gr.Textbox(label="密码（至少 8 位）", type="password")
                        new_role = gr.Radio(choices=["user", "admin"], value="user", label="角色")
                    create_btn = gr.Button("创建用户", variant="primary")
                    create_status = gr.Textbox(label="结果", lines=2)
                    users_table = gr.Dataframe(
                        headers=["id", "用户名", "角色", "状态", "任务数", "创建时间"],
                        label="用户列表",
                        interactive=False,
                    )
                    users_refresh = gr.Button("刷新用户列表")
                    with gr.Row():
                        manage_name = gr.Textbox(label="要管理的用户名")
                        reset_pass = gr.Textbox(label="重置为新密码", type="password")
                        reset_btn = gr.Button("重置密码")
                        disable_btn = gr.Button("停用")
                        enable_btn = gr.Button("启用")
                    create_btn.click(
                        do_create_user,
                        inputs=[user_state, new_name, new_user_pass, new_role],
                        outputs=[create_status, users_table],
                    )
                    users_refresh.click(refresh_users, inputs=[user_state], outputs=[users_table])
                    reset_btn.click(
                        do_reset_password,
                        inputs=[user_state, manage_name, reset_pass],
                        outputs=[create_status, users_table],
                    )
                    disable_btn.click(
                        lambda actor, name: do_disable_user(actor, name, True),
                        inputs=[user_state, manage_name],
                        outputs=[create_status, users_table],
                    )
                    enable_btn.click(
                        lambda actor, name: do_disable_user(actor, name, False),
                        inputs=[user_state, manage_name],
                        outputs=[create_status, users_table],
                    )

        change_pass_btn.click(
            do_change_password,
            inputs=[user_state, old_pass, new_pass],
            outputs=[pass_status],
        )
        login_btn.click(
            do_login,
            inputs=[login_user, login_pass],
            outputs=[user_state, account_box, jobs_table, job_pick, login_view, workspace, login_error],
        )
        try:
            login_pass.submit(
                do_login,
                inputs=[login_user, login_pass],
                outputs=[user_state, account_box, jobs_table, job_pick, login_view, workspace, login_error],
            )
        except Exception:
            pass
        logout_btn.click(
            do_logout,
            outputs=[user_state, account_box, jobs_table, job_pick, login_view, workspace, login_error],
        )
        jobs_refresh.click(refresh_jobs, inputs=[user_state], outputs=[jobs_table, job_pick])
        open_job_btn.click(
            open_selected_job,
            inputs=[job_pick, user_state, page_size],
            outputs=[batch_status, job_id_box, stored_paths, gallery, page_num, page_info, job_bar],
        )
        hist_export_btn.click(
            do_export,
            inputs=[hist_fmt, user_state, job_pick],
            outputs=[hist_export_status, hist_export_file],
        )
        try:
            tick = gr.Timer(2)
            tick.tick(
                tick_job,
                inputs=[job_id_box, stored_paths, page_num, page_size, user_state],
                outputs=[batch_status, stored_paths, gallery, page_num, page_info, job_bar],
            )
        except Exception:
            pass

        gr.Markdown(f"<span style='color:#94a3b8;font-size:12px'>模型 `{MODEL_ID}` · `{DEVICE}` gpu_id=`{GPU_ID}`</span>")
    return demo



if __name__ == "__main__":
    if CLEAN_CFG.get("on_startup"):
        cleanup(
            images_dir=DIRS["images"],
            annotations_dir=DIRS["annotations"],
            previews_dir=DIRS["previews"],
            exports_dir=DIRS["exports"],
            keep_days=float(CLEAN_CFG.get("keep_days", 7)),
            include_exports=bool(CLEAN_CFG.get("include_exports", True)),
            dry_run=False,
        )

    print("=" * 60, flush=True)
    print("Grounding DINO 预标注", flush=True)
    print(gpu_summary(), flush=True)
    if pg.postgres_enabled(CFG):
        try:
            print(pg.init_db(CFG), flush=True)
            print(pg.ping(CFG), flush=True)
            admin = (CFG.get("postgres") or {}).get("admin_username") or "admin"
            print(f"默认管理员：{admin} / 见 config.yaml postgres.admin_password", flush=True)
        except Exception as exc:
            print(f"PostgreSQL 初始化失败：{exc}", flush=True)
            print("界面仍可开，但登录 / 任务台账不可用。", flush=True)
    else:
        print("postgres.enabled=false，无账号体系。", flush=True)
    print(f"正在加载界面预览模型：{MODEL_ID}  device={DEVICE}  gpu_id={GPU_ID}", flush=True)
    get_detector()
    print(_status, flush=True)
    print(start_replica_workers(), flush=True)
    print("=" * 60, flush=True)

    demo = build_ui()
    host = str(UI_CFG.get("server_name") or "0.0.0.0")
    port = int(UI_CFG.get("server_port", 7860))
    lan_ip = host if host not in ("0.0.0.0", "::", "") else host_ipv4()
    print(f"Running on host URL:  http://{lan_ip}:{port}", flush=True)
    print(f"Bind: {host}:{port}", flush=True)

    launch_kw = dict(
        server_name=host,
        server_port=port,
        share=bool(UI_CFG.get("share", False)),
        inbrowser=bool(UI_CFG.get("inbrowser", False)),
        quiet=True,
        prevent_thread_lock=False,
        css=APP_CSS,
        fill_width=True,
    )
    try:
        demo.launch(**launch_kw)
    except TypeError:
        launch_kw.pop("fill_width", None)
        try:
            demo.launch(**launch_kw)
        except TypeError:
            launch_kw.pop("css", None)
            demo.launch(**launch_kw)
