#!/usr/bin/env bash
# 按本机路径改 APP_DIR / ENV_PY 后执行一次：
#   bash deploy/install-service.sh
set -euo pipefail

APP_DIR=/opt/grounddino_prelabel
ENV_PY=/opt/conda/envs/yolo/bin/python
UNIT_SRC="$APP_DIR/deploy/grounddino-prelabel.service"
UNIT_DST=/etc/systemd/system/grounddino-prelabel.service
LOG_DIR="$APP_DIR/logs"
APP_USER="${SUDO_USER:-$USER}"

if [[ ! -x "$ENV_PY" ]]; then
  echo "找不到 Python：$ENV_PY" >&2
  exit 1
fi
if [[ ! -f "$APP_DIR/app.py" ]]; then
  echo "找不到 app.py：$APP_DIR/app.py" >&2
  exit 1
fi
if [[ ! -f "$UNIT_SRC" ]]; then
  echo "找不到 service 文件：$UNIT_SRC" >&2
  exit 1
fi

mkdir -p "$LOG_DIR"
touch "$LOG_DIR/app.log" "$LOG_DIR/app.err.log"
chown -R "$APP_USER:$APP_USER" "$LOG_DIR"

sudo cp "$UNIT_SRC" "$UNIT_DST"
sudo systemctl daemon-reload
sudo systemctl enable grounddino-prelabel.service
sudo systemctl restart grounddino-prelabel.service
sleep 1
sudo systemctl --no-pager --full status grounddino-prelabel.service || true

echo
echo "开机自启已打开。"
echo "日志文件： $LOG_DIR/app.log"
echo "错误日志： $LOG_DIR/app.err.log"
echo "journal：  journalctl -u grounddino-prelabel -f"
