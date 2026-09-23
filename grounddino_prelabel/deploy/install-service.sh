#!/usr/bin/env bash
# 在 192.168.110.246 上以有 sudo 的账号执行一次：
#   bash /home/chenchen/ssd/grounddino_prelabel/deploy/install-service.sh
set -euo pipefail

APP_DIR=/home/chenchen/ssd/grounddino_prelabel
ENV_PY=/home/chenchen/miniconda3/envs/ai-mlops/bin/python
UNIT_SRC="$APP_DIR/deploy/grounddino-prelabel.service"
UNIT_DST=/etc/systemd/system/grounddino-prelabel.service
LOG_DIR="$APP_DIR/logs"

if [[ ! -x "$ENV_PY" ]]; then
  echo "找不到 conda 环境：$ENV_PY" >&2
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
chown -R chenchen:chenchen "$LOG_DIR"
touch "$LOG_DIR/app.log" "$LOG_DIR/app.err.log"
chown chenchen:chenchen "$LOG_DIR/app.log" "$LOG_DIR/app.err.log"

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
echo "界面：     http://192.168.110.246:7860"
