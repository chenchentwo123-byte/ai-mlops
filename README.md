# Grounding DINO / YOLOE 预标注

两种预标方式，写出**同一套规范 JSON**，再导出 YOLO / COCO / LabelMe：

| 方式 | 模型 | 怎么告诉它类别 |
| --- | --- | --- |
| 文本提示词 | [Grounding DINO](https://github.com/IDEA-Research/GroundingDINO) | `person. helmet. safety vest.` |
| 视觉提示 | [YOLOE](https://docs.ultralytics.com/models/yoloe/) | 在参考图上画框当样例 |

适合室内、工地、自定义类别还没有训练数据时，先批量预标，再人工改。

## 特点

- **本地权重**：`models/grounding-dino-tiny/` 与 `models/yoloe/`，缺文件直接报错，不会联网下载
- **每个框一个类**：DINO 按类别 token 跨度打分；长标签列表自动分组推理
- **视觉样例**：参考图点两次组框，YOLOE 注入类别后再扫文件夹
- **先存规范 JSON，导出时再转格式**；YOLO 导出为 `images/` + `labels/` + 相对路径 `data.yaml`，可直接训练
- **显卡可配置**：`auto` / `cpu` / `cuda:0` / `cuda:1`，界面和命令行都能选
- **同卡多模型**：一张卡上启动 `workers.replicas` 份 DINO，经 Redis 并行拉批量任务
- **账号 + PostgreSQL**：管理员建用户；每次上传是一个任务，进度落库
- **中断后续跑**：重启后从未完成的图重新入队，已完成的不重跑
- **定期清理**过期图片和标注（仅管理员）

## 架构

```
浏览器 / CLI
    │
    ▼
app.py (Gradio)  或  prelabel.py (单进程批量)
    │
    ├─ 文本单张 / 批量：Grounding DINO
    │     批量 + redis.enabled → gdino:queue → worker.py × replicas
    ├─ 视觉提示：YOLOE（第一次点检测才加载；批量在本进程串行，不进 Redis）
    └─ 进度 / 账号 → PostgreSQL；原图 / JSON / 预览 → 磁盘
```

登录后文件落在 `data/users/<用户名>/jobs/<任务id>/`。CLI 和不走账号时用 `data/images`、`data/annotations`、`data/previews`。

## 目录

```
app.py                      Gradio 界面（登录、预览、批量、视觉提示、导出）
prelabel.py                 命令行批量预标注（不经过 Redis）
worker.py                   Redis 上的一份 Grounding DINO 进程
start_workers.py            同一张卡拉起 replicas 份 worker
cleanup.py                  按天数清理过期文件
config.example.yaml         配置模板（复制为 config.yaml）
src/
  detector.py               Grounding DINO 推理、赋类、NMS、GPU 选择
  yoloe_detector.py         YOLOE 视觉提示推理（懒加载）
  prompts.py                文本提示词拆分
  visualize.py              画框
  store.py                  规范 JSON
  exporters.py              YOLO / COCO / LabelMe
  queue.py                  Redis 入队 / 出队 / 续跑
  db.py                     Postgres 用户与任务
  spawn.py                  拉起 / 停止 replica
  cleanup.py                过期文件删除
models/grounding-dino-tiny/ 本地 DINO 权重（须自行放入，约 690MB）
models/yoloe/               本地 YOLOE 权重（须自行放入，如 yoloe-11s-seg.pt）
data/                       图片、标注、预览、导出
deploy/                     systemd 示例
```

## 环境

**Python 3.12**（3.10 / 3.11 一般可用。不要用 3.14）。

先按机器 CUDA 安装 PyTorch，再装其余依赖。`requirements.txt` **不钉死 torch**，避免把已装好的 CUDA 版覆盖成 CPU 版。

```bash
# 示例：CUDA 12.8，按 https://pytorch.org 换成你的 index
python -m pip install torch torchvision --index-url https://download.pytorch.org/whl/cu128
python -m pip install -r requirements.txt
```

已验证组合（仅作参考）：Python 3.12、torch 2.x + cu128、transformers 5.x、gradio 6.x。

## 配置

```bash
cp config.example.yaml config.yaml
```

改 `model.device` / `gpu_id`、Redis、Postgres、`yoloe.gpu_id`。

**`config.yaml` 已在 `.gitignore`，不要提交。** 内网主机、数据库密码、管理员口令只写在这份本地文件里。仓库里的模板是 `config.example.yaml`（Postgres / Redis 默认 `127.0.0.1`，密码占位符 `CHANGE_ME`）。

## 权重（不联网）

把完整的 Hugging Face 快照放到：

```
models/grounding-dino-tiny/
  model.safetensors     # 必须正好 689359096 字节
  config.json
  tokenizer.json
  ...
```

来源：`IDEA-Research/grounding-dino-tiny`。加载时 `local_files_only=True`。目录缺失或文件不完整会直接报错，**不会去网上拉**。

部署到另一台机器时，把整个 `models/grounding-dino-tiny/` 一起拷过去，不要断点续传拼 `model.safetensors`。

自检：

```bash
python -c "from src.detector import GroundingDINODetector; d=GroundingDINODetector(); d.load(); print(d.device, d.model_id)"
```

应打印 `cuda:0`（或你配置的卡）和本地绝对路径。

## 提示词

用句号、逗号或换行分隔类别，下面三种等价：

```text
person. face. hand. sofa. tv.
person.face.hand.sofa.tv
person, face, hand, sofa, tv
```

内部会转成 Grounding DINO caption：`person. face. hand. sofa. tv.`

- 每个检测框只赋 **一个** 类别（该类 token 跨度最高分）
- 类别超过 `chunk_size` 会分组推理
- 类别名尽量用**英文名词**；中文不稳定
- 复合词写成一个短语更好：`coffee table`、`tv cabinet`、`safety vest`

漏检多：把 box threshold 降到 `0.25`。误检多：升到 `0.45`–`0.55`。同一物体多个框：NMS 降到 `0.4`。

## 命令行

不加载 Gradio、不依赖 Redis。适合本机文件夹直接预标。

```bash
python prelabel.py --images data/images --prompt "person. face. sofa."
python prelabel.py --images data/images --prompt "person. face. sofa." --format yolo
python prelabel.py --export-only --format all
python prelabel.py --device cuda --gpu-id 0 --prompt "person. sofa."
python cleanup.py --keep-days 7 --dry-run
```

Windows 可用 `.\run_prelabel.ps1`（脚本会优先用本机 conda `yolo` 环境，没有则回退到 `python`）。

| 参数 | 含义 | 默认 |
| --- | --- | --- |
| `--images` | 输入图片目录 | `data/images` |
| `--prompt` | 类别提示词 | 推理时必填 |
| `--format` | 推理后立刻导出 `yolo` / `coco` / `labelme` / `all` | 不导出 |
| `--export-only` | 只转换已有 JSON，不加载模型 | 关 |
| `--device` | `auto` / `cpu` / `cuda` / `cuda:N` | `auto` |
| `--box-threshold` | 框置信度 | `0.35` |
| `--text-threshold` | 文本匹配阈值 | `0.25` |
| `--nms-iou` | 同类 NMS | `0.50` |
| `--chunk-size` | 每组类别数 | 配置文件 |
| `--no-preview` | 不写画框图 | 关 |

## 可视化界面

```bash
python app.py
```

Windows：`.\run_ui.ps1`。默认监听 `0.0.0.0:7860`。

| 页签 | 作用 |
| --- | --- |
| 任务历史 | 打开已有任务、看可视化、导出已完成图 |
| 单张预览 | 上传一张图 + 提示词，立刻画框 |
| 文件夹批量预标注 | 浏览器选本机目录，上传后入队 |
| 视觉提示预标注 | 参考图上点两次组框，YOLOE 按样例扫文件夹（本进程串行，不进 Redis） |
| 导出 / 清理 | 从规范 JSON 转格式；管理员可按天数删过期文件 |
| 管理员 | 创建 / 停用用户、重置密码 |

`postgres.enabled: true` 时需要登录。默认管理员只在 `users` 表还没有该用户时写入，登录后立刻改密。普通用户只能看自己的任务。

`redis.enabled: true` 时，`app.py` 会按 `workers.replicas` 拉起 worker，不要再同时跑 `python start_workers.py`（显存会翻倍）。Redis 关闭时，批量在界面进程里串行跑。

## 同卡多模型

```yaml
redis:
  enabled: true
workers:
  gpu_id: 0
  replicas: 3
```

需要本机 Redis。单独拉 worker：

```bash
python start_workers.py
```

tiny 大约每份 1.5–2GB 显存，别把卡撑满。界面预览还占一份 `model.gpu_id` 上的模型；worker 卡和界面卡相同的话把 replicas 算上这份。

视觉提示用的 YOLOE **第一次点检测才加载**，不要和 `workers.gpu_id` 抢同一张已经堆满 replica 的卡。在 `config.yaml` 的 `yoloe.gpu_id` 里指定另一张卡。

## 视觉提示预标注

文本提示词找不到、或类别要用真实样例时，打开 **视觉提示预标注**：

1. 填写类别（逗号分隔，**保留大小写**），例如 `charging_nest, robot`
2. 上传参考图，在图上点两次组成框（左上 → 右下），选类别，点「加入视觉提示」。每个类别至少一框
3. 单张：再上传目标图点「检测当前图」
4. 批量：选文件夹或填服务器目录，点「开始视觉预标注」。**始终在界面进程串行**，不会进 `gdino:queue`

把权重放到 `models/yoloe/`：

```
models/yoloe/yoloe-11s-seg.pt
models/yoloe/yoloe-11m-seg.pt   # 可选
models/yoloe/yoloe-11l-seg.pt
models/yoloe/yoloe-26x-seg.pt
```

权重来自 Ultralytics YOLOE 发布文件，文件名不要改。缺文件会直接报错，不会去网上拉。依赖 `ultralytics==8.4.128`（`requirements.txt` 已钉死）。

写出的规范 JSON 与文本预标注相同，可用同一套任务历史 / 导出。`prompt` 字段形如 `visual: charging_nest, robot`。

## 标注与导出

规范标注（始终写入）：`data/annotations/<图名>.json` 或任务目录下的 `annotations/`。

```json
{
  "image": "sample.jpg",
  "width": 735,
  "height": 1280,
  "prompt": "person. face. sofa.",
  "classes": ["person", "face", "sofa"],
  "count": 1,
  "detections": [
    { "label": "person", "score": 0.772, "bbox_xyxy": [1, 251, 733, 1277] }
  ]
}
```

框是像素坐标 `x1 y1 x2 y2`。改阈值重新导出不必重跑模型，除非要重新检测。

导出目录：

```
data/exports/
  yolo/data.yaml          # path: .  train/val: images
  yolo/images/<原图>
  yolo/labels/<图名>.txt  # class_id cx cy w h（归一化 0–1）
  coco/annotations.json
  labelme/<图名>.json
  classes.json
```

YOLO 这一套可以直接给 Ultralytics 训练（`images/` 与 `labels/` 并列）。`data.yaml` 的 `names` 顺序优先用该任务写入 JSON 的 `classes`，不要把两套提示词的导出混在一个目录里。

## 中断后续跑

重启（Ctrl+C 后再 `python app.py`）**不会清库、不会删图**。Redis 队列会丢，但启动时会：

1. 把卡在 `running` 的图改回 `queued`（进程被掐死时没写完）
2. 只把未跑完的图重新推进 Redis
3. 已 `done` 的跳过，失败的不自动重试
4. 原图已经不在磁盘上的记为失败

## 定期清理

超过 `keep_days`（默认 7 天，按修改时间）会删 `images` / `annotations` / `previews`，以及可选的 `exports`。`.gitkeep` 不会删。先 `--dry-run` 再真删。

## 部署

1. 拷贝代码 + `models/grounding-dino-tiny/`（以及要用视觉提示时的 `models/yoloe/`）+ 自己的 `config.yaml`
2. Python 3.12，按服务器 CUDA 装 PyTorch，再 `pip install -r requirements.txt`
3. 需要批量并行时启动 Redis；需要账号时连已有 Postgres（启动时 `CREATE TABLE IF NOT EXISTS`）
4. `python app.py`，或参考 `deploy/grounddino-prelabel.service` 做成 systemd（先改工作目录、用户、Python 路径）
5. 不要把真实 IP、密码写进 README、示例配置或 service 文件

`HF_HUB_OFFLINE=1` 和 `TRANSFORMERS_OFFLINE=1` 可防止进程在缺文件时试图联网。

## 常见问题

**每个框都贴上全部标签**  
已按 token 跨度赋类。确认跑的是当前 `src/detector.py` / `src/prompts.py`，界面「解析出的类别」是多个类而不是一整句。

**报错本地模型不存在 / 权重不完整**  
确认 `model.safetensors` 在，且大小正好 689359096 字节。不要把 `model.id` 改回 Hugging Face 仓库名。

**配置了 cuda 却报错**  
`python -c "import torch; print(torch.cuda.is_available())"` 应为 True。驱动、CUDA 与 torch 的 cuXXX 要匹配。

**显存不够**  
减小 `replicas`、把 `chunk_size` 降到 4–6、确认 `dtype: auto`（GPU 上已是 fp16）。

**批量没有变快**  
确认 Redis 在跑、worker 已拉起、界面批量页显示任务在排队。

**中文提示词乱标**  
改用英文类别名。
