本地 Grounding DINO 权重（不联网）。

把 Hugging Face 仓库 `IDEA-Research/grounding-dino-tiny` 的完整快照放到本目录：

```
models/grounding-dino-tiny/
  model.safetensors     # 必须正好 689359096 字节
  config.json
  preprocessor_config.json
  tokenizer.json
  tokenizer_config.json
  vocab.txt
  special_tokens_map.json
  added_tokens.json
```

`config.yaml` 里 `model.local_files_only: true`。缺文件或体积不对会直接报错，不会去 Hugging Face 下载。

权重约 690MB，不进 Git。部署时把整个 `grounding-dino-tiny/` 目录拷到目标机器。

## YOLOE 视觉提示

把对应 `.pt` 放到 `models/yoloe/`（不进 Git）：

```
models/yoloe/yoloe-11s-seg.pt
```

规格：`s` → `yoloe-11s-seg.pt`，`m` → `yoloe-11m-seg.pt`，`l` → `yoloe-11l-seg.pt`，`x` → `yoloe-26x-seg.pt`。
权重来自 Ultralytics YOLOE 发布文件，文件名不要改。
