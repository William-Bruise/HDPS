# HIR-Diff (Runtime Guide Only)

本说明只保留：
1. 预训练模型下载  
2. 运行脚本  
3. 数据存储格式与键名规范

---

## 1) 下载预训练模型

下载预训练扩散模型：  
[I190000_E97_gen.pth](https://www.dropbox.com/sh/z6k5ixlhkpwgzt5/AAApBOGEUhHa4qZon0MxUfmua?dl=0)

将文件放到：
```bash
checkpoints/diffusion/I190000_E97_gen.pth
```

---

## 2) 运行脚本

### 参数说明（核心）
- `--rank`：子空间维度 `r`（适配器输出通道数，RRQR选带数）。
- `--posterior_update_steps`：每个采样步的后验内循环更新次数。
- `--adapter_lr`：适配器学习率。
- `--factor_lr`：矩阵因子加性微调学习率。
- `--adapter_hidden`：轻量适配器隐藏通道数。
- `--no_rrqr`：关闭 RRQR（关闭后改为等间隔选带）。

### Denoise
```bash
python main.py \
  -eta1 16 -eta2 10 --k 8 -step 20 \
  -dn Houston --task denoise --task_params 50 \
  --rank 6 --posterior_update_steps 1 \
  --adapter_lr 1e-4 --factor_lr 5e-3 --adapter_hidden 16 \
  -gpu 0 --beta_schedule exp
```

### Super-Resolution
```bash
python main.py \
  -eta1 500 -eta2 12 --k 8 -step 20 \
  -dn WDC --task sr --task_params 0.25 \
  --rank 6 --posterior_update_steps 1 \
  --adapter_lr 1e-4 --factor_lr 5e-3 --adapter_hidden 16 \
  -gpu 0 --beta_schedule exp
```

### Inpainting
```bash
python main.py \
  -eta1 8 -eta2 6 --k 5 -step 20 \
  -dn Salinas --task inpainting --task_params 0.8 \
  --rank 6 --posterior_update_steps 1 \
  --adapter_lr 1e-4 --factor_lr 5e-3 --adapter_hidden 16 \
  -gpu 0 --beta_schedule exp
```

### 关闭后验更新（只前向，不更新适配器/因子）
```bash
python main.py \
  -eta1 16 -eta2 10 --k 8 -step 20 \
  -dn Houston --task denoise --task_params 50 \
  --rank 6 --posterior_update_steps 0 \
  -gpu 0 --beta_schedule exp
```

---

## 3) 数据存储格式（`.mat`）

程序会从 `.mat` 中读取以下键名：

### 通用键（所有任务必须有）
- `input`：退化高光谱图像，shape = `[H, W, C]`
- `gt`：真值高光谱图像，shape = `[H, W, C]`
- `sigma`：噪声相关参数（标量或数组，按你数据生成流程保存）

### SR 任务额外键（`--task sr`）
- `scale`：下采样比例（例如 `0.25` 对应 x4 超分）

### Inpainting 任务额外键（`--task inpainting`）
- `mask`：掩膜，shape = `[H, W, C]`，建议 0/1

### 命名与类型建议
- 键名严格使用：`input`, `gt`, `sigma`, `scale`, `mask`
- `input` 和 `gt` 建议保存为 `float32`
- `input` 与 `gt` 的空间大小、波段数必须一致

---

## 4) 数据路径约定

`main.py` 会根据以下参数自动拼接测试路径：
- `--dataname` in `{Houston, WDC, Salinas}`
- `--task` in `{denoise, sr, inpainting}`
- `--task_params`（如 `50`, `0.25`, `0.8`）

请按项目现有目录结构组织数据文件。
