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

> 以上三个 `.sh` 脚本已内置这些参数；也可以在命令后追加额外参数覆盖，例如 `./denoise.sh --step 10`。
- `--rank`：子空间维度 `r`（适配器输出通道数，RRQR选带数）。
- `--posterior_update_steps`：每个采样步的后验内循环更新次数。
- `--adapter_lr`：适配器学习率。
- `--factor_lr`：矩阵因子加性微调学习率。
- `--adapter_hidden`：轻量适配器隐藏通道数。
- `--no_rrqr`：关闭 RRQR（关闭后改为等间隔选带）。
- `--dataroot`：可以是 `.mat` 文件路径，也可以是目录（如 `data/`）。
- `--data_file`：当 `--dataroot` 是目录时，指定文件名（如 `car.mat`）。

### 先看你的目录怎么传参（对应你截图）
你的目录是：
```bash
/home/wuweihao/HDPS/data/
```
而 `.mat` 文件在该目录下（如 `ironman.mat`, `car.mat`, `animal_garden.mat` ...），
所以运行时统一用：
```bash
--dataroot /home/wuweihao/HDPS/data --data_file <文件名.mat>
```

例如：
```bash
--dataroot /home/wuweihao/HDPS/data --data_file ironman.mat
```

### Denoise
```bash
./denoise.sh
```

### Super-Resolution
```bash
./super-resolution.sh
```

### Inpainting
```bash
./inpainting.sh
```

### 关闭后验更新（只前向，不更新适配器/因子）
```bash
python main.py \
  -eta1 16 -eta2 10 --k 8 -step 20 \
  -dn Houston --task denoise --task_params 50 \
  --dataroot data --data_file fruit.mat \
  --rank 6 --posterior_update_steps 0 \
  -gpu 0 --beta_schedule exp
```

### 直接跑你截图里的 5 个文件（示例）
> 下面以 denoise 为例；你可以把 `--task` 和 `--task_params` 改成 `sr` / `inpainting` 对应配置。

```bash
python main.py -dn Houston --task denoise --task_params 50 --rank 6 --posterior_update_steps 1 \
  --adapter_lr 1e-4 --factor_lr 5e-3 --adapter_hidden 16 --k 8 -step 20 -gpu 0 --beta_schedule exp \
  --dataroot /home/wuweihao/HDPS/data --data_file ironman.mat

python main.py -dn Houston --task denoise --task_params 50 --rank 6 --posterior_update_steps 1 \
  --adapter_lr 1e-4 --factor_lr 5e-3 --adapter_hidden 16 --k 8 -step 20 -gpu 0 --beta_schedule exp \
  --dataroot /home/wuweihao/HDPS/data --data_file car.mat

python main.py -dn Houston --task denoise --task_params 50 --rank 6 --posterior_update_steps 1 \
  --adapter_lr 1e-4 --factor_lr 5e-3 --adapter_hidden 16 --k 8 -step 20 -gpu 0 --beta_schedule exp \
  --dataroot /home/wuweihao/HDPS/data --data_file animal_garden.mat

python main.py -dn Houston --task denoise --task_params 50 --rank 6 --posterior_update_steps 1 \
  --adapter_lr 1e-4 --factor_lr 5e-3 --adapter_hidden 16 --k 8 -step 20 -gpu 0 --beta_schedule exp \
  --dataroot /home/wuweihao/HDPS/data --data_file fruit.mat

python main.py -dn Houston --task denoise --task_params 50 --rank 6 --posterior_update_steps 1 \
  --adapter_lr 1e-4 --factor_lr 5e-3 --adapter_hidden 16 --k 8 -step 20 -gpu 0 --beta_schedule exp \
  --dataroot /home/wuweihao/HDPS/data --data_file chaos_traffic.mat
```

---

## 3) 数据存储格式（`.mat`）

现在运行脚本会**根据 `gt` 自动合成 `input`**，所以 `.mat` 最少只需要：

### 必需键
- `gt`：真值高光谱图像，shape = `[H, W, C]`，建议 `float32`

### 按任务自动合成退化观测
- `--task denoise`：按 `--task_params` 指定的噪声强度（如 `50`）自动加高斯噪声生成 `input`
- `--task inpainting`：按 `--task_params` 指定的遮挡率（如 `0.8`）自动采样 mask，并生成 `input = gt * mask`
- `--task sr`：按 `--task_params` 指定的缩放比例（如 `0.25`）自动做 blur+downsample 生成 `input`

### 说明
- 你仍然可以在 `.mat` 里额外放其他字段，但当前脚本只依赖 `gt`。
- `gt` 会被加载为 `[1, C, H, W]` 张量参与后续流程。

---

## 4) 数据路径约定

`main.py` 读取优先级：
1. 若 `--dataroot` 直接是 `.mat` 文件，直接读取该文件。  
2. 若 `--dataroot` 是目录且指定了 `--data_file`，读取该文件。  
3. 若 `--dataroot` 是目录但未指定 `--data_file`，自动读取目录下按字母序第一个 `.mat` 文件。  
4. 若 `--dataroot` 不存在，才回退到历史的 `dataname/task/task_params` 路径拼接逻辑。
