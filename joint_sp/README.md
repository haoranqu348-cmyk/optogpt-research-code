# joint_sp — OptoGPT Joint s+p Polarization Inverse Design

**θ=60°下，输入284维[Rs(71),Ts(71),Rp(71),Tp(71)]，模型输出同一多层介电薄膜结构，TMM同时满足s和p高透射（最终由TMM物理验证保证，非模型自身保证）。**

> 角度边界：当前模型只在 60° 数据合同上训练。对 0–80° 的结论必须使用
> `scripts/evaluate_wide_angle.py` 对同一结构逐角度运行 s/p TMM；60° 单点成功不能
> 解释为宽角设计能力。

## 0–80° 宽角高透可靠性评估

先用少量候选做计算评估，不修改 checkpoint，也不生成训练数据：

```powershell
python -B .\joint_sp\scripts\evaluate_wide_angle.py `
  --model .\joint_sp\formal_checkpoints_500k_v2_20260725_03\optogpt_joint_sp_500k_v2_best.pt `
  --output_dir .\joint_sp\wide_angle_eval_20260727_smoke `
  --num_candidates 64 --seeds 42,43 `
  --coarse_angles 0:80:5 --dense_angles 0:80:1 `
  --coarse_top_k 16 --reliability_top_k 3 --mc_trials 50
```

正式候选筛选可将 `--num_candidates` 提高到 256 或 512，并增加种子。脚本先在
5° 网格粗排，再对前 `coarse_top_k` 个结构做 1° 网格扫描，最后对前
`reliability_top_k` 个结构做厚度、材料折射率和入射角扰动 Monte Carlo。默认名义
通过条件为：每个角度上 s/p 的波段均值均不低于 0.85、波段 p05 均不低于 0.80、
波段最差点均不低于 0.70。结果保存在 `wide_angle_summary.json` 和逐角度 CSV 中。

也可以绕过模型，直接评估已有结构 JSON：

```powershell
python -B .\joint_sp\scripts\evaluate_wide_angle.py `
  --structures .\candidate_structures_seed42.json .\candidate_structures_seed142.json `
  --output_dir .\joint_sp\wide_angle_eval_known_structures_20260727
```

Monte Carlo 通过率是给定 NK 数据库和扰动模型下的计算可靠性，不等同于实验认证。
制造可行性结论还需要实测材料色散、沉积偏差统计和样片角分辨光谱。

**不需要重新预训练OptoGPT。** 从`model/optogpt.pt`继承权重。

---

## 架构版本 (Architecture Versions)

由于原始 OptoGPT checkpoint (`optogpt.pt`) 的 `FullyConnectedLayers` 和 `PositionwiseFeedForward` 在训练时使用了**不同于当前 core 实现**的 forward 语义，state_dict 的 shape 相同但计算路径不同，因此必须通过架构版本显式区分。

### 三个架构版本

| 版本标识符 | 含义 | FC forward | FFN forward |
|---|---|---|---|
| `optogpt_legacy_v1` | 原始 142 维 optogpt checkpoint 语义 | `fc2(norm(fc1(x)))` | `w2(dropout(w1(x)))` |
| `joint_sp_legacy_v1` | 联合 s+p 模型，继承历史语义 (**默认**) | `fc2(norm(fc1(x)))` | `w2(dropout(w1(x)))` |
| `joint_sp_relu_v0` | 使用当前 core ReLU 实现的旧实验 | `fc2(dropout(relu(norm(fc1(x)))))` | `w2(dropout(relu(w1(x))))` |

### 为什么 state_dict shape 相同不代表兼容

两种语义的参数 shape 完全相同（都是 `input_dim→input_dim→out_dim`），因此 `load_state_dict` 不会报错。但 forward 计算路径不同：
- 历史语义：线性路径（仅 LayerNorm + Dropout）
- ReLU 语义：非线性路径（ReLU 激活 + 额外 Dropout）

**相同的权重在两种语义下产生不同的输出**，因此必须通过 `architecture_version` 显式指定。

### 无版本 checkpoint 的加载策略

- **已知原始 `optogpt.pt`**：虽然无架构版本字段，但通过完整 SHA-256 识别，自动采用 `joint_sp_legacy_v1`（历史语义）
- **其他无版本 checkpoint**：默认拒绝加载，必须由用户通过 `--architecture_override` 显式指定
- **已有明确版本的 checkpoint**：**不允许**使用 `--architecture_override` 覆盖为不同语义。state_dict shape 相同，错误语义不会触发 shape mismatch，因此必须拒绝

### 四个推理入口的 override 支持

所有推理/验证/部署入口均支持 `--architecture_override`：

```powershell
python -B joint_sp\scripts\validate.py `
  --model path\to\checkpoint.pt `
  --architecture_override joint_sp_relu_v0 `
  --test_spec ... --test_struct ...

python -B joint_sp\scripts\deploy.py `
  --model path\to\checkpoint.pt `
  --architecture_override joint_sp_relu_v0

python -B joint_sp\self_improving\prepare.py `
  --model path\to\checkpoint.pt `
  --architecture_override joint_sp_relu_v0 `
  --targets ...

python -B joint_sp\self_improving\run.py `
  --model_path path\to\checkpoint.pt `
  --architecture_override joint_sp_relu_v0 `
  ...
```

### pretrained_sha256 强制字段

所有通过 `save_sp_checkpoint()` 保存的 checkpoint **必须**包含：
- `pretrained_sha256`：64 位小写十六进制字符串，记录原始 pretrained checkpoint 的完整 SHA-256
- 保存时如果缺失、格式不合法或与 configs 冲突，拒绝保存

---

## 数据格式

> **重要迁移提示**：旧目录 `data_60deg_sp_1M_dielectric` 是把 s/p 单偏振样本沿样本轴拼接得到的 142 维数据，不是联合 284 维数据。当前训练入口会拒绝该目录，不能继续用于 joint_sp 训练、验证或部署。

### 磁盘Structure_*.pkl：纯膜层token（无BOS/EOS）

```python
["SiO2_100", "TiO2_50"]       # 正确
```

BOS/EOS由`PrepareDataAug.load_data()`自动添加，磁盘文件不应包含。

### 284维联合光谱

```
[Rs(71点), Ts(71), Rp(71), Tp(71)]  — 400-1100nm, step 10nm
```

---

## 快速验证

```bash
cd optogpt/
python -B -m unittest discover -s joint_sp/tests -p "test*.py" -v
```

---

## 完整运行命令

### Step 2: 构建联合数据

```bash
python joint_sp/scripts/build_joint_data.py \
    --s_dir data_60deg_s_500k_dielectric \
    --p_dir data_60deg_p_500k_dielectric \
    --out_dir data_60deg_sp_joint \
    --theta 60 --seed 42 --split 0.8 0.1 0.1 \
    --chunk_size 5000 --num_workers 4
```

源目录必须各自包含 `generation_config.json`，并明确声明 `polarization=s` 或 `polarization=p`。构建器会严格验证 token、元数据和抽样 TMM；任何 mismatch 都不会写 `BUILD_COMPLETE.json`。数据切分按结构 SHA-256 确定性完成，train/dev/test 无结构级泄漏。

中断后使用同样参数并增加 `--resume`。`chunk_size` 控制 checkpoint 频率，`num_workers` 控制 TMM 工作线程数。

### Step 4: 首次联合微调（从optogpt.pt开始）

```bash
python joint_sp/scripts/finetune.py \
    --data_dir data_60deg_sp_joint \
    --pretrained model/optogpt.pt \
    --epochs 10 --batch_size 16 --lr 3e-5 \
    --fusion_warmup_epochs 2 \
    --early_stopping --patience 5 \
    --output_name optogpt_60deg_sp_v1
```

### Resume训练

```bash
# 自动从latest恢复
python joint_sp/scripts/finetune.py \
    --data_dir data_60deg_sp_joint \
    --pretrained model/optogpt.pt \
    --epochs 15 --batch_size 16 --lr 3e-5 \
    --output_name optogpt_60deg_sp_v1 --resume

# 指定checkpoint
python joint_sp/scripts/finetune.py \
    --data_dir data_60deg_sp_ultimate \
    --pretrained joint_sp/models/optogpt_60deg_sp_v1_best.pt \
    --epochs 20 --batch_size 16 --lr 1e-5 \
    --output_name optogpt_60deg_sp_v1 \
    --resume_from joint_sp/models/optogpt_60deg_sp_v1_latest.pt
```

### Step 5: 联合Self-improving

```bash
python joint_sp/self_improving/run.py \
    --model_path joint_sp/models/optogpt_60deg_sp_v1_best.pt \
    --train_struct_path data_60deg_sp_joint/Structure_train.pkl \
    --dev_struct_path data_60deg_sp_joint/Structure_dev.pkl \
    --output_dir joint_sp/self_improving_output \
    --theta 60 --n_ood_targets 100
```

### Step 6: 终极合并

```bash
python joint_sp/final_merge_retrain.py \
    --original_dir data_60deg_sp_joint \
    --si_dir joint_sp/self_improving_output \
    --out_dir data_60deg_sp_ultimate \
    --aug_ratio 0.3
```

### Step 7: 终极微调（从联合checkpoint继续，fusion_warmup_epochs=0）

```bash
python joint_sp/scripts/finetune.py \
    --data_dir data_60deg_sp_ultimate \
    --pretrained joint_sp/models/optogpt_60deg_sp_v1_best.pt \
    --epochs 20 --batch_size 16 --lr 1e-5 \
    --fusion_warmup_epochs 0 \
    --early_stopping --patience 3 \
    --output_name optogpt_60deg_sp_ultimate
```

### Step 8: 联合验证

```bash
python joint_sp/scripts/validate.py \
    --model joint_sp/models/optogpt_60deg_sp_ultimate_best.pt \
    --test_spec data_60deg_sp_joint/Spectrum_test.pkl \
    --test_struct data_60deg_sp_joint/Structure_test.pkl \
    --output_dir joint_sp/validation_results \
    --num_samples 100 --num_candidates 32 \
    --mean_t_threshold 0.9 --p05_t_threshold 0.8
```

### 独立部署测试

```bash
python joint_sp/scripts/deploy.py \
    --model joint_sp/models/optogpt_60deg_sp_ultimate_best.pt \
    --target broadband_high_T --num_candidates 64
```

---

## 恢复方法

- 训练 checkpoint 含 optimizer 和 RNG 状态，支持 **epoch 边界**恢复；当前不承诺 batch 内精确恢复
- Self-improving `--resume` 会复用已原子发布的 OOD、候选和扰动中间文件
- 数据构建 `--resume` 从 `.build_checkpoint.pkl` 的下一 chunk 接续

---

## Windows Server 部署

文档中的服务器目录对应仓库根：

```text
D:\hrqu\optogpt_project\optogpt
```

在 Anaconda PowerShell Prompt 中运行：

```powershell
cd D:\hrqu\optogpt_project\optogpt

# RTX 4090D 推荐使用 CUDA 12.1 独立环境
conda env create -f joint_sp\windows\environment.yml
conda activate optogpt-joint-sp

# 环境、CUDA、TMM、NK、基础 checkpoint 预检
powershell -ExecutionPolicy Bypass -File joint_sp\windows\preflight.ps1

# 构建真正的 284 维联合数据
powershell -ExecutionPolicy Bypass -File joint_sp\windows\build_data.ps1 `
  -SDir data_60deg_s_500k_dielectric `
  -PDir data_60deg_p_500k_dielectric `
  -OutDir data_60deg_sp_joint -Workers 4 -ChunkSize 5000

# 训练前预检并开始训练
powershell -ExecutionPolicy Bypass -File joint_sp\windows\train.ps1 `
  -DataDir data_60deg_sp_joint -Pretrained model\optogpt.pt `
  -OutputName optogpt_60deg_sp_v1 -Epochs 10 -BatchSize 16 `
  -FusionWarmupEpochs 2

# 部署推理
powershell -ExecutionPolicy Bypass -File joint_sp\windows\deploy.ps1 `
  -Model joint_sp\models\optogpt_60deg_sp_v1_best.pt `
  -Target broadband_high_T -NumCandidates 64
```

`deploy.ps1` 会先加载并校验联合 checkpoint，再调用 284 维联合部署入口。它不会覆盖 `model\optogpt.pt`。

从已有联合 checkpoint 继续终极微调时使用 `-FusionWarmupEpochs 0`；resume 必须保持 batch size、学习率、smoothing、seed 和 warmup 与 checkpoint 一致。

---

## 禁止事项

1. 磁盘Structure文件含BOS/EOS/PAD/UNK
2. 浮点数去重
3. GA材料突变引入导体
4. 复制数据凑数量
5. 重新构建词表
