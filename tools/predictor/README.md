# OptoGPT 交互式预测器

输入目标 R/T 光谱 → 模型生成薄膜结构 → TMM 验证 → 误差分析 + 可视化

## 目录结构

```
predictor/
├── interactive_predictor.py   # 主程序：交互菜单 + 命令行
├── run_prediction.py           # 一键预测脚本（headless）
├── inputs/                     # 目标光谱输入文件
│   ├── target_spectrum.csv     # 示例：71点 R,T 光谱
│   └── target_joint_spectrum.csv # 示例：联合 Rs,Ts,Rp,Tp 光谱
└── outputs/                    # 预测结果
    ├── prediction_*.png        # 对比图（6面板）
    └── prediction_*.json       # 详细数据
```

## 快速开始

```powershell
cd predictor

# 交互模式（推荐）
python interactive_predictor.py

# 命令行一键预测
python run_prediction.py inputs/target_spectrum.csv

# 指定模型和条件
python run_prediction.py inputs/target_spectrum.csv --model ../optogpt/model/optogpt.pt --theta 0 --pol s

# 60° s-pol 介质模型
python run_prediction.py inputs/target_spectrum.csv --model ../optogpt/dielectric_60deg_s/models/optogpt_60deg_s_dielectric_best.pt --theta 60 --pol s

# 生成 12 个候选并调整随机采样
python run_prediction.py inputs/target_spectrum.csv --candidates 12 --top-k 20 --top-p 0.95 --temperature 0.8 --seed 42

# joint_sp 微调模型：输入必须包含 s、p 两个偏振
python run_prediction.py inputs/target_joint_spectrum.csv --model ../joint_sp/formal_checkpoints_500k_v2_20260725_03/optogpt_joint_sp_500k_v2_best.pt --candidates 8
```

### Checkpoint 架构

预测器默认使用 `--architecture auto`：已知原始 `optogpt.pt` 会按训练时的
历史无 ReLU 前向加载；当前项目微调产生的无版本 checkpoint 按 ReLU 前向加载。
对于来源不明的旧 checkpoint，请明确指定：

```powershell
python run_prediction.py inputs/target_spectrum.csv --model path/to/model.pt --architecture legacy
```

微调 checkpoint 可以只保存模型权重和配置；若其中没有词表，预测器会自动从
`../optogpt/model/optogpt.pt` 读取词表，也可用 `--pretrained` 指定其他词表来源。
模型权重采用严格加载，架构或词表不匹配时会直接报错。

`joint_sp` checkpoint 会根据 `model_type=joint_sp` 或 `fc_s/fc_p/fusion` 参数名
自动识别，并使用项目内的 `joint_sp.model.load_joint_sp_checkpoint` 严格加载。
它的输入是 284 维 `[Rs, Ts, Rp, Tp]`，不能使用三列单偏振 CSV。

每次预测包含一个确定性的 greedy 候选，其余候选使用 top-k/top-p 采样。
`--max-layers` 是膜层数量的硬上限。相同模型、光谱和 `--seed` 可复现采样结果。

介质模型会自动启用材料与厚度掩码，只允许以下材料：
`Al2O3, AlN, HfO2, MgF2, MgO, Si3N4, SiO2, Ta2O5, TiO2, ZnO`，
且每层厚度限制为 10-300 nm。

## CSV 输入格式

```csv
400,0.20,0.68
410,0.17,0.73
...
1100,0.31,0.69
```
每行: `波长(nm),反射率R,透射率T`，自动插值到 71 点 (400-1100nm, 步长10nm)

输入必须至少包含两个不同波长点并覆盖完整的 400-1100 nm 范围。程序会自动按
波长排序，但会拒绝重复波长、NaN/Inf、外推、超出 `[0,1]` 的 R/T，以及任一点
`R+T>1` 的非物理光谱。

联合 s+p 模型使用五列 CSV：

```csv
400,0.03,0.95,0.03,0.95
1100,0.03,0.95,0.03,0.95
```

每行依次为 `波长(nm),Rs,Ts,Rp,Tp`。也可以输入包含 284 维光谱的 `.pkl`
或 `.npy`，并用 `--index` 选择样本。程序分别检查 `Rs+Ts<=1` 与 `Rp+Tp<=1`。

## 依赖

TMM 验证使用标准 `tmm.inc_tmm`，需要 `tmm==0.1.8`：

```powershell
python -m pip install tmm==0.1.8
```

预测器会先使用当前 Python 环境中的 `tmm`；若未安装，也会自动查找项目内的
`../optogpt/.venv/Lib/site-packages`。两处都找不到时会给出明确安装提示。

## 支持的模型

| 模型 | 路径 |
|------|------|
| OptoGPT 原始 (θ=0°) | `../optogpt/model/optogpt.pt` |
| OptoGPT 60° s-pol | `../optogpt/model/optogpt_60deg_s_best.pt` |
| Dielectric 60° s-pol | `../optogpt/dielectric_60deg_s/models/optogpt_60deg_s_dielectric_best.pt` |
