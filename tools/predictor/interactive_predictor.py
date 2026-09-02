"""
OptoGPT 交互式预测器 — Interactive Predictor with TMM Validation

功能:
  1. 加载训练好的 OptoGPT 模型（支持多个模型切换）
  2. 输入目标反射率(R)和透射率(T)光谱（手动/文件/示例）
  3. 模型生成对应的多层薄膜结构
  4. TMM 计算预测结构的实际 R/T 光谱
  5. 计算目标与 TMM 之间的误差 (MAE)
  6. 生成对比图和结构可视化

用法:
    python interactive_predictor.py                          # 交互菜单
    python interactive_predictor.py --model model/optogpt.pt # 直接指定模型
    python interactive_predictor.py --model dielectric_60deg_s/models/optogpt_60deg_s_dielectric_best.pt --theta 60 --pol s

支持的模型:
    - model/optogpt.pt                     # 原始 OptoGPT (θ=0°, 通用)
    - model/optogpt_60deg_s_best.pt        # 60° s-pol 微调
    - dielectric_60deg_s/models/optogpt_60deg_s_dielectric_best.pt  # 介质专用 60° s-pol

输出目录: ./outputs/
"""

import os
import sys
import json
import argparse
import hashlib
import types
import warnings
from functools import lru_cache
import numpy as np
import torch
import pickle as pkl
from pathlib import Path
from datetime import datetime
from scipy.interpolate import interp1d

import matplotlib
matplotlib.use(os.environ.get("MPLBACKEND", "TkAgg"))
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec

plt.rcParams['font.sans-serif'] = ['DejaVu Sans', 'Arial Unicode MS']
plt.rcParams['axes.unicode_minus'] = False

# ============================================================
# 路径配置
# ============================================================
SCRIPT_DIR = Path(__file__).resolve().parent          # predictor/
PROJECT_ROOT = SCRIPT_DIR.parent                       # d:\BaiduSyncdisk\optogpt_project
OPTOGPT_ROOT = PROJECT_ROOT / "optogpt"
PKG_ROOT = OPTOGPT_ROOT / "optogpt"                    # core/ 所在目录
NK_DIR = PKG_ROOT / "nk"
OUTPUT_DIR = SCRIPT_DIR / "outputs"
INPUT_DIR = SCRIPT_DIR / "inputs"
OUTPUT_DIR.mkdir(exist_ok=True)
INPUT_DIR.mkdir(exist_ok=True)

sys.path.insert(0, str(PKG_ROOT))
sys.path.insert(0, str(OPTOGPT_ROOT))
sys.path.insert(0, str(PROJECT_ROOT))

from core.models.transformer import make_model_I, subsequent_mask

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

ARCH_AUTO = "auto"
ARCH_LEGACY = "legacy"
ARCH_RELU = "relu"
ORIGINAL_OPTOGPT_SHA256 = (
    "a7677602ae8dae60dababde9bd3981ad16be61430a22e797ba359b1d01921d85"
)

# ============================================================
# 波长配置
# ============================================================
WAVELENGTHS_UM = np.arange(0.4, 1.101, 0.01)    # μm, 400-1100 nm
WAVELENGTHS_NM = (WAVELENGTHS_UM * 1000).astype(int)
N_WL = len(WAVELENGTHS_NM)  # 71

# ============================================================
# 可用模型注册表
# ============================================================
MODEL_REGISTRY = {
    "1": {
        "name": "OptoGPT (原始, θ=0°)",
        "path": str(OPTOGPT_ROOT / "model" / "optogpt.pt"),
        "default_theta": 0,
        "default_pol": "s",
        "description": "通用多层薄膜反向设计模型，入射角 0°"
    },
    "2": {
        "name": "OptoGPT 60° s-pol",
        "path": str(OPTOGPT_ROOT / "model" / "optogpt_60deg_s_best.pt"),
        "default_theta": 60,
        "default_pol": "s",
        "description": "60° s-pol 微调模型"
    },
    "3": {
        "name": "Dielectric 60° s-pol",
        "path": str(OPTOGPT_ROOT / "dielectric_60deg_s" / "models" / "optogpt_60deg_s_dielectric_best.pt"),
        "default_theta": 60,
        "default_pol": "s",
        "description": "介质材料专用 60° s-pol 模型（仅允许介质材料）"
    },
    "custom": {
        "name": "自定义路径",
        "path": None,
        "default_theta": 0,
        "default_pol": "s",
        "description": "手动指定 checkpoint 路径"
    }
}

# ============================================================
# 材料信息
# ============================================================
SUBSTRATES = {'Glass_Substrate', 'SiO2_Substrate', 'Si_Substrate'}

BANNED_DIELECTRIC = {"Ag", "Al", "TiN", "Ge", "Si", "ITO", "ZnS", "ZnSe"}
ALLOWED_DIELECTRIC = {
    "Al2O3", "AlN", "HfO2", "MgF2", "MgO",
    "Si3N4", "SiO2", "Ta2O5", "TiO2", "ZnO",
}

# ============================================================
# 模型加载
# ============================================================

def _config_value(configs, key, default=None):
    if isinstance(configs, dict):
        return configs.get(key, default)
    return getattr(configs, key, default)


def _sha256_file(path, chunk_size=8 * 1024 * 1024):
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        while True:
            chunk = handle.read(chunk_size)
            if not chunk:
                break
            digest.update(chunk)
    return digest.hexdigest()


def _resolve_architecture(ckpt_path, configs, architecture=ARCH_AUTO):
    """Resolve checkpoint forward semantics without relying on tensor shapes."""
    if architecture in (ARCH_LEGACY, ARCH_RELU):
        return architecture
    if architecture != ARCH_AUTO:
        raise ValueError(f"未知架构: {architecture}")

    saved = _config_value(configs, "architecture_version")
    if saved in ("optogpt_legacy_v1", "joint_sp_legacy_v1", ARCH_LEGACY):
        return ARCH_LEGACY
    if saved in ("optogpt_relu_v0", "joint_sp_relu_v0", ARCH_RELU):
        return ARCH_RELU
    if saved is not None:
        raise ValueError(f"Checkpoint 包含未知 architecture_version: {saved}")

    if _sha256_file(ckpt_path) == ORIGINAL_OPTOGPT_SHA256:
        return ARCH_LEGACY

    warnings.warn(
        "Checkpoint 没有 architecture_version，且不是已知原始 optogpt.pt；"
        "按当前微调脚本的实现假定为 ReLU 架构。若这是历史旧模型，请指定 "
        "--architecture legacy。",
        RuntimeWarning,
        stacklevel=2,
    )
    return ARCH_RELU


def _legacy_fc_forward(self, x):
    return self.fc2(self.norm(self.fc1(x)))


def _legacy_ff_forward(self, x):
    return self.w_2(self.dropout(self.w_1(x)))


def _apply_model_architecture(model, architecture):
    """Apply the forward semantics used when the checkpoint was trained."""
    if architecture == ARCH_LEGACY:
        model.fc.forward = types.MethodType(_legacy_fc_forward, model.fc)
        for layer in model.decoder.layers:
            layer.feed_forward.forward = types.MethodType(
                _legacy_ff_forward, layer.feed_forward
            )
        model.architecture_version = "optogpt_legacy_v1"
    elif architecture == ARCH_RELU:
        model.architecture_version = "optogpt_relu_v0"
    else:
        raise ValueError(f"未知架构: {architecture}")
    return model


def _checkpoint_model_type(configs, state_dict):
    """Identify single-polarization and joint s+p checkpoints."""
    configured = str(_config_value(configs, "model_type", "")).lower()
    state_keys = set(state_dict)
    if configured == "joint_sp" or (
        any(key.startswith("fc_s.") for key in state_keys)
        and any(key.startswith("fc_p.") for key in state_keys)
        and any(key.startswith("fusion.") for key in state_keys)
    ):
        return "joint_sp"
    return "single"


def _joint_architecture_override(architecture):
    if architecture == ARCH_AUTO:
        return None
    if architecture == ARCH_LEGACY:
        return "joint_sp_legacy_v1"
    if architecture == ARCH_RELU:
        return "joint_sp_relu_v0"
    raise ValueError(f"未知架构: {architecture}")


def load_model_from_ckpt(ckpt_path, pretrained_path=None, architecture=ARCH_AUTO):
    """加载模型、词表与配置。"""
    ckpt_path = Path(ckpt_path).resolve()
    if not ckpt_path.is_file():
        raise FileNotFoundError(f"模型 checkpoint 不存在: {ckpt_path}")
    ckpt = torch.load(ckpt_path, map_location=DEVICE, weights_only=False)
    if "model_state_dict" not in ckpt:
        raise ValueError(f"Checkpoint 缺少 model_state_dict: {ckpt_path}")
    configs = ckpt.get("configs", {})
    model_type = _checkpoint_model_type(configs, ckpt["model_state_dict"])
    if model_type == "joint_sp":
        from joint_sp.model import load_joint_sp_checkpoint

        return load_joint_sp_checkpoint(
            str(ckpt_path),
            device=DEVICE,
            architecture_override=_joint_architecture_override(architecture),
        )
    resolved_architecture = _resolve_architecture(
        ckpt_path, configs, architecture=architecture
    )

    def _get(key, default=None):
        return _config_value(configs, key, default)

    word_dict = _get("struc_word_dict")
    index_dict = _get("struc_index_dict")

    # Fallback to pretrained for vocab
    pretrained_cfg = {}
    if word_dict is None:
        pretrained_path = pretrained_path or OPTOGPT_ROOT / "model" / "optogpt.pt"
        pretrained_path = Path(pretrained_path).resolve()
        if not pretrained_path.is_file():
            raise FileNotFoundError(f"词表回退 checkpoint 不存在: {pretrained_path}")
        pt = torch.load(pretrained_path, map_location=DEVICE, weights_only=False)
        pretrained_cfg = pt.get("configs", {})
        if isinstance(pretrained_cfg, dict):
            word_dict = pretrained_cfg.get("struc_word_dict")
            index_dict = pretrained_cfg.get("struc_index_dict")
        else:
            word_dict = getattr(pretrained_cfg, "struc_word_dict", None)
            index_dict = getattr(pretrained_cfg, "struc_index_dict", None)

    if not word_dict:
        raise ValueError("Checkpoint 中未找到 struc_word_dict！请用 --pretrained 指定预训练模型获取词表。")
    word_dict = {str(token): int(token_id) for token, token_id in word_dict.items()}
    if index_dict:
        index_dict = {int(token_id): str(token) for token_id, token in index_dict.items()}
    else:
        index_dict = {token_id: token for token, token_id in word_dict.items()}

    def get_hp(key, default):
        val = _get(key)
        if val is None:
            if isinstance(pretrained_cfg, dict):
                val = pretrained_cfg.get(key)
            elif hasattr(pretrained_cfg, key):
                val = getattr(pretrained_cfg, key)
        return val if val is not None else default

    model = make_model_I(
        src_vocab=get_hp("spec_dim", 142),
        tgt_vocab=get_hp("struc_dim", len(word_dict)),
        N=get_hp("layers", 6),
        d_model=get_hp("d_model", 1024),
        d_ff=get_hp("d_ff", 512),
        h=get_hp("head_num", 8),
        dropout=get_hp("dropout", 0.1),
    )
    model = _apply_model_architecture(model, resolved_architecture).to(DEVICE)
    model.load_state_dict(ckpt["model_state_dict"], strict=True)
    model.eval()
    return model, word_dict, index_dict, configs


# ============================================================
# 解码器 (基于 multi_candidate_decoder.py)
# ============================================================

def _token_material(token):
    if token in ("UNK", "PAD", "BOS", "EOS") or "_" not in token:
        return None
    material, thickness_text = token.rsplit("_", 1)
    try:
        thickness = float(thickness_text)
    except ValueError:
        return None
    return material, thickness


def build_logits_mask(word_dict, allowed_materials=None, banned_materials=None,
                      min_thickness=None, max_thickness=None):
    """Build a token mask that permits EOS and valid material-thickness tokens."""
    allowed_materials = set(allowed_materials) if allowed_materials else None
    banned_materials = set(banned_materials or ())
    mask = torch.zeros(len(word_dict), dtype=torch.bool, device=DEVICE)
    eos_id = word_dict.get("EOS", 3)
    mask[eos_id] = True

    for token, token_id in word_dict.items():
        parsed = _token_material(token)
        if parsed is None:
            continue
        material, thickness = parsed
        if allowed_materials is not None and material not in allowed_materials:
            continue
        if material in banned_materials:
            continue
        if min_thickness is not None and thickness < min_thickness:
            continue
        if max_thickness is not None and thickness > max_thickness:
            continue
        if not (NK_DIR / f"{material}.csv").is_file():
            continue
        mask[token_id] = True

    if int(mask.sum().item()) <= 1:
        raise ValueError("材料或厚度约束排除了所有结构 token")
    return mask


def _sample_tokens(logits, top_k=10, top_p=0.9, temperature=1.0):
    if top_k < 0:
        raise ValueError("top_k 必须大于等于 0")
    if temperature <= 0:
        raise ValueError("temperature 必须大于 0")
    logits = logits / temperature
    if 0 < top_k < logits.size(-1):
        cutoff = torch.topk(logits, top_k, dim=-1).values[:, -1:]
        logits = logits.masked_fill(logits < cutoff, float("-inf"))
    probabilities = torch.softmax(logits, dim=-1)
    if not 0 < top_p <= 1:
        raise ValueError("top_p 必须在 (0, 1] 范围")
    if top_p < 1.0:
        sorted_probs, sorted_indices = torch.sort(probabilities, descending=True, dim=-1)
        cumulative = torch.cumsum(sorted_probs, dim=-1)
        remove = cumulative - sorted_probs > top_p
        sorted_probs = sorted_probs.masked_fill(remove, 0.0)
        probabilities = torch.zeros_like(probabilities).scatter(
            -1, sorted_indices, sorted_probs
        )
        probabilities = probabilities / probabilities.sum(
            dim=-1, keepdim=True
        ).clamp_min(1e-12)
    return torch.multinomial(probabilities, 1).squeeze(-1)


def _decode(model, specs, word_dict, index_dict, max_layers=20, sampling=False,
            top_k=10, top_p=0.9, temperature=1.0, logits_mask=None):
    specs = np.asarray(specs, dtype=np.float32)
    if specs.ndim == 1:
        specs = specs.reshape(1, -1)
    expected_dim = 284 if hasattr(model, "fc_s") else 142
    if specs.ndim != 2 or specs.shape[1] != expected_dim:
        raise ValueError(
            f"模型输入必须是 (N, {expected_dim})，实际为 {specs.shape}"
        )
    if not np.isfinite(specs).all():
        raise ValueError("目标光谱包含 NaN 或 Inf")
    if max_layers < 1:
        raise ValueError("max_layers 必须大于 0")

    bos_id = word_dict.get("BOS", 2)
    eos_id = word_dict.get("EOS", 3)
    pad_id = word_dict.get("PAD", 1)
    batch_size = len(specs)
    src = torch.from_numpy(specs).to(DEVICE)
    ys = torch.full((batch_size, 1), bos_id, dtype=torch.long, device=DEVICE)
    finished = torch.zeros(batch_size, dtype=torch.bool, device=DEVICE)
    if logits_mask is None:
        logits_mask = build_logits_mask(word_dict)

    with torch.inference_mode():
        for step in range(max_layers + 1):
            trg_mask = subsequent_mask(ys.size(1)).to(DEVICE)
            out = model(src, ys, None, trg_mask)
            logits = model.generator.proj(out[:, -1])
            logits = logits.masked_fill(~logits_mask.unsqueeze(0), float("-inf"))
            if step == max_layers:
                logits.fill_(float("-inf"))
                logits[:, eos_id] = 0.0
            next_words = (
                _sample_tokens(logits, top_k, top_p, temperature)
                if sampling else torch.argmax(logits, dim=-1)
            )
            next_words = torch.where(
                finished, torch.full_like(next_words, pad_id), next_words
            )
            ys = torch.cat([ys, next_words.unsqueeze(1)], dim=1)
            finished |= next_words == eos_id
            if bool(finished.all()):
                break

    designs = []
    for row in ys[:, 1:].cpu().tolist():
        tokens = []
        for token_id in row:
            if token_id == eos_id:
                break
            token = index_dict.get(token_id, "UNK")
            if _token_material(token) is not None:
                tokens.append(token)
        designs.append(tokens)
    return designs


def greedy_decode(model, spec, word_dict, index_dict, max_len=22, logits_mask=None):
    """Compatibility wrapper for deterministic single-spectrum decoding."""
    return _decode(
        model, spec, word_dict, index_dict,
        max_layers=max(1, max_len - 2), logits_mask=logits_mask,
    )[0]


def generate_candidates(model, spec, word_dict, index_dict, num_candidates=8,
                        max_layers=20, top_k=10, top_p=0.9, temperature=1.0,
                        logits_mask=None, seed=42):
    """Generate one greedy and N-1 sampled, unique, valid structures."""
    if num_candidates < 1:
        raise ValueError("候选数量必须大于 0")
    if max_layers < 1:
        raise ValueError("max_layers 必须大于 0")
    if top_k < 0:
        raise ValueError("top_k 必须大于等于 0")
    if not 0 < top_p <= 1:
        raise ValueError("top_p 必须在 (0, 1] 范围")
    if temperature <= 0:
        raise ValueError("temperature 必须大于 0")
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    decoded = _decode(
        model, spec, word_dict, index_dict, max_layers=max_layers,
        logits_mask=logits_mask,
    )
    if num_candidates > 1:
        batch = np.repeat(np.asarray(spec, dtype=np.float32)[None, :],
                          num_candidates - 1, axis=0)
        decoded.extend(_decode(
            model, batch, word_dict, index_dict, max_layers=max_layers,
            sampling=True, top_k=top_k, top_p=top_p,
            temperature=temperature, logits_mask=logits_mask,
        ))

    candidates, seen = [], set()
    for attempt, tokens in enumerate(decoded):
        key = tuple(tokens)
        if not key or key in seen:
            continue
        seen.add(key)
        materials, thicknesses = parse_structure(tokens)
        if not materials or len(materials) != len(tokens) or len(materials) > max_layers:
            continue
        candidates.append({
            "tokens": tokens,
            "materials": materials,
            "thicknesses": thicknesses,
            "n_layers": len(materials),
            "decode_method": "greedy" if attempt == 0 else "sampling",
        })
    return candidates


def parse_structure(tokens):
    """解析 token 列表 -> (materials, thicknesses)。"""
    materials, thicknesses = [], []
    for tok in tokens:
        if "_" not in tok:
            continue
        parts = tok.rsplit("_", 1)
        if len(parts) != 2:
            continue
        mat, thick_str = parts
        try:
            thick = float(thick_str)
        except ValueError:
            continue
        materials.append(mat)
        thicknesses.append(thick)
    return materials, thicknesses


# ============================================================
# TMM 计算（使用项目训练/验证一致的 tmm.inc_tmm）
# ============================================================

@lru_cache(maxsize=1)
def _load_inc_tmm():
    try:
        from tmm import inc_tmm
        return inc_tmm
    except ModuleNotFoundError:
        bundled = OPTOGPT_ROOT / ".venv" / "Lib" / "site-packages"
        if bundled.is_dir() and str(bundled) not in sys.path:
            sys.path.append(str(bundled))
        try:
            from tmm import inc_tmm
            return inc_tmm
        except ModuleNotFoundError as exc:
            raise RuntimeError(
                "缺少 TMM 依赖。请在当前 Python 环境安装 tmm==0.1.8："
                "python -m pip install tmm==0.1.8"
            ) from exc


@lru_cache(maxsize=None)
def load_nk_interpolated(material_name):
    """加载材料 nk 并插值到标准波长。"""
    filepath = NK_DIR / f"{material_name}.csv"
    if not filepath.is_file():
        raise FileNotFoundError(f"缺少材料 NK 文件: {filepath}")
    import pandas as pd
    nk_df = pd.read_csv(filepath)
    nk_df.dropna(inplace=True)
    if not {"wl", "n", "k"}.issubset(nk_df.columns):
        raise ValueError(f"NK 文件必须包含 wl,n,k 三列: {filepath}")
    if len(nk_df) < 4:
        raise ValueError(f"NK 文件数据点不足: {filepath}")
    wl = nk_df['wl'].to_numpy()
    n = nk_df['n'].to_numpy()
    k = nk_df['k'].to_numpy()
    n_fn = interp1d(wl, n, bounds_error=False, fill_value='extrapolate', kind='cubic')
    k_fn = interp1d(wl, k, bounds_error=False, fill_value='extrapolate', kind='linear')
    result = n_fn(WAVELENGTHS_UM) + 1j * k_fn(WAVELENGTHS_UM)
    if not np.isfinite(result).all():
        raise ValueError(f"NK 插值产生 NaN/Inf: {material_name}")
    return result


def tmm_simulate(materials, thicknesses_nm, theta_deg=0.0, pol='s',
                 substrate='Glass_Substrate', substrate_thick_nm=500000):
    """Simulate coherent films on an incoherent substrate using inc_tmm."""
    if pol not in ("s", "p"):
        raise ValueError("偏振必须是 s 或 p")
    if not 0 <= theta_deg < 90:
        raise ValueError("入射角必须在 [0, 90) 度")
    if not materials or len(materials) != len(thicknesses_nm):
        raise ValueError("材料与厚度列表为空或长度不一致")
    if any((not np.isfinite(t)) or t <= 0 for t in thicknesses_nm):
        raise ValueError("所有膜层厚度必须是有限正数")

    inc_tmm = _load_inc_tmm()
    all_mats = set(materials) | {substrate}
    nk_cache = {m: load_nk_interpolated(m) for m in all_mats}
    theta_rad = np.deg2rad(theta_deg)
    thickness = [np.inf] + list(thicknesses_nm) + [substrate_thick_nm, np.inf]
    coherence = ["i"] + ["c"] * len(materials) + ["i", "i"]
    R, T = [], []
    for index, wavelength_nm in enumerate(WAVELENGTHS_NM):
        n_list = (
            [1.0]
            + [nk_cache[material][index] for material in materials]
            + [nk_cache[substrate][index], 1.0]
        )
        result = inc_tmm(
            pol, n_list, thickness, coherence, theta_rad, float(wavelength_nm)
        )
        R.append(float(result["R"]))
        T.append(float(result["T"]))

    R = np.asarray(R, dtype=np.float64)
    T = np.asarray(T, dtype=np.float64)
    A = 1.0 - R - T
    if not np.isfinite(np.concatenate([R, T, A])).all():
        raise ValueError("TMM 结果包含 NaN 或 Inf")
    if np.any(R < -1e-8) or np.any(T < -1e-8) or np.any(R + T > 1 + 1e-6):
        raise ValueError("TMM 结果违反能量守恒")
    return np.clip(R, 0, 1), np.clip(T, 0, 1), np.clip(A, 0, 1)


def tmm_simulate_from_tokens(tokens, theta_deg=0.0, pol='s',
                              substrate='Glass_Substrate'):
    """从 token 列表直接跑 TMM。"""
    materials, thicknesses = parse_structure(tokens)
    if not materials:
        raise ValueError("无法解析结构")
    return tmm_simulate(materials, thicknesses, theta_deg, pol, substrate)


# ============================================================
# 误差计算
# ============================================================

def compute_errors(spec_target, R_sim, T_sim):
    """计算目标光谱与 TMM 模拟结果的误差。"""
    spec_target = np.asarray(spec_target, dtype=np.float64).reshape(-1)
    if spec_target.shape != (142,) or not np.isfinite(spec_target).all():
        raise ValueError("目标光谱必须是有限的 142 维数组")
    half = len(spec_target) // 2
    R_target = np.array(spec_target[:half])
    T_target = np.array(spec_target[half:])
    R_sim = np.array(R_sim)
    T_sim = np.array(T_sim)
    if R_sim.shape != (71,) or T_sim.shape != (71,):
        raise ValueError("TMM 输出必须分别包含 71 个 R/T 点")
    if not np.isfinite(np.concatenate([R_sim, T_sim])).all():
        raise ValueError("TMM 输出包含 NaN 或 Inf")

    mae_R = float(np.mean(np.abs(R_sim - R_target)))
    mae_T = float(np.mean(np.abs(T_sim - T_target)))
    mae_total = (mae_R + mae_T) / 2.0

    # Per-wavelength errors
    R_error_per_wl = np.abs(R_sim - R_target)
    T_error_per_wl = np.abs(T_sim - T_target)

    return {
        "mae_R": mae_R, "mae_T": mae_T, "mae_total": mae_total,
        "R_error_per_wl": R_error_per_wl,
        "T_error_per_wl": T_error_per_wl,
        "R_target": R_target, "T_target": T_target,
        "R_sim": R_sim, "T_sim": T_sim,
    }


def compute_joint_errors(spec_target, R_s, T_s, R_p, T_p):
    """Compute per-polarization and joint errors for a 284-dim target."""
    target = np.asarray(spec_target, dtype=np.float64).reshape(-1)
    simulated = [np.asarray(value, dtype=np.float64) for value in (R_s, T_s, R_p, T_p)]
    if target.shape != (284,) or not np.isfinite(target).all():
        raise ValueError("联合目标光谱必须是有限的 284 维 [Rs, Ts, Rp, Tp]")
    if any(value.shape != (N_WL,) for value in simulated):
        raise ValueError("联合 TMM 输出必须分别包含 71 个 Rs/Ts/Rp/Tp 点")
    if not np.isfinite(np.concatenate(simulated)).all():
        raise ValueError("联合 TMM 输出包含 NaN 或 Inf")

    target_Rs, target_Ts, target_Rp, target_Tp = np.split(target, 4)
    mae_Rs = float(np.mean(np.abs(R_s - target_Rs)))
    mae_Ts = float(np.mean(np.abs(T_s - target_Ts)))
    mae_Rp = float(np.mean(np.abs(R_p - target_Rp)))
    mae_Tp = float(np.mean(np.abs(T_p - target_Tp)))
    mae_s = (mae_Rs + mae_Ts) / 2.0
    mae_p = (mae_Rp + mae_Tp) / 2.0
    return {
        "joint_sp": True,
        "mae_Rs": mae_Rs, "mae_Ts": mae_Ts,
        "mae_Rp": mae_Rp, "mae_Tp": mae_Tp,
        "mae_s": mae_s, "mae_p": mae_p,
        "mae_R": (mae_Rs + mae_Rp) / 2.0,
        "mae_T": (mae_Ts + mae_Tp) / 2.0,
        "mae_total": (mae_s + mae_p) / 2.0,
        "Rs_target": target_Rs, "Ts_target": target_Ts,
        "Rp_target": target_Rp, "Tp_target": target_Tp,
        "Rs_sim": R_s, "Ts_sim": T_s,
        "Rp_sim": R_p, "Tp_sim": T_p,
    }


# ============================================================
# 可视化
# ============================================================

def plot_comparison(result, save_path=None, show=True):
    """绘制目标 vs 预测光谱对比图。"""
    if result.get("joint_sp"):
        return _plot_joint_comparison(result, save_path=save_path, show=show)

    wl = WAVELENGTHS_NM
    R_tgt = result["R_target"]
    T_tgt = result["T_target"]
    R_sim = result["R_sim"]
    T_sim = result["T_sim"]
    mae_R = result["mae_R"]
    mae_T = result["mae_T"]

    fig = plt.figure(figsize=(16, 10))
    gs = GridSpec(3, 2, figure=fig, hspace=0.35, wspace=0.3,
                  height_ratios=[1.2, 1, 1])

    # --- 左上: R 对比 ---
    ax1 = fig.add_subplot(gs[0, 0])
    ax1.plot(wl, R_tgt * 100, 'b-', lw=2.0, label=f'Target R')
    ax1.plot(wl, R_sim * 100, 'r--', lw=2.0, label=f'Predicted R (MAE={mae_R:.4f})')
    ax1.fill_between(wl, 0, np.abs(R_sim - R_tgt) * 100, alpha=0.2, color='orange',
                     label=f'|ΔR|')
    ax1.set_ylabel('Reflectance (%)', fontsize=12, color='#E74C3C')
    ax1.set_ylim(-3, 105)
    ax1.set_xlim(wl[0], wl[-1])
    ax1.legend(loc='best', fontsize=9)
    ax1.grid(True, alpha=0.3, linestyle='--')
    ax1.set_title('Reflectance: Target vs Predicted', fontsize=13, fontweight='bold')

    # --- 右上: T 对比 ---
    ax2 = fig.add_subplot(gs[0, 1])
    ax2.plot(wl, T_tgt * 100, 'b-', lw=2.0, label=f'Target T')
    ax2.plot(wl, T_sim * 100, 'g--', lw=2.0, label=f'Predicted T (MAE={mae_T:.4f})')
    ax2.fill_between(wl, 0, np.abs(T_sim - T_tgt) * 100, alpha=0.2, color='orange',
                     label=f'|ΔT|')
    ax2.set_ylabel('Transmittance (%)', fontsize=12, color='#2ECC71')
    ax2.set_ylim(-3, 105)
    ax2.set_xlim(wl[0], wl[-1])
    ax2.legend(loc='best', fontsize=9)
    ax2.grid(True, alpha=0.3, linestyle='--')
    ax2.set_title('Transmittance: Target vs Predicted', fontsize=13, fontweight='bold')

    # --- 中左: R 逐波长误差 ---
    ax3 = fig.add_subplot(gs[1, 0])
    ax3.bar(wl, result["R_error_per_wl"] * 100, width=8, color='#E74C3C', alpha=0.6,
            edgecolor='#C0392B', linewidth=0.5)
    ax3.axhline(y=mae_R * 100, color='darkred', linestyle='--', lw=1.5,
                label=f'Mean R Error = {mae_R*100:.2f}%')
    ax3.set_ylabel('R Error (%)', fontsize=11)
    ax3.set_xlim(wl[0], wl[-1])
    ax3.legend(fontsize=9)
    ax3.grid(True, alpha=0.3, axis='y', linestyle='--')

    # --- 中右: T 逐波长误差 ---
    ax4 = fig.add_subplot(gs[1, 1])
    ax4.bar(wl, result["T_error_per_wl"] * 100, width=8, color='#2ECC71', alpha=0.6,
            edgecolor='#27AE60', linewidth=0.5)
    ax4.axhline(y=mae_T * 100, color='darkgreen', linestyle='--', lw=1.5,
                label=f'Mean T Error = {mae_T*100:.2f}%')
    ax4.set_ylabel('T Error (%)', fontsize=11)
    ax4.set_xlim(wl[0], wl[-1])
    ax4.legend(fontsize=9)
    ax4.grid(True, alpha=0.3, axis='y', linestyle='--')

    # --- 底部: 结构可视化 ---
    ax5 = fig.add_subplot(gs[2, :])
    _plot_structure_diagram(ax5, result.get("materials", []),
                            result.get("thicknesses", []),
                            result.get("theta_deg", 0), result.get("pol", "s"),
                            result.get("substrate", "Glass_Substrate"))

    # 总标题
    total_mae = result["mae_total"]
    info = result.get("info", "")
    fig.suptitle(f'OptoGPT Inverse Design — Total MAE = {total_mae:.4f}  {info}',
                 fontsize=14, fontweight='bold', y=0.99)

    fig.subplots_adjust(top=0.91)

    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"\n  📁 图像已保存: {save_path}")

    if show:
        plt.show()
    else:
        plt.close(fig)


def _plot_joint_comparison(result, save_path=None, show=True):
    """Plot s/p targets, simulations, errors, and the predicted structure."""
    wl = WAVELENGTHS_NM
    fig = plt.figure(figsize=(22, 12.5))
    gs = GridSpec(
        3, 4, figure=fig, hspace=0.42, wspace=0.28,
        height_ratios=[1.25, 0.9, 0.72],
    )
    panels = [
        ("Rs", "s-pol Reflectance", "#D64541", "R Error (%)"),
        ("Ts", "s-pol Transmittance", "#239B56", "T Error (%)"),
        ("Rp", "p-pol Reflectance", "#C0392B", "R Error (%)"),
        ("Tp", "p-pol Transmittance", "#148F77", "T Error (%)"),
    ]
    for index, (prefix, title, color, error_label) in enumerate(panels):
        target = result[f"{prefix}_target"]
        simulated = result[f"{prefix}_sim"]
        mae = result[f"mae_{prefix}"]

        ax_curve = fig.add_subplot(gs[0, index])
        ax_curve.plot(wl, target * 100, color="#1F4EAD", lw=2.2,
                      label=f"Target {prefix}")
        ax_curve.plot(wl, simulated * 100, color=color, ls="--", lw=2.2,
                      label=f"Predicted {prefix} (MAE={mae:.4f})")
        ax_curve.fill_between(
            wl, 0, np.abs(simulated - target) * 100,
            color="#F5B041", alpha=0.2, label=f"|Delta {prefix}|",
        )
        ax_curve.set_title(title, fontsize=13, fontweight="bold")
        ax_curve.set_ylabel("Spectrum (%)", fontsize=10)
        ax_curve.set_xlim(wl[0], wl[-1])
        ax_curve.set_ylim(-3, 105)
        ax_curve.grid(True, alpha=0.3, linestyle="--")
        ax_curve.legend(fontsize=8, loc="best")

        error = np.abs(simulated - target) * 100
        ax_error = fig.add_subplot(gs[1, index])
        ax_error.bar(
            wl, error, width=8, color=color, alpha=0.58,
            edgecolor=color, linewidth=0.45,
        )
        ax_error.axhline(
            mae * 100, color=color, linestyle="--", lw=1.6,
            label=f"Mean {prefix} Error = {mae * 100:.2f}%",
        )
        ax_error.set_ylabel(error_label, fontsize=10)
        ax_error.set_xlabel("Wavelength (nm)", fontsize=10)
        ax_error.set_xlim(wl[0], wl[-1])
        ax_error.set_ylim(bottom=0)
        ax_error.grid(True, alpha=0.3, axis="y", linestyle="--")
        ax_error.legend(fontsize=8, loc="best")

    ax_structure = fig.add_subplot(gs[2, :])
    _plot_joint_structure_diagram(
        ax_structure, result.get("materials", []), result.get("thicknesses", []),
        result.get("theta_deg", 60), result.get("substrate", "Glass_Substrate"),
    )
    info = result.get("info", "")
    fig.suptitle(
        f"OptoGPT Joint s+p Inverse Design | Joint MAE={result['mae_total']:.4f} "
        f"| s MAE={result['mae_s']:.4f} | p MAE={result['mae_p']:.4f} {info}",
        fontsize=16, fontweight="bold", y=0.985,
    )
    fig.subplots_adjust(top=0.91, bottom=0.055, left=0.055, right=0.985)
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches="tight")
        print(f"\n  📁 图像已保存: {save_path}")
    if show:
        plt.show()
    else:
        plt.close(fig)


def _plot_joint_structure_diagram(ax, materials, thicknesses, theta_deg, substrate):
    """Draw the joint-design layer sequence without overlapping thin-layer labels."""
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.axis("off")
    ax.set_title(
        f"Layer Structure ({len(materials)} layers, theta={theta_deg} deg, s+p)",
        fontsize=13, fontweight="bold", pad=10,
    )
    if not materials:
        ax.text(0.5, 0.45, "No structure", ha="center", va="center", fontsize=13)
        return

    left, right = 0.08, 0.92
    width = (right - left) / len(materials)
    cmap = plt.cm.tab20
    color_map = {
        material: cmap(index % 20 / 20)
        for index, material in enumerate(dict.fromkeys(materials))
    }
    for index, (material, thickness) in enumerate(zip(materials, thicknesses)):
        x = left + index * width
        rect = plt.Rectangle(
            (x, 0.22), width, 0.52, facecolor=color_map[material],
            edgecolor="#222222", linewidth=1.1, alpha=0.9,
        )
        ax.add_patch(rect)
        font_size = 9 if len(materials) <= 8 else 7 if len(materials) <= 14 else 6
        rotation = 0 if len(materials) <= 10 else 90
        ax.text(
            x + width / 2, 0.48, f"{index + 1}. {material}\n{thickness:.0f} nm",
            ha="center", va="center", fontsize=font_size,
            fontweight="bold", rotation=rotation,
        )
    ax.text(left - 0.015, 0.48, "Air", ha="right", va="center",
            fontsize=11, fontstyle="italic")
    ax.text(right + 0.015, 0.48, substrate, ha="left", va="center",
            fontsize=11, fontstyle="italic")
    ax.annotate("", xy=(right + 0.008, 0.82), xytext=(left - 0.008, 0.82),
                arrowprops={"arrowstyle": "->", "lw": 1.4, "color": "#444444"})
    ax.text(0.5, 0.88, "Incident direction / layer order", ha="center",
            va="bottom", fontsize=9, color="#444444")


def _plot_structure_diagram(ax, materials, thicknesses, theta_deg, pol, substrate):
    """在坐标轴上绘制薄膜结构示意图。"""
    if not materials:
        ax.text(0.5, 0.5, 'No structure', ha='center', va='center',
                transform=ax.transAxes, fontsize=14)
        return

    # 颜色映射
    unique_mats = list(dict.fromkeys(materials))
    cmap = plt.cm.tab20
    color_map = {}
    for i, m in enumerate(unique_mats):
        color_map[m] = cmap(i % 20 / 20)

    # 绘制各层
    y_bottom = 0
    total_thick = sum(thicknesses)
    for i, (mat, thick) in enumerate(zip(materials, thicknesses)):
        height = thick / max(total_thick, 1) * 0.6  # 归一化
        rect = plt.Rectangle((0.2, y_bottom), 0.6, height,
                             facecolor=color_map[mat], edgecolor='k',
                             linewidth=1.2, alpha=0.85)
        ax.add_patch(rect)
        # 标注
        mid = y_bottom + height / 2
        ax.text(0.5, mid, f"{mat}\n{thick:.0f}nm", ha='center', va='center',
                fontsize=9, fontweight='bold')
        y_bottom += height

    # 空气和基底
    ax.text(0.5, -0.08, 'Air (n=1)', ha='center', va='top', fontsize=10,
            transform=ax.transAxes, style='italic')
    ax.text(0.5, 1.02, f'Substrate: {substrate}', ha='center', va='bottom',
            fontsize=10, transform=ax.transAxes, style='italic')

    ax.set_xlim(0, 1)
    ax.set_ylim(-0.1, 0.7)
    ax.axis('off')
    pol_text = "s+p" if pol == "s+p" else f"{pol}-pol"
    ax.set_title(f'Layer Structure ({len(materials)} layers, θ={theta_deg}°, {pol_text})',
                 fontsize=12, fontweight='bold')


# ============================================================
# 光谱输入方式
# ============================================================

def input_spectrum_manual(joint_sp=False):
    """手动输入光谱：两种模式。"""
    print("\n" + "-" * 50)
    print("  光谱输入方式:")
    print("  1) 从测试数据 .pkl 文件加载样例")
    print("  2) 手动输入每波长 R/T 值（CSV 格式，400-1100nm, 步长10nm）")
    print("  3) 使用预定义目标（高透、高反、带通等）")
    choice = input("\n  请选择 [1/2/3]: ").strip()

    if choice == "1":
        return _load_joint_array_file() if joint_sp else _load_from_pkl()
    elif choice == "2":
        return _input_manual_csv(joint_sp=joint_sp)
    elif choice == "3":
        target = _predefined_targets()
        return np.concatenate([target, target]) if joint_sp else target
    else:
        print("  无效选择，使用预定义目标")
        target = _predefined_targets()
        return np.concatenate([target, target]) if joint_sp else target


def _load_from_pkl():
    """从 .pkl 文件加载测试光谱。"""
    print("\n  可用的测试数据目录:")
    candidates = []
    data_dir = OPTOGPT_ROOT / "data"
    if data_dir.exists():
        candidates.append(("data (通用)", data_dir))
    data_60 = OPTOGPT_ROOT / "data_60deg_s"
    if data_60.exists():
        candidates.append(("data_60deg_s", data_60))
    dielec_data = OPTOGPT_ROOT / "dielectric_60deg_s" / "data"
    if dielec_data.exists():
        candidates.append(("dielectric_60deg_s/data", dielec_data))

    for i, (label, path) in enumerate(candidates, 1):
        print(f"    {i}) {label}: {path}")

    idx = input(f"\n  选择数据集 [1-{len(candidates)}]: ").strip()
    try:
        selected = candidates[int(idx) - 1][1]
    except (ValueError, IndexError):
        print("  无效选择")
        return None

    spec_files = sorted(selected.glob("Spectrum_test.pkl"))
    if not spec_files:
        spec_files = sorted(selected.glob("Spectrum_train.pkl"))
    if not spec_files:
        print(f"  未找到 .pkl 文件于 {selected}")
        return None

    with open(spec_files[0], "rb") as f:
        specs = pkl.load(f)
    specs = np.array(specs)
    print(f"  加载了 {len(specs)} 条光谱")

    sample_idx = input(f"  输入样本索引 [0-{len(specs)-1}, 默认随机]: ").strip()
    if sample_idx:
        idx = int(sample_idx)
    else:
        idx = np.random.randint(0, len(specs))
    return specs[idx]


def _load_joint_array_file():
    """Load one 284-dimensional joint spectrum from a pickle or NumPy file."""
    path = Path(input("  联合光谱 .pkl/.npy 文件路径: ").strip()).expanduser()
    if path.suffix.lower() == ".npy":
        specs = np.load(path)
    elif path.suffix.lower() == ".pkl":
        with open(path, "rb") as handle:
            specs = pkl.load(handle)
    else:
        raise ValueError("联合光谱文件必须是 .pkl 或 .npy")
    specs = np.asarray(specs, dtype=np.float32)
    if specs.ndim == 1:
        return specs
    index_text = input(f"  样本索引 [0-{len(specs) - 1}, 默认 0]: ").strip()
    return specs[int(index_text) if index_text else 0]


def _input_manual_csv(joint_sp=False):
    """通过 CSV 粘贴或文件加载光谱。"""
    print("\n  提供光谱数据的方式:")
    columns = "波长,Rs,Ts,Rp,Tp" if joint_sp else "波长,R,T"
    print(f"    A) 粘贴 CSV 格式数据 ({columns})")
    print("    B) 提供 CSV 文件路径")
    choice = input("  选择 [A/B]: ").strip().upper()

    if choice == "A":
        paste_columns = "波长nm,Rs,Ts,Rp,Tp" if joint_sp else "波长nm,R,T"
        print(f"\n  请粘贴 CSV 数据 (格式: {paste_columns}, 每行一个):")
        print("  输入空行结束。示例:")
        print("    400,0.05,0.95")
        print("    410,0.06,0.94")
        print("    ...")
        lines = []
        while True:
            line = input()
            if not line.strip():
                break
            lines.append(line.strip())
        # 解析并插值到标准波长
        parser = _parse_joint_csv if joint_sp else _parse_and_interpolate_csv
        return parser("\n".join(lines))
    else:
        path = input("  CSV 文件路径: ").strip()
        with open(path, 'r') as f:
            content = f.read()
        parser = _parse_joint_csv if joint_sp else _parse_and_interpolate_csv
        return parser(content)


def _parse_and_interpolate_csv(csv_text):
    """从 CSV 文本解析 R,T 并插值到 71 点标准波长。"""
    data = []
    for line in csv_text.strip().split('\n'):
        parts = line.strip().split(',')
        if len(parts) >= 3:
            try:
                wl = float(parts[0])
                r = float(parts[1])
                t = float(parts[2])
                if not np.isfinite([wl, r, t]).all():
                    raise ValueError("CSV 包含 NaN 或 Inf")
                data.append((wl, r, t))
            except ValueError:
                if any(value.strip().lower() in {"nan", "+nan", "-nan", "inf", "+inf", "-inf"}
                       for value in parts[:3]):
                    raise ValueError("CSV 包含 NaN 或 Inf")
                continue

    if len(data) < 2:
        raise ValueError("CSV 至少需要两个不同波长的数据点")

    data = np.array(data)
    order = np.argsort(data[:, 0])
    data = data[order]
    if len(np.unique(data[:, 0])) != len(data):
        raise ValueError("CSV 包含重复波长")
    wl_input = data[:, 0]
    r_input = data[:, 1]
    t_input = data[:, 2]
    if wl_input[0] > WAVELENGTHS_NM[0] or wl_input[-1] < WAVELENGTHS_NM[-1]:
        raise ValueError("CSV 波长范围必须覆盖 400-1100 nm，禁止外推")
    if np.any((r_input < 0) | (r_input > 1) | (t_input < 0) | (t_input > 1)):
        raise ValueError("R 和 T 必须位于 [0, 1]")
    if np.any(r_input + t_input > 1 + 1e-6):
        raise ValueError("CSV 存在 R+T>1 的非物理数据")

    # 插值到标准波长
    r_fn = interp1d(wl_input, r_input, bounds_error=True)
    t_fn = interp1d(wl_input, t_input, bounds_error=True)
    r_std = r_fn(WAVELENGTHS_NM.astype(float))
    t_std = t_fn(WAVELENGTHS_NM.astype(float))
    if not np.isfinite(np.concatenate([r_std, t_std])).all():
        raise ValueError("CSV 插值结果包含 NaN 或 Inf")
    if np.any(r_std + t_std > 1 + 1e-6):
        raise ValueError("插值后存在 R+T>1 的非物理数据")

    full = np.concatenate([r_std, t_std])
    print(f"  ✅ 光谱已插值到 {N_WL} 点 (400-1100nm)")
    return full


def _parse_joint_csv(csv_text):
    """Parse wavelength, Rs, Ts, Rp, Tp CSV data into a 284-dim spectrum."""
    data = []
    for line in csv_text.strip().split("\n"):
        parts = [value.strip() for value in line.split(",")]
        if len(parts) < 5:
            continue
        try:
            row = tuple(float(value) for value in parts[:5])
        except ValueError:
            continue
        if not np.isfinite(row).all():
            raise ValueError("联合 CSV 包含 NaN 或 Inf")
        data.append(row)

    if len(data) < 2:
        raise ValueError("联合 CSV 至少需要两个不同波长的数据点")
    data = np.asarray(data, dtype=np.float64)
    data = data[np.argsort(data[:, 0])]
    wavelengths = data[:, 0]
    if len(np.unique(wavelengths)) != len(wavelengths):
        raise ValueError("联合 CSV 包含重复波长")
    if wavelengths[0] > WAVELENGTHS_NM[0] or wavelengths[-1] < WAVELENGTHS_NM[-1]:
        raise ValueError("联合 CSV 波长范围必须覆盖 400-1100 nm，禁止外推")

    columns = data[:, 1:]
    if np.any((columns < 0) | (columns > 1)):
        raise ValueError("联合 CSV 的 Rs/Ts/Rp/Tp 必须位于 [0, 1]")
    if np.any(columns[:, 0] + columns[:, 1] > 1 + 1e-6):
        raise ValueError("联合 CSV 存在 Rs+Ts>1 的非物理数据")
    if np.any(columns[:, 2] + columns[:, 3] > 1 + 1e-6):
        raise ValueError("联合 CSV 存在 Rp+Tp>1 的非物理数据")

    interpolated = []
    for values in columns.T:
        fn = interp1d(wavelengths, values, bounds_error=True)
        interpolated.append(fn(WAVELENGTHS_NM.astype(float)))
    result = np.concatenate(interpolated).astype(np.float32)
    if not np.isfinite(result).all():
        raise ValueError("联合 CSV 插值结果包含 NaN 或 Inf")
    for offset, label in ((0, "s"), (2 * N_WL, "p")):
        if np.any(result[offset:offset + N_WL] + result[
            offset + N_WL:offset + 2 * N_WL
        ] > 1 + 1e-6):
            raise ValueError(f"联合 CSV 插值后 {label} 偏振存在 R+T>1")
    print(f"  ✅ 联合 s+p 光谱已插值到 284 维 (400-1100nm)")
    return result


def _predefined_targets():
    """预定义的目标光谱。"""
    print("\n  预定义目标:")
    print("    1) 高透射 (T>90% 全波段)")
    print("    2) 高反射 (R>90% 全波段)")
    print("    3) 带通 500-600nm (500-600nm 高透, 其余高反)")
    print("    4) 长通 700nm (>700nm 高透, <700nm 高反)")
    print("    5) 短通 600nm (<600nm 高透, >600nm 高反)")
    print("    6) 陷波 550nm (550nm 附近低透, 其余高透)")
    choice = input("  选择 [1-6]: ").strip()

    n = N_WL
    wl = WAVELENGTHS_NM

    if choice == "1":
        R = np.full(n, 0.03)
        T = np.full(n, 0.95)
    elif choice == "2":
        R = np.full(n, 0.95)
        T = np.full(n, 0.03)
    elif choice == "3":
        R = np.where((wl >= 500) & (wl <= 600), 0.05, 0.95)
        T = np.where((wl >= 500) & (wl <= 600), 0.93, 0.03)
    elif choice == "4":
        R = np.where(wl >= 700, 0.05, 0.95)
        T = np.where(wl >= 700, 0.93, 0.03)
    elif choice == "5":
        R = np.where(wl <= 600, 0.05, 0.95)
        T = np.where(wl <= 600, 0.93, 0.03)
    elif choice == "6":
        R = np.where((wl >= 530) & (wl <= 570), 0.90, 0.05)
        T = np.where((wl >= 530) & (wl <= 570), 0.05, 0.93)
    else:
        print("  无效选择，使用高透射")
        R = np.full(n, 0.03)
        T = np.full(n, 0.95)

    return np.concatenate([R, T])


def select_test_data_dir(model_path, theta_deg=0, dielectric_only=False):
    """Choose the test set that matches the currently loaded checkpoint."""
    model_text = str(model_path).lower()
    if dielectric_only:
        return OPTOGPT_ROOT / "dielectric_60deg_s" / "data"
    if "60deg" in model_text or theta_deg == 60:
        return OPTOGPT_ROOT / "data_60deg_s"
    return OPTOGPT_ROOT / "data"


# ============================================================
# 主预测流程
# ============================================================

class InteractivePredictor:
    """交互式预测器主类。"""

    def __init__(self, model_path=None, pretrained_path=None, theta_deg=0, pol="s",
                 architecture=ARCH_AUTO):
        self.model_path = model_path
        self.pretrained_path = pretrained_path or str(OPTOGPT_ROOT / "model" / "optogpt.pt")
        self.theta_deg = theta_deg
        self.pol = pol
        self.architecture = architecture
        self.substrate = "Glass_Substrate"
        self.model = None
        self.word_dict = None
        self.index_dict = None
        self.configs = None
        self.logits_mask = None
        self.dielectric_only = False
        self.is_joint_sp = False
        self.history = []  # 预测历史

    def load_model(self):
        """加载模型。"""
        print(f"\n⏳ 加载模型: {self.model_path}")
        print(f"   Device: {DEVICE}")
        self.model, self.word_dict, self.index_dict, self.configs = load_model_from_ckpt(
            self.model_path, self.pretrained_path, architecture=self.architecture
        )
        self.is_joint_sp = _config_value(self.configs, "model_type", "") == "joint_sp"
        model_text = str(self.model_path).lower()
        description = str(_config_value(self.configs, "description", "")).lower()
        configured_materials = _config_value(self.configs, "allowed_materials")
        self.dielectric_only = (
            self.is_joint_sp
            or "dielectric" in model_text
            or "dielectric" in description
        )
        allowed_materials = (
            set(configured_materials) if configured_materials
            else ALLOWED_DIELECTRIC if self.dielectric_only else None
        )
        self.logits_mask = build_logits_mask(
            self.word_dict,
            allowed_materials=allowed_materials,
            banned_materials=BANNED_DIELECTRIC if self.dielectric_only else None,
            min_thickness=10 if self.dielectric_only and not self.is_joint_sp else None,
            max_thickness=300 if self.dielectric_only and not self.is_joint_sp else None,
        )
        if self.is_joint_sp:
            self.theta_deg = float(_config_value(self.configs, "theta_deg", 60))
            self.pol = "s+p"
        print(f"   ✅ 模型已加载 | 词表大小: {len(self.word_dict)} | "
              f"架构: {self.model.architecture_version}")
        if self.is_joint_sp:
            print("   模型类型: joint_sp | 输入: 284维 [Rs, Ts, Rp, Tp]")
            print(f"   联合条件: θ={self.theta_deg}°, s+p 双偏振")
        if allowed_materials:
            thickness_text = "checkpoint 词表范围" if self.is_joint_sp else "10-300 nm"
            print(f"   材料约束: {len(allowed_materials)} 种允许材料 | 厚度: {thickness_text}")

    def predict_and_validate(self, target_spec, num_candidates=8, top_k=10,
                             top_p=0.9, temperature=1.0, max_layers=20,
                             seed=42):
        """核心：预测 + TMM 验证。"""
        print(f"\n{'='*60}")
        mode = "s+p 联合偏振" if self.is_joint_sp else f"{self.pol}-pol"
        print(f"  🎯 开始预测 (θ={self.theta_deg}°, {mode})")
        print(f"{'='*60}")

        target_spec = np.asarray(target_spec, dtype=np.float32).reshape(-1)
        expected_dim = 284 if self.is_joint_sp else 142
        if target_spec.shape != (expected_dim,) or not np.isfinite(target_spec).all():
            layout = "[Rs, Ts, Rp, Tp]" if self.is_joint_sp else "[R, T]"
            raise ValueError(f"目标光谱必须是有限的 {expected_dim} 维 {layout}")
        branch_offsets = (0, 2 * N_WL) if self.is_joint_sp else (0,)
        for offset in branch_offsets:
            target_R = target_spec[offset:offset + N_WL]
            target_T = target_spec[offset + N_WL:offset + 2 * N_WL]
            if np.any((target_R < 0) | (target_R > 1)) or np.any(
                (target_T < 0) | (target_T > 1)
            ):
                raise ValueError("目标光谱的 R 和 T 必须在 [0, 1] 范围")
            if np.any(target_R + target_T > 1.0 + 1e-3):
                raise ValueError("目标光谱不满足能量守恒：存在 R+T>1")
        candidates = generate_candidates(
            self.model, target_spec, self.word_dict, self.index_dict,
            num_candidates=num_candidates, max_layers=max_layers,
            top_k=top_k, top_p=top_p, temperature=temperature,
            logits_mask=self.logits_mask, seed=seed,
        )
        results = []

        for i, candidate in enumerate(candidates):
            materials = candidate["materials"]
            thicknesses = candidate["thicknesses"]

            try:
                if self.is_joint_sp:
                    Rs, Ts, As = tmm_simulate(
                        materials, thicknesses, self.theta_deg, "s", self.substrate
                    )
                    Rp, Tp, Ap = tmm_simulate(
                        materials, thicknesses, self.theta_deg, "p", self.substrate
                    )
                    err = compute_joint_errors(target_spec, Rs, Ts, Rp, Tp)
                    absorption = {"s": As, "p": Ap}
                else:
                    R_sim, T_sim, absorption = tmm_simulate(
                        materials, thicknesses,
                        theta_deg=self.theta_deg, pol=self.pol,
                        substrate=self.substrate
                    )
                    err = compute_errors(target_spec, R_sim, T_sim)
            except Exception as e:
                print(f"    ⚠️ 候选 {i+1}: TMM 失败 - {e}")
                continue

            results.append({
                **candidate,
                "absorption": absorption,
                **err,
            })

        if not results:
            print("  ❌ 没有生成有效候选结构")
            return None

        # 按总 MAE 排序
        results.sort(key=lambda x: x["mae_total"])

        # 打印结果
        print(f"\n  📊 生成了 {len(results)} 个有效候选结构:\n")
        for rank, r in enumerate(results[:5]):
            struct_str = " | ".join(f"{m}({t:.0f}nm)" for m, t in zip(r["materials"], r["thicknesses"]))
            detail = (
                f"s={r['mae_s']:.4f}, p={r['mae_p']:.4f}"
                if self.is_joint_sp
                else f"R={r['mae_R']:.4f}, T={r['mae_T']:.4f}"
            )
            print(f"  [{rank+1}] MAE_total={r['mae_total']:.4f} "
                  f"({detail}) | {r['n_layers']}层: {struct_str[:80]}...")

        return results

    def run_interactive(self):
        """交互式主循环。"""
        condition = "s+p 双偏振" if self.is_joint_sp else f"{self.pol}-pol"
        print("\n" + "=" * 60)
        print("  🧪 OptoGPT 交互式预测器")
        print("=" * 60)
        print(f"  模型: {self.model_path}")
        print(f"  默认条件: θ={self.theta_deg}°, {condition}")
        print(f"  设备: {DEVICE}")

        while True:
            print("\n" + "-" * 60)
            print("  操作选项:")
            print("    1) 🎯 输入目标光谱 → 预测结构 → TMM验证")
            print("    2) 📂 批量测试 (从测试数据加载多条)")
            print("    3) ⚙️  修改模拟条件 (角度/偏振/基底)")
            print("    4) 📊 查看历史预测")
            print("    5) 🔄 切换模型")
            print("    6) 👋 退出")
            choice = input("\n  请选择 [1-6]: ").strip()

            if choice == "1":
                self._single_prediction()
            elif choice == "2":
                self._batch_test()
            elif choice == "3":
                self._change_conditions()
            elif choice == "4":
                self._show_history()
            elif choice == "5":
                self._switch_model()
            elif choice == "6":
                print("\n  👋 再见！")
                break
            else:
                print("  无效选择")

    def _single_prediction(self):
        """单次预测流程。"""
        target_spec = input_spectrum_manual(joint_sp=self.is_joint_sp)
        if target_spec is None:
            return

        num = input("  候选数量 [默认 8]: ").strip()
        num_candidates = int(num) if num else 8

        results = self.predict_and_validate(target_spec, num_candidates=num_candidates)
        if results is None:
            return

        best = results[0]

        # 显示详情
        print(f"\n  🏆 最佳结构 (MAE_total={best['mae_total']:.4f}):")
        if self.is_joint_sp:
            print(f"     s-pol MAE: {best['mae_s']:.4f}  |  "
                  f"p-pol MAE: {best['mae_p']:.4f}")
        else:
            print(f"     R MAE: {best['mae_R']:.4f}  |  T MAE: {best['mae_T']:.4f}")
        print(f"     层数: {best['n_layers']}")
        for i, (m, t) in enumerate(zip(best["materials"], best["thicknesses"])):
            print(f"       层{i+1}: {m:8s}  {t:6.0f} nm")

        # 保存结果
        best["theta_deg"] = self.theta_deg
        best["pol"] = self.pol
        best["substrate"] = self.substrate
        pol_text = "s+p" if self.is_joint_sp else f"{self.pol}-pol"
        best["info"] = f"| θ={self.theta_deg}° {pol_text} | model={Path(self.model_path).stem}"
        self.history.append(best)

        # 保存和绘图
        save_name = input("\n  保存图像文件名 [默认自动生成]: ").strip()
        if not save_name:
            ts = datetime.now().strftime("%Y%m%d_%H%M%S")
            save_name = f"prediction_{ts}.png"
        save_path = OUTPUT_DIR / save_name

        plot_comparison(best, save_path=str(save_path), show=True)

        # 保存 JSON
        json_path = OUTPUT_DIR / save_name.replace('.png', '.json')
        json_data = {
            "timestamp": datetime.now().isoformat(),
            "model": self.model_path,
            "theta_deg": self.theta_deg,
            "pol": self.pol,
            "substrate": self.substrate,
            "mae_R": best["mae_R"],
            "mae_T": best["mae_T"],
            "mae_total": best["mae_total"],
            "n_layers": best["n_layers"],
            "materials": best["materials"],
            "thicknesses": [float(t) for t in best["thicknesses"]],
            "tokens": best["tokens"],
        }
        if self.is_joint_sp:
            json_data.update({
                "model_type": "joint_sp",
                "mae_s": best["mae_s"], "mae_p": best["mae_p"],
                "mae_Rs": best["mae_Rs"], "mae_Ts": best["mae_Ts"],
                "mae_Rp": best["mae_Rp"], "mae_Tp": best["mae_Tp"],
                "target": {
                    "Rs": best["Rs_target"].tolist(),
                    "Ts": best["Ts_target"].tolist(),
                    "Rp": best["Rp_target"].tolist(),
                    "Tp": best["Tp_target"].tolist(),
                },
                "simulated": {
                    "Rs": best["Rs_sim"].tolist(),
                    "Ts": best["Ts_sim"].tolist(),
                    "Rp": best["Rp_sim"].tolist(),
                    "Tp": best["Tp_sim"].tolist(),
                },
            })
        with open(json_path, 'w') as f:
            json.dump(json_data, f, indent=2, ensure_ascii=False)
        print(f"  📁 JSON 已保存: {json_path}")

    def _batch_test(self):
        """批量测试模式。"""
        if self.is_joint_sp:
            print("\n  联合模型批量测试需要 284 维 Spectrum_test.pkl。")
            path = input("  文件路径 [回车取消]: ").strip()
            if not path:
                return
            spec_file = Path(path).expanduser()
            data_dir = spec_file.parent
        else:
            data_dir = select_test_data_dir(
                self.model_path, self.theta_deg, self.dielectric_only
            )
            spec_file = data_dir / "Spectrum_test.pkl"
        print("\n  批量测试: 从测试数据加载多条光谱进行预测验证")

        if not spec_file.exists():
            print("  ❌ 未找到测试数据文件")
            return

        with open(spec_file, "rb") as f:
            specs = np.array(pkl.load(f))
        print(f"  数据集: {data_dir}")
        print(f"  加载了 {len(specs)} 条测试光谱")

        n = input(f"  测试数量 [默认 10]: ").strip()
        n_samples = int(n) if n else 10
        n_samples = min(n_samples, len(specs))

        num = input("  每条候选数 [默认 4]: ").strip()
        num_candidates = int(num) if num else 4

        all_maes = []
        ts = datetime.now().strftime("%Y%m%d_%H%M%S")
        batch_dir = OUTPUT_DIR / f"batch_{ts}"
        batch_dir.mkdir(exist_ok=True)

        print(f"\n  ⏳ 批量测试 {n_samples} 条样本...")
        for i in range(n_samples):
            spec = specs[i]
            results = self.predict_and_validate(spec, num_candidates=num_candidates)
            if results:
                best = results[0]
                best["theta_deg"] = self.theta_deg
                best["pol"] = self.pol
                best["substrate"] = self.substrate
                pol_text = "s+p" if self.is_joint_sp else f"{self.pol}-pol"
                best["info"] = f"| Sample #{i} | θ={self.theta_deg}° {pol_text}"
                all_maes.append(best["mae_total"])
                save_path = batch_dir / f"sample_{i:03d}.png"
                plot_comparison(best, save_path=str(save_path), show=False)

        if all_maes:
            print(f"\n  📊 批量测试完成!")
            print(f"     样本数: {len(all_maes)}")
            print(f"     MAE 均值: {np.mean(all_maes):.4f}")
            print(f"     MAE 中位数: {np.median(all_maes):.4f}")
            print(f"     MAE 标准差: {np.std(all_maes):.4f}")
            print(f"     最佳 MAE: {np.min(all_maes):.4f}")
            print(f"     最差 MAE: {np.max(all_maes):.4f}")
            print(f"  📁 图像保存于: {batch_dir}")

    def _change_conditions(self):
        """修改模拟条件。"""
        pol_text = "s+p 双偏振" if self.is_joint_sp else f"{self.pol}-pol"
        print(f"\n  当前条件: θ={self.theta_deg}°, {pol_text}, substrate={self.substrate}")
        theta = input("  入射角(度): ").strip()
        if theta:
            self.theta_deg = float(theta)
        pol = input("  偏振 [s/p]: ").strip().lower()
        if not self.is_joint_sp and pol in ('s', 'p'):
            self.pol = pol
        elif self.is_joint_sp:
            self.pol = "s+p"
        sub = input("  基底 [Glass_Substrate/SiO2_Substrate/Si_Substrate]: ").strip()
        if sub in SUBSTRATES:
            self.substrate = sub
        pol_text = "s+p 双偏振" if self.is_joint_sp else f"{self.pol}-pol"
        print(f"  ✅ 更新: θ={self.theta_deg}°, {pol_text}, substrate={self.substrate}")

    def _show_history(self):
        """查看历史预测。"""
        if not self.history:
            print("\n  暂无历史记录")
            return
        print(f"\n  📊 历史预测 ({len(self.history)} 条):")
        for i, h in enumerate(self.history):
            struct = " | ".join(f"{m}({t:.0f})" for m, t in zip(h["materials"], h["thicknesses"]))
            print(f"  [{i}] MAE={h['mae_total']:.4f} | {h['n_layers']}层: {struct[:60]}...")
        idx = input("\n  输入序号查看详情 [回车跳过]: ").strip()
        if idx.isdigit() and int(idx) < len(self.history):
            h = self.history[int(idx)]
            plot_comparison(h, save_path=None, show=True)

    def _switch_model(self):
        """切换模型。"""
        print("\n  可用模型:")
        for key, info in MODEL_REGISTRY.items():
            if key != "custom":
                print(f"    {key}) {info['name']}")
                print(f"       {info['description']}")
        print(f"    C) 自定义路径")

        choice = input("\n  选择: ").strip().upper()
        if choice == "C":
            self.model_path = input("  Checkpoint 路径: ").strip()
            self.architecture = ARCH_AUTO
        elif choice in MODEL_REGISTRY:
            self.model_path = MODEL_REGISTRY[choice]["path"]
            self.theta_deg = MODEL_REGISTRY[choice]["default_theta"]
            self.pol = MODEL_REGISTRY[choice]["default_pol"]
            self.architecture = ARCH_AUTO
        else:
            print("  无效选择")
            return

        self.load_model()
        self.history.clear()
        print(f"  ✅ 已切换到: {self.model_path}")


# ============================================================
# 命令行入口
# ============================================================

def main():
    parser = argparse.ArgumentParser(
        description="OptoGPT 交互式预测器 — 输入光谱 → 生成结构 → TMM 验证"
    )
    parser.add_argument("--model", type=str, default=None,
                        help="模型 checkpoint 路径（不指定则启动交互菜单选择）")
    parser.add_argument("--pretrained", type=str, default=None,
                        help="预训练模型路径（用于回退获取词表）")
    parser.add_argument("--theta", type=float, default=None,
                        help="入射角 (度)")
    parser.add_argument("--pol", type=str, default=None, choices=['s', 'p'],
                        help="偏振")
    parser.add_argument("--spec_file", type=str, default=None,
                        help="直接指定目标光谱文件 (.pkl 或 .csv)")
    parser.add_argument("--spec_index", type=int, default=0,
                        help="光谱文件中的样本索引 (.pkl 时)")
    parser.add_argument("--num_candidates", type=int, default=8,
                        help="候选结构数量")
    parser.add_argument("--top_k", type=int, default=10,
                        help="采样 top-k")
    parser.add_argument("--top_p", type=float, default=0.9,
                        help="采样 top-p")
    parser.add_argument("--temperature", type=float, default=1.0,
                        help="采样温度")
    parser.add_argument("--max_layers", type=int, default=20,
                        help="最大膜层数")
    parser.add_argument("--seed", type=int, default=42,
                        help="随机种子")
    parser.add_argument("--output", type=str, default=None,
                        help="输出图像路径")
    parser.add_argument("--architecture", type=str, default=ARCH_AUTO,
                        choices=[ARCH_AUTO, ARCH_LEGACY, ARCH_RELU],
                        help="Checkpoint 前向架构；auto 会识别原始模型，其余无版本微调模型按 relu")

    args = parser.parse_args()

    # 确定模型
    if args.model is None:
        # 交互选择模型
        print("\n" + "=" * 60)
        print("  🧪 OptoGPT 交互式预测器")
        print("=" * 60)
        print("\n  可用模型:")
        for key, info in MODEL_REGISTRY.items():
            if key != "custom":
                print(f"    {key}) {info['name']}")
        print(f"    C) 自定义路径")

        choice = input("\n  选择模型 [默认 1]: ").strip().upper()
        if choice == "C" or choice == "" and not MODEL_REGISTRY.get("1"):
            model_path = input("  Checkpoint 路径: ").strip()
            theta = 0
            pol = "s"
        elif choice == "":
            choice = "1"
            info = MODEL_REGISTRY["1"]
            model_path = info["path"]
            theta = info["default_theta"]
            pol = info["default_pol"]
        elif choice in MODEL_REGISTRY:
            info = MODEL_REGISTRY[choice]
            model_path = info["path"]
            theta = info["default_theta"]
            pol = info["default_pol"]
        else:
            print("  无效选择，使用默认")
            info = MODEL_REGISTRY["1"]
            model_path = info["path"]
            theta = info["default_theta"]
            pol = info["default_pol"]
    else:
        model_path = args.model
        theta = args.theta if args.theta is not None else 0
        pol = args.pol if args.pol is not None else "s"

    pretrained = args.pretrained or str(OPTOGPT_ROOT / "model" / "optogpt.pt")

    predictor = InteractivePredictor(
        model_path=model_path,
        pretrained_path=pretrained,
        theta_deg=theta,
        pol=pol,
        architecture=args.architecture,
    )
    predictor.load_model()

    # 如果命令行直接指定了光谱文件，直接预测
    if args.spec_file:
        spec_path = Path(args.spec_file)
        if spec_path.suffix == '.pkl':
            with open(spec_path, 'rb') as f:
                specs = np.array(pkl.load(f))
            target = specs[args.spec_index]
            print(f"  使用光谱文件: {spec_path}, 索引: {args.spec_index}")
        elif spec_path.suffix == '.npy':
            specs = np.asarray(np.load(spec_path))
            target = specs if specs.ndim == 1 else specs[args.spec_index]
            print(f"  使用光谱文件: {spec_path}, 索引: {args.spec_index}")
        elif spec_path.suffix == '.csv':
            with open(spec_path, 'r') as f:
                parser = _parse_joint_csv if predictor.is_joint_sp else _parse_and_interpolate_csv
                target = parser(f.read())
            print(f"  使用光谱文件: {spec_path}")
        else:
            print(f"  不支持的文件格式: {spec_path.suffix}")
            return

        if target is not None:
            results = predictor.predict_and_validate(
                target,
                num_candidates=args.num_candidates,
                top_k=args.top_k,
                top_p=args.top_p,
                temperature=args.temperature,
                max_layers=args.max_layers,
                seed=args.seed,
            )
            if results:
                best = results[0]
                best["theta_deg"] = predictor.theta_deg
                best["pol"] = predictor.pol
                best["substrate"] = predictor.substrate
                pol_text = "s+p" if predictor.is_joint_sp else f"{predictor.pol}-pol"
                best["info"] = f"| θ={predictor.theta_deg}° {pol_text}"

                save_path = args.output or str(OUTPUT_DIR / f"prediction_{datetime.now().strftime('%Y%m%d_%H%M%S')}.png")
                plot_comparison(best, save_path=save_path, show=True)
        return

    # 否则进入交互模式
    predictor.run_interactive()


if __name__ == "__main__":
    main()
