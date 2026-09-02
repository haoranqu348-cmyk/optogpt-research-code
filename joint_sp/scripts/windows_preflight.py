"""Read-only Windows deployment preflight for the joint s+p pipeline."""

import argparse
import json
import pickle
import platform
import sys
from pathlib import Path

import numpy as np
import torch

_SCRIPT_DIR = Path(__file__).resolve().parent
_JOINT_SP = _SCRIPT_DIR.parent
_PKG_ROOT = _JOINT_SP.parent
if str(_PKG_ROOT) not in sys.path:
    sys.path.insert(0, str(_PKG_ROOT))

from joint_sp.constants import SPEC_DIM, THETA_DEG
from joint_sp.model import load_joint_sp_checkpoint


def _check_data(data_dir):
    data_dir = Path(data_dir)
    with open(data_dir / "BUILD_COMPLETE.json", encoding="utf-8") as handle:
        complete = json.load(handle)
    if complete.get("status") != "complete":
        raise RuntimeError("BUILD_COMPLETE.json does not report complete")
    with open(data_dir / "generation_config.json", encoding="utf-8") as handle:
        config = json.load(handle)
    if config.get("spec_layout") != ["Rs", "Ts", "Rp", "Tp"]:
        raise RuntimeError("Data layout is not [Rs, Ts, Rp, Tp]")
    if config.get("spec_dim") != SPEC_DIM or config.get("theta_deg") != THETA_DEG:
        raise RuntimeError("Data dimension/angle contract mismatch")
    for split in ("train", "dev", "test"):
        with open(data_dir / f"Spectrum_{split}.pkl", "rb") as handle:
            spectra = np.asarray(pickle.load(handle))
        if spectra.ndim != 2 or spectra.shape[1] != SPEC_DIM:
            raise RuntimeError(f"{split} spectra shape is {spectra.shape}")


def main():
    parser = argparse.ArgumentParser(description="Windows joint_sp deployment preflight")
    parser.add_argument("--model", default=None)
    parser.add_argument("--data_dir", default=None)
    args = parser.parse_args()

    print(f"Python: {sys.version.split()[0]} ({platform.platform()})")
    print(f"PyTorch: {torch.__version__}")
    print(f"CUDA available: {torch.cuda.is_available()}")
    if torch.cuda.is_available():
        print(f"CUDA device: {torch.cuda.get_device_name(0)}")

    import tmm
    print(f"TMM import: OK ({Path(tmm.__file__).resolve()})")
    nk_dir = _PKG_ROOT / "optogpt" / "nk"
    if not nk_dir.is_dir():
        raise FileNotFoundError(f"NK database missing: {nk_dir}")
    print(f"NK database: OK ({nk_dir})")

    pretrained = _PKG_ROOT / "model" / "optogpt.pt"
    if not pretrained.is_file():
        raise FileNotFoundError(f"Base checkpoint missing: {pretrained}")
    print(f"Base checkpoint: OK ({pretrained})")

    if args.data_dir:
        _check_data(args.data_dir)
        print(f"Joint dataset: OK ({Path(args.data_dir).resolve()})")
    if args.model:
        model, _word_dict, _index_dict, configs = load_joint_sp_checkpoint(
            args.model, device="cpu"
        )
        if model.training:
            raise RuntimeError("Loaded deployment model is unexpectedly in training mode")
        print(f"Joint checkpoint: OK ({configs['architecture_version']})")

    print("PREFLIGHT_OK")


if __name__ == "__main__":
    main()
