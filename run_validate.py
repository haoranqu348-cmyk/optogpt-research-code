"""
TMM 
"""

import sys, os
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent / "optogpt"))

CONFIG = {
    "model": "model/optogpt_60deg_s_best.pt",     # 要验证的模型
    "pretrained": "model/optogpt.pt",             # 原始预训练模型（用于词表）
    "test_spec": "data_60deg_s/Spectrum_test.pkl", # 测试光谱
    "test_struct": "data_60deg_s/Structure_test.pkl", # 测试结构（可选）
    "num_samples": 50,           # ← 验证多少条？建议 50~200
    "num_candidates": 10,        # 每条生成几个候选挑最优
    "output_dir": "validation_results",  # 结果存哪里
}
# ============================================================

if __name__ == "__main__":
    sys.argv = [
        "validate",
        "--model", CONFIG["model"],
        "--pretrained", CONFIG["pretrained"],
        "--test_spec", CONFIG["test_spec"],
        "--test_struct", CONFIG["test_struct"],
        "--num_samples", str(CONFIG["num_samples"]),
        "--num_candidates", str(CONFIG["num_candidates"]),
        "--output_dir", CONFIG["output_dir"],
    ]
    
    os.chdir(Path(__file__).resolve().parent)
    
    from optogpt.validate_60deg_s import main
    main()
