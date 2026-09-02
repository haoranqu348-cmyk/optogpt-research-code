"生成 60° "
import sys, os
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent / "optogpt"))

from core.datasets.generate_data import main as generate_main
import argparse


CONFIG = {
    "num_samples": 10000,       # ← 要生成多少条
    "output_dir": "data_60deg_s",  # 数据存哪里
    "chunk_size": 2000,         # 每多少条存一次（防止中断丢失）
    "split": [0.8, 0.1, 0.1],  # 训练:验证:测试 比例
    "seed": 42,                 # 随机种子（同样的种子生成同样的数据）
    "resume": False,            # True=从断点继续，False=重新开始
    "merge_only": False,        # True=只合并已有chunk不生成新数据
    "min_layers": 1,
    "max_layers": 10,  # ← 修复: 应为10层而非20层
}
# ============================================================

if __name__ == "__main__":
    # 模拟命令行参数
    class FakeArgs:
        def __init__(self, d):
            for k, v in d.items():
                setattr(self, k, v)
    
    import sys
    # 把 sys.argv 伪装成 argparse 期望的格式（让它以为从命令行调用）
    sys.argv = [
        "generate",
        "--num_samples", str(CONFIG["num_samples"]),
        "--output_dir", CONFIG["output_dir"],
        "--chunk_size", str(CONFIG["chunk_size"]),
        "--split", str(CONFIG["split"][0]), str(CONFIG["split"][1]), str(CONFIG["split"][2]),
        "--seed", str(CONFIG["seed"]),
        "--min_layers", str(CONFIG["min_layers"]),
        "--max_layers", str(CONFIG["max_layers"]),
    ]
    if CONFIG["resume"]:
        sys.argv.append("--resume")
    if CONFIG["merge_only"]:
        sys.argv.append("--merge_only")
    
    generate_main()
