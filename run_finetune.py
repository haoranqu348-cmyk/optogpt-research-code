import sys, os
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent / "optogpt"))


CONFIG = {
    "data_dir": "data_60deg_s",           # 数据目录
    "epochs": 10,                          # ← 训练几轮？建议 5~20
    "batch_size": 16,                      # 每批多少条（显存不够就减小）
    "lr": 3e-5,                            # ← 学习率，建议 1e-5 ~ 5e-5 之间
    "smoothing": 0.1,                      # label smoothing
    "seed": 42,
    "output_name": "optogpt_60deg_s",     # 模型保存的名字前缀
}
# ============================================================

if __name__ == "__main__":
    # 把参数伪装成命令行参数
    sys.argv = [
        "finetune",
        "--data_dir", CONFIG["data_dir"],
        "--epochs", str(CONFIG["epochs"]),
        "--batch_size", str(CONFIG["batch_size"]),
        "--lr", str(CONFIG["lr"]),
        "--smoothing", str(CONFIG["smoothing"]),
        "--seed", str(CONFIG["seed"]),
        "--output_name", CONFIG["output_name"],
    ]
    
    # 切到项目根目录（finetune 脚本用相对路径）
    os.chdir(Path(__file__).resolve().parent)
    
    # 直接 import 并运行 main
    from optogpt.finetune_60deg_s import main
    main()
