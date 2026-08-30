"""5.3 跨电芯实验完整矩阵运行器（10 个配置）。

三个视角 × 四种策略：
    视角 A   ：同分布跨电芯（14 训练 / 6 测试，按温度组分层）
    视角 B1  ：10 环境温训练 → 10 恒温箱测试（跨温度外推）
    视角 B2  ：10 恒温箱训练 → 10 环境温测试（跨温度外推）
    策略 trans      ：Severson 预训练 + SIT 微调（温度+物理全开）
         trans-nt   ：同上但关闭温度嵌入（温度消融）
         scratch    ：随机初始化 + SIT 训练（从头对照）
    零样本（1 次全量）：预训练模型不做 SIT 微调，直接测评全部 20 只
         SIT 电池；A/B1/B2 各视角的零样本结果由过滤预测表得到。

每个配置调用 finetune_sit.py，输出：
    results/runs/sec53_<name>.log          运行日志
    results/metrics/temperature_soh/sec53/ <name>_preds.parquet  逐片段预测
    models/temperature_soh/sec53/<name>.pt  训练后模型（zero 不保存）

运行（需要先完成 sit_cache.py --rebuild）：
```powershell
& "E:\conda\envs\battery-soh\python.exe" scripts/run_sec53_matrix.py
```
"""

from __future__ import annotations

import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
PYTHON = r"E:\conda\envs\battery-soh\python.exe"
SCRIPT = ROOT / "src" / "temperature_soh" / "Trainer" / "finetune_sit.py"
CACHE_DIR = ROOT / "data" / "processed" / "sit_cache"
LOG_DIR = ROOT / "results" / "runs"
PRED_DIR = ROOT / "results" / "metrics" / "temperature_soh" / "sec53"
MODEL_DIR = ROOT / "models" / "temperature_soh" / "sec53"

# 电池分组（与 5.1.1 一致）。
AMBIENT = ["001-1","001-2","001-3","001-4","001-5","001-6","001-7","001-8",
           "101-1","101-3"]
CHAMBER = ["002-1","002-2","002-3","002-4","002-5","002-7",
           "003-1","003-3","003-5","003-7"]
# 视角 A：seed=42 组内分层随机（见 5.3 设计）。
A_TRAIN = ["001-6","001-7","001-1","001-8","001-4","001-3","001-5",
           "002-5","003-5","002-3","003-1","002-7","003-7","003-3"]
A_TEST  = ["101-3","001-2","101-1","002-4","002-1","002-2"]

# (名称, 训练电池, 测试电池, init, 温度嵌入, 物理λ, epochs)
CONFIGS: list[tuple[str, list[str], list[str], str, bool, float, int]] = [
    ("A-trans",      A_TRAIN, A_TEST, "pretrained", True,  0.1, 30),
    ("A-trans-nt",   A_TRAIN, A_TEST, "pretrained", False, 0.1, 30),
    ("A-scratch",    A_TRAIN, A_TEST, "random",     True,  0.1, 30),
    ("B1-trans",     AMBIENT, CHAMBER, "pretrained", True,  0.1, 30),
    ("B1-trans-nt",  AMBIENT, CHAMBER, "pretrained", False, 0.1, 30),
    ("B1-scratch",   AMBIENT, CHAMBER, "random",     True,  0.1, 30),
    ("B2-trans",     CHAMBER, AMBIENT, "pretrained", True,  0.1, 30),
    ("B2-trans-nt",  CHAMBER, AMBIENT, "pretrained", False, 0.1, 30),
    ("B2-scratch",   CHAMBER, AMBIENT, "random",     True,  0.1, 30),
    # 零样本全量：一次测评全部 20 只，各视角按测试电池过滤。
    ("zero-all",     [],      AMBIENT + CHAMBER, "pretrained", False, 0.0, 0),
]


def run_one(name: str, train: list[str], test: list[str], init: str,
            use_temp: bool, phys: float, epochs: int) -> int:
    """跑一个配置，日志落盘，返回子进程退出码。"""
    cmd = [
        PYTHON, str(SCRIPT),
        "--init", init,
        "--cache-dir", str(CACHE_DIR),
        "--train-cells", ",".join(train),
        "--test-cells", ",".join(test),
        "--epochs", str(epochs),
        "--min-soh", "0.75",
        "--save-preds", str(PRED_DIR / f"{name}_preds.parquet"),
        "--out", str(MODEL_DIR / f"{name}.pt"),
    ]
    if use_temp:
        cmd.append("--use-temp-embed")
    if phys > 0:
        cmd += ["--phys-lambda", str(phys)]

    LOG_DIR.mkdir(parents=True, exist_ok=True)
    PRED_DIR.mkdir(parents=True, exist_ok=True)
    MODEL_DIR.mkdir(parents=True, exist_ok=True)
    log_path = LOG_DIR / f"sec53_{name}.log"
    with open(log_path, "w", encoding="utf-8") as log:
        print(f"\n===== {name} =====  (log -> {log_path})", flush=True)
        proc = subprocess.run(
            cmd, stdout=log, stderr=subprocess.STDOUT, text=True, encoding="utf-8",
        )
    print(f"完成 {name}: 退出码 {proc.returncode}", flush=True)
    return proc.returncode


def main() -> None:
    sys.stdout.reconfigure(encoding="utf-8")
    if not (CACHE_DIR / "temp_features.npy").exists():
        raise SystemExit(
            f"缓存缺少 temp_features.npy，请先运行 "
            "src/temperature_soh/Trainer/sit_cache.py --rebuild"
        )
    failed = []
    for cfg in CONFIGS:
        code = run_one(*cfg)
        if code != 0:
            failed.append(cfg[0])
    if failed:
        print(f"\n失败配置: {failed}", flush=True)
        raise SystemExit(1)
    print("\n全部 10 个配置完成。", flush=True)


if __name__ == "__main__":
    main()
