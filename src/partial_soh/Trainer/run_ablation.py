"""消融实验驱动器：顺序运行 5 个配置并汇总指标。

设计原则（软件工程）
-------------------
- 每个配置一个独立进程，跑完自动解析日志里的 MAE / RMSE；
- 所有配置共用相同的 seed、epoch 数、有效 batch 与每 epoch 样本量，
  只差“采样方式 / 损失项”，保证对比公平；
- 结果同时落盘为 JSON（机器可读）和 PNG 柱状图（报告可直接用）。

5 个配置
--------
1. baseline        基线（论文复现）：普通 shuffle，无创新损失；
2. grouped_control 只改采样：循环分组采样，但一致性损失权重=0；
3. consistency     同循环一致性约束；
4. recon           扩展自监督（掩码电压重建）；
5. full            完整方案：一致性 + 重建。

用法
----
& "E:\conda\envs\battery-soh\python.exe" "src/partial_soh/Trainer/run_ablation.py" `
    --epochs 10 --preload

输出
----
- results/runs/ablation_<name>.log   每个配置的完整训练日志；
- models/ablation_<name>.pt          每个配置的最终模型；
- results/metrics/ablation_consistency_ssl.json  汇总表；
- results/figures/ablation_consistency_ssl.png   测试 MAE / RMSE 柱状图。
"""

from __future__ import annotations

import argparse
import json
import re
import subprocess
import sys
import time
from dataclasses import dataclass
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[3]
TRAINER = ROOT / "src" / "partial_soh" / "Trainer" / "trainer.py"


@dataclass
class AblationConfig:
    """一个消融配置：名字 + 报告用中文名 + 传给 trainer 的额外参数。"""

    name: str
    label: str
    extra_args: list[str]


# 注意：--batch-groups / --group-size 对所有配置都传（普通模式会忽略），
# 这样 grouped 配置的有效 batch = batch_groups × group_size，与普通模式一致。
CONFIGS = [
    AblationConfig("baseline", "基线（论文复现）", []),
    AblationConfig(
        "grouped_control",
        "只改采样（对照）",
        ["--consistency", "--consist-lambda", "0"],
    ),
    AblationConfig("consistency", "同循环一致性", ["--consistency"]),
    AblationConfig("recon", "掩码电压重建", ["--recon-loss"]),
    AblationConfig(
        "full",
        "完整方案（一致+重建）",
        ["--consistency", "--recon-loss"],
    ),
]


def _parse_metrics(log_text: str) -> dict[str, float] | None:
    """从 trainer 日志中解析 训练/测试 的 MAE 和 RMSE（百分比）。

    trainer 输出顺序固定：
        训练集评估: MAE, RMSE
        测试集评估: MAE, RMSE
    所以第一次出现的两个 MAE 是训练集、后两个是测试集。
    """
    maes = re.findall(r"MAE\s*=\s*([\d.]+)%", log_text)
    rmses = re.findall(r"RMSE\s*=\s*([\d.]+)%", log_text)
    if len(maes) < 2 or len(rmses) < 2:
        return None
    return {
        "train_mae_pct": float(maes[0]),
        "train_rmse_pct": float(rmses[0]),
        "test_mae_pct": float(maes[1]),
        "test_rmse_pct": float(rmses[1]),
    }


def _run_one(
    cfg: AblationConfig,
    common_args: list[str],
    log_path: Path,
    model_out: Path,
) -> tuple[int, dict[str, float] | None]:
    """运行单个配置，返回 (返回码, 解析出的指标)。"""
    cmd = [
        sys.executable,
        str(TRAINER),
        *common_args,
        "--model-out",
        str(model_out),
        *cfg.extra_args,
    ]
    print(f"\n=== [{cfg.name}] {cfg.label} ===")
    print("命令: " + " ".join(str(c) for c in cmd))

    t0 = time.perf_counter()
    # 日志直接写入文件：既能完整保留训练过程，也避免 Windows 控制台编码问题。
    with open(log_path, "w", encoding="utf-8") as log_file:
        proc = subprocess.run(cmd, stdout=log_file, stderr=subprocess.STDOUT)
    elapsed = time.perf_counter() - t0

    if proc.returncode != 0:
        print(f"[{cfg.name}] 失败（返回码 {proc.returncode}），详见 {log_path}")
        return proc.returncode, None

    metrics = _parse_metrics(log_path.read_text(encoding="utf-8"))
    if metrics is None:
        print(f"[{cfg.name}] 日志中未找到完整的 MAE/RMSE，详见 {log_path}")
        return proc.returncode, None

    print(
        f"[{cfg.name}] 完成，耗时 {elapsed:.1f}s | "
        f"test MAE={metrics['test_mae_pct']:.4f}%  "
        f"RMSE={metrics['test_rmse_pct']:.4f}%"
    )
    return proc.returncode, metrics


def _save_plot(
    results: dict[str, dict[str, float] | None],
    out_path: Path,
    epochs: int,
) -> None:
    """把各配置的测试 MAE / RMSE 画成柱状图。"""
    try:
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt

        # Windows 上常见的中文字体；避免柱状图里中文变成方块。
        plt.rcParams["font.sans-serif"] = [
            "Microsoft YaHei",
            "SimHei",
            "Noto Sans CJK SC",
            "DejaVu Sans",
        ]
        plt.rcParams["axes.unicode_minus"] = False
    except ImportError:
        print("matplotlib 不可用，跳过画图")
        return

    names = [cfg.label for cfg in CONFIGS]
    mae = [results[cfg.name]["test_mae_pct"] for cfg in CONFIGS if results[cfg.name]]
    rmse = [results[cfg.name]["test_rmse_pct"] for cfg in CONFIGS if results[cfg.name]]
    labels = [cfg.label for cfg in CONFIGS if results[cfg.name]]
    if not mae:
        return

    x = np.arange(len(labels))
    width = 0.38
    fig, ax = plt.subplots(figsize=(9, 4.8))
    ax.bar(x - width / 2, mae, width, label="Test MAE", color="#4c72b0")
    ax.bar(x + width / 2, rmse, width, label="Test RMSE", color="#dd8452")
    ax.set_xticks(x)
    ax.set_xticklabels(labels, rotation=15, ha="right")
    ax.set_ylabel("误差（%）")
    ax.set_title(f"SOH 消融实验（{epochs} 预训练 + {epochs} 微调 epoch）")
    ax.legend()
    ax.grid(axis="y", alpha=0.3)
    fig.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=150)
    plt.close(fig)
    print(f"柱状图已保存: {out_path}")


def main() -> None:
    # line_buffering=True：每条 print 立即输出，方便后台监控进度。
    sys.stdout.reconfigure(encoding="utf-8", line_buffering=True)

    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--epochs", type=int, default=10, help="预训练与微调的 epoch 数（两边相同）")
    parser.add_argument("--batch-size", type=int, default=4096, help="普通模式 batch；分组模式有效 batch 由其对齐")
    parser.add_argument("--batch-groups", type=int, default=1024, help="分组模式每批循环数")
    parser.add_argument("--group-size", type=int, default=4, help="分组模式每个循环抽取的片段数")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--preload", action="store_true", help="预加载充电曲线到内存（全量训练建议开启）")
    parser.add_argument("--max-samples", type=int, default=None, help="冒烟验证：每个配置只取前 N 个样本")
    parser.add_argument("--index", type=Path, default=ROOT / "data" / "processed" / "partial_segments_index.parquet")
    parser.add_argument("--mat-dir", type=Path, default=ROOT / "data" / "external" / "matr")
    parser.add_argument(
        "--cache-dir",
        type=Path,
        default=ROOT / "data" / "processed" / "segments_cache",
        help="build_cache.py 生成的 memmap 缓存目录（训练提速，建议先用它构建）",
    )
    parser.add_argument("--log-dir", type=Path, default=ROOT / "results" / "runs")
    parser.add_argument("--model-dir", type=Path, default=ROOT / "models")
    parser.add_argument("--out", type=Path, default=ROOT / "results" / "metrics" / "ablation_consistency_ssl.json")
    parser.add_argument("--plot", type=Path, default=ROOT / "results" / "figures" / "ablation_consistency_ssl.png")
    args = parser.parse_args()

    common_args = [
        "--pretrain-epochs", str(args.epochs),
        "--finetune-epochs", str(args.epochs),
        "--batch-size", str(args.batch_size),
        "--batch-groups", str(args.batch_groups),
        "--group-size", str(args.group_size),
        "--seed", str(args.seed),
        "--index", str(args.index),
        "--mat-dir", str(args.mat_dir),
        "--cache-dir", str(args.cache_dir),
    ]
    if args.preload:
        common_args.append("--preload")
    if args.max_samples is not None:
        common_args += ["--max-samples", str(args.max_samples)]

    args.log_dir.mkdir(parents=True, exist_ok=True)
    args.model_dir.mkdir(parents=True, exist_ok=True)
    args.out.parent.mkdir(parents=True, exist_ok=True)

    print("消融实验开始：")
    print(f"  epochs={args.epochs}, batch_size={args.batch_size}, "
          f"batch_groups={args.batch_groups}, group_size={args.group_size}, "
          f"seed={args.seed}, preload={args.preload}")

    results: dict[str, dict[str, float] | None] = {}
    for cfg in CONFIGS:
        log_path = args.log_dir / f"ablation_{cfg.name}.log"
        model_out = args.model_dir / f"ablation_{cfg.name}.pt"
        _, metrics = _run_one(cfg, common_args, log_path, model_out)
        results[cfg.name] = metrics
        # 每跑完一个配置就落盘一次，中途中断也不丢已完成的结果。
        payload = {
            "hyperparameters": {
                "epochs": args.epochs,
                "batch_size": args.batch_size,
                "batch_groups": args.batch_groups,
                "group_size": args.group_size,
                "effective_batch": args.batch_groups * args.group_size,
                "seed": args.seed,
                "preload": args.preload,
            },
            "runs": results,
        }
        with open(args.out, "w", encoding="utf-8") as f:
            json.dump(payload, f, ensure_ascii=False, indent=2)

    print("\n========== 消融结果汇总 ==========")
    print(f"{'配置':<16}{'test MAE (%)':>16}{'test RMSE (%)':>16}")
    for cfg in CONFIGS:
        m = results[cfg.name]
        if m is None:
            print(f"{cfg.label:<16}{'失败':>16}{'':>16}")
        else:
            print(
                f"{cfg.label:<16}{m['test_mae_pct']:>16.4f}"
                f"{m['test_rmse_pct']:>16.4f}"
            )
    print(f"\n汇总 JSON: {args.out}")

    if all(m is not None for m in results.values()):
        _save_plot(results, args.plot, args.epochs)


if __name__ == "__main__":
    main()
