"""数据集下载入口。

用法:
    python scripts/download_data.py list
    python scripts/download_data.py stanford-summary
    python scripts/download_data.py stanford-raw --cells 3 17 45
    python scripts/download_data.py stanford-raw --all          # 约 16 GB
    python scripts/download_data.py matr --batches 20170512     # 单 batch 约 3 GB
    python scripts/download_data.py nasa-rw                     # 约 1 GB
"""

from __future__ import annotations

import argparse
import re
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from battery_soh.data.download import fetch, human_size  # noqa: E402

STANFORD_DRUID = "td676xr4322"
STANFORD_META_URL = f"https://purl.stanford.edu/{STANFORD_DRUID}.json"
STANFORD_FILE_URL = "https://stacks.stanford.edu/file/druid:{druid}/{name}?download=true"

MATR_FILES = {
    "20170512": ("https://data.matr.io/1/api/v1/file/5c86c0b5fa2ede00015ddf66/download", 3025320241),
    "20170630": ("https://data.matr.io/1/api/v1/file/5c86bf13fa2ede00015ddd82/download", None),
    "20180412": ("https://data.matr.io/1/api/v1/file/5c86bd64fa2ede00015ddbb2/download", None),
    "20190124": ("https://data.matr.io/1/api/v1/file/5dcef152110002c7215b2c90/download", 2601295745),
}

NASA_RW_URL = "https://phm-datasets.s3.amazonaws.com/NASA/11.+Randomized+Battery+Usage+Data+Set.zip"


def stanford_inventory() -> list[dict]:
    """从 Stanford SDR 元数据获取文件清单。"""
    import requests

    meta = requests.get(STANFORD_META_URL, timeout=60).json()
    files = []
    for fs in meta["structural"]["contains"]:
        for f in fs["structural"]["contains"]:
            files.append({"name": f["filename"], "size": f["size"]})
    return files


def download_stanford(dest: Path, pattern: str) -> None:
    files = [f for f in stanford_inventory() if re.fullmatch(pattern, f["name"])]
    if not files:
        raise SystemExit(f"未匹配到文件: {pattern}")
    total = sum(f["size"] for f in files)
    print(f"共 {len(files)} 个文件，约 {human_size(total)}")
    for f in files:
        url = STANFORD_FILE_URL.format(druid=STANFORD_DRUID, name=f["name"])
        fetch(url, dest / f["name"])


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("dataset", choices=["list", "stanford-summary", "stanford-raw", "matr", "nasa-rw"])
    parser.add_argument("--cells", type=int, nargs="*", default=None, help="Stanford raw 电池编号")
    parser.add_argument("--all", action="store_true", help="Stanford raw 全部下载（约 16 GB）")
    parser.add_argument("--batches", type=str, nargs="*", default=["20170512"], help="MATR 批次")
    parser.add_argument("--dest-root", type=Path, default=Path("data/external"))
    args = parser.parse_args()

    if args.dataset == "list":
        print("Stanford 动态循环数据集 (92 只电池, Geslin et al. Nat Energy 2024):")
        for f in stanford_inventory():
            print(f"  {f['name']}  {human_size(f['size'])}")
        print("\nMATR (Severson, 124 只 A123 LFP):")
        for name, (_, size) in MATR_FILES.items():
            print(f"  MATR_batch_{name}.mat  {human_size(size) if size else '(未知大小)'}")
        print(f"\nNASA Randomized Battery Usage: {NASA_RW_URL}")
        return

    if args.dataset == "stanford-summary":
        dest = args.dest_root / "stanford_dynamic"
        download_stanford(dest, r"README\.md")
        download_stanford(dest, r"Publishing_data/(aging_summary_cell_\d+\.csv|protocol_mapping_dic\.json|diagnostic_capacities\.pkl)")
    elif args.dataset == "stanford-raw":
        dest = args.dest_root / "stanford_dynamic"
        if args.all:
            download_stanford(dest, r"Publishing_data/raw_data_cell_\d+\.csv")
        elif args.cells:
            for c in args.cells:
                download_stanford(dest, rf"Publishing_data/raw_data_cell_{c:03d}\.csv")
        else:
            raise SystemExit("请用 --cells 指定电池编号，或 --all 全量下载")
    elif args.dataset == "matr":
        dest = args.dest_root / "matr"
        for b in args.batches:
            url, _ = MATR_FILES[b]
            fetch(url, dest / f"MATR_batch_{b}.mat")
    elif args.dataset == "nasa-rw":
        fetch(NASA_RW_URL, args.dest_root / "nasa_randomized" / "randomized_battery_usage.zip")


if __name__ == "__main__":
    main()
