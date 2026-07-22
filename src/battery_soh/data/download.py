"""通用下载工具：断点续传 + 进度显示 + 重试。"""

from __future__ import annotations

import sys
import time
from pathlib import Path

import requests

try:
    from tqdm import tqdm
except ImportError:
    tqdm = None

_CHUNK = 1 << 20  # 1 MiB


def human_size(n: float) -> str:
    for unit in ("B", "KB", "MB", "GB", "TB"):
        if n < 1024 or unit == "TB":
            return f"{n:.1f}{unit}" if unit != "B" else f"{int(n)}B"
        n /= 1024
    return f"{n:.1f}TB"


def fetch(url: str, dest: Path, retries: int = 5, timeout: int = 60) -> Path:
    """下载 url 到 dest；已存在部分文件时按 Range 续传。"""
    dest.parent.mkdir(parents=True, exist_ok=True)
    headers = {}
    pos = dest.stat().st_size if dest.exists() else 0
    if pos:
        headers["Range"] = f"bytes={pos}-"

    for attempt in range(1, retries + 1):
        try:
            with requests.get(url, stream=True, headers=headers, timeout=timeout) as resp:
                if pos and resp.status_code == 200:
                    pos = 0  # 服务端不支持续传，重新下载
                resp.raise_for_status()
                total = int(resp.headers.get("Content-Length", 0)) + pos
                mode = "ab" if pos else "wb"
                bar = (
                    tqdm(total=total, initial=pos, unit="B", unit_scale=True,
                         desc=dest.name, ncols=90)
                    if tqdm
                    else None
                )
                downloaded = pos
                t0 = time.time()
                with open(dest, mode) as f:
                    for chunk in resp.iter_content(chunk_size=_CHUNK):
                        if not chunk:
                            continue
                        f.write(chunk)
                        downloaded += len(chunk)
                        if bar:
                            bar.update(len(chunk))
                        elif time.time() - t0 > 10:
                            sys.stderr.write(
                                f"\r{dest.name}: {human_size(downloaded)}/{human_size(total)}"
                            )
                            sys.stderr.flush()
                            t0 = time.time()
                if bar:
                    bar.close()
                return dest
        except (requests.RequestException, OSError) as exc:
            pos = dest.stat().st_size if dest.exists() else 0
            headers["Range"] = f"bytes={pos}-"
            if attempt == retries:
                raise RuntimeError(f"下载失败（已重试 {retries} 次）: {url}") from exc
            wait = min(2**attempt, 30)
            print(f"[WARN] {dest.name}: {exc}；{wait}s 后重试（第 {attempt}/{retries} 次）",
                  file=sys.stderr)
            time.sleep(wait)
    return dest
