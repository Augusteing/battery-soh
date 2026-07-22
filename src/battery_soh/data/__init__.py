"""数据加载与预处理：原始实车/台架数据的解析、清洗、时间对齐与重采样。"""

from battery_soh.data.download import fetch
from battery_soh.data.stanford_dynamic import build_soh_table, load_aging_summary

__all__ = ["fetch", "build_soh_table", "load_aging_summary"]
