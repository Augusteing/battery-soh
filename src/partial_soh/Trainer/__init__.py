"""partial_soh 训练包。

各模块职责：

- dataset.py : 惰性加载的 PyTorch Dataset；
- model.py   : 共享编码器 LSTM + 电压头 / SOH 头；
- trainer.py : 预训练、微调与评估入口。
"""
