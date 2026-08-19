"""partial_soh 数据层包。

各模块只做一件明确的事情：

- mat_io.py    : 读取 MATLAB v7.3 .mat 原始文件；
- charge.py    : 从单个循环中提取充电阶段；
- segments.py  : 按额定容量窗口生成部分充电片段；
- labels.py    : 生成 SOH 标签；
- build_dataset.py : 编排上述模块，产出最终数据集。
"""
