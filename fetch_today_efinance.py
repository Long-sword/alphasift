"""
用 efinance 获取今日实时行情（备用数据源）
"""
import os
import sys
import time
import json
import pandas as pd
import numpy as np

# 实时行情接口
import efinance as ef

print(f"[INFO] efinance 版本: {ef.__version__ if hasattr(ef, '__version__') else 'unknown'}")
print(f"[INFO] 当前时间: {time.strftime('%Y-%m-%d %H:%M:%S')}")

# 获取A股实时行情
print("\n[INFO] 获取A股实时行情...")
try:
    df = ef.stock.get_realtime_quotes()
    print(f"[INFO] 实时行情数据形状: {df.shape}")
    print(f"[INFO] 列名: {list(df.columns)}")
    print(f"[INFO] 前5行:\n{df.head()}")
except Exception as e:
    print(f"[ERROR] 获取实时行情失败: {e}")
    df = None

if df is not None and len(df) > 0:
    # 保存到本地，作为daily数据的备用
    out_path = '/workspace/data/today_realtime.csv'
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    df.to_csv(out_path, index=False, encoding='utf-8-sig')
    print(f"\n[INFO] 实时行情已保存: {out_path}")
    print(f"[INFO] 总标的数: {len(df)}")

    # 简单统计
    if '涨跌幅' in df.columns:
        up = (df['涨跌幅'] > 0).sum()
        down = (df['涨跌幅'] < 0).sum()
        flat = (df['涨跌幅'] == 0).sum()
        print(f"[INFO] 上涨: {up} | 下跌: {down} | 平盘: {flat}")

    if '成交额' in df.columns:
        # 转换为数值
        df['成交额_num'] = pd.to_numeric(df['成交额'], errors='coerce')
        top10_amt = df.nlargest(10, '成交额_num')[['股票代码', '股票名称', '涨跌幅', '成交额_num']]
        print(f"\n[INFO] 成交额Top10:")
        print(top10_amt.to_string(index=False))
