"""
用 akshare 获取A股最新行情（备用数据源）
"""
import os
import time
import pandas as pd

try:
    import akshare as ak
    print(f"[INFO] akshare 版本: {ak.__version__ if hasattr(ak, '__version__') else 'unknown'}")
except ImportError:
    print("[ERROR] akshare 未安装")
    ak = None

print(f"[INFO] 当前时间: {time.strftime('%Y-%m-%d %H:%M:%S')}")

if ak is not None:
    # 实时行情接口
    print("\n[INFO] 尝试获取A股实时行情 (stock_zh_a_spot)...")
    try:
        df = ak.stock_zh_a_spot()
        print(f"[INFO] 数据形状: {df.shape}")
        print(f"[INFO] 列名: {list(df.columns)}")
        if len(df) > 0:
            print(f"[INFO] 前3行:\n{df.head(3)}")
            # 保存
            out_path = '/workspace/data/today_akshare.csv'
            os.makedirs(os.path.dirname(out_path), exist_ok=True)
            df.to_csv(out_path, index=False, encoding='utf-8-sig')
            print(f"\n[INFO] 已保存: {out_path}")

            # 统计
            for col in ['涨跌幅', '涨跌额', '成交量', '成交额', '最新价']:
                if col in df.columns:
                    df[col] = pd.to_numeric(df[col], errors='coerce')

            if '涨跌幅' in df.columns:
                up = (df['涨跌幅'] > 0).sum()
                down = (df['涨跌幅'] < 0).sum()
                flat = (df['涨跌幅'] == 0).sum()
                print(f"[INFO] 上涨: {up} | 下跌: {down} | 平盘: {flat}")

            if '成交额' in df.columns:
                top10 = df.nlargest(10, '成交额')
                cols = [c for c in ['代码', '名称', '最新价', '涨跌幅', '成交额'] if c in top10.columns]
                print(f"\n[INFO] 成交额Top10:")
                print(top10[cols].to_string(index=False))
    except Exception as e:
        print(f"[ERROR] stock_zh_a_spot 失败: {e}")

    # 尝试新接口
    print("\n[INFO] 尝试 stock_zh_a_spot_em (东财)...")
    try:
        df2 = ak.stock_zh_a_spot_em()
        print(f"[INFO] 数据形状: {df2.shape}")
        if len(df2) > 0:
            print(f"[INFO] 前3行:\n{df2.head(3)}")
            out_path2 = '/workspace/data/today_akshare_em.csv'
            df2.to_csv(out_path2, index=False, encoding='utf-8-sig')
            print(f"\n[INFO] 已保存: {out_path2}")
    except Exception as e:
        print(f"[ERROR] stock_zh_a_spot_em 失败: {e}")
