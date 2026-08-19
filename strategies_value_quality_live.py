#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
A股5策略选股 - 8月19日盘中实时数据
策略: dual_low / quality_value / blue_chip_income / low_volatility_quality / oversold_reversal
数据源: today_akshare.csv(盘中实时) + Tushare(历史K线/基本面)
"""

import tushare as ts
import pandas as pd
import numpy as np
import json
import time
import os

# ======================== 配置 ========================
TUSHARE_TOKEN = '9f640d421f866dde9a1888ab4193f77c43659d47446d717ffa343023'
TODAY_CSV = '/workspace/data/today_akshare.csv'
OUTPUT_JSON = '/workspace/reports/value_quality_live.json'
END_DATE = '20260818'       # 历史K线截止日期
TODAY_DATE = '20260819'     # 今日盘中日期
ONE_YEAR_AGO = '20250819'   # 次新过滤阈值(上市<1年排除)

os.makedirs(os.path.dirname(OUTPUT_JSON), exist_ok=True)
ts.set_token(TUSHARE_TOKEN)
pro = ts.pro_api()


# ======================== Step 1: 读取今日盘中数据 ========================
print("=" * 100)
print("【Step 1】读取今日盘中行情数据 (8月19日11:34实时)")
print("=" * 100)

df_today = pd.read_csv(TODAY_CSV, dtype={'代码': str})
print(f"  原始股票数: {len(df_today)}")

# 成交额转亿元
df_today['成交额亿'] = df_today['成交额'] / 1e8

# 过滤成交额>=2亿
df_today = df_today[df_today['成交额亿'] >= 2].copy()
print(f"  成交额>=2亿: {len(df_today)} 只")

# 过滤最新价<=0(停牌等)
df_today = df_today[df_today['最新价'] > 0].copy()
print(f"  有效最新价: {len(df_today)} 只")


# 代码转ts_code: bj920000 -> 920000.BJ, sh600000 -> 600000.SH, sz000001 -> 000001.SZ
def to_ts_code(code):
    prefix, num = code[:2], code[2:]
    if prefix == 'bj':
        return f'{num}.BJ'
    elif prefix == 'sh':
        return f'{num}.SH'
    elif prefix == 'sz':
        return f'{num}.SZ'
    return None

df_today['ts_code'] = df_today['代码'].apply(to_ts_code)
df_today = df_today.dropna(subset=['ts_code']).copy()

# 取成交额Top300
df_top = df_today.nlargest(300, '成交额亿').copy().reset_index(drop=True)
print(f"  成交额Top300: {len(df_top)} 只")


# ======================== Step 2: 股票基础信息(行业/上市日期) + 排除条件 ========================
print("\n" + "=" * 100)
print("【Step 2】获取股票基础信息 & 排除ST/退市/次新")
print("=" * 100)

stock_basic = pro.stock_basic(exchange='', list_status='L',
                              fields='ts_code,symbol,name,industry,list_date')
print(f"  Tushare在市股票: {len(stock_basic)} 只")

df_top = df_top.merge(stock_basic[['ts_code', 'industry', 'list_date']],
                      on='ts_code', how='left')
n_no_match = df_top['industry'].isna().sum()
if n_no_match > 0:
    print(f"  警告: {n_no_match} 只未匹配到基础信息(可能是BJ/新股), 将被排除")
    df_top = df_top.dropna(subset=['industry']).copy()

# 排除条件1: ST/*ST (名称含ST)
before = len(df_top)
df_top = df_top[~df_top['名称'].str.contains('ST', na=False)].copy()
print(f"  排除ST/*ST: -{before - len(df_top)}, 剩余 {len(df_top)}")

# 排除条件2: 次新(上市<1年, list_date > 20250819)
before = len(df_top)
df_top = df_top[df_top['list_date'].fillna('00000000') <= ONE_YEAR_AGO].copy()
print(f"  排除次新(上市<1年): -{before - len(df_top)}, 剩余 {len(df_top)}")

# 排除条件3: 退市已通过 list_status='L' 过滤
print(f"  退市: 已通过list_status='L'过滤")


# ======================== Step 3: 基本面数据(PE/PB/市值/换手率) ========================
print("\n" + "=" * 100)
print("【Step 3】获取基本面数据 (PE/PB/总市值/换手率, 日期=20260818)")
print("=" * 100)

daily_basic = pro.daily_basic(trade_date=END_DATE,
                              fields='ts_code,trade_date,turnover_rate,pe,pb,total_mv,circ_mv')
print(f"  daily_basic记录数: {len(daily_basic)}")

df_top = df_top.merge(daily_basic[['ts_code', 'turnover_rate', 'pe', 'pb', 'total_mv']],
                      on='ts_code', how='left')
df_top['总市值亿'] = df_top['total_mv'] / 10000.0  # 万元 -> 亿元
n_no_basic = df_top['pe'].isna().sum()
print(f"  合并基本面后: {len(df_top)} 只 (其中{n_no_basic}只无PE数据,将在策略硬筛中排除)")


# ======================== Step 4: 获取历史K线 & 计算技术指标 ========================
print("\n" + "=" * 100)
print("【Step 4】获取历史K线(近30日) + 附加今日盘中数据 + 计算技术指标")
print("=" * 100)

def calc_indicators(ts_code, today_close, today_vol):
    """
    获取Tushare近30日K线 -> 附加今日盘中数据 -> 计算技术指标
    返回dict: MA5/MA10/MA20/均线多头/20日波动率/20日涨幅/RSI/量比/距MA20%
    """
    try:
        df = pro.daily(ts_code=ts_code, end_date=END_DATE, limit=30)
    except Exception:
        return None

    if df is None or len(df) < 20:
        return None

    df = df.sort_values('trade_date').reset_index(drop=True)

    # 附加今日盘中数据 (成交量: 股 -> 手, 与Tushare对齐)
    today_row = pd.DataFrame([{
        'ts_code': ts_code,
        'trade_date': TODAY_DATE,
        'close': float(today_close),
        'vol': float(today_vol) / 100.0,
    }])
    df = pd.concat([df, today_row], ignore_index=True)

    closes = df['close'].values.astype(float)
    vols = df['vol'].values.astype(float)

    if len(closes) < 21:
        return None

    # MA5/MA10/MA20 (用近20日历史+今日盘中收盘)
    ma5 = float(closes[-5:].mean())
    ma10 = float(closes[-10:].mean())
    ma20 = float(closes[-20:].mean())

    # 均线多头: MA5 > MA10 > MA20
    ma_bull = bool(ma5 > ma10 > ma20)

    # 20日波动率: np.std(np.diff(np.log(closes[-21:])))*np.sqrt(252)*100
    log_ret = np.diff(np.log(closes[-21:]))
    vol_20 = float(np.std(log_ret) * np.sqrt(252) * 100)

    # 20日涨幅: (close[-1]/close[-21]-1)*100  (close[-1]为今日盘中价)
    ret_20 = float((closes[-1] / closes[-21] - 1) * 100)

    # RSI(14) - 标准RSI公式(简单平均法)
    delta = np.diff(closes[-15:])  # 最近15个收盘价, 14个差分
    gain = np.where(delta > 0, delta, 0.0)
    loss = np.where(delta < 0, -delta, 0.0)
    avg_gain = float(gain.mean())
    avg_loss = float(loss.mean())
    if avg_loss == 0:
        rsi = 100.0
    else:
        rs = avg_gain / avg_loss
        rsi = float(100 - 100 / (1 + rs))

    # 量比: 今日盘中vol / 20日均vol
    avg_vol_20 = float(vols[-21:-1].mean())  # 今日之前20日均量
    vol_ratio = float(vols[-1] / avg_vol_20) if avg_vol_20 > 0 else 0.0

    # 距MA20%: (close/MA20-1)*100
    dist_ma20 = float((closes[-1] / ma20 - 1) * 100)

    return {
        'MA5': round(ma5, 2),
        'MA10': round(ma10, 2),
        'MA20': round(ma20, 2),
        '均线多头': ma_bull,
        '20日波动率': round(vol_20, 2),
        '20日涨幅': round(ret_20, 2),
        'RSI': round(rsi, 2),
        '量比': round(vol_ratio, 2),
        '距MA20%': round(dist_ma20, 2),
    }


results = []
total = len(df_top)
fail_count = 0
for i, (_, row) in enumerate(df_top.iterrows()):
    ts_code = row['ts_code']
    today_close = row['最新价']
    today_vol = row['成交量']

    if i % 25 == 0:
        print(f"  进度: {i}/{total} (失败:{fail_count})...")

    indicators = calc_indicators(ts_code, today_close, today_vol)
    if indicators is None:
        fail_count += 1
        time.sleep(0.05)
        continue

    record = {
        'ts_code': ts_code,
        '代码': row['代码'],
        '名称': row['名称'],
        '行业': row['industry'] if pd.notna(row['industry']) else '未知',
        '最新价': float(today_close),
        '涨跌幅': float(row['涨跌幅']),
        '成交额亿': round(float(row['成交额亿']), 2),
        'PE': float(row['pe']) if pd.notna(row['pe']) else None,
        'PB': float(row['pb']) if pd.notna(row['pb']) else None,
        '总市值亿': round(float(row['总市值亿']), 2) if pd.notna(row['总市值亿']) else None,
        '换手率': float(row['turnover_rate']) if pd.notna(row['turnover_rate']) else None,
    }
    record.update(indicators)
    results.append(record)

    time.sleep(0.12)  # Tushare限流

df_all = pd.DataFrame(results)
print(f"\n  成功计算指标: {len(df_all)} 只 (失败:{fail_count})")


# ======================== Step 5: 5策略硬筛+评分 ========================
print("\n" + "=" * 100)
print("【Step 5】运行5个选股策略 (硬筛+评分)")
print("=" * 100)

def format_indicators(row):
    """格式化关键技术指标为紧凑字符串"""
    ma = "多头" if row['均线多头'] else "空头"
    return (f"MA{ma}|波动{row['20日波动率']:.1f}%|20日涨{row['20日涨幅']:+.1f}%|"
            f"RSI{row['RSI']:.0f}|量比{row['量比']:.1f}|距MA20{row['距MA20%']:+.1f}%")


# ----- 策略1: dual_low 双低选股（价值） -----
def strategy_dual_low(df):
    def calc(row):
        pe, pb = row['PE'], row['PB']
        amt, ret20, today_pct = row['成交额亿'], row['20日涨幅'], row['涨跌幅']
        mcap = row['总市值亿']
        if pe is None or pb is None or mcap is None:
            return None
        # 硬筛
        if not (0 < pe <= 30): return None
        if not (0 < pb <= 3): return None
        if amt < 5: return None
        if mcap < 50: return None
        # 评分
        s = 0
        s += max(0.0, 30.0 - pe)                              # PE越低分越高(30分封顶)
        s += max(0.0, 20.0 * (3 - pb) / 3)                   # PB越低分越高(20分)
        if amt >= 10: s += 10                                  # 成交额>=10亿加10分
        if 0 <= ret20 <= 15: s += 15                           # 20日涨幅0-15%加15分
        if -2 <= today_pct <= 3: s += 10                       # 当日涨幅-2~3%加10分
        if mcap >= 200: s += 15                                # 总市值>=200亿加15分
        return round(s, 2)
    df = df.copy()
    df['得分'] = df.apply(calc, axis=1)
    df['得分'] = pd.to_numeric(df['得分'], errors='coerce')
    return df.dropna(subset=['得分']).nlargest(10, '得分').reset_index(drop=True)


# ----- 策略2: quality_value 稳健价值（价值） -----
def strategy_quality_value(df):
    def calc(row):
        pe, pb = row['PE'], row['PB']
        amt, ret20, ma_bull = row['成交额亿'], row['20日涨幅'], row['均线多头']
        mcap, turnover, vol20 = row['总市值亿'], row['换手率'], row['20日波动率']
        if pe is None or pb is None or mcap is None or turnover is None:
            return None
        # 硬筛
        if not (10 <= pe <= 40): return None
        if not (0 < pb <= 5): return None
        if amt < 3: return None
        if mcap < 100: return None
        if vol20 > 50: return None
        # 评分
        s = 0
        if 15 <= pe <= 30: s += 15                             # PE 15-30加15分
        if pb < 3: s += 10                                     # PB<3加10分
        if 0 <= ret20 <= 15: s += 15                           # 20日涨幅0-15%加15分
        if ma_bull: s += 10                                    # 均线多头加10分
        if amt >= 10: s += 10                                  # 成交额>=10亿加10分
        if 3 <= turnover <= 8: s += 10                         # 换手率3-8%加10分
        if vol20 < 30: s += 10                                 # 20日波动率<30%加10分
        return round(s, 2)
    df = df.copy()
    df['得分'] = df.apply(calc, axis=1)
    df['得分'] = pd.to_numeric(df['得分'], errors='coerce')
    return df.dropna(subset=['得分']).nlargest(10, '得分').reset_index(drop=True)


# ----- 策略3: blue_chip_income 蓝筹收益质量（收益） -----
def strategy_blue_chip_income(df):
    def calc(row):
        pe, pb = row['PE'], row['PB']
        amt, ret20, today_pct = row['成交额亿'], row['20日涨幅'], row['涨跌幅']
        mcap, turnover, ma_bull = row['总市值亿'], row['换手率'], row['均线多头']
        if pe is None or pb is None or mcap is None or turnover is None:
            return None
        # 硬筛
        if mcap < 300: return None
        if amt < 5: return None
        if not (0 < pe <= 30): return None
        if not (0 < pb <= 5): return None
        # 评分
        s = 0
        if mcap >= 1000: s += 20                               # 总市值>=1000亿加20分
        if amt >= 20: s += 15                                  # 成交额>=20亿加15分
        if pe < 20: s += 15                                    # PE<20加15分
        if ma_bull: s += 10                                    # 均线多头加10分
        if 0 <= ret20 <= 10: s += 10                           # 20日涨幅0-10%加10分
        if -2 <= today_pct <= 2: s += 10                       # 当日涨幅-2~2%加10分
        if 1 <= turnover <= 5: s += 10                         # 换手率1-5%加10分
        return round(s, 2)
    df = df.copy()
    df['得分'] = df.apply(calc, axis=1)
    df['得分'] = pd.to_numeric(df['得分'], errors='coerce')
    return df.dropna(subset=['得分']).nlargest(10, '得分').reset_index(drop=True)


# ----- 策略4: low_volatility_quality 低波质量（质量） -----
def strategy_low_volatility_quality(df):
    def calc(row):
        pe, pb = row['PE'], row['PB']
        amt, ret20, today_pct = row['成交额亿'], row['20日涨幅'], row['涨跌幅']
        mcap, vol20, ma_bull = row['总市值亿'], row['20日波动率'], row['均线多头']
        dist_ma20 = row['距MA20%']
        if pe is None or pb is None or mcap is None:
            return None
        # 硬筛
        if vol20 > 30: return None
        if not (0 < pe <= 50): return None
        if not (0 < pb <= 5): return None
        if amt < 3: return None
        if mcap < 100: return None
        # 评分
        s = 0
        if vol20 <= 15: s += 25                                # 波动率<=15%加25分
        elif vol20 <= 25: s += 15                              # 15-25%加15分
        elif vol20 <= 30: s += 8                               # 25-30%加8分
        if ma_bull: s += 15                                    # 均线多头加15分
        if 0 <= ret20 <= 10: s += 15                           # 20日涨幅0-10%加15分
        if 10 <= pe <= 30: s += 10                             # PE 10-30加10分
        if amt >= 5: s += 10                                   # 成交额>=5亿加10分
        if 0 <= dist_ma20 <= 10: s += 10                       # 距MA20 0-10%加10分
        if -2 <= today_pct <= 2: s += 10                       # 当日涨跌-2~2%加10分
        return round(s, 2)
    df = df.copy()
    df['得分'] = df.apply(calc, axis=1)
    df['得分'] = pd.to_numeric(df['得分'], errors='coerce')
    return df.dropna(subset=['得分']).nlargest(10, '得分').reset_index(drop=True)


# ----- 策略5: oversold_reversal 超跌反转（反转） -----
def strategy_oversold_reversal(df):
    def calc(row):
        amt, ret20, mcap = row['成交额亿'], row['20日涨幅'], row['总市值亿']
        rsi, today_pct, vol_ratio = row['RSI'], row['涨跌幅'], row['量比']
        ma5, today_close = row['MA5'], row['最新价']
        if mcap is None:
            return None
        # 硬筛
        if ret20 > -10: return None                            # 20日跌幅<=-10%
        if amt < 2: return None
        if mcap < 30: return None
        if rsi > 40: return None
        # 评分
        s = 0
        if ret20 <= -20: s += 25                               # 20日跌幅<=-20%加25分
        elif ret20 <= -15: s += 20                             # -15~-20%加20分
        elif ret20 <= -10: s += 15                             # -10~-15%加15分
        if rsi < 25: s += 20                                   # RSI<25加20分
        elif rsi < 35: s += 15                                 # 25-35加15分
        elif rsi <= 40: s += 10                               # 35-40加10分
        if today_pct > 0: s += 15                              # 当日涨幅>0加15分(放量反弹信号)
        if vol_ratio > 1.5: s += 10                            # 量比>1.5加10分
        if today_close > ma5: s += 10                          # 均线站上MA5加10分
        return round(s, 2)
    df = df.copy()
    df['得分'] = df.apply(calc, axis=1)
    df['得分'] = pd.to_numeric(df['得分'], errors='coerce')
    return df.dropna(subset=['得分']).nlargest(10, '得分').reset_index(drop=True)


# 运行5个策略
strategies = [
    ("策略1", "dual_low", "双低选股（价值）", strategy_dual_low),
    ("策略2", "quality_value", "稳健价值（价值）", strategy_quality_value),
    ("策略3", "blue_chip_income", "蓝筹收益质量（收益）", strategy_blue_chip_income),
    ("策略4", "low_volatility_quality", "低波质量（质量）", strategy_low_volatility_quality),
    ("策略5", "oversold_reversal", "超跌反转（反转）", strategy_oversold_reversal),
]

strategy_results = {}
for label, key, desc, func in strategies:
    df_res = func(df_all)
    strategy_results[key] = df_res
    print(f"  {label} {desc}: {len(df_res)} 只入选 (候选池{len(df_all)})")


# ======================== 输出Top10表格 ========================
print("\n" + "=" * 100)
print("【输出】5策略 Top10 选股结果")
print("=" * 100)

def print_strategy_table(df_res, label, desc):
    print(f"\n{'─' * 100}")
    print(f"  {label}: {desc} - Top{len(df_res)}")
    print(f"{'─' * 100}")

    if len(df_res) == 0:
        print("  (无符合条件的股票)")
        return

    # 构建展示DataFrame
    display_cols = ['排名', '代码', '名称', '行业', '最新价', '涨跌%', '成交额亿',
                    'PE', 'PB', '总市值亿', '关键技术指标', '得分']
    display_data = []
    for i, (_, row) in enumerate(df_res.iterrows()):
        display_data.append({
            '排名': i + 1,
            '代码': row['ts_code'],
            '名称': row['名称'],
            '行业': row['行业'],
            '最新价': f"{row['最新价']:.2f}",
            '涨跌%': f"{row['涨跌幅']:+.2f}%",
            '成交额亿': f"{row['成交额亿']:.2f}",
            'PE': f"{row['PE']:.1f}" if row['PE'] is not None else "N/A",
            'PB': f"{row['PB']:.2f}" if row['PB'] is not None else "N/A",
            '总市值亿': f"{row['总市值亿']:.0f}" if row['总市值亿'] is not None else "N/A",
            '关键技术指标': format_indicators(row),
            '得分': f"{row['得分']:.1f}",
        })
    df_display = pd.DataFrame(display_data, columns=display_cols)
    # 使用to_string输出, index=False
    print(df_display.to_string(index=False, justify='left'))


for label, key, desc, _ in strategies:
    print_strategy_table(strategy_results[key], label, desc)


# ======================== 保存JSON ========================
print("\n" + "=" * 100)
print("【保存】JSON结果文件")
print("=" * 100)

def clean_for_json(obj):
    """递归清理数据使其JSON可序列化"""
    if isinstance(obj, dict):
        return {k: clean_for_json(v) for k, v in obj.items()}
    elif isinstance(obj, list):
        return [clean_for_json(v) for v in obj]
    elif isinstance(obj, (np.integer,)):
        return int(obj)
    elif isinstance(obj, (np.floating,)):
        return None if (pd.isna(obj)) else float(obj)
    elif isinstance(obj, (np.bool_,)):
        return bool(obj)
    elif isinstance(obj, float):
        return None if pd.isna(obj) else obj
    elif obj is None:
        return None
    else:
        return obj

json_output = {
    "生成时间": "2026-08-19 盘中(11:34数据)",
    "数据说明": {
        "盘中行情": "today_akshare.csv (8月19日11:34实时)",
        "历史K线": f"Tushare pro.daily end_date={END_DATE} limit=30",
        "基本面": f"Tushare pro.daily_basic trade_date={END_DATE}",
        "候选池数量": int(len(df_all)),
    },
    "策略结果": {}
}

for label, key, desc, _ in strategies:
    df_res = strategy_results[key]
    records = []
    for i, (_, row) in enumerate(df_res.iterrows()):
        rec = {
            "排名": i + 1,
            "代码": row['ts_code'],
            "名称": row['名称'],
            "行业": row['行业'],
            "最新价": row['最新价'],
            "涨跌幅": row['涨跌幅'],
            "成交额亿": row['成交额亿'],
            "PE": row['PE'],
            "PB": row['PB'],
            "总市值亿": row['总市值亿'],
            "换手率": row.get('换手率'),
            "MA5": row.get('MA5'),
            "MA10": row.get('MA10'),
            "MA20": row.get('MA20'),
            "均线多头": row.get('均线多头'),
            "20日波动率": row.get('20日波动率'),
            "20日涨幅": row.get('20日涨幅'),
            "RSI": row.get('RSI'),
            "量比": row.get('量比'),
            "距MA20%": row.get('距MA20%'),
            "得分": row['得分'],
        }
        records.append(rec)
    json_output["策略结果"][key] = {
        "策略名称": desc,
        "入选数量": len(records),
        "Top10": clean_for_json(records),
    }

with open(OUTPUT_JSON, 'w', encoding='utf-8') as f:
    json.dump(json_output, f, ensure_ascii=False, indent=2)
print(f"  JSON已保存: {OUTPUT_JSON}")


# ======================== 汇总 ========================
print("\n" + "=" * 100)
print("【汇总】5策略选股结果摘要")
print("=" * 100)
for label, key, desc, _ in strategies:
    df_res = strategy_results[key]
    if len(df_res) > 0:
        top1 = df_res.iloc[0]
        print(f"  {label} {desc}: {len(df_res)}只 | 首选: {top1['名称']}({top1['ts_code']}) "
              f"得分{top1['得分']} 价格{top1['最新价']:.2f} 涨{top1['涨跌幅']:+.2f}% 成交{top1['成交额亿']:.1f}亿")
    else:
        print(f"  {label} {desc}: 0只 (无符合条件的股票)")

print("\n" + "=" * 100)
print("  脚本执行完毕!")
print("=" * 100)
