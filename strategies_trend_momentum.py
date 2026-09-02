#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
A股趋势 / 动量 选股策略 (策略6-10)
- 策略6: volume_breakout  放量突破 (趋势)
- 策略7: shrink_pullback  缩量回踩 (趋势)
- 策略8: capital_heat     资金热度 (动量)
- 策略9: balanced_alpha   均衡多因子 (综合)
- 策略10:momentum_quality 趋势质量 (综合)

数据来源: Tushare pro
交易日: 20260820
"""

import os
import json
import time
import numpy as np
import pandas as pd
import tushare as ts

# ---------------- 配置 ----------------
TOKEN = '9f640d421f866dde9a1888ab4193f77c43659d47446d717ffa343023'
TRADE_DATE = '20260901'
REPORT_DIR = '/workspace/reports'
REPORT_PATH = os.path.join(REPORT_DIR, 'trend_momentum_result.json')

# 单位换算说明 (Tushare):
#   pro.daily.amount        -> 千元 ; 1亿 = 100000 千元
#   pro.daily_basic.total_mv-> 万元 ; 1亿 = 10000 万元
#   pct_chg / turnover_rate -> % (已是百分比)
AMT_2YI = 200000      # 千元
AMT_3YI = 300000
AMT_5YI = 500000
AMT_10YI = 1000000
AMT_20YI = 2000000
AMT_50YI = 5000000
MV_50YI = 500000      # 万元
MV_100YI = 1000000
MV_500YI = 5000000

ts.set_token(TOKEN)
pro = ts.pro_api()


# ---------------- 工具函数 ----------------
def num(v):
    """安全转 float, 失败/缺失返回 nan"""
    if v is None:
        return np.nan
    try:
        f = float(v)
        if np.isnan(f):
            return np.nan
        return f
    except (TypeError, ValueError):
        return np.nan


def r2(v, d=2):
    """四舍五入, nan -> None"""
    f = num(v)
    if np.isnan(f):
        return None
    return round(f, d)


def fetch_with_retry(func, retries=4, **kwargs):
    """带重试的 Tushare 调用"""
    for i in range(retries):
        try:
            return func(**kwargs)
        except Exception as e:  # noqa
            if i == retries - 1:
                print(f"    [API ERROR] {kwargs} -> {e}")
                return pd.DataFrame()
            time.sleep(0.5 * (i + 1))
    return pd.DataFrame()


def calc_rsi(closes, period=14):
    """Wilder RSI(14)"""
    closes = np.asarray(closes, dtype=float)
    deltas = np.diff(closes)
    if len(deltas) < period:
        return np.nan
    gains = np.where(deltas > 0, deltas, 0.0)
    losses = np.where(deltas < 0, -deltas, 0.0)
    avg_gain = gains[:period].mean()
    avg_loss = losses[:period].mean()
    for i in range(period, len(gains)):
        avg_gain = (avg_gain * (period - 1) + gains[i]) / period
        avg_loss = (avg_loss * (period - 1) + losses[i]) / period
    if avg_loss == 0:
        return 100.0
    rs = avg_gain / avg_loss
    return 100.0 - 100.0 / (1.0 + rs)


def calc_indicators(kline):
    """根据30日K线计算技术指标, 返回 dict 或 None"""
    kline = kline.sort_values('trade_date').reset_index(drop=True)
    n = len(kline)
    if n < 21:  # 至少21根才能算20日类指标
        return None
    closes = kline['close'].values.astype(float)
    highs = kline['high'].values.astype(float)
    lows = kline['low'].values.astype(float)
    vols = kline['vol'].values.astype(float)

    close = closes[-1]
    ma5 = closes[-5:].mean() if n >= 5 else np.nan
    ma10 = closes[-10:].mean() if n >= 10 else np.nan
    ma20 = closes[-20:].mean() if n >= 20 else np.nan
    bullish_ma = bool((num(ma5) > num(ma10)) and (num(ma10) > num(ma20))) \
        if not (np.isnan(ma5) or np.isnan(ma10) or np.isnan(ma20)) else False

    # 20日涨幅
    ret_20d = (closes[-1] / closes[-21] - 1) * 100
    # 20日波动率 (年化)
    log_ret = np.diff(np.log(closes[-21:]))
    vol_20d = np.std(log_ret) * np.sqrt(252) * 100
    # 突破20日高 % (与过去20日最高价比较, 不含今日)
    prior_high_20 = highs[-21:-1].max()
    breakout_20d = (close / prior_high_20 - 1) * 100
    # 量比 = 当日vol / 过去20日均vol(不含今日)
    vol_ratio = vols[-1] / vols[-21:-1].mean()
    # 距MA20%
    dist_ma20 = (close / ma20 - 1) * 100
    # RSI(14)
    rsi = calc_rsi(closes, 14)
    # 平台持续天数: 近20日内 (当日振幅/前收) < 3% 的天数
    cdays = 0
    for i in range(max(1, n - 20), n):
        pc = closes[i - 1]
        if pc > 0 and (highs[i] - lows[i]) / pc < 0.03:
            cdays += 1

    pct_chg = num(kline['pct_chg'].iloc[-1])

    return {
        'close': close,
        'ma5': ma5, 'ma10': ma10, 'ma20': ma20,
        'bullish_ma': bullish_ma,
        'above_ma20': bool(close > ma20),
        'ret_20d': ret_20d,
        'vol_20d': vol_20d,
        'breakout_20d': breakout_20d,
        'vol_ratio': vol_ratio,
        'dist_ma20': dist_ma20,
        'rsi': rsi,
        'consolidation_days': cdays,
        'pct_chg': pct_chg,
    }


# ---------------- 策略函数: 返回 (通过硬筛, 得分) ----------------
def s6_volume_breakout(r):
    """策略6 放量突破 (趋势)"""
    if not (r['bullish_ma'] and r['above_ma20']):
        return False, 0
    if not (r['vol_ratio'] >= 1.5):
        return False, 0
    if not (r['breakout_20d'] >= 0):
        return False, 0
    if not (r['pct_chg'] <= 7):
        return False, 0
    if not (r['amount'] >= AMT_3YI):
        return False, 0
    score = 0
    vr = r['vol_ratio']
    if 2 <= vr < 4:
        score += 25
    elif 1.5 <= vr < 2:
        score += 20
    elif 4 <= vr < 6:
        score += 15
    bo = r['breakout_20d']
    if 0 <= bo < 5:
        score += 15
    elif bo >= 5:
        score += 10
    cd = r['consolidation_days']
    if cd >= 15:
        score += 20
    elif cd >= 10:
        score += 25
    if 3 <= r['pct_chg'] <= 7:
        score += 10
    if r['amount'] >= AMT_10YI:
        score += 10
    pe = num(r['pe'])
    if not np.isnan(pe) and 10 <= pe <= 80:
        score += 5
    return True, score


def s7_shrink_pullback(r):
    """策略7 缩量回踩 (趋势)"""
    if not (r['bullish_ma'] and r['above_ma20']):
        return False, 0
    if not (r['vol_ratio'] <= 1.3):
        return False, 0
    if not (0 <= r['dist_ma20'] <= 5):
        return False, 0
    if not (r['vol_20d'] <= 40):
        return False, 0
    if not (r['amount'] >= AMT_2YI):
        return False, 0
    score = 0
    vr = r['vol_ratio']
    if 0.7 <= vr < 0.9:
        score += 30
    elif 0.9 <= vr < 1.1:
        score += 20
    elif 1.1 <= vr <= 1.3:
        score += 10
    dm = r['dist_ma20']
    if 0 <= dm < 2:
        score += 25
    elif 2 <= dm < 4:
        score += 15
    elif 4 <= dm <= 5:
        score += 8
    pc = r['pct_chg']
    if -2 <= pc < 0:
        score += 15
    elif 0 <= pc < 1:
        score += 10
    pe = num(r['pe'])
    if not np.isnan(pe) and 10 <= pe <= 50:
        score += 8
    pb = num(r.get('pb'))
    if not np.isnan(pb) and pb < 5:
        score += 4
    if r['amount'] >= AMT_5YI:
        score += 5
    return True, score


def s8_capital_heat(r):
    """策略8 资金热度 (动量)"""
    if not (r['amount'] >= AMT_10YI):
        return False, 0
    if not (r['vol_ratio'] >= 1.2):
        return False, 0
    if not (0 <= r['pct_chg'] <= 7):
        return False, 0
    tr = num(r.get('turnover_rate'))
    if not (tr >= 3):
        return False, 0
    if not r['bullish_ma']:
        return False, 0
    score = 0
    amt = r['amount']
    if amt >= AMT_50YI:
        score += 25
    elif AMT_20YI <= amt < AMT_50YI:
        score += 20
    elif AMT_10YI <= amt < AMT_20YI:
        score += 15
    vr = r['vol_ratio']
    if 1.5 <= vr < 3:
        score += 20
    elif 3 <= vr < 5:
        score += 15
    if 3 <= r['pct_chg'] <= 7:
        score += 15
    if 5 <= tr < 10:
        score += 15
    elif 3 <= tr < 5:
        score += 10
    if 5 <= r['ret_20d'] <= 20:
        score += 15
    if r['bullish_ma']:
        score += 10
    return True, score


def s9_balanced_alpha(r):
    """策略9 均衡多因子 (综合)"""
    if not (r['amount'] >= AMT_3YI):
        return False, 0
    tmv = num(r.get('total_mv'))
    if not (tmv >= MV_50YI):
        return False, 0
    pe = num(r['pe'])
    if not (0 < pe <= 80):
        return False, 0
    if not (-3 <= r['pct_chg'] <= 5):
        return False, 0
    pb = num(r.get('pb'))
    tr = num(r.get('turnover_rate'))
    score = 0
    # 估值因子 (20)
    if 10 <= pe < 30:
        score += 15
    elif 30 <= pe <= 50:
        score += 10
    if not np.isnan(pb) and pb < 3:
        score += 5
    # 动量因子 (25)
    if 5 <= r['ret_20d'] <= 15:
        score += 15
    elif 0 <= r['ret_20d'] < 5:
        score += 10
    if r['bullish_ma']:
        score += 10
    # 资金因子 (20)
    if r['amount'] >= AMT_10YI:
        score += 10
    if 1 <= r['vol_ratio'] <= 2:
        score += 5
    if not np.isnan(tr) and 3 <= tr <= 8:
        score += 5
    # 稳定性因子 (20)
    if r['vol_20d'] < 30:
        score += 10
    if 0 <= r['dist_ma20'] <= 10:
        score += 10
    # 反转因子 (15)
    if 0 <= r['pct_chg'] <= 3:
        score += 8
    if not np.isnan(r['rsi']) and 40 <= r['rsi'] <= 60:
        score += 7
    return True, score


def s10_momentum_quality(r):
    """策略10 趋势质量 (综合)"""
    if not (r['bullish_ma'] and r['above_ma20']):
        return False, 0
    if not (0 <= r['dist_ma20'] <= 15):
        return False, 0
    if not (0 <= r['ret_20d'] <= 20):
        return False, 0
    pe = num(r['pe'])
    if not (0 < pe <= 60):
        return False, 0
    pb = num(r.get('pb'))
    if not (pb <= 6):
        return False, 0
    if not (r['amount'] >= AMT_2YI):
        return False, 0
    tmv = num(r.get('total_mv'))
    if not (tmv >= MV_50YI):
        return False, 0
    if not (-3 <= r['pct_chg'] <= 7):
        return False, 0
    score = 0
    score += 15  # 均线多头
    score += 10  # 站上MA20
    dm = r['dist_ma20']
    if 0 <= dm <= 8:
        score += 10
    elif 8 < dm <= 15:
        score += 5
    if 5 <= r['ret_20d'] <= 15:
        score += 15
    elif 0 <= r['ret_20d'] < 5:
        score += 8
    if 10 <= pe <= 35:
        score += 10
    elif pe < 10:
        score += 8
    if not np.isnan(pb) and pb < 3:
        score += 5
    if r['amount'] >= AMT_10YI:
        score += 10
    elif AMT_5YI <= r['amount'] < AMT_10YI:
        score += 8
    if tmv >= MV_500YI:
        score += 8
    elif MV_100YI <= tmv < MV_500YI:
        score += 5
    pc = r['pct_chg']
    if 0 <= pc <= 4:
        score += 10
    elif -2 <= pc < 0:
        score += 5
    if 0.8 <= r['vol_ratio'] <= 1.8:
        score += 5
    return True, score


# ---------------- 主流程 ----------------
def main():
    os.makedirs(REPORT_DIR, exist_ok=True)

    print("=" * 90)
    print(f"A股趋势/动量选股策略  交易日: {TRADE_DATE}")
    print("=" * 90)

    # 1. 获取全市场行情 + 基础数据
    print("\n[1/4] 获取全市场当日行情 / 估值 / 股票列表 ...")
    daily = fetch_with_retry(pro.daily, trade_date=TRADE_DATE)
    daily_basic = fetch_with_retry(pro.daily_basic, trade_date=TRADE_DATE)
    stock_basic = fetch_with_retry(pro.stock_basic)
    print(f"  daily: {len(daily)} 行 | daily_basic: {len(daily_basic)} 行 | stock_basic: {len(stock_basic)} 行")
    if len(daily) == 0:
        print("  [FATAL] 当日行情为空, 退出")
        return

    # 合并 (stock_basic 默认只返回在市股票, 已排除退市)
    sb = stock_basic[['ts_code', 'name', 'industry', 'list_date']].copy()
    df = daily.merge(daily_basic, on='ts_code', how='left', suffixes=('', '_db'))
    df = df.merge(sb, on='ts_code', how='left')

    # 2. 排除过滤 (name 为空 = 已退市不在 stock_basic, 一并剔除)
    print("\n[2/4] 执行排除过滤 (ST/*ST | 退市 | 次新<1年 | 成交额<2亿) ...")
    before = len(df)
    df = df[df['name'].notna()]
    df = df[~df['name'].astype(str).str.contains('ST', na=False)]
    cutoff = '20250818'  # 上市满1年
    df = df[df['list_date'].astype(str) < cutoff]
    df = df[df['amount'].astype(float) >= AMT_2YI]
    print(f"  过滤前 {before} -> 过滤后 {len(df)} 只")
    print(f"  抽样校验单位: 成交额最大={df['amount'].max()/100000:.1f}亿  总市值最大={df['total_mv'].max()/10000:.1f}亿")

    # 取成交额 Top300
    df = df.sort_values('amount', ascending=False).head(300).reset_index(drop=True)
    print(f"  选取成交额 Top300 活跃股作为计算样本")

    # 3. 逐个获取30日K线 + 技术指标
    print("\n[3/4] 逐个获取 Top300 的 30日K线并计算技术指标 ...")
    records = []
    fail = 0
    for i, row in df.iterrows():
        code = row['ts_code']
        kline = fetch_with_retry(pro.daily, ts_code=code, end_date=TRADE_DATE, limit=30)
        if kline is None or len(kline) == 0:
            fail += 1
            continue
        ind = calc_indicators(kline)
        if ind is None:
            fail += 1
            continue
        # PE: 优先 pe_ttm, 退而求其次 pe
        pe_val = row.get('pe_ttm')
        if num(pe_val) and not np.isnan(num(pe_val)):
            pe_val = pe_val
        else:
            pe_val = row.get('pe')
        rec = {
            'ts_code': code,
            'name': row['name'],
            'industry': row.get('industry', ''),
            'close': num(row['close']),
            'pct_chg': num(row['pct_chg']),
            'amount': num(row['amount']),           # 千元
            'pe': num(pe_val),
            'pb': num(row.get('pb')),
            'total_mv': num(row.get('total_mv')),    # 万元
            'turnover_rate': num(row.get('turnover_rate')),
        }
        rec.update(ind)
        records.append(rec)
        time.sleep(0.08)
        if (i + 1) % 50 == 0:
            print(f"  进度 {i+1}/300  已成功 {len(records)}  失败 {fail}")
    print(f"  完成: 成功计算指标 {len(records)} 只, 失败 {fail} 只")

    if len(records) == 0:
        print("  [FATAL] 无可用记录, 退出")
        return

    # 4. 跑5个策略
    print("\n[4/4] 跑5个策略并输出 Top10 ...")
    strategies = [
        ('volume_breakout',  '策略6 放量突破 (趋势)', s6_volume_breakout),
        ('shrink_pullback',  '策略7 缩量回踩 (趋势)', s7_shrink_pullback),
        ('capital_heat',     '策略8 资金热度 (动量)', s8_capital_heat),
        ('balanced_alpha',   '策略9 均衡多因子 (综合)', s9_balanced_alpha),
        ('momentum_quality', '策略10 趋势质量 (综合)', s10_momentum_quality),
    ]

    json_result = {
        'trade_date': TRADE_DATE,
        'universe_top300_computed': len(records),
        'strategies': {},
    }

    for key, title, func in strategies:
        scored = []
        for r in records:
            passed, sc = func(r)
            if passed:
                rr = dict(r)
                rr['_score'] = sc
                scored.append(rr)
        # 排序: 得分降序, 成交额降序
        scored.sort(key=lambda x: (-x['_score'], -x['amount']))
        top10 = scored[:10]

        print("\n" + "-" * 100)
        print(f"  {title}   通过硬筛 {len(scored)} 只, Top10 如下:")
        print("-" * 100)
        if len(top10) == 0:
            print("  (无标的满足硬筛条件)")
        else:
            print_table(top10, key)

        json_result['strategies'][key] = {
            'name_cn': title,
            'passed_count': len(scored),
            'top10': [to_json_item(r) for r in top10],
        }

    # 写 JSON
    with open(REPORT_PATH, 'w', encoding='utf-8') as f:
        json.dump(json_result, f, ensure_ascii=False, indent=2)
    print("\n" + "=" * 100)
    print(f"  JSON 结果已保存: {REPORT_PATH}")
    print("=" * 100)


def print_table(top10, strategy_key):
    """打印中文表头表格"""
    rows = []
    for r in top10:
        rows.append({
            '代码': r['ts_code'],
            '名称': r['name'],
            '行业': r.get('industry', '') or '',
            '收盘': r2(r['close'], 2),
            '涨跌%': r2(r['pct_chg'], 2),
            '成交额亿': r2(num(r['amount']) / 100000.0, 2),
            'PE': (r2(r['pe'], 1) if not np.isnan(num(r['pe'])) else None),
            'PB': (r2(r.get('pb'), 2) if not np.isnan(num(r.get('pb'))) else None),
            '总市值亿': (r2(num(r.get('total_mv')) / 10000.0, 1) if not np.isnan(num(r.get('total_mv'))) else None),
            '关键技术指标': key_indicators_str(r, strategy_key),
            '得分': r['_score'],
        })
    disp = pd.DataFrame(rows, columns=[
        '代码', '名称', '行业', '收盘', '涨跌%', '成交额亿',
        'PE', 'PB', '总市值亿', '关键技术指标', '得分'])
    # 用 to_string 保证对齐
    with pd.option_context('display.unicode.east_asian_width', True,
                           'display.max_columns', None,
                           'display.width', 200):
        print(disp.to_string(index=False))


def key_indicators_str(r, strategy_key):
    vr = r2(r['vol_ratio'], 2)
    bo = r2(r['breakout_20d'], 2)
    cd = r['consolidation_days']
    dm = r2(r['dist_ma20'], 2)
    vol20 = r2(r['vol_20d'], 1)
    r20 = r2(r['ret_20d'], 2)
    rsi = r2(r['rsi'], 0)
    pe = r2(r['pe'], 1)
    pb = r2(r.get('pb'), 2)
    tr = r2(r.get('turnover_rate'), 2)
    bull = '是' if r['bullish_ma'] else '否'
    if strategy_key == 'volume_breakout':
        return f"量比:{vr} 突破20日高:{bo}% 平台:{cd}天 均线多头:{bull}"
    if strategy_key == 'shrink_pullback':
        return f"量比:{vr} 距MA20:{dm}% 波动率:{vol20}% 均线多头:{bull}"
    if strategy_key == 'capital_heat':
        return f"量比:{vr} 换手:{tr}% 20日涨:{r20}% 均线多头:{bull}"
    if strategy_key == 'balanced_alpha':
        return f"PE:{pe} 20日涨:{r20}% 波动率:{vol20}% RSI:{rsi}"
    if strategy_key == 'momentum_quality':
        return f"距MA20:{dm}% 20日涨:{r20}% PE:{pe} PB:{pb} 量比:{vr}"
    return ''


def to_json_item(r):
    """单条记录转 JSON 友好结构"""
    return {
        'ts_code': r['ts_code'],
        'name': r['name'],
        'industry': r.get('industry', '') or '',
        'close': r2(r['close'], 2),
        'pct_chg': r2(r['pct_chg'], 2),
        'amount_yi': r2(num(r['amount']) / 100000.0, 2),
        'pe': (r2(r['pe'], 1) if not np.isnan(num(r['pe'])) else None),
        'pb': (r2(r.get('pb'), 2) if not np.isnan(num(r.get('pb'))) else None),
        'total_mv_yi': (r2(num(r.get('total_mv')) / 10000.0, 1) if not np.isnan(num(r.get('total_mv'))) else None),
        'turnover_rate': (r2(r.get('turnover_rate'), 2) if not np.isnan(num(r.get('turnover_rate'))) else None),
        'vol_ratio': r2(r['vol_ratio'], 2),
        'ret_20d': r2(r['ret_20d'], 2),
        'dist_ma20': r2(r['dist_ma20'], 2),
        'vol_20d': r2(r['vol_20d'], 1),
        'rsi': (r2(r['rsi'], 1) if not np.isnan(num(r['rsi'])) else None),
        'breakout_20d': r2(r['breakout_20d'], 2),
        'consolidation_days': int(r['consolidation_days']),
        'bullish_ma': r['bullish_ma'],
        'score': r['_score'],
    }


if __name__ == '__main__':
    main()
