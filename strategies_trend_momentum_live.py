#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
A股趋势 / 动量 选股策略 (策略6-10) — 8月19日盘中实时版
- 策略6: volume_breakout  放量突破 (趋势)
- 策略7: shrink_pullback  缩量回踩 (趋势)
- 策略8: capital_heat     资金热度 (动量)
- 策略9: balanced_alpha   均衡多因子 (综合)
- 策略10:momentum_quality 趋势质量 (综合)

数据源:
  1. 当日盘中行情: /workspace/data/today_akshare.csv (akshare 8月19日11:34实时行情)
     - 成交额单位元 -> 亿元: /1e8 ; 成交量单位股 -> 手: /100 ; 涨跌幅单位%
     - 代码格式 bj920000/sh600000/sz000001 -> tushare 920000.BJ/600000.SH/000001.SZ
  2. 历史30日K线: Tushare pro.daily(ts_code, end_date='20260818', limit=30)
     - 将今日盘中数据作为最后一行附加到历史K线上算技术指标
  3. 基本面: Tushare pro.daily_basic(trade_date='20260818') -> PE/PB/市值/换手率
  4. 股票列表/行业: Tushare pro.stock_basic()
"""

import os
import json
import time
import numpy as np
import pandas as pd
import tushare as ts

# ---------------- 配置 ----------------
TOKEN = '9f640d421f866dde9a1888ab4193f77c43659d47446d717ffa343023'
TODAY_CSV = '/workspace/data/today_akshare.csv'
TODAY_DATE = '20260819'   # 盘中日期
HIST_END = '20260818'      # 历史K线截止日(上一交易日)
REPORT_DIR = '/workspace/reports'
REPORT_PATH = os.path.join(REPORT_DIR, 'trend_momentum_live.json')

ts.set_token(TOKEN)
pro = ts.pro_api()

# 阈值 (亿元 / %)
AMT_2YI, AMT_3YI, AMT_5YI = 2.0, 3.0, 5.0
AMT_10YI, AMT_20YI, AMT_50YI = 10.0, 20.0, 50.0
MV_50YI, MV_100YI, MV_500YI = 50.0, 100.0, 500.0
LIST_DATE_CUTOFF = '20250819'   # 上市满1年: list_date < 此值


# ---------------- 工具函数 ----------------
def num(v):
    """安全转 float, 失败/缺失返回 nan"""
    if v is None:
        return np.nan
    try:
        f = float(v)
        return f if not np.isnan(f) else np.nan
    except (TypeError, ValueError):
        return np.nan


def r2(v, d=2):
    """四舍五入, nan -> None"""
    f = num(v)
    return None if np.isnan(f) else round(f, d)


def ak_to_ts(ak_code):
    """akshare代码(bj920000) -> tushare代码(920000.BJ)"""
    p = ak_code[:2].upper()
    d = ak_code[2:]
    return f"{d}.{p}"


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
    """根据 30日历史K线 + 今日盘中(最后一行) 计算技术指标, 返回 dict 或 None"""
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

    # 20日涨幅 (close[-1]/close[-21]-1)*100
    ret_20d = (closes[-1] / closes[-21] - 1) * 100
    # 20日波动率 (年化) np.std(np.diff(np.log(closes[-21:])))*sqrt(252)*100
    log_ret = np.diff(np.log(closes[-21:]))
    vol_20d = np.std(log_ret) * np.sqrt(252) * 100
    # 突破20日高% : close / 过去20日最高价(不含今日) - 1
    prior_high_20 = highs[-21:-1].max()
    breakout_20d = (close / prior_high_20 - 1) * 100
    # 量比 = 今日盘中vol / 过去20日均vol(不含今日)
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
    if not (r['amount_yi'] >= AMT_3YI):
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
    if r['amount_yi'] >= AMT_10YI:
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
    if not (r['amount_yi'] >= AMT_2YI):
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
    if r['amount_yi'] >= AMT_5YI:
        score += 5
    return True, score


def s8_capital_heat(r):
    """策略8 资金热度 (动量)"""
    if not (r['amount_yi'] >= AMT_10YI):
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
    amt = r['amount_yi']
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
    if not (r['amount_yi'] >= AMT_3YI):
        return False, 0
    tmv = num(r.get('total_mv_yi'))
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
    if r['amount_yi'] >= AMT_10YI:
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
    if not (r['amount_yi'] >= AMT_2YI):
        return False, 0
    tmv = num(r.get('total_mv_yi'))
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
    if r['amount_yi'] >= AMT_10YI:
        score += 10
    elif AMT_5YI <= r['amount_yi'] < AMT_10YI:
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


# ---------------- 表格/JSON 输出 ----------------
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


def print_table(top10, strategy_key):
    """打印中文表头表格"""
    rows = []
    for r in top10:
        rows.append({
            '代码': r['ts_code'],
            '名称': r['name'],
            '行业': r.get('industry', '') or '',
            '最新价': r2(r['close'], 2),
            '涨跌%': r2(r['pct_chg'], 2),
            '成交额亿': r2(num(r['amount_yi']), 2),
            'PE': (r2(r['pe'], 1) if not np.isnan(num(r['pe'])) else None),
            'PB': (r2(r.get('pb'), 2) if not np.isnan(num(r.get('pb'))) else None),
            '总市值亿': (r2(num(r.get('total_mv_yi')), 1) if not np.isnan(num(r.get('total_mv_yi'))) else None),
            '关键技术指标': key_indicators_str(r, strategy_key),
            '得分': r['_score'],
        })
    disp = pd.DataFrame(rows, columns=[
        '代码', '名称', '行业', '最新价', '涨跌%', '成交额亿',
        'PE', 'PB', '总市值亿', '关键技术指标', '得分'])
    with pd.option_context('display.unicode.east_asian_width', True,
                           'display.max_columns', None,
                           'display.width', 220):
        print(disp.to_string(index=False))


def to_json_item(r):
    """单条记录转 JSON 友好结构"""
    return {
        'ts_code': r['ts_code'],
        'name': r['name'],
        'industry': r.get('industry', '') or '',
        'close': r2(r['close'], 2),
        'pct_chg': r2(r['pct_chg'], 2),
        'amount_yi': r2(num(r['amount_yi']), 2),
        'pe': (r2(r['pe'], 1) if not np.isnan(num(r['pe'])) else None),
        'pb': (r2(r.get('pb'), 2) if not np.isnan(num(r.get('pb'))) else None),
        'total_mv_yi': (r2(num(r.get('total_mv_yi')), 1) if not np.isnan(num(r.get('total_mv_yi'))) else None),
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


# ---------------- 主流程 ----------------
def main():
    os.makedirs(REPORT_DIR, exist_ok=True)

    print("=" * 110)
    print(f"A股趋势/动量选股策略  8月19日盘中实时版 (策略6-10)  盘中数据: {TODAY_DATE}")
    print("=" * 110)

    # [1/5] 读取今日盘中行情
    print("\n[1/5] 读取今日盘中行情 (today_akshare.csv) ...")
    ak = pd.read_csv(TODAY_CSV, encoding='utf-8-sig')
    ak['成交额亿'] = ak['成交额'].astype(float) / 1e8     # 元 -> 亿元
    ak['成交量手'] = ak['成交量'].astype(float) / 100.0     # 股 -> 手 (与Tushare vol一致)
    ak['ts_code'] = ak['代码'].apply(ak_to_ts)
    print(f"  读取 {len(ak)} 只股票")

    # [2/5] Tushare 基本面 + 股票列表
    print("\n[2/5] 获取Tushare基本面/股票列表 ...")
    daily_basic = fetch_with_retry(pro.daily_basic, trade_date=HIST_END)
    stock_basic = fetch_with_retry(pro.stock_basic)
    print(f"  daily_basic: {len(daily_basic)} 行 | stock_basic: {len(stock_basic)} 行")
    if len(daily_basic) == 0 or len(stock_basic) == 0:
        print("  [FATAL] 基本面数据为空, 退出")
        return

    db = daily_basic[['ts_code', 'pe', 'pe_ttm', 'pb', 'total_mv', 'turnover_rate']].copy()
    sb = stock_basic[['ts_code', 'name', 'industry', 'list_date']].copy()

    # 合并基本面 (left join: akshare为主, 补充tushare估值/行业/上市日)
    df = ak.merge(db, on='ts_code', how='left').merge(sb, on='ts_code', how='left')
    # 名称优先用akshare, 缺失再用stock_basic
    df['name'] = df['名称'].where(df['名称'].notna(), df['name'])
    # 总市值(万元) -> 亿元
    df['total_mv_yi'] = num_vec_safe(df, 'total_mv') / 10000.0

    # [3/5] 排除过滤
    print("\n[3/5] 排除过滤 (ST/*ST | 退市 | 次新<1年 | 成交额<2亿) ...")
    before = len(df)
    df = df[df['name'].notna()]                                  # 退市: 不在stock_basic且akshare无名称
    df = df[~df['name'].astype(str).str.contains('ST', na=False)]  # ST/*ST
    df = df[df['list_date'].astype(str).str.replace('-', '', regex=False) < LIST_DATE_CUTOFF]  # 次新
    df = df[df['成交额亿'].astype(float) >= AMT_2YI]             # 成交额>=2亿
    print(f"  过滤前 {before} -> 过滤后 {len(df)} 只")
    if len(df) > 0:
        print(f"  抽样校验: 成交额最大={df['成交额亿'].max():.1f}亿  成交量(手)最大={df['成交量手'].max():.0f}")

    # 取成交额 Top300
    df = df.sort_values('成交额亿', ascending=False).head(300).reset_index(drop=True)
    print(f"  选取成交额 Top300 活跃股作为计算样本")

    # [4/5] 逐个获取30日K线 + 追加今日盘中 + 计算指标
    print("\n[4/5] 逐个获取 Top300 的 30日K线, 追加今日盘中数据, 计算技术指标 ...")
    records = []
    fail = 0
    for i, row in df.iterrows():
        code = row['ts_code']
        kline = fetch_with_retry(pro.daily, ts_code=code, end_date=HIST_END, limit=30)
        if kline is None or len(kline) == 0:
            fail += 1
            continue
        # 追加今日盘中数据作为最后一行
        today_row = pd.DataFrame([{
            'ts_code': code,
            'trade_date': TODAY_DATE,
            'open': num(row['今开']),
            'high': num(row['最高']),
            'low': num(row['最低']),
            'close': num(row['最新价']),
            'pre_close': num(row['昨收']),
            'change': num(row['涨跌额']),
            'pct_chg': num(row['涨跌幅']),
            'vol': num(row['成交量手']),        # 手, 与Tushare vol单位一致
            'amount': num(row['成交额']) / 1000.0,  # 元->千元, 与Tushare amount一致
        }])
        kline = pd.concat([kline, today_row], ignore_index=True)

        ind = calc_indicators(kline)
        if ind is None:
            fail += 1
            continue

        # PE: 优先 pe_ttm, 退而求其次 pe
        pe_ttm = num(row.get('pe_ttm'))
        pe_val = pe_ttm if not np.isnan(pe_ttm) else num(row.get('pe'))

        rec = {
            'ts_code': code,
            'name': row['name'],
            'industry': row.get('industry', '') or '',
            'close': num(row['最新价']),
            'pct_chg': num(row['涨跌幅']),
            'amount_yi': num(row['成交额亿']),            # 亿元
            'pe': pe_val,
            'pb': num(row.get('pb')),
            'total_mv_yi': num(row.get('total_mv_yi')),    # 亿元
            'turnover_rate': num(row.get('turnover_rate')),
        }
        rec.update(ind)
        records.append(rec)
        time.sleep(0.12)
        if (i + 1) % 50 == 0:
            print(f"  进度 {i+1}/300  已成功 {len(records)}  失败 {fail}")
    print(f"  完成: 成功计算指标 {len(records)} 只, 失败 {fail} 只")

    if len(records) == 0:
        print("  [FATAL] 无可用记录, 退出")
        return

    # [5/5] 跑5个策略并输出 Top10
    print("\n[5/5] 跑5个策略并输出 Top10 ...")
    strategies = [
        ('volume_breakout',  '策略6 放量突破 (趋势)', s6_volume_breakout),
        ('shrink_pullback',  '策略7 缩量回踩 (趋势)', s7_shrink_pullback),
        ('capital_heat',     '策略8 资金热度 (动量)', s8_capital_heat),
        ('balanced_alpha',   '策略9 均衡多因子 (综合)', s9_balanced_alpha),
        ('momentum_quality', '策略10 趋势质量 (综合)', s10_momentum_quality),
    ]

    json_result = {
        'trade_date': TODAY_DATE,
        'intraday_snapshot': 'akshare 8月19日11:34实时行情',
        'hist_kline_end': HIST_END,
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
        scored.sort(key=lambda x: (-x['_score'], -x['amount_yi']))
        top10 = scored[:10]

        print("\n" + "-" * 120)
        print(f"  {title}   通过硬筛 {len(scored)} 只, Top10 如下:")
        print("-" * 120)
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
    print("\n" + "=" * 120)
    print(f"  JSON 结果已保存: {REPORT_PATH}")
    print("=" * 120)


def num_vec_safe(df, col):
    """对DataFrame列安全转数值"""
    if col not in df.columns:
        return pd.Series([np.nan] * len(df), index=df.index)
    return pd.to_numeric(df[col], errors='coerce')


if __name__ == '__main__':
    main()
