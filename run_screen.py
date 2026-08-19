# -*- coding: utf-8 -*-
"""13:30 盘中选股策略报告 - 10策略筛选管线 (tushare 直连版)"""
import tushare as ts
import pandas as pd
import numpy as np
import sys, json, time, os
from datetime import datetime, timedelta

TOKEN = '9f640d421f866dde9a1888ab4193f77c43659d47446d717ffa343023'
ts.set_token(TOKEN)
pro = ts.pro_api()

TODAY = '20260819'
REPORT_DATE = '20260818'   # 最新已发布交易日
HIST_DAYS = 72             # 拉取的历史交易日数
OUT_DIR = '/workspace/reports'
os.makedirs(OUT_DIR, exist_ok=True)

def log(*a):
    print(*a, file=sys.stderr, flush=True)

# ---------- 1. 取交易日历 ----------
log('[1] 取交易日历...')
cal = pro.trade_cal(exchange='SSE', start_date='20251101', end_date=REPORT_DATE)
cal = cal[cal['is_open'] == 1].sort_values('cal_date')
trade_dates = cal['cal_date'].tolist()
hist_dates = trade_dates[-HIST_DAYS:]
log(f'  历史交易日 {len(hist_dates)} 个, 范围 {hist_dates[0]}~{hist_dates[-1]}')

# ---------- 2. 拉取历史日线(全市场, 按交易日) ----------
log('[2] 拉取历史日线...')
CACHE = '/workspace/_daily_cache.pkl'
if os.path.exists(CACHE):
    with open(CACHE,'rb') as f:
        daily, db, sb, hist_dates = __import__('pickle').load(f)
    log(f'  [cache] 日线 {len(daily)} 行, 交易日 {len(hist_dates)}')
else:
    frames = []
    for i, d in enumerate(hist_dates):
        try:
            df = pro.daily(trade_date=d)
            if df is not None and len(df):
                frames.append(df)
        except Exception as e:
            log(f'  {d} err {e}')
        if (i+1) % 15 == 0:
            log(f'  进度 {i+1}/{len(hist_dates)}')
        time.sleep(0.12)
    daily = pd.concat(frames, ignore_index=True)
    daily = daily[['ts_code','trade_date','open','high','low','close','pre_close','change','pct_chg','vol','amount']]
    daily['trade_date'] = daily['trade_date'].astype(str)
    daily = daily.sort_values(['ts_code','trade_date']).reset_index(drop=True)
    log(f'  日线合计 {len(daily)} 行, 覆盖 {daily.trade_date.nunique()} 个交易日')
    # ---------- 3. stock_basic ----------
    log('[3] 取 stock_basic...')
    sb = pro.stock_basic(exchange='', list_status='L',
                         fields='ts_code,symbol,name,industry,market,list_date,delist_date')
    sb['list_date'] = sb['list_date'].astype(str)
    # ---------- 4. daily_basic (最新交易日) ----------
    log('[4] 取 daily_basic...')
    db = pro.daily_basic(trade_date=REPORT_DATE)
    db = db[['ts_code','trade_date','turnover_rate','volume_ratio','pe','pe_ttm','pb','ps','ps_ttm',
             'dv_ratio','dv_ttm','total_share','float_share','free_share','total_mv','circ_mv']]
    db['trade_date'] = db['trade_date'].astype(str)
    log(f'  daily_basic {len(db)} 行')
    with open(CACHE,'wb') as f:
        __import__('pickle').dump((daily, db, sb, hist_dates), f)
    log('  [cache] 已保存')

# 次新: 上市<1年 排除
cutoff_list = (datetime.strptime(REPORT_DATE,'%Y%m%d') - timedelta(days=365)).strftime('%Y%m%d')
sb['is_new'] = sb['list_date'] > cutoff_list
log(f'  股票总数 {len(sb)}, 次新(<1年) {sb.is_new.sum()}')

# ---------- 5. 合并 + 技术指标 ----------
log('[5] 计算技术指标...')
m = daily.merge(sb[['ts_code','name','industry','market','list_date','is_new']], on='ts_code', how='left')
m = m.merge(db[['ts_code','turnover_rate','volume_ratio','pe_ttm','pb','dv_ttm','total_mv','circ_mv']],
            on='ts_code', how='left')

m = m.sort_values(['ts_code','trade_date']).reset_index(drop=True)
g = m.groupby('ts_code', group_keys=False)

# 均线
m['ma5']  = g['close'].transform(lambda x: x.rolling(5, min_periods=5).mean())
m['ma10'] = g['close'].transform(lambda x: x.rolling(10, min_periods=10).mean())
m['ma20'] = g['close'].transform(lambda x: x.rolling(20, min_periods=20).mean())

# MACD (transform 对齐安全)
ema12 = g['close'].transform(lambda x: x.ewm(span=12, adjust=False).mean())
ema26 = g['close'].transform(lambda x: x.ewm(span=26, adjust=False).mean())
m['dif'] = ema12 - ema26
m['dea'] = g['dif'].transform(lambda x: x.ewm(span=9, adjust=False).mean())
m['macd_hist'] = (m['dif'] - m['dea']) * 2

# RSI14
def rsi_series(c):
    d = c.diff()
    up = d.clip(lower=0); dn = (-d).clip(lower=0)
    au = up.ewm(alpha=1/14, adjust=False).mean()
    ad = dn.ewm(alpha=1/14, adjust=False).mean()
    rs = au / ad.replace(0, np.nan)
    return (100 - 100/(1+rs)).fillna(50)
m['rsi14'] = g['close'].transform(rsi_series)

# 量比: 当日成交量 / 过去5日平均成交量
m['vol_ma5']  = g['vol'].transform(lambda x: x.rolling(5, min_periods=5).mean().shift(1))
m['vol_ma20'] = g['vol'].transform(lambda x: x.rolling(20, min_periods=20).mean().shift(1))
m['vol_ratio_calc'] = m['vol'] / m['vol_ma5'].replace(0, np.nan)
m['vol_ratio_20d']  = m['vol'] / m['vol_ma20'].replace(0, np.nan)

# 20日突破幅度 (close vs 过去20日最高价, 不含当日)
m['high_20'] = g['high'].transform(lambda x: x.rolling(20, min_periods=20).max().shift(1))
m['breakout_20d_pct'] = (m['close'] - m['high_20']) / m['high_20'] * 100
# 20日振幅
hh = g['high'].transform(lambda x: x.rolling(20, min_periods=20).max())
ll = g['low'].transform(lambda x: x.rolling(20, min_periods=20).min())
m['range_20d_pct'] = (hh - ll) / m['close'] * 100

# 20日波动率 (日收益率std, %)
m['volatility_20d'] = g['pct_chg'].transform(lambda x: x.rolling(20, min_periods=20).std())

# 20日最大回撤
m['max_drawdown_20d'] = g['close'].transform(
    lambda c: (c / c.rolling(20, min_periods=20).max() - 1).rolling(20, min_periods=20).min())

# ATR20%
m['_tr'] = pd.concat([(m['high']-m['low']).abs(),
                      (m['high']-m['pre_close']).abs(),
                      (m['low']-m['pre_close']).abs()], axis=1).max(axis=1)
m['atr_20_pct'] = g['_tr'].transform(lambda x: x.rolling(20, min_periods=20).mean()) / m['close'] * 100

# 60日涨跌幅
m['close_60d'] = g['close'].transform(lambda x: x.shift(60))
m['change_60d'] = (m['close'] / m['close_60d'] - 1) * 100

# 距MA20%
m['pullback_to_ma20_pct'] = (m['close'] - m['ma20']) / m['ma20'] * 100

# 均线多头 / 站上MA20
m['ma_bullish'] = (m['ma5'] > m['ma10']) & (m['ma10'] > m['ma20'])
m['price_above_ma20'] = m['close'] > m['ma20']

# 实体占比
hl = (m['high'] - m['low']).replace(0, np.nan)
m['body_pct'] = ((m['close'] - m['open']).abs() / hl).fillna(0)

# MACD金叉 (当日dif上穿dea)
m['dif_prev'] = g['dif'].transform(lambda x: x.shift(1))
m['dea_prev'] = g['dea'].transform(lambda x: x.shift(1))
m['macd_golden'] = (m['dif'] > m['dea']) & (m['dif_prev'] <= m['dea_prev'])
m['macd_status'] = np.where(m['dif'] > m['dea'], 'bullish', 'bearish')

# 整理天数 (近20日内 |close-ma20|/ma20 < 2% 的天数)
m['_consol_flag'] = (((m['close'] - m['ma20']).abs() / m['ma20']) < 0.02).astype(float)
m['consolidation_days_20d'] = g['_consol_flag'].transform(lambda x: x.rolling(20, min_periods=20).sum())

log('  技术指标计算完成')

# ---------- 6. 最新交易日快照 ----------
snap = m[m['trade_date'] == REPORT_DATE].copy()
# 通用排除: ST / 次新 / 停牌(vol=0)
snap = snap[~snap['name'].str.contains('ST', na=False)]
snap = snap[~snap['is_new'].astype(bool)]
snap = snap[(snap['vol'] > 0) & (snap['amount'] > 0)]
snap['amount_yi'] = snap['amount'] / 100000.0   # 千元 -> 亿
snap['total_mv_yi'] = snap['total_mv'] / 10000.0  # 万元 -> 亿
log(f'[6] 最新交易日快照 {len(snap)} 只 (已排除ST/次新/停牌)')

# 板块强度(今日行业平均涨跌幅) -> theme_heat 百分位
ind_avg = snap.groupby('industry')['pct_chg'].mean().sort_values()
ind_rank = ind_avg.rank(pct=True)
snap['theme_heat'] = snap['industry'].map(ind_rank).fillna(50) * 100

# signal_score
def signal_score(r):
    s = 0
    if r['ma_bullish']: s += 15
    if r['price_above_ma20']: s += 10
    if r['close'] > r['ma5']: s += 5
    s += min(10, max(0, r['pct_chg']) * 2)
    if r['macd_status'] == 'bullish': s += 12
    if r['macd_golden']: s += 8
    rsi = r['rsi14']
    if 45 <= rsi <= 70: s += 10
    elif 30 <= rsi < 80: s += 5
    vr = r['vol_ratio_calc']
    if vr >= 2: s += 10
    elif vr >= 1.5: s += 7
    elif vr >= 1: s += 4
    if r['breakout_20d_pct'] > 0: s += 8
    if r['change_60d'] > 0: s += 2
    return min(100, s)
snap['signal_score'] = snap.apply(signal_score, axis=1)

# ---------- 因子分(0-100) ----------
def f_value(pe, pb, pe_max, pb_max):
    if pe <= 0 or np.isnan(pe): pe_s = 0
    else: pe_s = 100*(1 - min(pe, pe_max)/pe_max)
    pb_s = 100*(1 - min(pb, pb_max)/pb_max) if pb>0 else 0
    return 0.5*pe_s + 0.5*pb_s
def f_stability(vol20, mdd, atr20):
    s = 100 - (vol20 or 0)*1.5 - abs(mdd or 0)*1.2 - (atr20 or 0)*3
    return max(0, min(100, s))
def f_liquidity(amt_yi):
    if amt_yi <= 0: return 0
    s = (np.log10(amt_yi) - np.log10(0.5)) / (np.log10(50) - np.log10(0.5)) * 100
    return max(0, min(100, s))
def f_momentum(pct, chg60, ma_bull, chase_start, slope):
    base = 50 + pct*4 + (chg60 or 0)*0.3
    if ma_bull: base += 10
    if pct > chase_start: base -= (pct - chase_start) * slope
    if pct < -3: base -= (abs(pct)-3)*4
    return max(0, min(100, base))
def f_activity(vr, tr, ideal_vr, ideal_tr):
    vr_s = 100 - abs(vr - ideal_vr)*15
    tr_s = 100 - abs(tr - ideal_tr)*8
    s = 0.5*max(0,min(100,vr_s)) + 0.5*max(0,min(100,tr_s))
    return max(0, min(100, s))
def f_reversal(pct, ideal, collapse_start, slope):
    s = 100 - abs(pct - ideal)*10
    if pct < collapse_start: s -= (collapse_start - pct) * slope
    return max(0, min(100, s))
def f_size(mv_yi):
    if mv_yi < 20: s = mv_yi/20*50
    elif mv_yi < 500: s = 50 + (mv_yi-20)/480*50
    else: s = max(0, 100 - (mv_yi-500)/2000*100)
    return max(0, min(100, s))

log('[7] 执行 10 策略...')

# ---------- 策略硬筛+评分 ----------
def base_filter(df, ex):
    d = df.copy()
    if ex.get('exclude_st', True):
        d = d[~d['name'].str.contains('ST', na=False)]
    if 'amount_min' in ex:
        d = d[d['amount'] * 1000 >= ex['amount_min']]   # amount千元*1000=元
    if 'pe_ttm_min' in ex: d = d[d['pe_ttm'] >= ex['pe_ttm_min']]
    if 'pe_ttm_max' in ex: d = d[(d['pe_ttm'] <= ex['pe_ttm_max']) | d['pe_ttm'].isna()] if ex.get('pe_na_ok',False) else d[d['pe_ttm'] <= ex['pe_ttm_max']]
    if 'pb_min' in ex: d = d[d['pb'] >= ex['pb_min']]
    if 'pb_max' in ex: d = d[d['pb'] <= ex['pb_max']]
    if 'market_cap_min' in ex: d = d[d['total_mv'] >= ex['market_cap_min']/10000]
    if 'market_cap_max' in ex: d = d[d['total_mv'] <= ex['market_cap_max']/10000]
    if 'price_min' in ex: d = d[d['close'] >= ex['price_min']]
    if 'price_max' in ex: d = d[d['close'] <= ex['price_max']]
    if 'change_pct_min' in ex: d = d[d['pct_chg'] >= ex['change_pct_min']]
    if 'change_pct_max' in ex: d = d[d['pct_chg'] <= ex['change_pct_max']]
    if 'turnover_rate_min' in ex: d = d[d['turnover_rate'] >= ex['turnover_rate_min']]
    if 'volume_ratio_min' in ex:
        vr = d['volume_ratio'].fillna(d['vol_ratio_calc'])
        d = d.assign(_vr=vr)
        d = d[d['_vr'] >= ex['volume_ratio_min']]
    if 'require_ma_bullish' in ex and ex['require_ma_bullish']:
        d = d[d['ma_bullish'] == True]
    if 'require_price_above_ma20' in ex and ex['require_price_above_ma20']:
        d = d[d['price_above_ma20'] == True]
    if 'signal_score_min' in ex: d = d[d['signal_score'] >= ex['signal_score_min']]
    if 'volume_ratio_20d_min' in ex: d = d[d['vol_ratio_20d'] >= ex['volume_ratio_20d_min']]
    if 'volume_ratio_20d_max' in ex: d = d[d['vol_ratio_20d'] <= ex['volume_ratio_20d_max']]
    if 'breakout_20d_pct_min' in ex: d = d[d['breakout_20d_pct'] >= ex['breakout_20d_pct_min']]
    if 'range_20d_pct_max' in ex: d = d[d['range_20d_pct'] <= ex['range_20d_pct_max']]
    if 'volatility_20d_pct_max' in ex: d = d[d['volatility_20d'] <= ex['volatility_20d_pct_max']]
    if 'max_drawdown_20d_pct_min' in ex: d = d[d['max_drawdown_20d'] >= ex['max_drawdown_20d_pct_min']]
    if 'atr_20_pct_max' in ex: d = d[d['atr_20_pct'] <= ex['atr_20_pct_max']]
    if 'change_60d_min' in ex: d = d[d['change_60d'] >= ex['change_60d_min']]
    if 'change_60d_max' in ex: d = d[d['change_60d'] <= ex['change_60d_max']]
    if 'body_pct_min' in ex: d = d[d['body_pct'] >= ex['body_pct_min']]
    if 'consolidation_days_20d_min' in ex: d = d[d['consolidation_days_20d'] >= ex['consolidation_days_20d_min']]
    if 'pullback_to_ma20_pct_min' in ex: d = d[d['pullback_to_ma20_pct'] >= ex['pullback_to_ma20_pct_min']]
    if 'pullback_to_ma20_pct_max' in ex: d = d[d['pullback_to_ma20_pct'] <= ex['pullback_to_ma20_pct_max']]
    if 'macd_status_whitelist' in ex: d = d[d['macd_status'].isin(ex['macd_status_whitelist'])]
    return d

def score_row(r, fw, sp, risk, tech_weight, pe_max, pb_max):
    # 因子分
    fs = {}
    fs['value']      = f_value(r['pe_ttm'], r['pb'], pe_max, pb_max)
    fs['stability']  = f_stability(r['volatility_20d'], r['max_drawdown_20d'], r['atr_20_pct'])
    fs['liquidity']  = f_liquidity(r['amount_yi'])
    fs['momentum']   = f_momentum(r['pct_chg'], r['change_60d'], r['ma_bullish'],
                                  sp.get('momentum_chase_start_pct',5), sp.get('momentum_chase_penalty_slope',14))
    fs['activity']   = f_activity(r['vol_ratio_calc'], r['turnover_rate'],
                                  sp.get('activity_ideal_volume_ratio',1.5), sp.get('activity_ideal_turnover_rate',2.5))
    fs['reversal']   = f_reversal(r['pct_chg'], sp.get('reversal_ideal_change_pct',-3.5),
                                  sp.get('reversal_collapse_start_pct',-7), sp.get('reversal_collapse_penalty_slope',12))
    fs['size']       = f_size(r['total_mv_yi'])
    fs['theme_heat'] = r['theme_heat']
    wsum = sum(fw.get(k,0) for k in fs)
    fact = sum(fw.get(k,0)*fs[k] for k in fs) / wsum if wsum>0 else 0
    final = (1-tech_weight)*fact + tech_weight*r['signal_score']
    # 风险扣分
    if r['pct_chg'] > risk.get('chase_change_pct',9): final -= 8
    if r['vol_ratio_calc'] > risk.get('abnormal_volume_ratio',6): final -= 6
    if r['turnover_rate'] > risk.get('high_turnover_rate',15): final -= 5
    return max(0, min(100, final))

# 10 策略参数 (取自 /workspace/strategies/*.yaml)
STRATEGIES = {
 'dual_low': dict(disp='双低选股', cat='价值', ex=dict(amount_min=50000000,pe_ttm_max=15,pb_max=2.0,
        market_cap_min=5000000000,market_cap_max=300000000000,price_min=3,price_max=80,
        change_pct_min=-4.5,change_pct_max=4.5),
        fw=dict(value=0.34,stability=0.20,liquidity=0.14,momentum=0.10,activity=0.10,reversal=0.06,size=0.06),
        sp=dict(momentum_chase_start_pct=2.5,momentum_chase_penalty_slope=18,activity_ideal_volume_ratio=1.2,activity_ideal_turnover_rate=2.0),
        risk=dict(chase_change_pct=5.0,abnormal_volume_ratio=4.0,high_turnover_rate=8.0), tech=0.2, pemax=15, pbmax=2.0),
 'quality_value': dict(disp='稳健价值', cat='价值', ex=dict(amount_min=80000000,pe_ttm_max=25,pb_max=4.0,
        market_cap_min=10000000000,market_cap_max=800000000000,change_pct_min=-3.5,change_pct_max=5.0,price_min=3,price_max=180),
        fw=dict(value=0.32,stability=0.24,liquidity=0.18,momentum=0.08,activity=0.08,reversal=0.04,size=0.06),
        sp=dict(momentum_chase_start_pct=3.0,momentum_chase_penalty_slope=16,activity_ideal_volume_ratio=1.4,activity_ideal_turnover_rate=2.5),
        risk=dict(chase_change_pct=5.5,abnormal_volume_ratio=4.5,high_turnover_rate=10.0), tech=0.15, pemax=25, pbmax=4.0),
 'blue_chip_income': dict(disp='蓝筹收益质量', cat='收益', ex=dict(amount_min=120000000,market_cap_min=30000000000,
        pe_ttm_max=22,pb_max=3.2,turnover_rate_min=0.4,volume_ratio_min=0.6,change_pct_min=-3.0,change_pct_max=4.0,price_min=3,price_max=180),
        fw=dict(value=0.30,stability=0.26,liquidity=0.18,size=0.12,activity=0.06,momentum=0.05,reversal=0.03),
        sp=dict(momentum_chase_start_pct=2.8,momentum_chase_penalty_slope=18,activity_ideal_volume_ratio=1.1,activity_ideal_turnover_rate=1.8),
        risk=dict(chase_change_pct=4.8,abnormal_volume_ratio=3.5,high_turnover_rate=8.0), tech=0.18, pemax=22, pbmax=3.2),
 'low_volatility_quality': dict(disp='低波质量', cat='质量', ex=dict(amount_min=100000000,market_cap_min=12000000000,
        pe_ttm_max=45,pb_max=5.0,change_pct_min=-3.0,change_pct_max=5.0,price_min=4,price_max=180,
        change_60d_min=-10.0,change_60d_max=35.0,signal_score_min=55,range_20d_pct_max=28.0,
        volatility_20d_pct_max=32.0,max_drawdown_20d_pct_min=-8.0,atr_20_pct_max=4.5),
        fw=dict(stability=0.30,value=0.20,liquidity=0.15,momentum=0.12,activity=0.08,size=0.08,theme_heat=0.05,reversal=0.02),
        sp=dict(momentum_chase_start_pct=4.0,momentum_chase_penalty_slope=18,activity_ideal_volume_ratio=1.4,activity_ideal_turnover_rate=2.5),
        risk=dict(chase_change_pct=5.5,abnormal_volume_ratio=4.5,high_turnover_rate=10.0), tech=0.30, pemax=45, pbmax=5.0),
 'volume_breakout': dict(disp='放量突破', cat='趋势', ex=dict(amount_min=100000000,turnover_rate_min=3.0,
        volume_ratio_min=2.0,change_pct_min=2.0,change_pct_max=9.9,require_price_above_ma20=True,signal_score_min=60,
        macd_status_whitelist=['bullish','neutral'],breakout_20d_pct_min=-1.0,range_20d_pct_max=35.0,
        volume_ratio_20d_min=1.3,body_pct_min=0.5,consolidation_days_20d_min=8),
        fw=dict(momentum=0.32,activity=0.28,liquidity=0.22,theme_heat=0.08,stability=0.10),
        sp=dict(momentum_chase_start_pct=7.0,momentum_chase_penalty_slope=11,activity_ideal_volume_ratio=3.0,activity_ideal_turnover_rate=6.0),
        risk=dict(chase_change_pct=9.8,abnormal_volume_ratio=9.0,high_turnover_rate=24.0), tech=0.6, pemax=80, pbmax=8.0),
 'shrink_pullback': dict(disp='缩量回踩', cat='趋势', ex=dict(amount_min=80000000,turnover_rate_min=1.0,
        require_ma_bullish=True,require_price_above_ma20=True,signal_score_min=65,volume_ratio_20d_max=1.5,
        pullback_to_ma20_pct_min=-2.0,pullback_to_ma20_pct_max=8.0,volatility_20d_pct_max=45.0,
        max_drawdown_20d_pct_min=-12.0,atr_20_pct_max=6.5),
        fw=dict(momentum=0.32,stability=0.22,activity=0.18,liquidity=0.15,value=0.08,reversal=0.05),
        sp=dict(momentum_chase_start_pct=5.0,momentum_chase_penalty_slope=14,activity_ideal_volume_ratio=1.0,activity_ideal_turnover_rate=2.5),
        risk=dict(chase_change_pct=6.5,abnormal_volume_ratio=4.5,high_turnover_rate=12.0), tech=0.5, pemax=80, pbmax=8.0),
 'capital_heat': dict(disp='资金热度', cat='动量', ex=dict(amount_min=300000000,turnover_rate_min=2.0,
        volume_ratio_min=1.5,change_pct_min=1.0,change_pct_max=9.5,price_min=3,price_max=220),
        fw=dict(momentum=0.32,activity=0.28,liquidity=0.16,theme_heat=0.10,stability=0.10,reversal=0.04),
        sp=dict(momentum_chase_start_pct=6.5,momentum_chase_penalty_slope=12,activity_ideal_volume_ratio=2.8,activity_ideal_turnover_rate=6.0),
        risk=dict(chase_change_pct=9.3,abnormal_volume_ratio=8.0,high_turnover_rate=22.0), tech=0.65, pemax=80, pbmax=8.0),
 'oversold_reversal': dict(disp='超跌反转', cat='反转', ex=dict(amount_min=80000000,turnover_rate_min=1.0,
        change_pct_min=-8.0,change_pct_max=-1.0,pe_ttm_max=80,pb_max=8.0,price_min=3,price_max=180),
        fw=dict(reversal=0.40,stability=0.20,liquidity=0.18,value=0.16,activity=0.06),
        sp=dict(reversal_ideal_change_pct=-3.5,reversal_collapse_start_pct=-7.0,reversal_collapse_penalty_slope=12,
                activity_ideal_volume_ratio=1.6,activity_ideal_turnover_rate=3.0,momentum_chase_start_pct=5,momentum_chase_penalty_slope=14),
        risk=dict(chase_change_pct=5.0,abnormal_volume_ratio=5.0,high_turnover_rate=12.0), tech=0.45, pemax=80, pbmax=8.0),
 'balanced_alpha': dict(disp='均衡多因子', cat='综合', ex=dict(amount_min=100000000,market_cap_min=5000000000,
        pe_ttm_max=80,pb_max=8.0,change_pct_min=-4.0,change_pct_max=8.5,price_min=3,price_max=220),
        fw=dict(value=0.22,liquidity=0.18,momentum=0.20,activity=0.15,stability=0.12,reversal=0.05,theme_heat=0.05,size=0.03),
        sp=dict(momentum_chase_start_pct=5.5,momentum_chase_penalty_slope=12,activity_ideal_volume_ratio=2.0,activity_ideal_turnover_rate=4.0),
        risk=dict(chase_change_pct=8.0,abnormal_volume_ratio=6.0,high_turnover_rate=15.0), tech=0.35, pemax=80, pbmax=8.0),
 'momentum_quality': dict(disp='趋势质量', cat='综合', ex=dict(amount_min=200000000,market_cap_min=8000000000,
        pe_ttm_max=60,pb_max=8.0,change_pct_min=-3.5,change_pct_max=7.5,price_min=4,price_max=220),
        fw=dict(momentum=0.28,value=0.15,liquidity=0.18,activity=0.15,stability=0.12,reversal=0.04,theme_heat=0.06,size=0.02),
        sp=dict(momentum_chase_start_pct=5.0,momentum_chase_penalty_slope=14,activity_ideal_volume_ratio=2.2,activity_ideal_turnover_rate=4.5),
        risk=dict(chase_change_pct=7.5,abnormal_volume_ratio=5.5,high_turnover_rate=14.0), tech=0.45, pemax=60, pbmax=8.0),
}

results = {}
for name, cfg in STRATEGIES.items():
    d = base_filter(snap, cfg['ex'])
    # 放宽: 若通过硬筛<5, 逐步放宽 signal/tech 类阈值取最接近的前10
    relaxed = False
    if len(d) < 5:
        relaxed = True
        ex2 = {k:v for k,v in cfg['ex'].items() if k not in ('signal_score_min','volume_ratio_min',
               'volume_ratio_20d_min','consolidation_days_20d_min','turnover_rate_min','body_pct_min',
               'breakout_20d_pct_min','require_ma_bullish','require_price_above_ma20','macd_status_whitelist')}
        d = base_filter(snap, ex2)
    if len(d) == 0:
        results[name] = (cfg, [], relaxed)
        log(f'  {name}: 0 命中')
        continue
    d = d.copy()
    d['score'] = d.apply(lambda r: score_row(r, cfg['fw'], cfg['sp'], cfg['risk'], cfg['tech'], cfg['pemax'], cfg['pbmax']), axis=1)
    d = d.sort_values('score', ascending=False).head(10)
    results[name] = (cfg, d, relaxed)
    log(f'  {name}: 命中 {len(d)} (relaxed={relaxed})')

# ---------- 持久化中间结果给报告脚本 ----------
import pickle
with open('/workspace/_screen_cache.pkl','wb') as f:
    pickle.dump(dict(snap=snap, results=results, ind_avg=ind_avg, trade_dates=hist_dates), f)
log('[done] 缓存已保存 /workspace/_screen_cache.pkl')

# 输出紧凑摘要到 stdout
summary = {}
for name,(cfg,d,rel) in results.items():
    rows=[]
    for _,r in d.iterrows():
        rows.append(dict(code=r['ts_code'], name=r['name'], ind=r['industry'],
                         close=round(float(r['close']),2), pct=round(float(r['pct_chg']),2),
                         amt=round(float(r['amount_yi']),2), pe=(None if pd.isna(r['pe_ttm']) else round(float(r['pe_ttm']),1)),
                         pb=round(float(r['pb']),2), mv=round(float(r['total_mv_yi']),0), score=round(float(r['score']),1)))
    summary[name]=dict(disp=cfg['disp'],cat=cfg['cat'],relaxed=rel,n=len(rows),top3=rows[:3],top10=rows)
print(json.dumps(summary, ensure_ascii=False))
