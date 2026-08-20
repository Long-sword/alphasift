# -*- coding: utf-8 -*-
"""
每日10:00盘中选股策略报告 (method B: tushare 直连)
跑10个策略: dual_low / quality_value / blue_chip_income / low_volatility_quality /
volume_breakout / shrink_pullback / capital_heat / oversold_reversal /
balanced_alpha / momentum_quality
"""
import os, sys, time, json, math
import numpy as np
import pandas as pd
import tushare as ts
from datetime import datetime, timedelta

# ---------------- 配置 ----------------
TOKEN = '9f640d421f866dde9a1888ab4193f77c43659d47446d717ffa343023'
ts.set_token(TOKEN)
pro = ts.pro_api()

LATEST_DATE = '20260819'          # 最新可用交易日(收盘数据)
TODAY       = '20260820'          # 今日(盘中)
CACHE_DIR   = '/workspace/.cache'
REPORT_DIR  = '/workspace/reports'
os.makedirs(CACHE_DIR, exist_ok=True)
os.makedirs(REPORT_DIR, exist_ok=True)

NEED_DAYS = 66                    # 计算指标所需历史交易日
NEW_STOCK_CUT = '20250820'        # 上市晚于此 -> 次新(上市<1年)排除

# 单位换算说明:
#   daily.amount      = 千元  -> amount_min(yuan) / 1000
#   daily_basic.total_mv = 万元 -> market_cap_min(yuan) / 10000
#   daily_basic.turnover_rate, volume_ratio, pe, pb 直接使用

# ---------------- 数据获取(带缓存) ----------------
def fetch_stock_basic():
    cache = os.path.join(CACHE_DIR, 'stock_basic.parquet')
    if os.path.exists(cache):
        return pd.read_parquet(cache)
    df = pro.stock_basic(exchange='', list_status='L',
            fields='ts_code,symbol,name,area,industry,list_date')
    df.to_parquet(cache)
    return df

def fetch_trade_dates():
    cache = os.path.join(CACHE_DIR, 'trade_dates.parquet')
    if os.path.exists(cache):
        cal = pd.read_parquet(cache)
    else:
        cal = pro.trade_cal(exchange='SSE', start_date='20250101', end_date=LATEST_DATE)
        cal = cal[cal['is_open']==1].sort_values('cal_date').reset_index(drop=True)
        cal.to_parquet(cache)
    dates = cal['cal_date'].tolist()
    return dates[-NEED_DAYS:]

def fetch_daily_panel(trade_dates):
    frames = []
    for i, td in enumerate(trade_dates):
        cache = os.path.join(CACHE_DIR, f'daily_{td}.parquet')
        if os.path.exists(cache):
            df = pd.read_parquet(cache)
        else:
            for attempt in range(4):
                try:
                    df = pro.daily(trade_date=td)
                    break
                except Exception as e:
                    print(f'  retry daily {td} ({attempt}): {e}')
                    time.sleep(1.5)
            df.to_parquet(cache)
            time.sleep(0.18)
        df['trade_date'] = td
        frames.append(df)
        if (i+1) % 10 == 0:
            print(f'  daily panel {i+1}/{len(trade_dates)}')
    panel = pd.concat(frames, ignore_index=True)
    return panel

def fetch_daily_basic(date):
    cache = os.path.join(CACHE_DIR, f'daily_basic_{date}.parquet')
    if os.path.exists(cache):
        return pd.read_parquet(cache)
    df = pro.daily_basic(trade_date=date,
        fields='ts_code,trade_date,close,turnover_rate,turnover_rate_f,volume_ratio,'
               'pe,pe_ttm,pb,ps,ps_ttm,dv_ratio,dv_ttm,total_share,float_share,'
               'free_share,total_mv,circ_mv,limit_status')
    df.to_parquet(cache)
    return df

# ---------------- 技术指标 ----------------
def ema(s, span):
    return s.ewm(span=span, adjust=False).mean()

def compute_indicators(panel):
    """对每只股票计算技术指标, 返回最新一日指标表(index=ts_code)"""
    panel = panel.sort_values(['ts_code','trade_date']).reset_index(drop=True)
    g = panel.groupby('ts_code', group_keys=False)
    # 均线
    panel['ma5']  = g['close'].transform(lambda s: s.rolling(5, min_periods=5).mean())
    panel['ma10'] = g['close'].transform(lambda s: s.rolling(10, min_periods=10).mean())
    panel['ma20'] = g['close'].transform(lambda s: s.rolling(20, min_periods=20).mean())
    # pct_chg 已存在; 兼容补算
    if 'pct_chg' not in panel.columns:
        panel['pct_chg'] = g['close'].pct_change()*100
    panel['pct_chg'] = panel['pct_chg'].fillna(0)
    # MACD
    panel['ema12'] = g['close'].transform(lambda s: ema(s,12))
    panel['ema26'] = g['close'].transform(lambda s: ema(s,26))
    panel['dif']   = panel['ema12'] - panel['ema26']
    panel['dea']   = g['dif'].transform(lambda s: ema(s,9))
    panel['macd_hist'] = (panel['dif'] - panel['dea'])*2
    # RSI(14)
    def rsi(s, p=14):
        d = s.diff()
        up = d.clip(lower=0).rolling(p, min_periods=p).mean()
        dn = (-d.clip(upper=0)).rolling(p, min_periods=p).mean()
        rs = up / dn.replace(0, np.nan)
        return 100 - 100/(1+rs)
    panel['rsi14'] = g['close'].transform(rsi)
    # 真实波幅 / ATR20
    panel['prev_close'] = g['close'].shift(1)
    panel['tr'] = pd.concat([
        panel['high']-panel['low'],
        (panel['high']-panel['prev_close']).abs(),
        (panel['low']-panel['prev_close']).abs()
    ], axis=1).max(axis=1)
    panel['atr20'] = g['tr'].transform(lambda s: s.rolling(20, min_periods=15).mean())
    # 20日最高/最低(不含当日用于突破判定)
    panel['high20_max'] = g['high'].transform(lambda s: s.rolling(20, min_periods=10).max())
    panel['low20_min']  = g['low'].transform(lambda s: s.rolling(20, min_periods=10).min())
    panel['high20_prior']= g['high'].transform(lambda s: s.shift(1).rolling(19, min_periods=8).max())
    # 20日波动/振幅
    panel['vol20_std'] = g['pct_chg'].transform(lambda s: s.rolling(20, min_periods=10).std())
    # 成交量均量
    panel['vol20_mean'] = g['vol'].transform(lambda s: s.rolling(20, min_periods=10).mean().shift(0))
    panel['vol5_mean']  = g['vol'].transform(lambda s: s.rolling(5, min_periods=5).mean())
    # 60日前收盘(用于change_60d)
    panel['close60_ago'] = g['close'].transform(lambda s: s.shift(60))
    # 20日累计最高(用于最大回撤)
    panel['cummax20'] = g['close'].transform(lambda s: s.rolling(20, min_periods=10).max())
    panel['dd20'] = (panel['close'] - panel['cummax20']) / panel['cummax20'] * 100
    # 实体占比
    rng = (panel['high'] - panel['low']).replace(0, np.nan)
    panel['body_pct'] = (panel['close']-panel['open']).abs() / rng
    # 整理天数: 近20日中 |close-ma20|/ma20 < 6% 的天数
    panel['dev_ma20'] = (panel['close']-panel['ma20'])/panel['ma20']*100
    panel['consol'] = (panel['dev_ma20'].abs() < 6).astype(int)
    panel['consol20'] = g['consol'].transform(lambda s: s.rolling(20, min_periods=10).sum())

    # 取最新一日
    latest = panel[panel['trade_date']==LATEST_DATE].copy()
    # 计算衍生指标
    latest['ma_bullish'] = (latest['ma5']>latest['ma10']) & (latest['ma10']>latest['ma20'])
    latest['price_above_ma20'] = latest['close'] > latest['ma20']
    latest['macd_status'] = np.where(latest['dif']>latest['dea'], 'bullish', 'bearish')
    latest['macd_hist_pos'] = latest['macd_hist'] > 0
    latest['dist_ma20_pct'] = (latest['close']-latest['ma20'])/latest['ma20']*100
    latest['pullback_to_ma20_pct'] = latest['dist_ma20_pct']
    latest['volatility_20d_pct'] = latest['vol20_std'] * 100
    latest['range_20d_pct'] = (latest['high20_max']-latest['low20_min'])/latest['low20_min']*100
    latest['breakout_20d_pct'] = (latest['close']-latest['high20_prior'])/latest['high20_prior']*100
    latest['volume_ratio_20d'] = latest['vol'] / latest['vol20_mean'].replace(0, np.nan)
    latest['change_60d'] = (latest['close']-latest['close60_ago'])/latest['close60_ago']*100
    latest['max_drawdown_20d_pct'] = latest['dd20']
    latest['atr_20_pct'] = latest['atr20']/latest['close']*100
    return latest, panel

# ---------------- 信号综合分(0-100) ----------------
def signal_score(row):
    s = 0
    if row['ma_bullish']: s += 22
    if row['price_above_ma20']: s += 14
    if row['macd_status']=='bullish': s += 18
    if row['macd_hist_pos']: s += 6
    rsi = row['rsi14']
    if pd.notna(rsi):
        if 45 <= rsi <= 70: s += 16
        elif 35 <= rsi < 45 or 70 < rsi <= 80: s += 10
        elif rsi < 35: s += 4
    vr = row.get('volume_ratio_20d')
    if pd.notna(vr):
        if 1.0 <= vr <= 2.5: s += 12
        elif 0.6 <= vr < 1.0 or 2.5 < vr <= 4: s += 7
    # 动量: 5日趋势
    if row.get('change_60d',0) and pd.notna(row['change_60d']) and 0 < row['change_60d'] < 35: s += 12
    if pd.notna(row['pct_chg']) and 0 <= row['pct_chg'] <= 5: s += 6
    return min(s, 100)

# ---------------- 因子分(0-100) ----------------
def f_value(r):
    pe = r.get('pe_ttm'); pb = r.get('pb')
    sc = []
    if pd.notna(pe) and pe > 0:
        sc.append(max(20, min(95, 95 - (pe/60.0)*55)))
    if pd.notna(pb) and pb > 0:
        sc.append(max(20, min(95, 95 - (pb/8.0)*55)))
    return np.mean(sc) if sc else 40

def f_liquidity(r):
    amt = r.get('amount_yi')  # 成交额(亿元)
    if pd.notna(amt):
        return float(np.clip(20 + 28*np.log1p(amt)/np.log1p(20), 20, 95))
    return 35

def f_momentum(r):
    p = r.get('pct_chg', 0) or 0
    c60 = r.get('change_60d', 0) or 0
    base = 50
    base += max(-15, min(25, p*3))
    base += max(-15, min(20, c60*0.6))
    if r.get('ma_bullish'): base += 8
    return float(np.clip(base, 15, 95))

def f_activity(r):
    vr = r.get('volume_ratio'); tr = r.get('turnover_rate')
    ideal_vr = r.get('_ideal_vr', 2.0); ideal_tr = r.get('_ideal_tr', 4.0)
    s = 50
    if pd.notna(vr):
        d = abs(vr-ideal_vr); s -= min(35, d*9)
    if pd.notna(tr):
        d = abs(tr-ideal_tr); s -= min(30, d*4)
    return float(np.clip(s, 15, 95))

def f_stability(r):
    vol = r.get('volatility_20d_pct'); dd = r.get('max_drawdown_20d_pct'); atr = r.get('atr_20_pct')
    s = 90
    if pd.notna(vol): s -= min(45, max(0, vol-2)*2.0)
    if pd.notna(dd): s -= min(40, abs(min(dd,0))*3.0)
    if pd.notna(atr): s -= min(30, max(0, atr-2)*6.0)
    p = r.get('pct_chg',0) or 0
    if abs(p) > 6: s -= 12
    return float(np.clip(s, 15, 95))

def f_reversal(r):
    p = r.get('pct_chg',0) or 0
    ideal = -3.5
    s = 90 - min(50, abs(p-ideal)*7)
    if p <= -7: s -= 25   # collapse penalty
    rsi = r.get('rsi14')
    if pd.notna(rsi) and rsi < 35: s += 12
    return float(np.clip(s, 15, 95))

def f_theme_heat(r):
    vr = r.get('volume_ratio'); tr = r.get('turnover_rate')
    s = 40
    if pd.notna(vr): s += min(35, vr*5)
    if pd.notna(tr): s += min(25, tr*2)
    return float(np.clip(s, 15, 95))

def f_size(r):
    mv = r.get('total_mv_yi')  # 总市值(亿元)
    if pd.notna(mv):
        return float(np.clip(30 + 30*np.log1p(mv)/np.log1p(2000), 30, 92))
    return 45

FACTOR_FN = {
    'value': f_value, 'liquidity': f_liquidity, 'momentum': f_momentum,
    'activity': f_activity, 'stability': f_stability, 'reversal': f_reversal,
    'theme_heat': f_theme_heat, 'size': f_size,
}

# ---------------- 策略配置(精炼自YAML) ----------------
STRATS = {
 'dual_low': dict(name='双低选股', cat='价值', tech=0.20,
   hf=dict(amount_min=5e7, pe_ttm_max=15, pb_max=2.0, mcap_min=5e9, mcap_max=3e11,
           price_min=3, price_max=80, chg_min=-4.5, chg_max=4.5),
   fw=dict(value=.34, stability=.20, liquidity=.14, momentum=.10, activity=.10, reversal=.06, size=.06),
   ideal_vr=1.2, ideal_tr=2.0, hint='低PE+低PB基础,叠加当日活跃度与形态确认,避免长期只命中静态低估值票'),
 'quality_value': dict(name='稳健价值', cat='价值', tech=0.15,
   hf=dict(amount_min=8e7, pe_ttm_max=25, pb_max=4.0, mcap_min=1e10, mcap_max=8e11,
           price_min=3, price_max=180, chg_min=-3.5, chg_max=5.0),
   fw=dict(value=.32, stability=.24, liquidity=.18, momentum=.08, activity=.08, reversal=.04, size=.06),
   ideal_vr=1.4, ideal_tr=2.5, hint='估值合理+流动性充足+波动不过热,需温和活跃度确认'),
 'blue_chip_income': dict(name='蓝筹收益质量', cat='收益', tech=0.18,
   hf=dict(amount_min=1.2e8, mcap_min=3e10, pe_ttm_max=22, pb_max=3.2,
           tr_min=0.4, vr_min=0.6, price_min=3, price_max=180, chg_min=-3.0, chg_max=4.0),
   fw=dict(value=.30, stability=.26, liquidity=.18, size=.12, activity=.06, momentum=.05, reversal=.03),
   ideal_vr=1.1, ideal_tr=1.8, hint='高流动性大盘蓝筹,合理估值+稳定成交+防守型持有'),
 'low_volatility_quality': dict(name='低波质量', cat='质量', tech=0.30,
   hf=dict(amount_min=1e8, mcap_min=1.2e10, pe_ttm_max=45, pb_max=5.0,
           price_min=4, price_max=180, chg_min=-3.0, chg_max=5.0,
           change_60d_min=-10.0, change_60d_max=35.0, signal_min=55,
           range20_max=28.0, vol20_max=32.0, dd20_min=-8.0, atr20_max=4.5),
   fw=dict(stability=.30, value=.20, liquidity=.15, momentum=.12, activity=.08, size=.08, theme_heat=.05, reversal=.02),
   ideal_vr=1.4, ideal_tr=2.5, hint='低波动+浅回撤+估值不过热,防守型质量候选'),
 'volume_breakout': dict(name='放量突破', cat='趋势', tech=0.60,
   hf=dict(amount_min=1e8, tr_min=3.0, vr_min=2.0, chg_min=2.0, chg_max=9.9,
           above_ma20=True, signal_min=60, macd_wl=['bullish','neutral'],
           breakout20_min=-1.0, range20_max=35.0, vr20_min=1.3, body_min=0.5, consol_min=8),
   fw=dict(momentum=.32, activity=.28, liquidity=.22, theme_heat=.08, stability=.10),
   ideal_vr=3.0, ideal_tr=6.0, hint='放量突破关键阻力位,需前期横盘整理+量价同步+站上MA20'),
 'shrink_pullback': dict(name='缩量回踩', cat='趋势', tech=0.50,
   hf=dict(amount_min=8e7, tr_min=1.0, ma_bull=True, above_ma20=True, signal_min=65,
           vr20_max=1.5, pull_min=-2.0, pull_max=8.0, vol20_max=45.0, dd20_min=-12.0, atr20_max=6.5),
   fw=dict(momentum=.32, stability=.22, activity=.18, liquidity=.15, value=.08, reversal=.05),
   ideal_vr=1.0, ideal_tr=2.5, hint='均线多头+缩量回踩MA20支撑,趋势延续入场'),
 'capital_heat': dict(name='资金热度', cat='动量', tech=0.65,
   hf=dict(amount_min=3e8, tr_min=2.0, vr_min=1.5, chg_min=1.0, chg_max=9.5, price_min=3, price_max=220),
   fw=dict(momentum=.32, activity=.28, liquidity=.16, theme_heat=.10, stability=.10, reversal=.04),
   ideal_vr=2.8, ideal_tr=6.0, hint='资金活跃+量价同步但未极端过热的短线候选'),
 'oversold_reversal': dict(name='超跌反转', cat='反转', tech=0.45,
   hf=dict(amount_min=8e7, tr_min=1.0, chg_min=-8.0, chg_max=-1.0, pe_ttm_max=80, pb_max=8.0,
           price_min=3, price_max=180),
   fw=dict(reversal=.40, stability=.20, liquidity=.18, value=.16, activity=.06),
   ideal_vr=1.6, ideal_tr=3.0, hint='跌幅可控+流动性仍在+具备修复观察价值'),
 'balanced_alpha': dict(name='均衡多因子', cat='综合', tech=0.35,
   hf=dict(amount_min=1e8, mcap_min=5e9, pe_ttm_max=80, pb_max=8.0, price_min=3, price_max=220, chg_min=-4.0, chg_max=8.5),
   fw=dict(value=.22, liquidity=.18, momentum=.20, activity=.15, stability=.12, reversal=.05, theme_heat=.05, size=.03),
   ideal_vr=2.0, ideal_tr=4.0, hint='估值+资金+动量+稳定性综合,多维度都不差'),
 'momentum_quality': dict(name='趋势质量', cat='综合', tech=0.45,
   hf=dict(amount_min=2e8, mcap_min=8e9, pe_ttm_max=60, pb_max=8.0, chg_min=-3.5, chg_max=7.5, price_min=4, price_max=220),
   fw=dict(momentum=.28, value=.15, liquidity=.18, activity=.15, stability=.12, reversal=.04, theme_heat=.06, size=.02),
   ideal_vr=2.2, ideal_tr=4.5, hint='趋势确认+基本面质量约束的中线候选'),
}

# ---------------- 硬筛 ----------------
def is_excluded_global(r, sb_map):
    name = sb_map.get(r['ts_code'], {}).get('name','')
    if 'ST' in name or '退' in name:
        return True
    list_date = sb_map.get(r['ts_code'], {}).get('list_date','')
    if list_date and list_date >= NEW_STOCK_CUT:
        return True  # 次新
    if pd.isna(r.get('vol')) or r.get('vol',0) == 0 or r.get('amount',0) == 0:
        return True  # 停牌
    return False

def pass_hard(r, hf):
    p = r.get('pct_chg',0) or 0
    if p < hf.get('chg_min', -999) or p > hf.get('chg_max', 999): return False
    if r.get('close',0) < hf.get('price_min',0) or r.get('close',0) > hf.get('price_max',1e9): return False
    amt = r.get('amount',0)*1000  # 千元->元
    if amt < hf.get('amount_min',0): return False
    pe = r.get('pe_ttm'); 
    if 'pe_ttm_max' in hf and (pd.isna(pe) or pe > hf['pe_ttm_max'] or pe < hf.get('pe_ttm_min',0)): 
        if pd.isna(pe) or pe<=0: return False
        if pe > hf['pe_ttm_max']: return False
        if pe < hf.get('pe_ttm_min',0): return False
    pb = r.get('pb')
    if 'pb_max' in hf and (pd.isna(pb) or pb > hf['pb_max'] or pb < hf.get('pb_min',0)):
        if pd.isna(pb) or pb<=0: return False
        if pb > hf['pb_max']: return False
    mv = r.get('total_mv',0)*10000  # 万元->元
    if mv < hf.get('mcap_min',0): return False
    if 'mcap_max' in hf and mv > hf['mcap_max']: return False
    if 'tr_min' in hf and (pd.isna(r.get('turnover_rate')) or r['turnover_rate'] < hf['tr_min']): return False
    if 'vr_min' in hf and (pd.isna(r.get('volume_ratio')) or r['volume_ratio'] < hf['vr_min']): return False
    if hf.get('above_ma20') and not r.get('price_above_ma20', False): return False
    if hf.get('ma_bull') and not r.get('ma_bullish', False): return False
    if 'signal_min' in hf and r.get('signal_score',0) < hf['signal_min']: return False
    if 'macd_wl' in hf and r.get('macd_status') not in hf['macd_wl']: return False
    if 'breakout20_min' in hf and (pd.isna(r.get('breakout_20d_pct')) or r['breakout_20d_pct'] < hf['breakout20_min']): return False
    if 'range20_max' in hf and (pd.isna(r.get('range_20d_pct')) or r['range_20d_pct'] > hf['range20_max']): return False
    if 'vr20_min' in hf and (pd.isna(r.get('volume_ratio_20d')) or r['volume_ratio_20d'] < hf['vr20_min']): return False
    if 'vr20_max' in hf and (pd.isna(r.get('volume_ratio_20d')) or r['volume_ratio_20d'] > hf['vr20_max']): return False
    if 'body_min' in hf and (pd.isna(r.get('body_pct')) or r['body_pct'] < hf['body_min']): return False
    if 'consol_min' in hf and (pd.isna(r.get('consol20')) or r['consol20'] < hf['consol_min']): return False
    if 'change_60d_min' in hf:
        c60 = r.get('change_60d')
        if pd.isna(c60) or c60 < hf['change_60d_min'] or c60 > hf.get('change_60d_max',999): return False
    if 'vol20_max' in hf and (pd.isna(r.get('volatility_20d_pct')) or r['volatility_20d_pct'] > hf['vol20_max']): return False
    if 'dd20_min' in hf and (pd.isna(r.get('max_drawdown_20d_pct')) or r['max_drawdown_20d_pct'] < hf['dd20_min']): return False
    if 'atr20_max' in hf and (pd.isna(r.get('atr_20_pct')) or r['atr_20_pct'] > hf['atr20_max']): return False
    if 'pull_min' in hf:
        pl = r.get('pullback_to_ma20_pct')
        if pd.isna(pl) or pl < hf['pull_min'] or pl > hf.get('pull_max',999): return False
    return True

# ---------------- 评分 ----------------
def score_row(r, fw, ideal_vr, ideal_tr):
    r = dict(r); r['_ideal_vr']=ideal_vr; r['_ideal_tr']=ideal_tr
    factor_vals = {k: fn(r) for k,fn in FACTOR_FN.items() if k in fw}
    tech = r.get('signal_score',50)/100.0
    base = sum(factor_vals[k]*fw[k] for k in factor_vals)  # 加权0-95
    final = base*(1-0.0) + tech*0  # 主因子加权为主
    # 加入技术分作为加分项(小权重平滑)
    final = base*0.85 + tech*100*0.15
    return round(float(final),2)

# ---------------- 主流程 ----------------
def main():
    print('[1] 拉取 stock_basic ...')
    sb = fetch_stock_basic()
    sb_map = sb.set_index('ts_code').to_dict('index')
    print(f'    上市股票数: {len(sb)}')

    print('[2] 获取交易日历 ...')
    dates = fetch_trade_dates()
    print(f'    取最近 {len(dates)} 个交易日: {dates[0]} ~ {dates[-1]}')

    print('[3] 拉取日线面板(全市场) ...')
    panel = fetch_daily_panel(dates)
    print(f'    日线记录数: {len(panel)}')

    print('[4] 拉取 daily_basic (最新交易日) ...')
    db = fetch_daily_basic(LATEST_DATE)
    print(f'    daily_basic 记录数: {len(db)}')

    print('[5] 计算技术指标 ...')
    latest, full = compute_indicators(panel)
    # 合并 daily_basic 基本面
    latest = latest.merge(db[['ts_code','turnover_rate','turnover_rate_f','volume_ratio',
        'pe','pe_ttm','pb','ps_ttm','dv_ttm','total_mv','circ_mv','limit_status']],
        on='ts_code', how='left', suffixes=('','_db'))
    # 衍生数值
    latest['amount_yi'] = latest['amount']/100000.0        # 千元->亿元
    latest['total_mv_yi'] = latest['total_mv']/10000.0    # 万元->亿元
    latest['signal_score'] = latest.apply(signal_score, axis=1)
    print(f'    最新日标的数: {len(latest)}')

    # 市场概况
    mkt = latest.dropna(subset=['pct_chg'])
    up = (mkt['pct_chg']>0).sum(); down=(mkt['pct_chg']<0).sum(); flat=(mkt['pct_chg']==0).sum()
    avg_chg = mkt['pct_chg'].mean(); med_chg = mkt['pct_chg'].median()
    limit_up = ((mkt['pct_chg']>=9.8)).sum()
    print(f'    市场: 上涨{up} 下跌{down} 平{flat} 均涨跌{avg_chg:.2f}%')

    # 行业强弱
    latest['_industry'] = latest['ts_code'].map(lambda c: sb_map.get(c,{}).get('industry','未知'))
    ind_grp = latest.groupby('_industry')['pct_chg'].agg(['mean','count']).rename(columns={'mean':'avg_chg'})
    ind_grp = ind_grp[ind_grp['count']>=10].sort_values('avg_chg', ascending=False)

    # 跑策略
    results = {}
    for key, st in STRATS.items():
        hf = st['hf']; fw = st['fw']; ivr=st['ideal_vr']; itr=st['ideal_tr']
        cand = []
        for _, r in latest.iterrows():
            if is_excluded_global(r, sb_map):
                continue
            if not pass_hard(r, hf):
                continue
            sc = score_row(r, fw, ivr, itr)
            cand.append((r['ts_code'], sc, r))
        cand.sort(key=lambda x: x[1], reverse=True)
        top = cand[:10]
        if not top:  # 放宽: 取最接近的前5
            relaxed = []
            for _, r in latest.iterrows():
                if is_excluded_global(r, sb_map): continue
                sc = score_row(r, fw, ivr, itr)
                relaxed.append((r['ts_code'], sc, r))
            relaxed.sort(key=lambda x:x[1], reverse=True)
            top = relaxed[:5]
            results[key] = (st, top, True)  # relaxed
        else:
            results[key] = (st, top, False)
        print(f"    [{key}] {st['name']}: 命中{len(cand)}只, Top={top[0][0] if top else '-'} ({top[0][1] if top else '-'})")

    # 保存中间结果
    out = []
    for key,(st,top,relaxed) in results.items():
        rows=[]
        for code,sc,r in top:
            rows.append(dict(ts_code=code, score=sc,
                name=sb_map.get(code,{}).get('name',''),
                industry=sb_map.get(code,{}).get('industry',''),
                close=round(r.get('close',0),2), pct_chg=round(r.get('pct_chg',0),2),
                amount_yi=round(r.get('amount_yi',0),2),
                pe_ttm=round(r.get('pe_ttm',0),1) if pd.notna(r.get('pe_ttm')) else None,
                pb=round(r.get('pb',0),2) if pd.notna(r.get('pb')) else None,
                total_mv_yi=round(r.get('total_mv_yi',0),0) if pd.notna(r.get('total_mv_yi')) else None,
                turnover_rate=round(r.get('turnover_rate',0),2) if pd.notna(r.get('turnover_rate')) else None,
                volume_ratio=round(r.get('volume_ratio',0),2) if pd.notna(r.get('volume_ratio')) else None,
                rsi14=round(r.get('rsi14',0),1) if pd.notna(r.get('rsi14')) else None,
                macd_status=r.get('macd_status'),
                ma_bullish=bool(r.get('ma_bullish',False)),
                dist_ma20=round(r.get('dist_ma20_pct',0),2) if pd.notna(r.get('dist_ma20_pct')) else None,
                signal_score=round(r.get('signal_score',0),1)))
        out.append(dict(key=key, name=st['name'], cat=st['cat'], hint=st['hint'], relaxed=relaxed, rows=rows))
    summary = dict(
        latest_date=LATEST_DATE, today=TODAY,
        market=dict(up=int(up), down=int(down), flat=int(flat), avg_chg=round(float(avg_chg),2),
                    med_chg=round(float(med_chg),2), limit_up=int(limit_up), total=int(len(mkt))),
        industries=[dict(industry=i, avg_chg=round(float(v['avg_chg']),2), count=int(v['count']))
                    for i,v in ind_grp.head(8).iterrows()],
        industries_weak=[dict(industry=i, avg_chg=round(float(v['avg_chg']),2), count=int(v['count']))
                    for i,v in ind_grp.tail(5).iterrows()],
        strategies=out,
    )
    with open(os.path.join(CACHE_DIR,'result.json'),'w') as f:
        json.dump(summary, f, ensure_ascii=False, indent=2, default=str)
    print('[DONE] 结果已保存 .cache/result.json')
    return summary

if __name__ == '__main__':
    main()
