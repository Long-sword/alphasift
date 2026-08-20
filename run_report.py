# -*- coding: utf-8 -*-
"""
13:30 盘中选股策略报告 - 10策略并行筛选
数据源:
  - 盘中实时行情(最新价/涨跌/成交额/换手率/量比/PE): 腾讯财经 qt.gtimg.cn (今日盘中)
  - 历史日线(技术指标计算): Tushare (前61交易日)
  - 估值(PB/市值/股息率): Tushare daily_basic (最近已发布交易日)
"""
import os, sys, json, time, math, re, warnings, datetime
warnings.filterwarnings("ignore")
import numpy as np
import pandas as pd
import requests
import tushare as ts

TS_TOKEN = '9f640d421f866dde9a1888ab4193f77c43659d47446d717ffa343023'
ts.set_token(TS_TOKEN)
pro = ts.pro_api()

TODAY = datetime.date.today().strftime("%Y%m%d")
HIST_DAYS = 62  # 历史交易日长度(前61日用于技术指标, 第62日=今日实时)
GLOBAL_AMOUNT_MIN = 2e8  # 全局成交额下限 2亿(元)

def log(*a):
    print(*a, file=sys.stderr, flush=True)

# ---------------- 1. 交易日历(确认今日交易) & 历史日期 ----------------
cal = pro.trade_cal(exchange='SSE', start_date='20260501', end_date=TODAY)
cal = cal[cal['is_open'] == 1].sort_values('cal_date').reset_index(drop=True)
today_open = TODAY in set(cal['cal_date'].tolist())
log('今日', TODAY, 'is_open=', 1 if today_open else 0)

# 今日实时基准日: TODAY; 历史用今日之前最近 HIST_DAYS-1 个交易日
hist_dates = cal[cal['cal_date'] < TODAY]['cal_date'].tolist()[-(HIST_DAYS-1):]
if len(hist_dates) < 20:
    hist_dates = cal['cal_date'].tolist()[-(HIST_DAYS-1):]
PREV_TRADE_DATE = hist_dates[-1]  # 昨一交易日(估值基准)
log('历史交易日(技术面):', len(hist_dates), '从', hist_dates[0], '至', hist_dates[-1],
    '| 今日实时基准:', TODAY, '| 估值基准日:', PREV_TRADE_DATE)
latest = TODAY  # 报告日期=今日

# ---------------- 2. 拉取历史日线(按交易日, 多线程) ----------------
from concurrent.futures import ThreadPoolExecutor
DAILY_FIELDS = 'ts_code,trade_date,open,high,low,close,pre_close,pct_chg,vol,amount'

def fetch_daily(d):
    for _ in range(3):
        try:
            df = pro.daily(trade_date=d, fields=DAILY_FIELDS.split(','))
            return df
        except Exception as e:
            time.sleep(1.0)
    return pd.DataFrame()

t0 = time.time()
with ThreadPoolExecutor(max_workers=8) as ex:
    res = list(ex.map(fetch_daily, hist_dates))
daily_hist = pd.concat([r for r in res if r is not None and len(r)], ignore_index=True)
log('历史日线拉取完成:', len(daily_hist), '行 耗时%.1fs' % (time.time()-t0))
daily_hist['trade_date'] = daily_hist['trade_date'].astype(str)

# ---------------- 2b. 腾讯财经: 全市场盘中实时行情 ----------------
def fetch_qq_realtime(codes_qq):
    """codes_qq: ['sh600519','sz000001',...] 返回 DataFrame"""
    all_rows = []
    for i in range(0, len(codes_qq), 200):
        batch = codes_qq[i:i+200]
        try:
            r = requests.get('http://qt.gtimg.cn/q='+','.join(batch), timeout=15)
            for line in r.text.strip().split(';\n'):
                if not line.strip(): continue
                m = re.search(r'v_(sh|sz)(\d+)="([^"]*)"', line)
                if not m: continue
                f = m.group(3).split('~')
                if len(f) < 40: continue
                def num(x):
                    try: return float(x)
                    except: return np.nan
                code6 = m.group(2)
                last = num(f[3])
                if not (last > 0): continue  # 停牌/无效
                pre_close = num(f[4])
                opn = num(f[5])
                vol_hand = num(f[6])  # 手
                ts = f[30] if len(f)>30 else ''
                pct = num(f[32])
                high = num(f[33]) if len(f)>33 else np.nan
                low = num(f[34]) if len(f)>34 else np.nan
                turnover = num(f[38]) if len(f)>38 else np.nan  # 换手率%
                pe = num(f[39]) if len(f)>39 else np.nan
                amt_wan = num(f[37]) if len(f)>37 else np.nan  # 成交额(万元) - 腾讯统一万元单位
                amount = amt_wan * 1e4  # 元
                all_rows.append(dict(code6=code6, name_qq=f[1] if len(f)>1 else '',
                    last=last, pre_close=pre_close, open=opn, high=high, low=low,
                    vol_hand=vol_hand, amount=amount, pct=pct, turnover=turnover,
                    pe_ttm=pe, ts=ts))
        except Exception as e:
            log('  腾讯批量失败 i=%d %s' % (i, str(e)[:80]))
    return pd.DataFrame(all_rows)

# 股票列表
sb = pro.stock_basic(list_status='L',
    fields=['ts_code','symbol','name','industry','market','list_date','delist_date'])
sb['code6'] = sb['ts_code'].str[:6]
def to_qq(tc): return ('sh' if tc.endswith('.SH') else 'sz') + tc[:6]
codes_qq = sb['ts_code'].apply(to_qq).tolist()
t0 = time.time()
rt = fetch_qq_realtime(codes_qq)
log('腾讯实时拉取: %d 只 耗时%.1fs' % (len(rt), time.time()-t0))
if len(rt) == 0:
    log('FATAL: 腾讯实时数据为空, 无法生成盘中报告')
    sys.exit(1)
# 验证最新时间戳
sample_ts = rt['ts'].dropna().iloc[0] if len(rt) else ''
log('实时数据时间戳样本:', sample_ts)

# ---------------- 2c. 把今日实时数据拼成"今日K线"接到历史末尾 ----------------
# 今日K线: open/high/low/close=最新价, pre_close=昨收, pct_chg=涨跌幅, vol=手, amount=元
rt_map = rt.set_index('code6')
today_klines = []
for _, srow in sb.iterrows():
    c6 = srow['code6']
    if c6 not in rt_map.index: continue
    rr = rt_map.loc[c6]
    if isinstance(rr, pd.DataFrame): rr = rr.iloc[0]
    today_klines.append(dict(
        ts_code=srow['ts_code'], trade_date=TODAY,
        open=rr['open'], high=rr['high'], low=rr['low'], close=rr['last'],
        pre_close=rr['pre_close'], pct_chg=rr['pct'],
        vol=rr['vol_hand'], amount=rr['amount'],
        turnover=rr['turnover'], pe_ttm_rt=rr['pe_ttm']))
today_df_rt = pd.DataFrame(today_klines)
log('今日实时K线拼装:', len(today_df_rt), '只')

# 合并历史+今日
daily_hist = pd.concat([daily_hist, today_df_rt[['ts_code','trade_date','open','high','low','close','pre_close','pct_chg','vol','amount']]],
                       ignore_index=True)
daily_hist['vol'] = daily_hist['vol'].astype(float)
daily_hist['amount'] = daily_hist['amount'].astype(float)
daily_hist = daily_hist.sort_values(['ts_code','trade_date']).reset_index(drop=True)

# 当日(今日实时)行情 & 昨日(用于成交额对比)
day_df = today_df_rt.copy()
day_df['pct_chg'] = day_df['pct_chg'].astype(float)
day_df['amount'] = day_df['amount'].astype(float)
prev_date = hist_dates[-1]
prev_df = daily_hist[daily_hist['trade_date'] == prev_date].copy()
log('最新交易日(今日实时)股票数:', len(day_df), ' 前一交易日:', prev_date, len(prev_df))

# ---------------- 3. 估值(daily_basic 用昨交易日, PB/市值/股息率日内变化小) & 股票列表 ----------------
db = pro.daily_basic(trade_date=PREV_TRADE_DATE,
    fields=['ts_code','trade_date','close','turnover_rate','turnover_rate_f','volume_ratio',
            'pe','pe_ttm','pb','ps','ps_ttm','dv_ratio','dv_ttm','total_share','float_share',
            'free_share','total_mv','circ_mv'])
log('daily_basic 行数:', 0 if db is None else len(db))
db['total_mv'] = db['total_mv'].astype(float)  # 万元
db['circ_mv'] = db['circ_mv'].astype(float)
db['pe_ttm'] = db['pe_ttm'].astype(float)
db['pb'] = db['pb'].astype(float)
db['turnover_rate'] = db['turnover_rate'].astype(float)
db['volume_ratio'] = db['volume_ratio'].astype(float)
db['dv_ttm'] = db['dv_ttm'].astype(float)

sb = pro.stock_basic(exchange='', list_status='L',
    fields=['ts_code','symbol','name','area','industry','market','list_date','delist_date'])
sb['list_date'] = sb['list_date'].astype(str)
log('stock_basic 上市股票数:', len(sb))

# ---------------- 4. 向量化技术指标(pivot 宽表) ----------------
log('计算技术指标...')
# ffill 价格序列(停牌日用前收盘填充),避免 EWM/EMA 出现 NaN 传播
pclose = daily_hist.pivot(index='trade_date', columns='ts_code', values='close').astype(float).ffill()
phigh  = daily_hist.pivot(index='trade_date', columns='ts_code', values='high').astype(float).ffill()
plow   = daily_hist.pivot(index='trade_date', columns='ts_code', values='low').astype(float).ffill()
popen  = daily_hist.pivot(index='trade_date', columns='ts_code', values='open').astype(float).ffill()
ppre   = daily_hist.pivot(index='trade_date', columns='ts_code', values='pre_close').astype(float).ffill()
pvol   = daily_hist.pivot(index='trade_date', columns='ts_code', values='vol').astype(float).fillna(0)
pamt   = daily_hist.pivot(index='trade_date', columns='ts_code', values='amount').astype(float).fillna(0)
ppct   = daily_hist.pivot(index='trade_date', columns='ts_code', values='pct_chg').astype(float).fillna(0)

codes = pclose.columns
last_close = pclose.iloc[-1]
# 有效历史天数(最近30个交易日非空收盘)
valid30 = pclose.iloc[-30:].notna().sum()

def last_series(s):
    return s.iloc[-1] if isinstance(s, pd.Series) else s

# 均线
ma5  = pclose.iloc[-5:].mean()
ma10 = pclose.iloc[-10:].mean()
ma20 = pclose.iloc[-20:].mean()
ma_bullish = (ma5 > ma10) & (ma10 > ma20)
above_ma20 = last_close > ma20
dist_ma20 = (last_close - ma20) / ma20 * 100
pct_today = ppct.iloc[-1]

# 收益
ret_5d  = (last_close / pclose.iloc[-6] - 1) * 100
ret_20d = (last_close / pclose.iloc[-21] - 1) * 100
ret_60d = (last_close / pclose.iloc[-61] - 1) * 100 if len(pclose) >= 61 else pd.Series(np.nan, index=codes)

# MACD (EMA)
ema12 = pclose.ewm(span=12, adjust=False, ignore_na=True).mean()
ema26 = pclose.ewm(span=26, adjust=False, ignore_na=True).mean()
dif = ema12 - ema26
dea = dif.ewm(span=9, adjust=False, ignore_na=True).mean()
hist_macd = (dif - dea) * 2
dif_v = dif.iloc[-1]; dea_v = dea.iloc[-1]; hist_v = hist_macd.iloc[-1]
dif_prev = dif.iloc[-2]; dea_prev = dea.iloc[-2]
macd_bullish = (dif_v > dea_v) & (dif_v > 0)
macd_neutral = (dif_v > dea_v) & (dif_v <= 0)
macd_bearish = (dif_v < dea_v)
macd_hist_pos = hist_v > 0
macd_golden = (dif_v > dea_v) & (dif_prev <= dea_prev)

# RSI 14 (SMA 式)
delta = pclose.diff()
gain = delta.clip(lower=0)
loss = (-delta).clip(lower=0)
avg_gain = gain.iloc[-14:].mean()
avg_loss = loss.iloc[-14:].mean()
rs = avg_gain / avg_loss.replace(0, np.nan)
rsi = 100 - 100 / (1 + rs)
rsi = rsi.where(avg_loss > 0, 100.0)
rsi = rsi.where(~((avg_gain == 0) & (avg_loss == 0)), np.nan)

# 波动率20日(年化%); ppct 已为百分数, std*sqrt(250) 即年化波动率%
vol_20 = ppct.iloc[-20:].std(ddof=1) * np.sqrt(250)

# 20日振幅
hi20 = phigh.iloc[-20:].max()
lo20 = plow.iloc[-20:].min()
range_20 = (hi20 - lo20) / lo20 * 100

# 20日最大回撤
c20 = pclose.iloc[-20:]
rmax = c20.cummax()
dd = (c20 - rmax) / rmax * 100
mdd = dd.min()

# ATR 14 (%)
tr_arr = np.maximum.reduce([
    (phigh - plow).values,
    np.abs(phigh - ppre).values,
    np.abs(plow - ppre).values
])
tr = pd.DataFrame(tr_arr, index=phigh.index, columns=phigh.columns)
atr = tr.iloc[-14:].mean()
atr_pct = atr / last_close * 100

# 20日突破幅度(对比前20日最高, 不含今日)
prior_hi = phigh.iloc[-21:-1].max()
breakout_20 = (last_close - prior_hi) / prior_hi * 100

# 20日量比(今日量/前20日均量)
vrat_20 = pvol.iloc[-1] / pvol.iloc[-21:-1].mean()

# 实体幅度%
body = (last_close - popen.iloc[-1]).abs() / ppre.iloc[-1] * 100

# 20日整理天数(日内振幅<4%)
rng_day = (phigh - plow) / ppre * 100
consol = (rng_day.iloc[-20:] < 4).sum()

log('技术指标计算完成, 有效标的数:', int((valid30 >= 20).sum()))

# ---------------- 5. 组装特征表 ----------------
feat = pd.DataFrame(index=codes).reset_index().rename(columns={'index':'ts_code'})
feat['ts_code'] = codes
feat = feat.merge(sb[['ts_code','name','industry','market','list_date']], on='ts_code', how='left')
feat['close'] = last_close.values
feat['pct_chg'] = pct_today.values
feat['amount'] = pamt.iloc[-1].values  # 元(实时)
feat['amount_yi'] = feat['amount'] / 1e8  # 亿
feat['prev_amount_yi'] = (pamt.iloc[-2].reindex(codes).values * 1e3) / 1e8  # 千元->元->亿
# 今日盘中实时值(腾讯): PE_TTM、换手率；量比=今日成交额/近5日均成交额(自算)
rt_feat = today_df_rt.set_index('ts_code')
feat['pe_ttm'] = rt_feat['pe_ttm_rt'].reindex(codes).values  # 实时PE(腾讯)
feat['turnover_rate'] = rt_feat['turnover'].reindex(codes).values  # 实时换手率
# 量比: 今日成交额(腾讯,元) / 前5日均成交额(Tushare,千元*1e3=元) -> 同单位
amt_today = pamt.iloc[-1].reindex(codes)  # 今日(元)
amt_5d_avg = (pamt.iloc[-6:-1] * 1e3).mean().reindex(codes)  # 前5日 千元->元 后平均
feat['vol_ratio'] = (amt_today / amt_5d_avg).values
# PB/市值/股息率用昨交易日daily_basic(日内稳定)
feat['pb'] = db.set_index('ts_code')['pb'].reindex(codes).values
feat['total_mv_yi'] = db.set_index('ts_code')['total_mv'].reindex(codes).values / 10000.0  # 亿
feat['circ_mv_yi'] = db.set_index('ts_code')['circ_mv'].reindex(codes).values / 10000.0
feat['dv_ttm'] = db.set_index('ts_code')['dv_ttm'].reindex(codes).values
feat['ma5']=ma5.values; feat['ma10']=ma10.values; feat['ma20']=ma20.values
feat['ma_bullish']=ma_bullish.values; feat['above_ma20']=above_ma20.values
feat['dist_ma20']=dist_ma20.values
feat['ret_5d']=ret_5d.values; feat['ret_20d']=ret_20d.values; feat['ret_60d']=ret_60d.values
feat['dif']=dif_v.values; feat['dea']=dea_v.values; feat['macd_hist']=hist_v.values
feat['macd_bullish']=macd_bullish.values; feat['macd_neutral']=macd_neutral.values
feat['macd_bearish']=macd_bearish.values; feat['macd_hist_pos']=macd_hist_pos.values
feat['macd_golden']=macd_golden.values
feat['rsi']=rsi.values
feat['vol_20']=vol_20.values; feat['range_20']=range_20.values; feat['mdd']=mdd.values
feat['atr_pct']=atr_pct.values; feat['breakout_20']=breakout_20.values
feat['vrat_20']=vrat_20.values; feat['body']=body.values; feat['consol']=consol.values
feat['valid30']=valid30.values

# 行业当日平均涨跌(主题热度)
ind_ret = day_df.merge(sb[['ts_code','industry']], on='ts_code', how='left').groupby('industry')['pct_chg'].mean()
feat['ind_ret'] = feat['industry'].map(ind_ret).fillna(0)

# ---------------- 6. 全局排除 ----------------
# 次新(上市<1年)
cutoff_list = (datetime.datetime.strptime(latest,'%Y%m%d') - datetime.timedelta(days=365)).strftime('%Y%m%d')
is_st = feat['name'].astype(str).str.contains('ST', na=False) | feat['name'].astype(str).str.contains('退', na=False)
is_subnew = feat['list_date'].astype(str).fillna('00000000') >= cutoff_list
low_amt = feat['amount_yi'] * 1e8 < GLOBAL_AMOUNT_MIN  # 元
no_trade = feat['amount'].fillna(0) <= 0
short_hist = feat['valid30'] < 20
# 停牌: 当日无成交已含于 no_trade
global_keep = (~is_st) & (~is_subnew) & (~low_amt) & (~no_trade) & (~short_hist) & feat['close'].notna()
log('全局排除后剩余:', int(global_keep.sum()), ' ST:', int(is_st.sum()), ' 次新:', int(is_subnew.sum()),
    ' 成交<2亿:', int(low_amt.sum()), ' 历史不足:', int(short_hist.sum()))
feat_g = feat[global_keep].copy()

# ---------------- 7. 策略配置 ----------------
STRATEGIES = {
 'dual_low': dict(name='dual_low', display='双低选股', category='value', tech=0.20,
   hf=dict(amount_min=5e7, pe_max=15, pb_max=2.0, mcap_min=5e9, mcap_max=3e11, price_min=3, price_max=80, chg_min=-4.5, chg_max=4.5),
   fw=dict(value=.34, stability=.20, liquidity=.14, momentum=.10, activity=.10, reversal=.06, size=.06),
   sp=dict(chase_start=2.5, slope=18, downside_start=-2.5, ideal_vr=1.2, ideal_tr=2.0, high_vr=4.0, high_tr=8.0, sta_hot=4.0),
   rp=dict(chase=5.0, abn_vr=4.0, high_tr=8.0),
   sc=dict(vq_value=72, vq_stab=65, vq_bonus=3.0, vol_spike=4.0, vol_spike_pen=1.5)),
 'quality_value': dict(name='quality_value', display='稳健价值', category='value', tech=0.15,
   hf=dict(amount_min=8e7, pe_max=25, pb_max=4.0, mcap_min=1e10, mcap_max=8e11, chg_min=-3.5, chg_max=5.0, price_min=3, price_max=180),
   fw=dict(value=.32, stability=.24, liquidity=.18, momentum=.08, activity=.08, reversal=.04, size=.06),
   sp=dict(chase_start=3.0, slope=16, ideal_vr=1.4, ideal_tr=2.5, sta_hot=4.5),
   rp=dict(chase=5.5, abn_vr=4.5, high_tr=10),
   sc=dict(vq_value=72, vq_stab=68, vq_bonus=2.8, vol_spike=4.0, vol_spike_pen=1.5)),
 'blue_chip_income': dict(name='blue_chip_income', display='蓝筹收益质量', category='income', tech=0.18,
   hf=dict(amount_min=1.2e8, mcap_min=3e10, pe_max=22, pb_max=3.2, tr_min=0.4, vr_min=0.6, chg_min=-3.0, chg_max=4.0, price_min=3, price_max=180),
   fw=dict(value=.30, stability=.26, liquidity=.18, size=.12, activity=.06, momentum=.05, reversal=.03),
   sp=dict(chase_start=2.8, slope=18, downside_start=-3.0, ideal_vr=1.1, ideal_tr=1.8, high_vr=3.5, high_tr=7.0, sta_hot=4.0),
   rp=dict(chase=4.8, abn_vr=3.5, high_tr=8.0),
   sc=dict(vq_value=74, vq_stab=70, vq_bonus=2.5, vol_spike=3.5, vol_spike_pen=1.5)),
 'low_volatility_quality': dict(name='low_volatility_quality', display='低波质量', category='quality', tech=0.30,
   hf=dict(amount_min=1e8, mcap_min=1.2e10, pe_max=45, pb_max=5.0, chg_min=-3.0, chg_max=5.0, price_min=4, price_max=180,
           chg60_min=-10, chg60_max=35, signal_min=55, range20_max=28, vol20_max=32, mdd_min=-8, atr_max=4.5),
   fw=dict(stability=.30, value=.20, liquidity=.15, momentum=.12, activity=.08, size=.08, theme_heat=.05, reversal=.02),
   sp=dict(chase_start=4.0, slope=18, ideal_vr=1.4, ideal_tr=2.5, sta_hot=5.0, high_vol=30, mdd_floor=-8, high_atr=4.2),
   rp=dict(chase=5.5, abn_vr=4.5, high_tr=10, weak_signal=52),
   sc=dict(vq_stab=72, vq_bonus=2.2, hot_money_pen=2.8, vol_spike=4.0, vol_spike_pen=1.8, hot_money_min=92)),
 'volume_breakout': dict(name='volume_breakout', display='放量突破', category='trend', tech=0.60,
   hf=dict(amount_min=1e8, tr_min=3.0, vr_min=2.0, chg_min=2.0, chg_max=9.9, above_ma20=True, signal_min=60,
           macd_wl=['bullish','neutral'], breakout_min=-1.0, range20_max=35, vrat20_min=1.3, body_min=0.5, consol_min=8),
   fw=dict(momentum=.32, activity=.28, liquidity=.22, theme_heat=.08, stability=.10),
   sp=dict(chase_start=7.0, slope=11, ideal_vr=3.0, high_vr=9.0, ideal_tr=6.0, high_tr=20, sta_hot=8.8),
   rp=dict(chase=9.8, abn_vr=9.0, high_tr=24, weak_signal=45),
   sc=dict(cap_bonus=2.2, vol_spike=9.0, hot_money_min=96, hot_money_pen=1.4)),
 'shrink_pullback': dict(name='shrink_pullback', display='缩量回踩', category='trend', tech=0.50,
   hf=dict(amount_min=8e7, tr_min=1.0, ma_bull=True, above_ma20=True, signal_min=65, vrat20_max=1.5,
           pullback_min=-2.0, pullback_max=8.0, vol20_max=45, mdd_min=-12, atr_max=6.5),
   fw=dict(momentum=.32, stability=.22, activity=.18, liquidity=.15, value=.08, reversal=.05),
   sp=dict(chase_start=5.0, slope=14, ideal_vr=1.0, ideal_tr=2.5, high_vr=4.5, sta_hot=5.5, high_vol=40, mdd_floor=-10, high_atr=5.5),
   rp=dict(chase=6.5, abn_vr=4.5, high_tr=12, weak_signal=60),
   sc=dict(cap_bonus=1.4, vol_spike=4.5, vol_spike_pen=1.5)),
 'capital_heat': dict(name='capital_heat', display='资金热度', category='momentum', tech=0.65,
   hf=dict(amount_min=3e8, tr_min=2.0, vr_min=1.5, chg_min=1.0, chg_max=9.5, price_min=3, price_max=220),
   fw=dict(momentum=.32, activity=.28, liquidity=.16, theme_heat=.10, stability=.10, reversal=.04),
   sp=dict(chase_start=6.5, slope=12, ideal_vr=2.8, high_vr=7.0, ideal_tr=6.0, high_tr=18, sta_hot=8.5, theme_overheat=92, theme_slope=0.35),
   rp=dict(chase=9.3, abn_vr=8.0, high_tr=22, weak_signal=40),
   sc=dict(cap_bonus=2.4, vol_spike=8.0, vol_spike_pen=1.5, hot_money_min=95, hot_money_pen=1.5)),
 'oversold_reversal': dict(name='oversold_reversal', display='超跌反转', category='reversal', tech=0.45,
   hf=dict(amount_min=8e7, tr_min=1.0, chg_min=-8.0, chg_max=-1.0, pe_max=80, pb_max=8.0, price_min=3, price_max=180),
   fw=dict(reversal=.40, stability=.20, liquidity=.18, value=.16, activity=.06),
   sp=dict(rev_ideal=-3.5, rev_collapse=-7.0, rev_slope=12, rev_chase=0.0, rsi_bonus=14, downside_start=-4.0, ideal_vr=1.6, ideal_tr=3.0),
   rp=dict(breakdown=-8.5, chase=5.0, abn_vr=5.0, high_tr=12, weak_signal=35),
   sc=dict(ctrl_rev_min=70, ctrl_rev_bonus=2.0, vol_spike=5.5, vol_spike_pen=1.5)),
 'balanced_alpha': dict(name='balanced_alpha', display='均衡多因子', category='framework', tech=0.35,
   hf=dict(amount_min=1e8, mcap_min=5e9, pe_max=80, pb_max=8.0, chg_min=-4.0, chg_max=8.5, price_min=3, price_max=220),
   fw=dict(value=.22, liquidity=.18, momentum=.20, activity=.15, stability=.12, reversal=.05, theme_heat=.05, size=.03),
   sp=dict(chase_start=5.5, slope=12, ideal_vr=2.0, ideal_tr=4.0, sta_hot=7.0),
   rp=dict(chase=8.0, abn_vr=6.0, high_tr=15),
   sc=dict(vq_bonus=2.2, cap_bonus=1.8, vol_spike=5.5, vol_spike_pen=1.5, hot_money_pen=2.2, hot_money_min=92)),
 'momentum_quality': dict(name='momentum_quality', display='趋势质量', category='framework', tech=0.45,
   hf=dict(amount_min=2e8, mcap_min=8e9, pe_max=60, pb_max=8.0, chg_min=-3.5, chg_max=7.5, price_min=4, price_max=220),
   fw=dict(momentum=.28, value=.15, liquidity=.18, activity=.15, stability=.12, reversal=.04, theme_heat=.06, size=.02),
   sp=dict(chase_start=5.0, slope=14, ideal_vr=2.2, ideal_tr=4.5, sta_hot=6.5),
   rp=dict(chase=7.5, abn_vr=5.5, high_tr=14),
   sc=dict(vq_bonus=1.8, cap_bonus=2.4, vol_spike=5.0, vol_spike_pen=1.5, hot_money_pen=2.5, hot_money_min=92)),
}
STRATEGY_ORDER = ['dual_low','quality_value','blue_chip_income','low_volatility_quality',
                  'volume_breakout','shrink_pullback','capital_heat','oversold_reversal',
                  'balanced_alpha','momentum_quality']

def clip(s, lo, hi):
    return s.clip(lower=lo, upper=hi)

def factor_scores(df):
    """返回各因子原始分(0-100) DataFrame, 同索引"""
    f = pd.DataFrame(index=df.index)
    pe = df['pe_ttm']; pb = df['pb']
    pe_max = df.attrs.get('pe_max', 80); pb_max = df.attrs.get('pb_max', 8.0)
    pe_s = np.where(pe > 0, clip(100*(1 - pe/pe_max), 0, 100), 35.0)
    pb_s = clip(100*(1 - pb/pb_max), 0, 100)
    f['value'] = 0.6*pe_s + 0.4*pb_s
    # momentum
    mom = 50 + clip(df['ret_20d'],-20,20)*1.2 + clip(df['ret_5d'],-10,10)*0.6
    mom += np.where(df['ma_bullish'], 8, -5)
    mom += np.where(df['macd_hist_pos'], 6, -3)
    mom += np.where(df['macd_golden'], 4, 0)
    mom += np.where(df['above_ma20'], 4, -6)
    f['momentum'] = clip(mom, 0, 100)
    # activity
    ivr = df.attrs['ideal_vr']; itr = df.attrs['ideal_tr']
    vr = df['vol_ratio'].fillna(1.0); tr = df['turnover_rate'].fillna(itr)
    vr_s = 100*np.exp(-((vr - ivr)/(ivr*1.3))**2)
    tr_s = 100*np.exp(-((tr - itr)/(itr*1.3))**2)
    f['activity'] = clip(0.5*vr_s + 0.5*tr_s, 0, 100)
    # liquidity
    f['liquidity'] = clip(100*(1 - np.exp(-df['amount_yi']/20.0)), 0, 100)
    # stability
    vol_s = clip(100 - df['vol_20']*1.5, 0, 100)
    dd_s = clip(100 + df['mdd']*1.5, 0, 100)
    atr_s = clip(100 - df['atr_pct']*10, 0, 100)
    f['stability'] = clip(0.4*vol_s + 0.35*dd_s + 0.25*atr_s, 0, 100)
    # reversal
    f['reversal'] = clip((50 - df['rsi'])*2, 0, 100)
    # size
    f['size'] = clip(100*(1 - np.exp(-df['total_mv_yi']/500.0)), 0, 100)
    # theme_heat
    f['theme_heat'] = clip(50 + df['ind_ret']*8, 0, 100)
    # signal score (tech composite)
    trend = 50 + np.where(df['ma_bullish'],15,0) + np.where(df['macd_bullish'],12,np.where(df['macd_neutral'],5,-10)) \
            + np.where(df['above_ma20'],8,-8) + clip(df['ret_20d'],-15,15)*0.5
    f['signal'] = clip(0.4*f['momentum'] + 0.35*f['activity'] + 0.25*trend, 0, 100)
    return f

def apply_strategy(cfg, universe):
    hf = cfg['hf']; fw = cfg['fw']; sp = cfg['sp']; rp = cfg['rp']; sc = cfg['sc']; tw = cfg['tech']
    df = universe.copy()
    # 硬筛
    m = (df['amount_yi']*1e8 >= hf['amount_min']) & df['close'].notna()
    if 'pe_max' in hf: m &= (df['pe_ttm'] > 0) & (df['pe_ttm'] <= hf['pe_max'])
    if 'pb_max' in hf: m &= (df['pb'] >= 0) & (df['pb'] <= hf['pb_max'])
    if 'mcap_min' in hf: m &= (df['total_mv_yi']*1e8 >= hf['mcap_min'])
    if 'mcap_max' in hf: m &= (df['total_mv_yi']*1e8 <= hf['mcap_max'])
    if 'price_min' in hf: m &= (df['close'] >= hf['price_min'])
    if 'price_max' in hf: m &= (df['close'] <= hf['price_max'])
    if 'chg_min' in hf: m &= (df['pct_chg'] >= hf['chg_min'])
    if 'chg_max' in hf: m &= (df['pct_chg'] <= hf['chg_max'])
    if 'tr_min' in hf: m &= (df['turnover_rate'] >= hf['tr_min'])
    if 'vr_min' in hf: m &= (df['vol_ratio'] >= hf['vr_min'])
    if hf.get('above_ma20'): m &= df['above_ma20']
    if hf.get('ma_bull'): m &= df['ma_bullish']
    if 'chg60_min' in hf: m &= (df['ret_60d'] >= hf['chg60_min'])
    if 'chg60_max' in hf: m &= (df['ret_60d'] <= hf['chg60_max'])
    if 'range20_max' in hf: m &= (df['range_20'] <= hf['range20_max'])
    if 'vol20_max' in hf: m &= (df['vol_20'] <= hf['vol20_max'])
    if 'mdd_min' in hf: m &= (df['mdd'] >= hf['mdd_min'])
    if 'atr_max' in hf: m &= (df['atr_pct'] <= hf['atr_max'])
    if 'breakout_min' in hf: m &= (df['breakout_20'] >= hf['breakout_min'])
    if 'vrat20_min' in hf: m &= (df['vrat_20'] >= hf['vrat20_min'])
    if 'vrat20_max' in hf: m &= (df['vrat_20'] <= hf['vrat20_max'])
    if 'pullback_min' in hf: m &= (df['dist_ma20'] >= hf['pullback_min'])
    if 'pullback_max' in hf: m &= (df['dist_ma20'] <= hf['pullback_max'])
    if 'body_min' in hf: m &= (df['body'] >= hf['body_min'])
    if 'consol_min' in hf: m &= (df['consol'] >= hf['consol_min'])
    df_f = df[m].copy()
    relaxed = False
    # 放宽: 若通过硬筛过少(<5), 保留结构性技术筛(ma_bullish/above_ma20), 放宽数量化阈值
    relax_lvl = 0
    if len(df_f) < 5:
        relaxed = True; relax_lvl = 1
        m2 = (df['amount_yi']*1e8 >= hf['amount_min']) & df['close'].notna()
        if 'pe_max' in hf: m2 &= (df['pe_ttm'] > 0) & (df['pe_ttm'] <= hf['pe_max'])
        if 'pb_max' in hf: m2 &= (df['pb'] <= hf['pb_max'])
        if 'mcap_min' in hf: m2 &= (df['total_mv_yi']*1e8 >= hf['mcap_min'])
        if 'chg_min' in hf: m2 &= (df['pct_chg'] >= hf['chg_min'])
        if 'chg_max' in hf: m2 &= (df['pct_chg'] <= hf['chg_max'])
        if 'tr_min' in hf: m2 &= (df['turnover_rate'] >= hf['tr_min'])
        if 'vr_min' in hf: m2 &= (df['vol_ratio'] >= hf['vr_min'])
        # 结构性技术筛保留
        if hf.get('above_ma20'): m2 &= df['above_ma20']
        if hf.get('ma_bull'): m2 &= df['ma_bullish']
        # 数量化阈值放宽
        if 'pullback_min' in hf: m2 &= (df['dist_ma20'] >= hf['pullback_min']-3)
        if 'pullback_max' in hf: m2 &= (df['dist_ma20'] <= hf['pullback_max']+5)
        if 'vrat20_max' in hf: m2 &= (df['vrat_20'] <= hf['vrat20_max']+0.5)
        if 'vrat20_min' in hf: m2 &= (df['vrat_20'] >= hf['vrat20_min']-0.2)
        if 'vol20_max' in hf: m2 &= (df['vol_20'] <= hf['vol20_max']+10)
        if 'mdd_min' in hf: m2 &= (df['mdd'] >= hf['mdd_min']-5)
        if 'atr_max' in hf: m2 &= (df['atr_pct'] <= hf['atr_max']+1.5)
        if 'range20_max' in hf: m2 &= (df['range_20'] <= hf['range20_max']+10)
        if 'breakout_min' in hf: m2 &= (df['breakout_20'] >= hf['breakout_min']-2)
        if 'chg60_min' in hf: m2 &= (df['ret_60d'] >= hf['chg60_min']-5)
        if 'chg60_max' in hf: m2 &= (df['ret_60d'] <= hf['chg60_max']+10)
        df_f = df[m2].copy()
    # Level 2: 仍不足5, 丢弃所有技术筛, 仅保留基本面+流动性+涨跌幅, 取最接近的候选
    if len(df_f) < 5:
        relaxed = True; relax_lvl = 2
        m3 = (df['amount_yi']*1e8 >= hf['amount_min']) & df['close'].notna()
        if 'pe_max' in hf: m3 &= (df['pe_ttm'] > 0) & (df['pe_ttm'] <= hf['pe_max'])
        if 'pb_max' in hf: m3 &= (df['pb'] <= hf['pb_max'])
        if 'mcap_min' in hf: m3 &= (df['total_mv_yi']*1e8 >= hf['mcap_min'])
        if 'chg_min' in hf: m3 &= (df['pct_chg'] >= hf['chg_min'])
        if 'chg_max' in hf: m3 &= (df['pct_chg'] <= hf['chg_max'])
        if 'tr_min' in hf: m3 &= (df['turnover_rate'] >= hf['tr_min'])
        if 'vr_min' in hf: m3 &= (df['vol_ratio'] >= hf['vr_min'])
        if 'price_min' in hf: m3 &= (df['close'] >= hf['price_min'])
        if 'price_max' in hf: m3 &= (df['close'] <= hf['price_max'])
        df_f = df[m3].copy()
        log(f"  [{cfg['name']}] Level2 放宽(丢弃技术筛) 候选数={len(df_f)}")
    df_f.attrs['relax_lvl'] = relax_lvl
    if len(df_f) == 0:
        return df_f, relaxed, relax_lvl, 0
    # 评分
    df_f.attrs['pe_max'] = hf.get('pe_max', 80); df_f.attrs['pb_max'] = hf.get('pb_max', 8.0)
    df_f.attrs['ideal_vr'] = sp['ideal_vr']; df_f.attrs['ideal_tr'] = sp['ideal_tr']
    fs = factor_scores(df_f)
    base = np.zeros(len(df_f))
    for k, w in fw.items():
        if k in fs.columns:
            base += w * fs[k].values
    # tech tilt
    tech_signal = 0.5*fs['signal'].values + 0.5*fs['activity'].values
    final = base*(1 - tw*0.3) + (tw*0.3)*tech_signal
    # bonuses / penalties
    # value_quality_bonus
    if 'vq_value' in sc:
        mask = (fs['value'].values >= sc['vq_value']) & (fs['stability'].values >= sc.get('vq_stab',0))
        final = np.where(mask, final + sc['vq_bonus'], final)
    if 'vq_stab' in sc and 'vq_value' not in sc:
        mask = fs['stability'].values >= sc['vq_stab']
        final = np.where(mask, final + sc['vq_bonus'], final)
    # dividend bonus (value/income)
    if cfg['category'] in ('value','income'):
        final = final + np.clip(df_f['dv_ttm'].fillna(0).values*4, 0, 12)
    # capital confirmed
    if 'cap_bonus' in sc:
        final = np.where(fs['activity'].values >= 80, final + sc['cap_bonus']*0.5, final)
    # stability penalties
    final = np.where(df_f['pct_chg'].values > sp.get('sta_hot', 99), final - 3, final)
    if 'high_vol' in sp:
        final = np.where(df_f['vol_20'].values > sp['high_vol'], final - 3, final)
    if 'mdd_floor' in sp:
        final = np.where(df_f['mdd'].values < sp['mdd_floor'], final - 3, final)
    if 'high_atr' in sp:
        final = np.where(df_f['atr_pct'].values > sp['high_atr'], final - 3, final)
    # volume spike penalty
    if 'vol_spike' in sc:
        final = np.where(df_f['vol_ratio'].fillna(0).values > sc['vol_spike'], final - sc.get('vol_spike_pen',1.5), final)
    # hot money penalty
    if 'hot_money_pen' in sc:
        final = np.where(fs['activity'].values >= sc.get('hot_money_min',95), final - sc['hot_money_pen'], final)
    # theme overheat penalty
    if 'theme_overheat' in sp:
        final = np.where(fs['theme_heat'].values > sp['theme_overheat'], final - (fs['theme_heat'].values - sp['theme_overheat'])*sp['theme_slope'], final)
    # risk penalties
    final = np.where(df_f['pct_chg'].values > rp.get('chase',99), final - 5, final)
    final = np.where(df_f['vol_ratio'].fillna(0).values > rp.get('abn_vr',99), final - 3, final)
    final = np.where(df_f['turnover_rate'].fillna(0).values > rp.get('high_tr',99), final - 2, final)
    # chase penalty on momentum
    chase_pen = np.where(df_f['pct_chg'].values > sp.get('chase_start', 99),
                         (df_f['pct_chg'].values - sp.get('chase_start', 99))*sp.get('slope', 12)/10, 0)
    final = final - chase_pen
    if 'downside_start' in sp:
        final = final - np.where(df_f['pct_chg'].values < sp['downside_start'],
                                 (sp['downside_start'] - df_f['pct_chg'].values)*0.4, 0)
    # oversold reversal specifics
    if cfg['name'] == 'oversold_reversal':
        final = final + np.where(df_f['rsi'].fillna(50).values < 30, sp['rsi_bonus'], 0)
        final = final - np.where(df_f['pct_chg'].values < sp['rev_collapse'],
                                 (sp['rev_collapse'] - df_f['pct_chg'].values)*sp['rev_slope']/10, 0)
        final = np.where((df_f['pct_chg'].values >= sp['rev_collapse']) & (df_f['pct_chg'].values <= 0),
                         final + sc.get('ctrl_rev_bonus',0), final)
    final = np.clip(final, 0, 100)
    df_f = df_f.assign(final_score=final,
                      signal_score=fs['signal'].values,
                      value_s=fs['value'].values, mom_s=fs['momentum'].values,
                      act_s=fs['activity'].values, liq_s=fs['liquidity'].values,
                      sta_s=fs['stability'].values, rev_s=fs['reversal'].values)
    # 应用 signal_min 硬筛(若非放宽)
    if 'signal_min' in hf and not relaxed:
        df_f = df_f[df_f['signal_score'] >= hf['signal_min']]
    df_f = df_f.sort_values('final_score', ascending=False)
    return df_f, relaxed, relax_lvl, int(m.sum())

# ---------------- 8. 运行所有策略 ----------------
results = {}
for sname in STRATEGY_ORDER:
    cfg = STRATEGIES[sname]
    res, relaxed, relax_lvl, npass = apply_strategy(cfg, feat_g)
    top_n = 5 if relax_lvl >= 2 else 10
    top = res.head(top_n).copy()
    results[sname] = dict(cfg=cfg, df=res, top=top, relaxed=relaxed, relax_lvl=relax_lvl, npass=npass, nres=len(res))
    log(f"[{cfg['display']}] 通过硬筛:{npass} 命中:{len(res)} 放宽:{relaxed} Top1:",
        (top.iloc[0]['name'] if len(top) else '无'))

# ---------------- 9. 市场概况 ----------------
# 指数: 优先腾讯实时指数, 回退昨日收盘
idx_codes = {'000001.SH':'上证指数','000300.SH':'沪深300','399001.SZ':'深证成指','399006.SZ':'创业板指','000905.SH':'中证500'}
def to_qq_idx(tc): return ('sh' if tc.endswith('.SH') else 'sz') + tc[:6]
idx_map = {}
try:
    qq_idx_codes = ','.join([to_qq_idx(c) for c in idx_codes.keys()])
    r = requests.get('http://qt.gtimg.cn/q='+qq_idx_codes, timeout=10)
    for line in r.text.strip().split(';\n'):
        if not line.strip(): continue
        m = re.search(r'v_(sh|sz)(\d+)="([^"]*)"', line)
        if not m: continue
        f = m.group(3).split('~')
        if len(f) < 5: continue
        ts_code_full = ('%s.SH'%m.group(2)) if m.group(1)=='sh' else ('%s.SZ'%m.group(2))
        nm = idx_codes.get(ts_code_full)
        if not nm: continue
        def num(x):
            try: return float(x)
            except: return np.nan
        last = num(f[3]); pre_close = num(f[4])
        if last > 0 and pre_close > 0:
            idx_map[nm] = dict(close=last, pct=(last/pre_close-1)*100)
except Exception as e:
    log('qq index fail', e)
# 回退: 用 Tushare 昨日 index_daily
if not idx_map:
    try:
        idx_prev = pro.index_daily(trade_date=prev_date)
        for code, nm in idx_codes.items():
            rp_ = idx_prev[idx_prev['ts_code']==code]
            if len(rp_):
                idx_map[nm] = dict(close=float(rp_.iloc[0]['close']), pct=0.0)
    except Exception as e:
        log('index fallback fail', e)
log('指数实时:', {k:round(v['pct'],2) for k,v in idx_map.items()})

# 涨跌家数
up = int((day_df['pct_chg']>0).sum())
down = int((day_df['pct_chg']<0).sum())
flat = int((day_df['pct_chg']==0).sum())
limit_up = int((day_df['pct_chg']>=9.9).sum())
limit_down = int((day_df['pct_chg']<=-9.9).sum())
avg_chg = float(day_df['pct_chg'].mean())
med_chg = float(day_df['pct_chg'].median())
# 今日amount来自腾讯(元), 昨日amount来自Tushare(千元) -> 统一转亿
total_amt_yi = float(day_df['amount'].sum()/1e8)
prev_total_amt_yi = float(prev_df['amount'].sum()/1e5)  # 千元->亿
amt_chg_pct = (total_amt_yi/prev_total_amt_yi-1)*100 if prev_total_amt_yi else 0

# 板块强弱(行业) - day_df['amount']为元(腾讯), 转/1e8 -> 亿
ind_stat = day_df.merge(sb[['ts_code','industry']], on='ts_code', how='left')
ind_grp = ind_stat.groupby('industry').agg(
    avg_chg=('pct_chg','mean'), cnt=('ts_code','size'),
    total_amt_yi=('amount','sum')).reset_index()
ind_grp['total_amt_yi'] = ind_grp['total_amt_yi']/1e8  # 元->亿
ind_top = ind_grp.sort_values('avg_chg', ascending=False).head(10)
ind_bot = ind_grp.sort_values('avg_chg').head(8)

# ---------------- 10. 跨策略共振 ----------------
hit = {}
for sname in STRATEGY_ORDER:
    for _, r in results[sname]['top'].iterrows():
        hit.setdefault(r['ts_code'], []).append(sname)
resonance = sorted(hit.items(), key=lambda x: len(x[1]), reverse=True)
resonance_top = [(c, s) for c, s in resonance if len(s) >= 2][:15]

# ---------------- 11. 10点报告对比 ----------------
import glob
rep10_files = sorted(glob.glob(f'/workspace/reports/*{latest[:4]}*10*策略报告.md') +
                     glob.glob(f'/workspace/reports/{latest}_10*策略报告.md') +
                     glob.glob('/workspace/reports/*10*策略报告.md'))
rep10_prev = []
if rep10_files:
    try:
        with open(rep10_files[-1], encoding='utf-8') as f:
            rep10_prev = f.read()
    except Exception:
        pass
log('10点报告文件:', rep10_files[-1] if rep10_files else '无')

# ---------------- 12. 写 Markdown 报告 ----------------
def fmt(x, d=2):
    if x is None or (isinstance(x,float) and (math.isnan(x))): return '-'
    try: return f'{float(x):.{d}f}'
    except: return str(x)

lines = []
lines.append(f"# 13:30 盘中选股策略报告 ({latest})")
lines.append("")
rt_time_str = sample_ts
lines.append(f"> **报告日期**: {latest}  | **执行时点**: 13:30 盘中  | **数据基准**: 今日盘中实时行情")
lines.append(f"> **实时数据源**: 腾讯财经 qt.gtimg.cn (最新价/涨跌/成交额/换手率/PE_TTM)，实时时间戳: {rt_time_str}")
lines.append(f"> **历史数据**: Tushare 前{len(hist_dates)}交易日未复权日线(技术指标计算) | **估值基准**: {PREV_TRADE_DATE} daily_basic(PB/总市值/股息率, 日内稳定)")
lines.append(f"> 今日为A股交易日(is_open=1)，本报告基于**盘中实时数据**生成；量比按今日成交额/前5日均成交额自算。")
lines.append(f"> **评分方式**: 本地多因子评分(技术面+基本面)")
lines.append("")

# 市场概况
lines.append("## 一、市场概况")
lines.append("")
lines.append(f"- **涨跌家数**: 上涨 {up} 家 / 下跌 {down} 家 / 平盘 {flat} 家；涨停 {limit_up} 家，跌停 {limit_down} 家")
lines.append(f"- **平均涨跌**: 全市场均值 {fmt(avg_chg)}% / 中位数 {fmt(med_chg)}%")
lines.append(f"- **成交额**: 两市合计 {fmt(total_amt_yi)} 亿元，较前一交易日({prev_date} {fmt(prev_total_amt_yi)}亿) {('+' if amt_chg_pct>=0 else '')}{fmt(amt_chg_pct)}%")
if idx_map:
    lines.append("- **主要指数**:")
    lines.append("")
    lines.append("| 指数 | 收盘 | 涨跌幅 |")
    lines.append("|---|---|---|")
    for nm, v in idx_map.items():
        lines.append(f"| {nm} | {fmt(v['close'])} | {fmt(v['pct'])}% |")
lines.append("")
lines.append("**领涨行业(Top10)**")
lines.append("")
lines.append("| 行业 | 平均涨幅 | 成分数 | 成交额(亿) |")
lines.append("|---|---|---|---|")
for _, r in ind_top.iterrows():
    lines.append(f"| {r['industry']} | {fmt(r['avg_chg'])}% | {int(r['cnt'])} | {fmt(r['total_amt_yi'])} |")
lines.append("")
lines.append("**领跌行业(Top8)**")
lines.append("")
lines.append("| 行业 | 平均跌幅 | 成分数 | 成交额(亿) |")
lines.append("|---|---|---|---|")
for _, r in ind_bot.iterrows():
    lines.append(f"| {r['industry']} | {fmt(r['avg_chg'])}% | {int(r['cnt'])} | {fmt(r['total_amt_yi'])} |")
lines.append("")
lines.append(f"**全局排除**: ST/*ST/退市、次新(上市<1年)、停牌、成交额<2亿元、历史不足20日。")
lines.append("")

# 10策略
lines.append("## 二、10策略筛选结果")
lines.append("")
COLS = ['ts_code','name','industry','close','pct_chg','amount_yi','vol_ratio','turnover_rate',
        'ma_bullish','macd_hist_pos','rsi','ret_20d','vol_20','breakout_20','dist_ma20',
        'pe_ttm','pb','total_mv_yi','final_score']
HEADERS = ['代码','名称','行业','收盘','涨跌%','成交额(亿)','量比','换手%','均线多头','MACD柱','RSI','20日涨%','20日波动%','20日突破%','距MA20%','PE_TTM','PB','总市值(亿)','得分']

for sname in STRATEGY_ORDER:
    cfg = STRATEGIES[sname]; top = results[sname]['top']
    lines.append(f"### {STRATEGY_ORDER.index(sname)+1}. {cfg['display']}（{cfg['category']}）")
    lines.append("")
    cat_desc = {'value':'价值','quality':'质量','income':'收益','trend':'趋势','momentum':'动量','reversal':'反转','framework':'综合'}
    rl = results[sname]['relax_lvl']
    relax_tag = '' if rl == 0 else (' | ⚠️硬筛偏严，已放宽数量化阈值' if rl == 1 else ' | ⚠️硬筛无标的，列出最接近5只(技术筛已全部放宽)')
    lines.append(f"- **策略逻辑**: {cat_desc.get(cfg['category'],'')}类 | 技术权重 {cfg['tech']} | 通过硬筛 {results[sname]['npass']} 只，命中 {results[sname]['nres']} 只{relax_tag}")
    lines.append("")
    if len(top) == 0:
        lines.append("> 无标的通过筛选，建议关注市场环境或放宽参数。")
        lines.append("")
        continue
    lines.append("| " + " | ".join(HEADERS) + " |")
    lines.append("|" + "---|" * len(HEADERS))
    for _, r in top.iterrows():
        row = [
            r['ts_code'], str(r['name']), str(r['industry']),
            fmt(r['close']), fmt(r['pct_chg']), fmt(r['amount_yi']),
            fmt(r['vol_ratio']), fmt(r['turnover_rate']),
            '是' if r['ma_bullish'] else '否',
            '+' if r['macd_hist_pos'] else '-',
            fmt(r['rsi'],1), fmt(r['ret_20d']), fmt(r['vol_20']),
            fmt(r['breakout_20']), fmt(r['dist_ma20']),
            fmt(r['pe_ttm'],1) if r['pe_ttm']>0 else '亏损',
            fmt(r['pb']), fmt(r['total_mv_yi'],0), fmt(r['final_score'],1),
        ]
        lines.append("| " + " | ".join(row) + " |")
    lines.append("")
    # 解读
    top1 = top.iloc[0]
    interps = {
     'dual_low': f"低PE+低PB价值修复为主，{top1['name']}({top1['industry']})估值与活跃度兼顾，得分{fmt(top1['final_score'],1)}。",
     'quality_value': f"估值合理+流动性稳健，{top1['name']}量价温和未过热，适合深度跟踪。",
     'blue_chip_income': f"大市值蓝筹红利方向，{top1['name']}成交稳定、估值未透支，防守属性突出。",
     'low_volatility_quality': f"低波动+浅回撤防守仓，{top1['name']}波动率{fmt(top1['vol_20'])}%、回撤可控。",
     'volume_breakout': f"放量突破关键阻力，{top1['name']}量比{fmt(top1['vol_ratio'])}、涨{fmt(top1['pct_chg'])}%站上MA20，趋势启动。",
     'shrink_pullback': f"多头排列缩量回踩MA20，{top1['name']}距MA20{fmt(top1['dist_ma20'])}%，洗盘续涨结构。",
     'capital_heat': f"资金活跃量价同步，{top1['name']}量比{fmt(top1['vol_ratio'])}换手{fmt(top1['turnover_rate'])}%，热度未透支。",
     'oversold_reversal': f"超跌修复候选，{top1['name']}跌{fmt(top1['pct_chg'])}%、RSI{fmt(top1['rsi'],1)}，跌幅可控流动性仍在。",
     'balanced_alpha': f"多因子均衡，{top1['name']}估值/资金/动量/稳定性均衡无短板。",
     'momentum_quality': f"趋势确认+质量支撑，{top1['name']}趋势确立但未进入追涨极端区。",
    }
    lines.append(f"**解读**: {interps.get(sname,'')}")
    lines.append("")

# 跨策略共振
lines.append("## 三、跨策略共振标的（多策略同时命中）")
lines.append("")
lines.append("以下标的被2个及以上策略同时选中，共振信号较强，建议重点关注：")
lines.append("")
if resonance_top:
    lines.append("| 代码 | 名称 | 行业 | 收盘 | 涨跌% | 命中策略数 | 命中策略 |")
    lines.append("|---|---|---|---|---|---|---|")
    # 取每个共振标的的明细
    for code, slist in resonance_top:
        info = feat_g[feat_g['ts_code']==code]
        if len(info)==0: continue
        info = info.iloc[0]
        disp_names = [STRATEGIES[s]['display'] for s in slist]
        lines.append(f"| {code} | {info['name']} | {info['industry']} | {fmt(info['close'])} | {fmt(info['pct_chg'])}% | {len(slist)} | {'、'.join(disp_names)} |")
    lines.append("")
else:
    lines.append("> 本期无标的被2个及以上策略同时命中。")
    lines.append("")

# 市场信号 + 次日关注 + 风险
lines.append("## 四、市场信号总结与次日关注")
lines.append("")
# 综合判断
breadth_strong = up > down * 1.5
breadth_weak = down > up * 1.5
amt_warm = total_amt_yi > 1.0e4  # >1万亿
sig = []
if breadth_strong: sig.append("市场 breadth 偏强(上涨家数明显占优)")
elif breadth_weak: sig.append("市场 breadth 偏弱(下跌家数明显占优)")
else: sig.append("市场 breadth 中性(涨跌家数接近)")
if avg_chg >= 0:
    if amt_warm: sig.append(f"成交额 {fmt(total_amt_yi,0)} 亿放量上行，资金参与积极")
    else: sig.append(f"成交额 {fmt(total_amt_yi,0)} 亿缩量上行，上攻动能待观察")
else:
    if amt_warm: sig.append(f"成交额 {fmt(total_amt_yi,0)} 亿放量下跌，资金出逃/避险情绪浓，谨防抛压扩散")
    else: sig.append(f"成交额 {fmt(total_amt_yi,0)} 亿缩量下跌，卖压有限但做多意愿不足")
if idx_map:
    sh = idx_map.get('上证指数',{}).get('pct',0)
    cy = idx_map.get('创业板指',{}).get('pct',0)
    if sh>0 and cy>0: sig.append("主板与创业板同涨，风险偏好回升")
    elif sh<0 and cy<0: sig.append("主板与创业板同跌，避险情绪升温")
    elif cy>sh: sig.append("成长相对占优(创业板强于主板)")
    else: sig.append("价值相对占优(主板强于创业板)")
if limit_down >= 100:
    sig.append(f"跌停 {limit_down} 家，跌停潮显现，系统性风险升温，操作以防御为主")
lines.append("**市场信号**:")
lines.append("")
for s in sig:
    lines.append(f"- {s}")
lines.append("")

# 次日关注标的(从共振 + 各策略Top1 中挑选)
next_watch = []
seen = set()
for code, slist in resonance_top[:5]:
    if code not in seen:
        next_watch.append((code, len(slist), slist)); seen.add(code)
for sname in STRATEGY_ORDER:
    top = results[sname]['top']
    if len(top):
        c = top.iloc[0]['ts_code']
        if c not in seen:
            next_watch.append((c, 1, [sname])); seen.add(c)
    if len(next_watch) >= 12: break
lines.append("**次日重点观察标的**（共振优先 + 各策略龙头）:")
lines.append("")
lines.append("| 代码 | 名称 | 行业 | 逻辑 |")
lines.append("|---|---|---|---|")
for code, n, slist in next_watch:
    info = feat_g[feat_g['ts_code']==code]
    if len(info)==0: continue
    info = info.iloc[0]
    disp = '、'.join([STRATEGIES[s]['display'] for s in slist])
    tag = f"共振({n}策略)" if n>=2 else disp
    lines.append(f"| {code} | {info['name']} | {info['industry']} | {tag} |")
lines.append("")

lines.append("**风险提示**:")
lines.append("")
lines.append(f"- 本报告基于 {latest} 盘中实时行情(腾讯财经，时间戳{rt_time_str})生成，行情仍在变动，尾盘与收盘可能继续变化，次日开盘需结合集合竞价复核。")
lines.append("- 技术指标采用未复权日线计算，除权除息日附近存在一定失真。")
lines.append("- 策略为量化筛选结果，不构成投资建议；实际操作需结合个股公告、行业政策与大盘环境。")
lines.append("- 放宽条件的策略(已标注)命中质量下降，需人工二次确认。")
lines.append("- 跨策略共振标的集中度高时注意板块联动风险，避免单一主题过度暴露。")
lines.append("")

if rep10_files:
    lines.append("## 五、与10点报告对比")
    lines.append("")
    lines.append(f"> 参考文件: {rep10_files[-1]}")
    lines.append("")
    lines.append("(注: 自动文本对比仅供参考，新增/退出以代码匹配为准)")
    lines.append("")
else:
    lines.append("## 五、与10点报告对比")
    lines.append("")
    lines.append(f"> 未发现 {latest} 的10点报告文件，本期无对比基准。")
    lines.append("")

lines.append("---")
lines.append(f"*报告生成时间: {datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')} | 数据基准日: {latest} | 历史窗口: {len(hist_dates)}交易日*")
lines.append("")

report_text = "\n".join(lines)
os.makedirs('/workspace/reports', exist_ok=True)
rep_path = f'/workspace/reports/{latest}_13点30策略报告.md'
with open(rep_path, 'w', encoding='utf-8') as f:
    f.write(report_text)
log('报告已写入:', rep_path)

# ---------------- 13. 输出摘要 JSON 供推送 ----------------
summary = dict(latest=latest, today=TODAY, rt_source='腾讯财经qt.gtimg.cn', rt_timestamp=sample_ts, is_realtime=True,
    market=dict(up=up, down=down, flat=flat, limit_up=limit_up, limit_down=limit_down,
                avg_chg=avg_chg, total_amt_yi=total_amt_yi, prev_total_amt_yi=prev_total_amt_yi, amt_chg_pct=amt_chg_pct,
                idx=idx_map,
                top_industries=[(r['industry'], round(r['avg_chg'],2)) for _,r in ind_top.head(5).iterrows()]),
    strategies={}, resonance=[], next_watch=[])
for sname in STRATEGY_ORDER:
    top = results[sname]['top'].head(3)
    summary['strategies'][sname] = dict(display=STRATEGIES[sname]['display'],
        npass=results[sname]['npass'], nres=results[sname]['nres'], relaxed=results[sname]['relaxed'],
        relax_lvl=results[sname]['relax_lvl'],
        top3=[dict(code=r['ts_code'], name=str(r['name']), industry=str(r['industry']),
                   close=round(float(r['close']),2), pct=round(float(r['pct_chg']),2),
                   score=round(float(r['final_score']),1)) for _,r in top.iterrows()])
for code, slist in resonance_top[:8]:
    info = feat_g[feat_g['ts_code']==code]
    if len(info)==0: continue
    info = info.iloc[0]
    summary['resonance'].append(dict(code=code, name=str(info['name']), industry=str(info['industry']),
        n=len(slist), strategies=[STRATEGIES[s]['display'] for s in slist]))
for code, n, slist in next_watch[:8]:
    info = feat_g[feat_g['ts_code']==code]
    if len(info)==0: continue
    info = info.iloc[0]
    summary['next_watch'].append(dict(code=code, name=str(info['name']), industry=str(info['industry']), n=n))
with open('/workspace/reports/_summary_13.json','w',encoding='utf-8') as f:
    json.dump(summary, f, ensure_ascii=False, indent=2)
log('摘要JSON已写入: /workspace/reports/_summary_13.json')
log('完成。')
