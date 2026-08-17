"""
PCB印制电路板板块 趋势质量 momentum_quality 策略
"""
import os
import tushare as ts
from dotenv import load_dotenv
import pandas as pd
import numpy as np
from datetime import datetime

load_dotenv('/workspace/.env')
TOKEN = os.getenv('TUSHARE_TOKEN') or '9f640d421f866dde9a1888ab4193f77c43659d47446d717ffa343023'
ts.set_token(TOKEN)
pro = ts.pro_api()

def _get_latest_trade_date():
    cal = pro.trade_cal(exchange='SSE', end_date=pd.Timestamp.now().strftime('%Y%m%d'),
                        limit=10, fields='cal_date,is_open')
    cal = cal[cal['is_open'] == 1].sort_values('cal_date', ascending=False)
    for d in cal['cal_date'].tolist():
        probe = pro.daily(ts_code='000001.SZ', trade_date=d, limit=1)
        if len(probe) and not probe.empty:
            return d
    return cal['cal_date'].iloc[0]

last_trade = _get_latest_trade_date()
print(f"[INFO] 最新交易日: {last_trade}")

# PCB板块：按行业+关键词筛选
basic = pro.stock_basic(fields='ts_code,name,industry,list_status')
pcb = basic[(basic['industry'].str.contains('元器件|印刷电路|PCB|覆铜板|铜箔|CCL', na=False))
            | (basic['name'].str.contains('生益|深南|沪电|胜宏|景旺|崇达|奥士|兴森|超声|方邦|东山|四会|博敏|中京|广东骏亚|依顿|世运|超华|金安国纪|华正|建滔|超声电子', na=False))]
# 排除非PCB（按名称精确过滤）
exclude_names = []
pcb = pcb[~pcb['name'].str.contains(r'ST|\*ST|退', na=False, regex=True)]
pcb = pcb[pcb['list_status'] == 'L']
print(f"[INFO] PCB板块标的数: {len(pcb)}")
print(pcb[['ts_code','name','industry']].to_string(index=False))

rows = []
for _, r in pcb.iterrows():
    code = r['ts_code']
    try:
        q = pro.daily(ts_code=code, end_date=last_trade, limit=30)
        if len(q) < 21:
            continue
        q = q.sort_values('trade_date').reset_index(drop=True)
        today = q.iloc[-1]
        closes = q['close'].values
        vols = q['vol'].values
        highs = q['high'].values

        ma5 = closes[-5:].mean()
        ma10 = closes[-10:].mean()
        ma20 = closes[-20:].mean()
        ma_bullish = bool(ma5 > ma10 > ma20)
        above_ma20 = bool(today['close'] > ma20)

        dist_ma20_pct = (today['close']/ma20 - 1)*100
        chg_20d = (closes[-1]/closes[-21]-1)*100 if len(closes)>=21 else None
        avg_vol20 = vols[-20:].mean()
        vol_ratio = today['vol']/avg_vol20 if avg_vol20 > 0 else None
        high20 = highs[-20:].max()
        breakout_20d = (today['close']/high20 - 1)*100
        rets = np.diff(np.log(closes[-21:]))
        volatility = np.std(rets)*np.sqrt(252)*100

        b = pro.daily_basic(ts_code=code, end_date=last_trade, limit=1)
        rec = {
            'code': code.split('.')[0],
            'name': r['name'],
            'industry': r['industry'],
            'close': today['close'],
            'pct_chg': round(today['pct_chg'], 2),
            'amount': round(today['amount']/10000, 1),
            'ma5': round(ma5, 2),
            'ma10': round(ma10, 2),
            'ma20': round(ma20, 2),
            'ma_bullish': ma_bullish,
            'above_ma20': above_ma20,
            'dist_ma20_pct': round(dist_ma20_pct, 2),
            'chg_20d': round(chg_20d, 2) if chg_20d else None,
            'vol_ratio': round(vol_ratio, 2) if vol_ratio else None,
            'breakout_20d_pct': round(breakout_20d, 2),
            'volatility_20d': round(volatility, 1),
        }
        pe = None; pb = None; tv = None
        if len(b):
            pe = b.iloc[0]['pe']; pb = b.iloc[0]['pb']; tv = b.iloc[0]['total_mv']
            rec['pe'] = round(pe, 1) if pe is not None and not np.isnan(pe) else None
            rec['pb'] = round(pb, 2) if pb is not None and not np.isnan(pb) else None
            rec['total_mv'] = round(tv/10000, 0) if tv else None

        # 评分（即使未通过硬筛也计算）
        s = 0
        if ma_bullish: s += 15
        if above_ma20: s += 10
        if 0 <= dist_ma20_pct <= 8: s += 10
        elif 8 < dist_ma20_pct <= 15: s += 5
        elif dist_ma20_pct > 15: s -= 3
        if chg_20d is not None:
            if 5 <= chg_20d <= 15: s += 15
            elif 0 < chg_20d < 5: s += 8
            elif 15 < chg_20d <= 20: s += 5
        if pe is not None and not np.isnan(pe):
            if 10 <= pe <= 35: s += 10
            elif 0 < pe < 10: s += 8
            elif 35 < pe <= 60: s += 5
        if pb is not None and not np.isnan(pb):
            if 0 < pb < 3: s += 5
            elif 3 <= pb < 5: s += 3
        amt = rec['amount']
        if amt >= 10: s += 10
        elif amt >= 5: s += 8
        elif amt >= 2: s += 5
        mv = rec.get('total_mv')
        if mv and mv >= 300: s += 8
        elif mv and mv >= 100: s += 5
        p = today['pct_chg']
        if 0 < p <= 4: s += 10
        elif -2 <= p <= 0: s += 5
        elif 4 < p <= 7: s += 5
        if vol_ratio and 0.8 <= vol_ratio <= 1.8: s += 5
        rec['score'] = s
        rows.append(rec)
    except Exception as e:
        continue

df = pd.DataFrame(rows)
pd.set_option('display.unicode.east_asian_width', True)

print(f"\n{'='*80}")
print(f"PCB板块 趋势质量 momentum_quality ({last_trade})")
print(f"{'='*80}")
print(f"扫描: {len(pcb)} | 命中(全部纳入评分): {len(df)}")

# 严格符合：均线多头 + 站上MA20 + 距MA20 0-15% + 20日涨幅 0-20%
strict = df[(df['ma_bullish']) & (df['above_ma20']) & (df['dist_ma20_pct']>=0) & (df['dist_ma20_pct']<=15) & (df['chg_20d']>0) & (df['chg_20d']<=20)].copy()
print(f"严格符合趋势质量: {len(strict)}")
if len(strict) > 0:
    strict = strict.sort_values('score', ascending=False).reset_index(drop=True)
    print(f"\n--- 严格符合 标的 ---")
    cols = ['code','name','industry','close','pct_chg','amount','dist_ma20_pct','chg_20d','vol_ratio','breakout_20d_pct','pe','pb','total_mv','score']
    print(strict[cols].to_string(index=False))

# 全部PCB板块评分排序（看整体强弱）
print(f"\n--- PCB板块全部标的评分排序 ---")
df_sorted = df.sort_values('score', ascending=False).reset_index(drop=True)
cols2 = ['code','name','close','pct_chg','amount','ma_bullish','above_ma20','dist_ma20_pct','chg_20d','vol_ratio','breakout_20d_pct','pe','pb','total_mv','score']
print(df_sorted[cols2].to_string(index=False))

# 板块整体统计
print(f"\n--- 板块统计 ---")
print(f"上涨: {(df['pct_chg']>0).sum()} | 下跌: {(df['pct_chg']<0).sum()} | 平均涨幅: {df['pct_chg'].mean():.2f}%")
print(f"均线多头: {df['ma_bullish'].sum()} | 站上MA20: {df['above_ma20'].sum()} | 突破20日高: {(df['breakout_20d_pct']>=0).sum()}")
print(f"成交额≥5亿: {(df['amount']>=5).sum()} | 成交额≥10亿: {(df['amount']>=10).sum()}")
