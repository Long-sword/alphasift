"""
趋势质量 momentum_quality 选股策略
核心逻辑：
1. 均线多头：MA5 > MA10 > MA20
2. 站上MA20
3. 距MA20 0-15%（不追高）
4. 20日涨幅 0-20%
5. PE 0-60，PB <= 6
6. 成交额 >= 2亿，总市值 >= 50亿
7. 当日涨幅 -3% ~ 7%
8. 排除ST、次新、停牌
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

basic = pro.stock_basic(fields='ts_code,name,industry,list_date,list_status')
stocks = basic[basic['list_status'] == 'L'].copy()
list_cutoff = (datetime.strptime(last_trade, '%Y%m%d') - pd.Timedelta(days=365)).strftime('%Y%m%d')
stocks = stocks[stocks['list_date'] <= list_cutoff]
stocks = stocks[~stocks['name'].str.contains(r'ST|\*ST|退', na=False, regex=True)]
print(f"[INFO] 待扫描: {len(stocks)} (排除次新/ST)")

daily_all = pro.daily(trade_date=last_trade, fields='ts_code,close,pct_chg,vol,amount')
daily_all = daily_all[daily_all['amount'] >= 20000]
daily_all = daily_all[daily_all['pct_chg'] <= 7]
daily_all = daily_all[daily_all['pct_chg'] >= -3]
daily_all = daily_all.sort_values('amount', ascending=False)
scan_list = stocks[stocks['ts_code'].isin(daily_all['ts_code'])].copy()
print(f"[INFO] 按成交/涨跌过滤后: {len(scan_list)}")

rows = []
for i, r in scan_list.iterrows():
    code = r['ts_code']
    try:
        q = pro.daily(ts_code=code, end_date=last_trade, limit=30)
        if len(q) < 21:
            continue
        q = q.sort_values('trade_date').reset_index(drop=True)
        today = q.iloc[-1]
        closes = q['close'].values
        vols = q['vol'].values

        ma5 = closes[-5:].mean()
        ma10 = closes[-10:].mean()
        ma20 = closes[-20:].mean()
        ma_bullish = bool(ma5 > ma10 > ma20)
        above_ma20 = bool(today['close'] > ma20)
        if not (ma_bullish and above_ma20):
            continue

        dist_ma20_pct = (today['close']/ma20 - 1)*100
        if not (0 <= dist_ma20_pct <= 15):
            continue

        chg_20d = (closes[-1]/closes[-21]-1)*100
        if not (0 < chg_20d <= 20):
            continue

        avg_vol20 = vols[-20:].mean()
        vol_ratio = today['vol']/avg_vol20 if avg_vol20 > 0 else None

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
            'dist_ma20_pct': round(dist_ma20_pct, 2),
            'chg_20d': round(chg_20d, 2),
            'vol_ratio': round(vol_ratio, 2) if vol_ratio else None,
        }
        pe = None; pb = None; tv = None
        if len(b):
            pe = b.iloc[0]['pe']; pb = b.iloc[0]['pb']; tv = b.iloc[0]['total_mv']
            if pe is not None and not np.isnan(pe):
                if not (0 < pe <= 60):
                    continue
            if pb is not None and not np.isnan(pb):
                if not (0 < pb <= 6):
                    continue
            tv_yi = round(tv/10000, 0) if tv else None
            if tv_yi is None or tv_yi < 50:
                continue
            rec['pe'] = round(pe, 1) if pe is not None and not np.isnan(pe) else None
            rec['pb'] = round(pb, 2) if pb is not None and not np.isnan(pb) else None
            rec['total_mv'] = tv_yi

        # 评分
        s = 0
        if ma_bullish: s += 15
        if above_ma20: s += 10
        if 0 <= dist_ma20_pct <= 8: s += 10
        elif 8 < dist_ma20_pct <= 15: s += 5
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
            elif 5 <= pb <= 6: s += 1
        amt = rec['amount']
        if amt >= 10: s += 10
        elif amt >= 5: s += 8
        elif amt >= 2: s += 5
        mv = rec.get('total_mv')
        if mv and mv >= 500: s += 8
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
print(f"趋势质量 momentum_quality 策略 ({last_trade})")
print(f"{'='*80}")
print(f"扫描: {len(scan_list)} | 命中: {len(df)}")
if len(df) > 0:
    df = df.sort_values('score', ascending=False).reset_index(drop=True)
    print(f"\nTop 20:")
    cols = ['code','name','industry','close','pct_chg','amount','dist_ma20_pct','chg_20d','vol_ratio','pe','pb','total_mv','score']
    print(df[cols].head(20).to_string(index=False))
    print(f"\n行业分布:")
    print(df['industry'].value_counts().head(10).to_string())
