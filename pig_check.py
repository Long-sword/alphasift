import os
import tushare as ts
from dotenv import load_dotenv
import pandas as pd

load_dotenv('/workspace/.env')
ts.set_token(os.getenv('TUSHARE_TOKEN'))
pro = ts.pro_api()

# 猪肉养殖板块核心成分股（生猪为主，含部分禽类龙头）
pig_stocks = {
    '002714.SZ':'牧原股份','300498.SZ':'温氏股份','000876.SZ':'新希望','002157.SZ':'正邦科技',
    '002124.SZ':'天邦食品','603363.SH':'傲农生物','002567.SZ':'唐人神','002100.SZ':'天康生物',
    '600975.SH':'新五丰','000735.SZ':'罗牛山','603477.SH':'巨星农牧','002840.SZ':'华统股份',
    '001201.SZ':'东瑞股份','000048.SZ':'京基智农','300761.SZ':'立华股份','002458.SZ':'益生股份',
    '002299.SZ':'圣农发展','002234.SZ':'民和股份','002746.SZ':'仙坛股份','002982.SZ':'湘佳股份',
    '002548.SZ':'金新农','603609.SH':'禾丰股份','002311.SZ':'海大集团','002385.SZ':'大北农',
}
codes = list(pig_stocks.keys())

# 自动找最近有数据的交易日（向前找）
last_trade = None
for d in ['20260730','20260729','20260728','20260727','20260726','20260725','20260724']:
    dd = pro.daily(ts_code=codes[0], start_date=d, end_date=d)
    if len(dd) > 0:
        last_trade = d
        break
print(f"最近有数据交易日: {last_trade}")

# 批量获取行情（逐只查，合并）
rows = []
for code in codes:
    q = pro.daily(ts_code=code, start_date=last_trade, end_date=last_trade)
    b = pro.daily_basic(ts_code=code, start_date=last_trade, end_date=last_trade)
    rec = {'ts_code': code, 'name': pig_stocks[code]}
    if len(q):
        rec['close'] = q.iloc[0]['close']; rec['pct_chg'] = q.iloc[0]['pct_chg']; rec['amount'] = q.iloc[0]['amount']
    if len(b):
        rec['pe'] = b.iloc[0]['pe']; rec['pb'] = b.iloc[0]['pb']; rec['turnover_rate'] = b.iloc[0]['turnover_rate']; rec['total_mv'] = b.iloc[0]['total_mv']
    rows.append(rec)
df = pd.DataFrame(rows)
df = df.dropna(subset=['close']).sort_values('total_mv', ascending=False)
df['amount'] = (df['amount']/10000).round(0)
df['total_mv'] = (df['total_mv']/10000).round(0)
df = df.rename(columns={'ts_code':'代码','name':'名称','close':'现价','pct_chg':'涨跌幅%','amount':'成交额(万)','turnover_rate':'换手率%','pe':'PE','pb':'PB','total_mv':'总市值(亿)'})
pd.set_option('display.unicode.east_asian_width', True); pd.set_option('display.max_rows', None)
print(df.to_string(index=False))

# 分类统计
print(f"\n生猪养殖龙头: 牧原/温氏/新希望/正邦/天邦/傲农/唐人神/天康/新五丰/罗牛山/巨星/华统/东瑞/京基智农")
print(f"禽类龙头: 立华/益生/圣农/民和/仙坛/湘佳")
print(f"饲料兼营猪: 海大/大北农/金新农/禾丰")
