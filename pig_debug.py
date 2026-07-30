import os
import tushare as ts
from dotenv import load_dotenv

load_dotenv('/workspace/.env')
ts.set_token(os.getenv('TUSHARE_TOKEN'))
pro = ts.pro_api()

df = pro.stock_basic(list_status='L', fields='ts_code,symbol,name,industry,market')
print("stock_basic 总数:", len(df))
print("industry 非空数:", df['industry'].notna().sum())
print("\nindustry 唯一值(含'养'或'畜'或'饲'或'猪'):")
mask = df['industry'].str.contains('养|畜|饲|猪', na=False)
print(df[mask][['ts_code','name','industry']].to_string())
print("\n所有 industry 唯一值:")
print(df['industry'].unique())
