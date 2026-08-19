# -*- coding: utf-8 -*-
"""根据筛选结果生成 13:30 策略报告 Markdown"""
import json, pickle, os
from collections import Counter, defaultdict
import pandas as pd

SUMMARY = json.load(open('/workspace/_summary.json'))
cache = pickle.load(open('/workspace/_screen_cache.pkl','rb'))
snap = cache['snap']
ind_avg = cache['ind_avg']
daily_all = snap  # 0818 快照
# 全市场日线(用于成交额对比)
import tushare as ts  # not needed; use cache daily? cache only has snap. 用 hist 通过另一缓存
# 改用 _daily_cache 拿前一日成交额
daily_cache = pickle.load(open('/workspace/_daily_cache.pkl','rb'))
daily_df = daily_cache[0]
REPORT_DATE = '20260818'
TODAY = '20260819'
all_dates = sorted(daily_df['trade_date'].unique())
prev_date = all_dates[-2]

# ---------- 市场概况 ----------
up = int((snap['pct_chg'] > 0).sum())
down = int((snap['pct_chg'] < 0).sum())
flat = int((snap['pct_chg'] == 0).sum())
avg_chg = float(snap['pct_chg'].mean())
med_chg = float(snap['pct_chg'].median())
total_amt_yi = float(snap['amount_yi'].sum())
prev_amt_yi = float((daily_df[daily_df['trade_date']==prev_date]['amount'].sum())/100000.0)
amt_chg_pct = (total_amt_yi/prev_amt_yi - 1)*100 if prev_amt_yi>0 else 0
limit_up = int((snap['pct_chg'] >= 9.8).sum())
limit_dn = int((snap['pct_chg'] <= -9.8).sum())

# 板块强弱
ind_stats = snap.groupby('industry').agg(avg=('pct_chg','mean'), n=('ts_code','count'),
                                         amt=('amount_yi','sum')).reset_index()
ind_stats = ind_stats[ind_stats['n'] >= 5].sort_values('avg', ascending=False)
top_ind = ind_stats.head(8)
bot_ind = ind_stats.tail(8).iloc[::-1]

# ---------- 代码->技术指标映射 ----------
tech = {}
for _, r in snap.iterrows():
    tech[r['ts_code']] = r

def tech_str(c):
    r = tech.get(c)
    if r is None: return '-'
    ma = 'MA多头' if bool(r['ma_bullish']) else ('MA走平' if abs(r['close']-r['ma20'])/r['ma20']<0.01 else 'MA空头')
    if bool(r['macd_golden']): mc = 'MACD金叉'
    elif r['macd_status']=='bullish': mc = 'MACD红柱'
    else: mc = 'MACD绿柱'
    vr = r['vol_ratio_calc']
    vr_s = f"量比{vr:.1f}" if pd.notna(vr) else "量比NA"
    rsi = r['rsi14']
    rsi_s = f"RSI{rsi:.0f}" if pd.notna(rsi) else "RSINA"
    d2 = (r['close']-r['ma20'])/r['ma20']*100 if pd.notna(r['ma20']) else 0
    return f"{ma}|{mc}|{vr_s}|{rsi_s}|距MA20{d2:+.1f}%"

# ---------- 跨策略共振 ----------
appear = defaultdict(list)   # code -> [strategy names]
for sname, s in SUMMARY.items():
    for row in s['top10']:
        appear[row['code']].append(sname)
resonance = [(c, len(v), v) for c, v in appear.items() if len(v) >= 2]
resonance.sort(key=lambda x: -x[1])

# ---------- 10点报告对比 ----------
rep_dir = '/workspace/reports'
prev10_codes = set()
for fn in os.listdir(rep_dir) if os.path.isdir(rep_dir) else []:
    if '10' in fn and fn.endswith('.md'):
        # 简易提取代码
        with open(os.path.join(rep_dir,fn)) as f:
            import re
            prev10_codes = set(re.findall(r'(\d{6}\.[SHSZBJ])', f.read()))
has_10 = bool(prev10_codes)

# 今日全部命中代码
today_codes = set(appear.keys())
new_codes = today_codes - prev10_codes if has_10 else set()
exit_codes = prev10_codes - today_codes if has_10 else set()

# ---------- 生成 Markdown ----------
L = []
L.append(f"# A股 13:30 盘中选股策略报告")
L.append("")
L.append(f"- **报告日期**：{TODAY}（周三）13:30 盘中")
L.append(f"- **数据基准**：{TODAY} 为交易日，但 EOD 日线尚未发布；本报告采用最新已发布交易日 **{REPORT_DATE}** 收盘数据作为 13:30 盘中代理快照")
L.append(f"- **样本范围**：{len(snap)} 只（已排除 ST/*ST、退市、次新上市<1年、停牌）")
L.append(f"- **策略数量**：10 个（价值/质量/趋势/动量/反转/综合）")
L.append(f"- **数据源**：Tushare Pro（日线 + daily_basic）")
L.append("")
L.append("---")
L.append("")
L.append("## 一、市场概况")
L.append("")
L.append(f"| 指标 | 数值 |")
L.append(f"|---|---|")
L.append(f"| 上涨 / 下跌 / 平 | {up} / {down} / {flat} |")
L.append(f"| 平均涨跌幅 | {avg_chg:+.2f}%（中位 {med_chg:+.2f}%） |")
L.append(f"| 涨停(近似≥9.8%) / 跌停(近似≤-9.8%) | {limit_up} / {limit_dn} |")
L.append(f"| 全市场成交额 | {total_amt_yi:,.0f} 亿元（上一交易日 {prev_date}：{prev_amt_yi:,.0f} 亿，{amt_chg_pct:+.1f}%） |")
L.append("")
L.append("**板块强弱（按行业平均涨幅，含≥5只成分）**")
L.append("")
L.append("| 领涨板块 | 平均涨幅 | 成分数 | | 领跌板块 | 平均涨幅 | 成分数 |")
L.append("|---|---|---|---|---|---|---|")
for i in range(8):
    t = top_ind.iloc[i]
    b = bot_ind.iloc[i]
    L.append(f"| {t['industry']} | {t['avg']:+.2f}% | {int(t['n'])} | | {b['industry']} | {b['avg']:+.2f}% | {int(b['n'])} |")
L.append("")
L.append("> 解读：板块结构显示资金当日主攻方向与避险方向，结合后续策略命中可判断风格切换。")
L.append("")
L.append("---")
L.append("")
L.append("## 二、10 策略选股结果")
L.append("")

STRATEGY_ORDER = ['dual_low','quality_value','blue_chip_income','low_volatility_quality',
                  'volume_breakout','shrink_pullback','capital_heat','oversold_reversal',
                  'balanced_alpha','momentum_quality']
STRATEGY_DESC = {
 'dual_low':'低 PE + 低 PB 的稳健低估值修复候选，叠加当日活跃度与形态确认。',
 'quality_value':'估值合理、流动性充足、波动不过热的稳健价值候选。',
 'blue_chip_income':'高流动性大盘蓝筹，强调合理估值与稳定成交的防守型持有。',
 'low_volatility_quality':'20日波动与回撤可控、估值不过热的防守型质量候选。',
 'volume_breakout':'成交量放大突破关键阻力位，趋势启动信号。',
 'shrink_pullback':'上升趋势中缩量回踩均线支撑，趋势延续入场机会。',
 'capital_heat':'资金活跃、量价同步但未极端过热的短线动量候选。',
 'oversold_reversal':'跌幅可控、流动性仍在、具备修复观察价值的反转候选。',
 'balanced_alpha':'综合估值、资金、动量、稳定性的均衡多因子候选。',
 'momentum_quality':'兼顾趋势确认与基本面质量的中线候选。',
}

for sname in STRATEGY_ORDER:
    s = SUMMARY[sname]
    L.append(f"### {s['disp']}（{sname}）")
    L.append("")
    L.append(f"> 类别：{s['cat']} ｜ {STRATEGY_DESC[sname]} ｜ 命中 {s['n']} 只{'（已放宽阈值）' if s['relaxed'] else ''}")
    L.append("")
    L.append("| # | 代码 | 名称 | 行业 | 收盘 | 涨跌% | 成交额(亿) | 关键技术指标 | PE | PB | 总市值(亿) | 得分 |")
    L.append("|---|---|---|---|---|---|---|---|---|---|---|---|")
    for i, row in enumerate(s['top10'], 1):
        pe = f"{row['pe']:.1f}" if row['pe'] is not None else "亏损"
        L.append(f"| {i} | {row['code']} | {row['name']} | {row['ind']} | {row['close']:.2f} | {row['pct']:+.2f} | {row['amt']:.2f} | {tech_str(row['code'])} | {pe} | {row['pb']:.2f} | {row['mv']:.0f} | {row['score']:.1f} |")
    L.append("")
    # 简要解读
    top = s['top10'][0]
    ind_cnt = Counter(r['ind'] for r in s['top10'])
    ind_str = '、'.join(f"{k}({v})" for k,v in ind_cnt.most_common(3))
    L.append(f"**解读**：榜首 {top['name']}（{top['code']}）得分 {top['score']:.1f}，{top['ind']}行业。命中集中行业：{ind_str}。")
    L.append("")
    L.append("---")
    L.append("")

# ---------- 跨策略共振 ----------
L.append("## 三、跨策略共振汇总")
L.append("")
L.append("多策略同时命中的标的共振信号更强，建议重点关注。")
L.append("")
L.append("| 代码 | 名称 | 行业 | 命中策略数 | 命中策略 | 收盘 | 涨跌% | PE | PB | 总市值(亿) |")
L.append("|---|---|---|---|---|---|---|---|---|---|")
for code, n, strats in resonance[:20]:
    r = tech[code]
    pe = f"{r['pe_ttm']:.1f}" if pd.notna(r['pe_ttm']) else "亏损"
    sname_disp = '、'.join(SUMMARY[s]['disp'] for s in strats)
    L.append(f"| {code} | {r['name']} | {r['industry']} | {n} | {sname_disp} | {r['close']:.2f} | {r['pct_chg']:+.2f}% | {pe} | {r['pb']:.2f} | {r['total_mv_yi']:.0f} |")
L.append("")

# ---------- 与10点报告对比 ----------
L.append("## 四、与 10:00 报告对比")
L.append("")
if has_10:
    L.append(f"- **当日新增命中**（{len(new_codes)} 只）：{('、'.join(sorted(new_codes))) if new_codes else '无'}")
    L.append(f"- **当日退出命中**（{len(exit_codes)} 只）：{('、'.join(sorted(exit_codes))) if exit_codes else '无'}")
else:
    L.append("- 未找到今日 10:00 策略报告文件，暂无法进行新旧对比。建议先运行 10:00 报告以启用盘中-盘初共振追踪。")
L.append("")

# ---------- 市场信号总结 + 次日关注 + 风险 ----------
L.append("## 五、市场信号总结")
L.append("")
# 信号判断
val_strong = up > down * 1.5
L.append(f"- **多空结构**：上涨 {up} 家 vs 下跌 {down} 家，{'多头占优' if val_strong else ('空头占优' if down>up*1.5 else '多空均衡')}，平均涨幅 {avg_chg:+.2f}%。")
L.append(f"- **量能**：成交 {total_amt_yi:,.0f} 亿，较上日 {amt_chg_pct:+.1f}%，{'放量' if amt_chg_pct>5 else ('缩量' if amt_chg_pct<-5 else '量能持平')}。")
lead_ind = top_ind.iloc[0]['industry'] if len(top_ind) else '-'
weak_ind = bot_ind.iloc[0]['industry'] if len(bot_ind) else '-'
L.append(f"- **风格**：领涨板块 {lead_ind}，领跌板块 {weak_ind}；价值/蓝筹策略共振高度集中在银行、石油石化、海运，反映资金避险+红利属性；动量策略集中在元器件/半导体（面板、消费电子链）。")
L.append("")
L.append("## 六、次日关注标的")
L.append("")
L.append("结合跨策略共振强度、流动性与估值，建议次日重点观察：")
L.append("")
focus = []
seen = set()
# 优先 6 策略共振
for code, n, strats in resonance:
    if n >= 5 and code not in seen:
        focus.append((code, n, strats)); seen.add(code)
for code, n, strats in resonance:
    if n >= 3 and code not in seen:
        focus.append((code, n, strats)); seen.add(code)
focus = focus[:8]
L.append("| 代码 | 名称 | 行业 | 共振策略数 | 收盘 | 涨跌% | PE | PB | 关注逻辑 |")
L.append("|---|---|---|---|---|---|---|---|---|")
logic_map = {
 '600926.SH':'银行+低估值+低波多策略共振，红利防守首选',
 '601919.SH':'海运周期+低估值+低波共振，量价温和',
 '600968.SH':'石油开采低估值修复，多策略共振',
 '002648.SZ':'化工原料趋势+缩量回踩+价值共振',
 '000725.SZ':'元器件资金热度+趋势质量共振，量能爆发',
 '600251.SH':'农业综合资金热度+多因子共振',
 '688169.SH':'家电龙头资金热度+趋势质量共振',
 '601665.SH':'银行低估值蓝筹多策略共振',
}
for code, n, strats in focus:
    r = tech[code]
    pe = f"{r['pe_ttm']:.1f}" if pd.notna(r['pe_ttm']) else "亏损"
    L.append(f"| {code} | {r['name']} | {r['industry']} | {n} | {r['close']:.2f} | {r['pct_chg']:+.2f}% | {pe} | {r['pb']:.2f} | {logic_map.get(code,'多策略共振，量价基本面均衡')} |")
L.append("")
L.append("## 七、风险提示")
L.append("")
L.append("- 本报告数据基准为上一交易日收盘（EOD），13:30 实时盘口未接入，次日开盘可能跳空，需结合竞价修正。")
L.append("- 动量/突破类策略命中标的当日涨幅已较高，追高需严格控制止损；超跌反转类需确认无基本面利空。")
L.append("- 银行股在多策略中高度集中，存在风格拥挤风险，若板块轮动需及时调整。")
L.append("- 评分基于本地因子模型（LLM 排名未启用），技术与基本面因子为近似量化，仅供参考，不构成投资建议。")
L.append("- 涨停/跌停为近似统计（≥9.8%），科创板/创业板 20% 板块可能偏差。")
L.append("")
L.append("---")
L.append(f"*报告由 alphasift 10 策略管线（tushare 直连模式）于 {TODAY} 13:30 自动生成。*")

report_path = f'/workspace/reports/{TODAY}_13点30策略报告.md'
os.makedirs('/workspace/reports', exist_ok=True)
with open(report_path, 'w', encoding='utf-8') as f:
    f.write('\n'.join(L))
print('REPORT_SAVED:', report_path)
print('RESONANCE_TOP:', [(c,n) for c,n,_ in resonance[:5]])
print('MARKET:', dict(up=up,down=down,avg=round(avg_chg,2),amt_yi=round(total_amt_yi,0),prev_amt=round(prev_amt_yi,0),limit_up=limit_up))
