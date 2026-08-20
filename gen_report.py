# -*- coding: utf-8 -*-
"""读取 .cache/result.json 生成 Markdown 选股报告 + 跨策略共振分析"""
import json, os
from collections import Counter, defaultdict

CACHE = '/workspace/.cache/result.json'
REPORT_DIR = '/workspace/reports'
os.makedirs(REPORT_DIR, exist_ok=True)

with open(CACHE) as f:
    d = json.load(f)

ld = d['latest_date']; today = d['today']
mkt = d['market']
strats = d['strategies']

# 跨策略命中统计(区分硬命中/含放宽)
hit_hard = Counter()      # 仅非放宽策略命中
hit_all  = Counter()       # 含放宽
hit_strats_hard = defaultdict(list)
hit_strats_all  = defaultdict(list)
for s in strats:
    for r in s['rows']:
        hit_all[r['ts_code']] += 1
        hit_strats_all[r['ts_code']].append(s['name'])
        if not s['relaxed']:
            hit_hard[r['ts_code']] += 1
            hit_strats_hard[r['ts_code']].append(s['name'])

# 行业统计(全策略命中)
ind_count = Counter()
for s in strats:
    for r in s['rows']:
        ind_count[r['industry']] += 1

lines = []
def w(x=''): lines.append(x)

w(f"# 每日10:00盘中选股策略报告 · {ld}")
w()
w(f"> 报告日期(盘中): **{today}** ｜ 数据基准: **{ld}收盘**(最新可用交易日) ｜ 策略数: **10** ｜ 数据源: Tushare")
w(f"> 生成时间: 2026-08-20 10:00 盘中 ｜ 评分方式: 本地多因子加权(技术面+基本面)")
w()
w("---")
w()
w("## 一、市场概况")
w()
w(f"今日为A股交易日(基于{today}盘中判断,使用上一交易日{ld}收盘数据)。")
w()
w(f"| 指标 | 数值 |")
w(f"|---|---|")
w(f"| 上涨家数 | **{mkt['up']}** |")
w(f"| 下跌家数 | **{mkt['down']}** |")
w(f"| 平盘家数 | {mkt['flat']} |")
w(f"| 平均涨跌 | **{mkt['avg_chg']}%** |")
w(f"| 中位涨跌 | {mkt['med_chg']}% |")
w(f"| 涨停家数 | {mkt['limit_up']} |")
w(f"| 全市场样本 | {mkt['total']} |")
w()
w("**市场定调:** 全市场明显普跌,下跌家数远超上涨(5069:449),平均跌幅-4.02%,属于显著的风险释放/风险偏好下行日。资金明显向防御与低估值板块避险。")
w()
w("### 板块强弱")
w()
w("**强势板块(平均涨幅居前):**")
w()
w("| 板块 | 平均涨跌 | 成分股数 |")
w("|---|---|---|")
for it in d['industries']:
    w(f"| {it['industry']} | {it['avg_chg']:+.2f}% | {it['count']} |")
w()
w("**弱势板块(平均跌幅居前):**")
w()
w("| 板块 | 平均涨跌 | 成分股数 |")
w("|---|---|---|")
for it in d['industries_weak']:
    w(f"| {it['industry']} | {it['avg_chg']:+.2f}% | {it['count']} |")
w()
w("**板块信号:** 银行(+1.46%)、港口(+0.69%)、黄金(+0.47%)、煤炭(+0.41%)、水运(+0.19%)等防御/低估值/资源板块逆势走强;半导体(-7.28%)、通信设备(-6.88%)、元器件(-6.79%)、机床(-7.55%)等成长/科技板块领跌。典型的避险行情。")
w()
w("---")
w()
w("## 二、10个策略选股结果")
w()
w("每个策略呈现 Top10(不足则按实际命中数;硬筛无命中时按得分放宽列前5)。")
w()

for s in strats:
    w(f"### {s['name']}（{s['cat']}）  `策略: {s['key']}`")
    w()
    w(f"> {s['hint']}")
    if s['relaxed']:
        w(">")
        w("> ⚠️ 当日硬筛无标的通过(市场普跌致趋势/低波条件难满足),已**放宽**按综合得分列前5只最接近候选。")
    w()
    w("| # | 代码 | 名称 | 行业 | 收盘 | 涨跌% | 成交额(亿) | PE(TTM) | PB | 总市值(亿) | 换手% | 量比 | RSI14 | MACD | 均线多头 | 距MA20% | 信号分 | 得分 |")
    w("|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|")
    for i, r in enumerate(s['rows'], 1):
        pe = r['pe_ttm'] if r['pe_ttm'] is not None else '-'
        pb = r['pb'] if r['pb'] is not None else '-'
        macd = '多头' if r['macd_status']=='bullish' else '空头'
        mab = '是' if r['ma_bullish'] else '否'
        w(f"| {i} | {r['ts_code']} | {r['name']} | {r['industry']} | {r['close']} | {r['pct_chg']:+.2f} | {r['amount_yi']:.2f} | {pe} | {pb} | {r['total_mv_yi']:.0f} | {r['turnover_rate']} | {r['volume_ratio']} | {r['rsi14']} | {macd} | {mab} | {r['dist_ma20']:+.2f} | {r['signal_score']} | **{r['score']}** |")
    w()
    # 简要解读
    top3 = s['rows'][:3]
    names = "、".join(f"{r['name']}({r['ts_code']})" for r in top3)
    w(f"**解读:** 本策略Top3为 {names}。")
    if s['key'] in ('dual_low','quality_value','blue_chip_income'):
        w("价值/蓝筹类策略在大跌日集中命中银行、电力等低估值防御板块,符合避险逻辑;估值(PE/PB)与流动性匹配,均线多头且站上MA20,形态健康。")
    elif s['key'] in ('low_volatility_quality','shrink_pullback'):
        w("该策略要求低波动/趋势多头,在大跌日极少标的满足硬条件,放宽后仍指向银行等低波防御标的,表明当前趋势性标的多集中在防御板块。")
    elif s['key'] == 'volume_breakout':
        w("放量突破标的集中在化工/电器/焦炭等周期小盘,量价同步放大且站上MA20,但需警惕普跌环境下的假突破与追高风险。")
    elif s['key'] == 'capital_heat':
        w("资金热度标的以黄金、铝、化工、水运等资源/周期为主,资金活跃且量价同步,反映避险资金向资源品集中。")
    elif s['key'] == 'oversold_reversal':
        w("超跌反转标的跌幅可控(-3%~-3.5%)、成交仍在、多数仍维持均线多头,属可控回撤范畴,具备修复观察价值。")
    elif s['key'] in ('balanced_alpha','momentum_quality'):
        w("综合策略兼顾估值/资金/动量/稳定性,结果仍以银行、电力、铝业等防御+资源为主,体现多维度共振的稳健候选。")
    w()
    w("---")
    w()

# 跨策略共振
w("## 三、跨策略共振标的（重点关注）")
w()
w("统计各标的在10个策略Top名单中的命中次数。命中越多,多策略共振越强,建议重点关注。")
w()
w("### 硬命中共振（仅统计通过硬筛的策略,不含放宽）")
w()
hard_sorted = hit_hard.most_common()
w("| 代码 | 名称 | 行业 | 命中策略数 | 命中策略 |")
w("|---|---|---|---|---|")
# 名称/行业映射
namemap={}; indmap={}
for s in strats:
    for r in s['rows']:
        namemap[r['ts_code']]=r['name']; indmap[r['ts_code']]=r['industry']
for code, n in hard_sorted:
    if n < 2: continue
    w(f"| {code} | {namemap.get(code,'')} | {indmap.get(code,'')} | **{n}** | {'、'.join(hit_strats_hard[code])} |")
w()
w("### 含放宽命中共振（含放宽策略,供参考）")
w()
w("| 代码 | 名称 | 行业 | 命中策略数 |")
w("|---|---|---|---|")
for code, n in hit_all.most_common():
    if n < 3: continue
    w(f"| {code} | {namemap.get(code,'')} | {indmap.get(code,'')} | **{n}** |")
w()
w("**共振结论:** 银行板块(江苏银行、上海银行、中远海控、国电电力、青岛银行)在多个价值/蓝筹/综合策略中反复命中,形成最强共振,是当日避险资金的核心承接方向;南山铝业、海兴电力在动量/趋势类策略中形成次级共振,代表资源与出口链的活跃标的。")
w()
w("---")
w()
w("## 四、市场信号总结与风险提示")
w()
w("**市场信号:**")
w()
w(f"1. **风险偏好显著下行**:全市场平均跌幅-4.02%,下跌5069家 vs 上涨449家,属于明显的风险释放日,短线情绪偏空。")
w("2. **风格切换**:资金从半导体/通信/元器件等成长科技板块流出,集中涌入银行/港口/黄金/煤炭/水运等低估值防御与资源板块,呈现典型的避险行情。")
w("3. **策略共振指向防御**:10个策略的Top标的高度集中在银行、电力、资源,趋势类策略在大跌日硬命中极少,说明多头趋势标的稀缺,防御价值标的占优。")
w("4. **涨停梯队仍有44家**:局部题材仍活跃(如化工/资源脉冲),但需区分情绪脉冲与基本面支撑。")
w()
w("**风险提示:**")
w()
w("- 普跌行情下趋势类策略(放量突破/缩量回踩/低波质量)硬命中大幅减少,放宽结果仅供参考,不宜在趋势破坏环境中追涨。")
w("- 银行等防御板块虽共振强烈,但需关注其是避险脉冲还是基本面驱动;若后续风险偏好修复,资金可能从防御板块流出。")
w("- 放量突破类标的(化工/焦炭小盘)在弱势环境中假突破概率升高,需结合次日量能确认。")
w("- 超跌反转标的虽有修复观察价值,但若基本面或板块逻辑恶化,跌幅可能继续扩大,需严格控制仓位。")
w("- 本报告基于上一交易日收盘数据,10:00盘中实时行情可能已变化,仅供研究参考,不构成投资建议。")
w()
w("---")
w()
w(f"*数据口径: Tushare daily/daily_basic/stock_basic,基准交易日 {ld};技术指标基于近66个交易日计算(MA5/10/20、MACD、RSI14、ATR20、20日波动/振幅/回撤、量比、距MA20等)。*")
w(f"*评分: 各策略按其YAML factor_weights 多因子加权 + 技术信号分(0-100),本地评分(未调用LLM排名)。*")

report_path = os.path.join(REPORT_DIR, f'{ld}_10点策略报告.md')
with open(report_path, 'w') as f:
    f.write('\n'.join(lines))
print('REPORT_SAVED', report_path)
print('LINES', len(lines))

# 打印 Top3 摘要供推送
print('\n=== TOP3 SUMMARY ===')
for s in strats:
    t3 = s['rows'][:3]
    print(f"[{s['name']}] " + ' / '.join(f"{r['name']}({r['ts_code']}) {r['pct_chg']:+.2f}%" for r in t3))
print('\n=== RESONANCE (hard>=2) ===')
for code, n in hard_sorted:
    if n >= 2:
        print(f"{namemap.get(code)}({code}) x{n}: {'、'.join(hit_strats_hard[code])}")
