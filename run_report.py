# -*- coding: utf-8 -*-
"""10:00 盘中选股策略报告批量执行脚本。

一次抓取全市场快照（Tushare 最近可用交易日收盘数据），复用快照跑 10 个策略，
生成中文 Markdown 报告。
"""
from __future__ import annotations

import json
import logging
import os
import sys
import traceback
from collections import Counter, defaultdict
from datetime import datetime
from pathlib import Path

import pandas as pd
from dotenv import load_dotenv

# 加载 .env（含 TUSHARE_TOKEN、TUSHARE_TRADE_DATE 等）
load_dotenv("/workspace/.env")

# 导入 alphasift 模块
from alphasift import pipeline, snapshot as snap_mod
from alphasift.pipeline import screen
from alphasift.models import Pick, ScreenResult

logging.basicConfig(level=logging.WARNING, format="%(levelname)s %(name)s: %(message)s")
logger = logging.getLogger("report")

WORKSPACE = Path("/workspace")
REPORTS_DIR = WORKSPACE / "reports"
REPORTS_DIR.mkdir(parents=True, exist_ok=True)

# 10 个策略（按报告展示顺序）
STRATEGIES = [
    ("dual_low", "双低选股"),
    ("quality_value", "稳健价值"),
    ("blue_chip_income", "蓝筹收益质量"),
    ("low_volatility_quality", "低波质量"),
    ("volume_breakout", "放量突破"),
    ("shrink_pullback", "缩量回踩"),
    ("capital_heat", "资金热度"),
    ("oversold_reversal", "超跌反转"),
    ("balanced_alpha", "均衡多因子"),
    ("momentum_quality", "趋势质量"),
]

MAX_OUTPUT = 10


def fetch_snapshot_once():
    """抓取一次全市场快照。"""
    from alphasift.config import Config
    cfg = Config.from_env()
    df = snap_mod.fetch_snapshot_with_fallback(
        cfg.snapshot_source_priority,
        required_columns=None,
        fallback_snapshot_path=cfg.fallback_snapshot_path,
        fallback_max_age_hours=cfg.snapshot_fallback_max_age_hours,
        market="cn",
    )
    source = str(df.attrs.get("snapshot_source", ""))
    print(f"[snapshot] source={source} rows={len(df)}")
    return df, source


def patch_snapshot_fetcher(cached_df: pd.DataFrame):
    """Monkeypatch pipeline.fetch_snapshot_with_fallback 返回缓存快照副本。

    必须返回 copy，否则前序策略对 DataFrame 的就地改写（新增得分列、过滤）
    会污染后续策略的快照输入，导致 daily_k 类策略硬筛异常归零。
    """
    base_source = str(cached_df.attrs.get("snapshot_source", "tushare_cached"))

    def _return_cached(*args, **kwargs):
        df = cached_df.copy()
        df.attrs["snapshot_source"] = base_source
        return df
    pipeline.fetch_snapshot_with_fallback = _return_cached


def run_strategy(name: str) -> ScreenResult | None:
    """跑单个策略。

    显式传 daily_enrich=False：仅当策略本身必需日 K（daily_needed=True，即
    low_volatility_quality / volume_breakout / shrink_pullback）时才抓日 K。
    避免 7 个非 daily 策略也各自抓 80 条日 K 累计触发 Tushare 限流，
    进而导致后续 daily_k 策略增强失败、硬筛全归零。
    """
    try:
        result = screen(
            name,
            market="cn",
            max_output=MAX_OUTPUT,
            use_llm=False,
            post_analyzers=[],
            explain_filters=False,
            daily_enrich=False,
        )
        return result
    except Exception as exc:
        logger.error("策略 %s 执行失败: %s", name, exc)
        traceback.print_exc()
        return None


def build_market_overview(df: pd.DataFrame) -> dict:
    """从快照计算市场概况。"""
    out = {}
    # 涨跌家数（剔除停牌/无价格）
    valid = df[df["price"].notna() & (df["price"] > 0)].copy()
    chg = valid["change_pct"]
    out["total"] = int(len(valid))
    out["advancing"] = int((chg > 0).sum())
    out["declining"] = int((chg < 0).sum())
    out["flat"] = int((chg == 0).sum())
    out["limit_up"] = int((chg >= 9.8).sum())
    out["limit_down"] = int((chg <= -9.8).sum())
    out["avg_change"] = round(float(chg.mean()), 3) if len(chg) else 0.0
    out["median_change"] = round(float(chg.median()), 3) if len(chg) else 0.0

    # 成交额（亿元）
    if "amount" in valid.columns:
        amt = valid["amount"].fillna(0)
        out["total_amount_yi"] = round(float(amt.sum()) / 1e8, 1)
    else:
        out["total_amount_yi"] = 0.0

    # 板块强弱：按行业聚合平均涨跌幅
    if "industry" in valid.columns:
        ind = valid[valid["industry"].notna() & (valid["industry"].astype(str).str.len() > 0)]
        if not ind.empty:
            grp = ind.groupby("industry")["change_pct"].agg(["mean", "count"])
            grp = grp[grp["count"] >= 3].sort_values("mean", ascending=False)
            out["top_sectors"] = [
                {"name": k, "avg_change": round(float(v["mean"]), 2), "count": int(v["count"])}
                for k, v in grp.head(10).iterrows()
            ]
            out["bottom_sectors"] = [
                {"name": k, "avg_change": round(float(v["mean"]), 2), "count": int(v["count"])}
                for k, v in grp.tail(10).iterrows()
            ]
        else:
            out["top_sectors"] = []
            out["bottom_sectors"] = []
    else:
        out["top_sectors"] = []
        out["bottom_sectors"] = []
    return out


def fmt_amt(v) -> str:
    """成交额格式化为亿元。"""
    try:
        f = float(v or 0)
        return f"{f / 1e8:.2f}" if f > 0 else "-"
    except Exception:
        return "-"


def fmt_mv(v) -> str:
    """总市值格式化为亿元。"""
    try:
        f = float(v or 0)
        return f"{f / 1e8:.1f}" if f > 0 else "-"
    except Exception:
        return "-"


def fmt_num(v, nd=2) -> str:
    try:
        if v is None:
            return "-"
        f = float(v)
        if f == 0 or pd.isna(f):
            return "-"
        return f"{f:.{nd}f}"
    except Exception:
        return "-"


def tech_summary(p: Pick) -> str:
    """关键技术指标摘要。"""
    parts = []
    if p.ma_bullish is not None:
        parts.append("均线多头" if p.ma_bullish else "均线非多头")
    if p.macd_status:
        parts.append(f"MACD:{p.macd_status}")
    if p.rsi_status:
        parts.append(f"RSI:{p.rsi_status}")
    if p.volume_ratio is not None and p.volume_ratio:
        parts.append(f"量比{p.volume_ratio:.2f}")
    if p.breakout_20d_pct is not None and p.breakout_20d_pct:
        parts.append(f"20日突破{p.breakout_20d_pct:+.1f}%")
    if p.volatility_20d_pct is not None and p.volatility_20d_pct:
        parts.append(f"波动率{p.volatility_20d_pct:.1f}%")
    if p.pullback_to_ma20_pct is not None and p.pullback_to_ma20_pct:
        parts.append(f"距MA20{p.pullback_to_ma20_pct:+.1f}%")
    return " / ".join(parts) if parts else "-"


def render_strategy_section(name: str, display: str, result: ScreenResult | None) -> str:
    """渲染单个策略章节。"""
    lines = []
    lines.append(f"### {display}（{name}）")
    if result is None:
        lines.append("> 策略执行失败，跳过。\n")
        return "\n".join(lines)

    lines.append(
        f"- 快照标的：{result.snapshot_count} | 硬筛后：{result.after_filter_count} | "
        f"数据源：{result.snapshot_source or '-'} | 命中数：{len(result.picks)}"
    )
    if result.degradation:
        # 只展示前 3 条降级说明
        short = [d for d in result.degradation if d][:3]
        if short:
            lines.append(f"- 备注：{' | '.join(short)}")

    picks = result.picks
    if not picks:
        lines.append("\n> 本策略无标的通过硬筛条件。\n")
        return "\n".join(lines)

    lines.append("")
    lines.append("| 排名 | 代码 | 名称 | 行业 | 收盘价 | 涨跌% | 成交额(亿) | PE | PB | 总市值(亿) | 关键技术指标 | 得分 |")
    lines.append("|---|---|---|---|---|---|---|---|---|---|---|---|")
    for p in picks:
        lines.append(
            f"| {p.rank} | {p.code} | {p.name} | {p.industry or '-'} | "
            f"{fmt_num(p.price)} | {fmt_num(p.change_pct)} | {fmt_amt(p.amount)} | "
            f"{fmt_num(p.pe_ratio)} | {fmt_num(p.pb_ratio)} | {fmt_mv(p.total_mv)} | "
            f"{tech_summary(p)} | {fmt_num(p.final_score, 1)} |"
        )
    lines.append("")
    # 简要解读
    if picks:
        top3 = ", ".join(f"{p.name}({p.code})" for p in picks[:3])
        lines.append(f"**解读**：本策略 Top3 为 {top3}。{brief_commentary(name, picks)}")
    lines.append("")
    return "\n".join(lines)


def brief_commentary(name: str, picks: list[Pick]) -> str:
    """针对策略风格的简要点评。"""
    if name == "dual_low":
        return "聚焦低 PE+低 PB 价值锚，适合防守型配置，注意规避价值陷阱与周期顶点。"
    if name == "quality_value":
        return "估值合理且流动性充足，兼顾温和活跃度确认，攻守相对均衡。"
    if name == "blue_chip_income":
        return "高流动性大盘蓝筹，强调收益质量与稳定成交，适合底仓配置。"
    if name == "low_volatility_quality":
        return "低波动浅回撤防守选股，适合震荡市或风险偏好下行阶段。"
    if name == "volume_breakout":
        return "放量突破关键阻力位，趋势启动信号，需配合后续量能延续性验证。"
    if name == "shrink_pullback":
        return "上升趋势中缩量回踩均线支撑，趋势延续入场机会，关注回踩不破。"
    if name == "capital_heat":
        return "资金活跃且量价同步，短线动量机会，需警惕过度追涨风险。"
    if name == "oversold_reversal":
        return "跌幅可控且流动性修复的反转候选，左侧布局需严格止损纪律。"
    if name == "balanced_alpha":
        return "估值+资金+动量+稳定性均衡打分，跨风格综合候选，分散度较高。"
    if name == "momentum_quality":
        return "趋势确认叠加基本面质量约束，中线候选，避免单一题材过度集中。"
    return ""


def build_cross_strategy_summary(results: dict) -> str:
    """跨策略汇总：多策略同时命中的标的。"""
    code_hits = defaultdict(list)  # code -> [(strategy, rank, name, industry)]
    for name, (display, result) in results.items():
        if not result or not result.picks:
            continue
        for p in result.picks:
            code_hits[p.code].append((name, p.rank, p.name, p.industry, p.final_score, p.change_pct))

    multi = {c: v for c, v in code_hits.items() if len(v) >= 2}
    if not multi:
        return "### 跨策略共振汇总\n\n本次无标的同时命中 2 个及以上策略。\n"

    # 按命中策略数降序，再按平均排名升序
    sorted_codes = sorted(
        multi.items(),
        key=lambda kv: (-len(kv[1]), sum(r for _, r, *_ in kv[1]) / len(kv[1])),
    )

    lines = ["### 跨策略共振汇总（多策略同时命中）", ""]
    lines.append("| 代码 | 名称 | 行业 | 命中策略数 | 命中策略(排名) | 最新涨跌% | 平均得分 |")
    lines.append("|---|---|---|---|---|---|---|")
    for code, hits in sorted_codes[:20]:
        name = hits[0][2]
        industry = hits[0][3] or "-"
        n = len(hits)
        strat_str = ", ".join(f"{s}#{r}" for s, r, *_ in hits)
        avg_score = sum(h[4] for h in hits) / n
        chg = hits[0][5]
        lines.append(
            f"| {code} | {name} | {industry} | {n} | {strat_str} | {fmt_num(chg)} | {fmt_num(avg_score, 1)} |"
        )
    lines.append("")
    lines.append(
        "> 重点关注命中 3 个及以上策略的标的——多因子共振通常意味着"
        "趋势、资金、估值与稳定性多重验证，具备更高的中线胜率与容错空间。"
    )
    lines.append("")
    return "\n".join(lines)


def render_market_overview(ov: dict, trade_date: str) -> str:
    lines = ["## 一、市场概况", ""]
    lines.append(
        f"- 数据基准日：{trade_date}（最近可用交易日收盘数据，10:00 盘中口径）"
    )
    lines.append(
        f"- 全市场有效标的：{ov['total']} | 上涨 {ov['advancing']} / 下跌 {ov['declining']} / 平盘 {ov['flat']}"
    )
    lines.append(
        f"- 涨停 {ov['limit_up']} / 跌停 {ov['limit_down']} | "
        f"平均涨跌 {ov['avg_change']:+.2f}% | 中位数 {ov['median_change']:+.2f}%"
    )
    lines.append(f"- 全市场成交额：{ov['total_amount_yi']:.1f} 亿元")

    if ov.get("top_sectors"):
        lines.append("")
        lines.append("**领涨板块 Top10**（按行业平均涨幅，样本≥3）")
        lines.append("")
        lines.append("| 板块 | 平均涨幅% | 标的数 |")
        lines.append("|---|---|---|")
        for s in ov["top_sectors"]:
            lines.append(f"| {s['name']} | {s['avg_change']:+.2f} | {s['count']} |")

    if ov.get("bottom_sectors"):
        lines.append("")
        lines.append("**领跌板块 Top10**")
        lines.append("")
        lines.append("| 板块 | 平均涨幅% | 标的数 |")
        lines.append("|---|---|---|")
        for s in ov["bottom_sectors"]:
            lines.append(f"| {s['name']} | {s['avg_change']:+.2f} | {s['count']} |")
    lines.append("")
    return "\n".join(lines)


def build_signal_summary(ov: dict, results: dict) -> str:
    """市场信号总结 + 风险提示。"""
    lines = ["## 四、市场信号总结与风险提示", ""]
    # 大势判断
    adv = ov["advancing"]
    dec = ov["declining"]
    avg = ov["avg_change"]
    if adv > dec * 1.5 and avg > 0.5:
        regime = "偏多（涨多跌少，赚钱效应回升）"
    elif dec > adv * 1.5 and avg < -0.5:
        regime = "偏空（跌多涨少，风险释放中）"
    else:
        regime = "震荡（多空相对均衡，结构分化）"
    lines.append(f"- **大势研判**：{regime}，上涨/下跌 = {adv}/{dec}，平均涨跌 {avg:+.2f}%。")

    # 统计各策略命中数
    hit_counts = {name: (len(r.picks) if r else 0) for name, (disp, r) in results.items()}
    active = sum(1 for c in hit_counts.values() if c >= 5)
    lines.append(
        f"- **策略广度**：10 个策略中 {active} 个策略命中≥5 只候选，"
        f"反映当前市场结构性机会{'较充足' if active >= 6 else '相对收敛' if active >= 3 else '明显不足'}。"
    )

    # 跨策略共振
    code_hits = defaultdict(int)
    for name, (disp, r) in results.items():
        if not r:
            continue
        for p in r.picks:
            code_hits[p.code] += 1
    multi3 = [c for c, n in code_hits.items() if n >= 3]
    lines.append(
        f"- **共振信号**：{len(multi3)} 只标的命中≥3 个策略，"
        f"{'多因子共振明显，中线机会值得重点跟踪' if multi3 else '共振信号偏弱，以结构性机会为主'}。"
    )

    lines.append("")
    lines.append("**风险提示**")
    lines.append(
        "- 本报告基于最近交易日收盘数据生成，10:00 盘中实际行情可能已发生变化，"
        "盘中价格波动、集合竞价异动等无法在本报告中体现，下单前请以实时盘口为准。"
    )
    lines.append("- 策略结果为量化筛选候选，不构成投资建议；个股需结合基本面、公告、行业景气与个人风险承受能力独立判断。")
    lines.append("- 趋势/动量类策略在追涨阶段需严格设置止损；价值类策略需警惕周期顶点与价值陷阱。")
    lines.append("- 涨停/跌停标的流动性受限，实盘成交难度大，追高需谨慎。")
    lines.append("")
    return "\n".join(lines)


def main():
    print("=" * 60)
    print("10:00 盘中选股策略报告 - 批量执行")
    print("=" * 60)

    # 1. 抓取快照一次
    cached_df, snap_source = fetch_snapshot_once()
    trade_date = os.getenv("TUSHARE_TRADE_DATE", datetime.now().strftime("%Y%m%d"))

    # 2. 计算市场概况
    print("[overview] 计算市场概况...")
    overview = build_market_overview(cached_df)

    # 3. Patch 快照抓取器，复用缓存
    patch_snapshot_fetcher(cached_df)

    # 4. 跑 10 个策略
    results: dict = {}  # name -> (display, ScreenResult)
    raw_results: dict = {}
    for name, display in STRATEGIES:
        print(f"[strategy] {name} ({display}) ...")
        result = run_strategy(name)
        results[name] = (display, result)
        raw_results[name] = result
        n_picks = len(result.picks) if result else 0
        n_filter = result.after_filter_count if result else 0
        print(f"  -> after_filter={n_filter} picks={n_picks}")

    # 5. 持久化原始结果（便于复盘）
    runs_dir = REPORTS_DIR / "runs"
    runs_dir.mkdir(parents=True, exist_ok=True)
    for name, (display, result) in results.items():
        if not result:
            continue
        out = {
            "strategy": name,
            "display_name": display,
            "trade_date": trade_date,
            "snapshot_source": result.snapshot_source,
            "snapshot_count": result.snapshot_count,
            "after_filter_count": result.after_filter_count,
            "picks": [
                {
                    "rank": p.rank,
                    "code": p.code,
                    "name": p.name,
                    "industry": p.industry,
                    "price": p.price,
                    "change_pct": p.change_pct,
                    "amount": p.amount,
                    "pe_ratio": p.pe_ratio,
                    "pb_ratio": p.pb_ratio,
                    "total_mv": p.total_mv,
                    "turnover_rate": p.turnover_rate,
                    "volume_ratio": p.volume_ratio,
                    "final_score": p.final_score,
                    "ma_bullish": p.ma_bullish,
                    "macd_status": p.macd_status,
                    "rsi_status": p.rsi_status,
                    "breakout_20d_pct": p.breakout_20d_pct,
                    "volatility_20d_pct": p.volatility_20d_pct,
                    "pullback_to_ma20_pct": p.pullback_to_ma20_pct,
                }
                for p in result.picks
            ],
        }
        (runs_dir / f"{trade_date}_{name}.json").write_text(
            json.dumps(out, ensure_ascii=False, indent=2), encoding="utf-8"
        )

    # 6. 生成 Markdown 报告
    print("[report] 生成 Markdown 报告...")
    report_lines = []
    report_lines.append(f"# A股 10:00 盘中选股策略报告（{trade_date}）")
    report_lines.append("")
    report_lines.append(
        f"> 生成时间：{datetime.now().strftime('%Y-%m-%d %H:%M')} | "
        f"数据基准日：{trade_date}（Tushare 最近可用交易日收盘） | "
        f"策略数：10 | 数据源：{snap_source}"
    )
    report_lines.append("")

    report_lines.append(render_market_overview(overview, trade_date))

    report_lines.append("## 二、10 策略选股结果")
    report_lines.append("")
    for name, display in STRATEGIES:
        result = raw_results.get(name)
        report_lines.append(render_strategy_section(name, display, result))

    report_lines.append("## 三、跨策略汇总")
    report_lines.append("")
    report_lines.append(build_cross_strategy_summary(results))

    report_lines.append(build_signal_summary(overview, results))

    report_lines.append("---")
    report_lines.append("*本报告由 alphasift 量化框架自动生成，10 个策略并行筛选，本地 score 评分排序。*")

    report_path = REPORTS_DIR / f"{trade_date}_10点策略报告.md"
    report_path.write_text("\n".join(report_lines), encoding="utf-8")
    print(f"[done] 报告已保存：{report_path}")

    # 7. 输出摘要到 stdout（供推送）
    print("\n" + "=" * 60)
    print("报告摘要（每策略 Top3 + 市场信号 + 共振标的）")
    print("=" * 60)
    summary = {
        "trade_date": trade_date,
        "market": {
            "advancing": overview["advancing"],
            "declining": overview["declining"],
            "avg_change": overview["avg_change"],
            "total_amount_yi": overview["total_amount_yi"],
        },
        "strategies_top3": {},
        "cross_resonance": [],
    }
    for name, display in STRATEGIES:
        r = raw_results.get(name)
        top3 = []
        if r and r.picks:
            for p in r.picks[:3]:
                top3.append({
                    "code": p.code, "name": p.name, "industry": p.industry,
                    "change_pct": round(p.change_pct, 2),
                    "final_score": round(p.final_score, 1),
                })
        summary["strategies_top3"][f"{display}({name})"] = top3

    # 共振标的
    code_hits = defaultdict(list)
    for name, (display, r) in results.items():
        if not r:
            continue
        for p in r.picks:
            code_hits[p.code].append((name, p.rank, p.name, p.industry))
    for code, hits in sorted(code_hits.items(), key=lambda kv: (-len(kv[1]),)):
        if len(hits) >= 2:
            summary["cross_resonance"].append({
                "code": code,
                "name": hits[0][2],
                "industry": hits[0][3],
                "hit_count": len(hits),
                "strategies": [h[0] for h in hits],
            })

    summary_path = REPORTS_DIR / f"{trade_date}_摘要.json"
    summary_path.write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
