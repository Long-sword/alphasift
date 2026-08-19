# -*- coding: utf-8 -*-
"""
A股策略选股脚本 - 5个策略各输出Top10
策略1: dual_low 双低选股（价值）
策略2: quality_value 稳健价值（价值）
策略3: blue_chip_income 蓝筹收益质量（收益）
策略4: low_volatility_quality 低波质量（质量）
策略5: oversold_reversal 超跌反转（反转）
"""
import os
import json
import time
import numpy as np
import pandas as pd
import tushare as ts

# ========== 基础配置 ==========
TUSHARE_TOKEN = "9f640d421f866dde9a1888ab4193f77c43659d47446d717ffa343023"
TRADE_DATE = "20260819"
TOP_ACTIVE = 300           # 成交额Top活跃股
KLINE_DAYS = 30            # 取30日K线
KLINE_LIMIT = 40           # 实际请求条数（多取点保险）
TOP_N = 10                 # 每策略输出Top10
REPORT_DIR = "/workspace/reports"
JSON_PATH = os.path.join(REPORT_DIR, "value_quality_result.json")

os.makedirs(REPORT_DIR, exist_ok=True)

ts.set_token(TUSHARE_TOKEN)
pro = ts.pro_api()


# ========== 工具函数 ==========
def safe_call(func, *args, **kwargs):
    """带简单重试的接口调用"""
    for i in range(3):
        try:
            return func(*args, **kwargs)
        except Exception as e:
            print(f"  [retry {i+1}/3] 接口异常: {e}")
            time.sleep(1.5)
    return None


def calc_rsi(closes, period=14):
    """标准RSI"""
    closes = np.asarray(closes, dtype=float)
    if len(closes) < period + 1:
        return np.nan
    deltas = np.diff(closes)
    gains = np.where(deltas > 0, deltas, 0.0)
    losses = np.where(deltas < 0, -deltas, 0.0)
    # 用SMA（Wilder平滑）近似
    avg_gain = np.mean(gains[-period:])
    avg_loss = np.mean(losses[-period:])
    if avg_loss == 0:
        return 100.0
    rs = avg_gain / avg_loss
    return 100 - (100 / (1 + rs))


def calc_indicators(kline_df, daily_row):
    """
    根据K线计算技术指标
    kline_df: 包含 close/vol, 按trade_date升序
    daily_row: 当日daily_basic的数据
    """
    closes = kline_df["close"].values.astype(float)
    vols = kline_df["vol"].values.astype(float)
    if len(closes) < 21:
        return None

    ma5 = np.mean(closes[-5:])
    ma10 = np.mean(closes[-10:])
    ma20 = np.mean(closes[-20:])
    last_close = closes[-1]

    # 20日波动率
    log_ret = np.diff(np.log(closes[-21:]))
    vol_20 = np.std(log_ret) * np.sqrt(252) * 100

    # 20日涨幅
    ret_20 = (last_close / closes[-21] - 1) * 100

    # 当日涨跌幅（pct_chg已经存）
    pct_chg = float(daily_row.get("pct_chg", 0) or 0)

    # 成交额（万元）
    amount = float(daily_row.get("amount", 0) or 0)
    amount_yi = amount / 10000.0  # 万元 -> 亿元

    # 总市值（万元）
    total_mv = float(daily_row.get("total_mv", 0) or 0)
    total_mv_yi = total_mv / 10000.0  # 万元 -> 亿元

    # PE/PB/换手率/股息率
    pe = float(daily_row.get("pe", 0) or 0)
    pe_ttm = float(daily_row.get("pe_ttm", 0) or 0)
    pb = float(daily_row.get("pb", 0) or 0)
    turnover = float(daily_row.get("turnover_rate", 0) or 0)
    dv_ttm = float(daily_row.get("dv_ttm", 0) or 0)

    # RSI(14)
    rsi14 = calc_rsi(closes, 14)

    # 量比
    avg_vol_20 = np.mean(vols[-21:-1]) if len(vols) >= 21 else np.mean(vols[:-1])
    vol_ratio = (vols[-1] / avg_vol_20) if avg_vol_20 > 0 else np.nan

    # 距MA20%
    dist_ma20 = (last_close / ma20 - 1) * 100

    # 均线多头
    ma_bull = ma5 > ma10 > ma20

    return {
        "close": last_close,
        "ma5": ma5,
        "ma10": ma10,
        "ma20": ma20,
        "ma_bull": ma_bull,
        "vol_20": vol_20,
        "ret_20": ret_20,
        "pct_chg": pct_chg,
        "amount_yi": amount_yi,
        "total_mv_yi": total_mv_yi,
        "pe": pe,
        "pe_ttm": pe_ttm,
        "pb": pb,
        "turnover": turnover,
        "dv_ttm": dv_ttm,
        "rsi14": rsi14,
        "vol_ratio": vol_ratio,
        "dist_ma20": dist_ma20,
    }


# ========== 数据准备 ==========
def prepare_data():
    print("\n========== 数据准备 ==========")
    # 1. 全市场当日行情
    print(f"[1/4] 拉取 pro.daily(trade_date={TRADE_DATE}) 全市场行情...")
    daily_df = safe_call(pro.daily, trade_date=TRADE_DATE)
    if daily_df is None or daily_df.empty:
        raise RuntimeError("pro.daily 返回空，请确认交易日或token权限")
    print(f"  全市场当日行情：{len(daily_df)} 只")

    # 2. 全市场PE/PB/市值/换手率
    print(f"[2/4] 拉取 pro.daily_basic(trade_date={TRADE_DATE}) 估值数据...")
    basic_val_df = safe_call(pro.daily_basic, trade_date=TRADE_DATE)
    if basic_val_df is None or basic_val_df.empty:
        raise RuntimeError("pro.daily_basic 返回空")
    print(f"  估值数据：{len(basic_val_df)} 只")

    # 3. 股票列表/行业
    print(f"[3/4] 拉取 pro.stock_basic() 股票基础信息...")
    stock_basic_df = safe_call(pro.stock_basic)
    if stock_basic_df is None or stock_basic_df.empty:
        raise RuntimeError("pro.stock_basic 返回空")
    print(f"  股票基础信息：{len(stock_basic_df)} 只")

    # 字段处理
    stock_basic_df = stock_basic_df[["ts_code", "name", "industry", "list_date"]].copy()

    # 合并行情+估值+基础
    merged = daily_df.merge(
        basic_val_df, on="ts_code", how="left", suffixes=("", "_bv")
    )
    merged = merged.merge(stock_basic_df, on="ts_code", how="left")

    # 排除条件
    # ST/*ST/退市
    def is_excluded_name(name):
        if not isinstance(name, str):
            return True
        n = name.upper()
        return ("ST" in n) or ("*ST" in n) or (n.endswith("退"))

    name_mask = merged["name"].apply(is_excluded_name)
    # 次新：上市<1年
    list_date_str = TRADE_DATE
    cutoff_listdate = (pd.Timestamp(TRADE_DATE) - pd.DateOffset(years=1)).strftime("%Y%m%d")
    not_new_mask = merged["list_date"].fillna("19900101") < cutoff_listdate
    # 成交额>=2亿
    amt_mask = (merged["amount"].fillna(0) / 10000.0) >= 2.0  # amount单位:千元 -> 亿元

    keep = (~name_mask) & not_new_mask & amt_mask
    filtered = merged[keep].copy()
    print(f"  排除ST/次新/<2亿后剩：{len(filtered)} 只")

    # 4. 取成交额Top300
    filtered = filtered.sort_values("amount", ascending=False).head(TOP_ACTIVE).reset_index(drop=True)
    print(f"  取成交额Top{TOP_ACTIVE}：{len(filtered)} 只")

    return filtered


def fetch_klines(filtered_df):
    """对每只股票拉取近30日K线"""
    print(f"\n[4/4] 对Top{len(filtered_df)}只活跃股逐个拉取{KLINE_DAYS}日K线...")
    rows = []
    total = len(filtered_df)
    for i, r in filtered_df.iterrows():
        code = r["ts_code"]
        if (i + 1) % 20 == 0 or i == 0:
            print(f"  进度 {i+1}/{total} ...")
        kdf = safe_call(
            pro.daily, ts_code=code, end_date=TRADE_DATE, limit=KLINE_LIMIT
        )
        if kdf is None or len(kdf) < 21:
            continue
        # 按日期升序
        kdf = kdf.sort_values("trade_date").reset_index(drop=True)
        ind = calc_indicators(kdf, r)
        if ind is None:
            continue
        rec = {
            "ts_code": code,
            "name": r.get("name", ""),
            "industry": r.get("industry", ""),
            "pct_chg": r.get("pct_chg", 0),
            "amount": r.get("amount", 0),
        }
        rec.update(ind)
        rows.append(rec)
        time.sleep(0.05)  # 轻微节流
    df = pd.DataFrame(rows)
    print(f"  成功获取K线并算出指标：{len(df)} 只")
    return df


# ========== 5个策略 ==========
def strat_dual_low(df):
    """策略1: dual_low 双低选股（价值）"""
    cond = (
        (df["pe"] > 0) & (df["pe"] <= 30) &
        (df["pb"] > 0) & (df["pb"] <= 3) &
        (df["amount_yi"] >= 5) &
        (df["total_mv_yi"] >= 50)
    )
    sub = df[cond].copy()

    def score(row):
        s = 0.0
        # PE越低分越高(30分封顶): PE=1时30分，PE=30时0分
        pe_s = max(0, min(30, (30 - row["pe"]) / 29 * 30))
        s += pe_s
        # PB越低分越高(20分): PB=0时20分，PB=3时0分
        pb_s = max(0, min(20, (3 - row["pb"]) / 3 * 20))
        s += pb_s
        if row["amount_yi"] >= 10:
            s += 10
        if 0 <= row["ret_20"] <= 15:
            s += 15
        if -2 <= row["pct_chg"] <= 3:
            s += 10
        if row["total_mv_yi"] >= 200:
            s += 15
        return s

    sub["score"] = sub.apply(score, axis=1)
    return sub.sort_values("score", ascending=False).head(TOP_N)


def strat_quality_value(df):
    """策略2: quality_value 稳健价值（价值）"""
    cond = (
        (df["pe"] >= 10) & (df["pe"] <= 40) &
        (df["pb"] > 0) & (df["pb"] <= 5) &
        (df["amount_yi"] >= 3) &
        (df["total_mv_yi"] >= 100) &
        (df["vol_20"] <= 50)
    )
    sub = df[cond].copy()

    def score(row):
        s = 0.0
        if 15 <= row["pe"] <= 30:
            s += 15
        if row["pb"] < 3:
            s += 10
        if 0 <= row["ret_20"] <= 15:
            s += 15
        if row["ma_bull"]:
            s += 10
        if row["amount_yi"] >= 10:
            s += 10
        if 3 <= row["turnover"] <= 8:
            s += 10
        if row["vol_20"] < 30:
            s += 10
        return s

    sub["score"] = sub.apply(score, axis=1)
    return sub.sort_values("score", ascending=False).head(TOP_N)


def strat_blue_chip_income(df):
    """策略3: blue_chip_income 蓝筹收益质量（收益）"""
    cond = (
        (df["total_mv_yi"] >= 300) &
        (df["amount_yi"] >= 5) &
        (df["pe"] > 0) & (df["pe"] <= 30) &
        (df["pb"] <= 5)
    )
    sub = df[cond].copy()

    def score(row):
        s = 0.0
        if row["total_mv_yi"] >= 1000:
            s += 20
        if row["amount_yi"] >= 20:
            s += 15
        if row["pe"] < 20:
            s += 15
        if row["dv_ttm"] >= 2:
            s += 15
        if row["ma_bull"]:
            s += 10
        if 0 <= row["ret_20"] <= 10:
            s += 10
        if -2 <= row["pct_chg"] <= 2:
            s += 10
        if 1 <= row["turnover"] <= 5:
            s += 10
        return s

    sub["score"] = sub.apply(score, axis=1)
    return sub.sort_values("score", ascending=False).head(TOP_N)


def strat_low_volatility_quality(df):
    """策略4: low_volatility_quality 低波质量（质量）"""
    cond = (
        (df["vol_20"] <= 30) &
        (df["pe"] > 0) & (df["pe"] <= 50) &
        (df["pb"] > 0) & (df["pb"] <= 5) &
        (df["amount_yi"] >= 3) &
        (df["total_mv_yi"] >= 100)
    )
    sub = df[cond].copy()

    def score(row):
        s = 0.0
        if row["vol_20"] <= 15:
            s += 25
        elif row["vol_20"] <= 25:
            s += 15
        elif row["vol_20"] <= 30:
            s += 8
        if row["ma_bull"]:
            s += 15
        if 0 <= row["ret_20"] <= 10:
            s += 15
        if 10 <= row["pe"] <= 30:
            s += 10
        if row["amount_yi"] >= 5:
            s += 10
        if 0 <= row["dist_ma20"] <= 10:
            s += 10
        if -2 <= row["pct_chg"] <= 2:
            s += 10
        return s

    sub["score"] = sub.apply(score, axis=1)
    return sub.sort_values("score", ascending=False).head(TOP_N)


def strat_oversold_reversal(df):
    """策略5: oversold_reversal 超跌反转（反转）"""
    cond = (
        (df["ret_20"] <= -10) &
        (df["amount_yi"] >= 2) &
        (df["total_mv_yi"] >= 30) &
        (df["rsi14"] <= 40)
    )
    sub = df[cond].copy()

    def score(row):
        s = 0.0
        if row["ret_20"] <= -20:
            s += 25
        elif row["ret_20"] <= -15:
            s += 20
        elif row["ret_20"] <= -10:
            s += 15
        rsi = row["rsi14"]
        if rsi < 25:
            s += 20
        elif rsi < 35:
            s += 15
        elif rsi <= 40:
            s += 10
        if row["pct_chg"] > 0:
            s += 15
        if row["vol_ratio"] > 1.5:
            s += 10
        if row["close"] > row["ma5"]:
            s += 10
        return s

    sub["score"] = sub.apply(score, axis=1)
    return sub.sort_values("score", ascending=False).head(TOP_N)


# ========== 输出 ==========
def fmt_df(sub, key_metric_cols):
    """格式化为展示表"""
    show = sub[[
        "ts_code", "name", "industry", "close", "pct_chg",
        "amount_yi", "pe", "pb", "total_mv_yi"
    ] + key_metric_cols + ["score"]].copy()
    show["close"] = show["close"].round(2)
    show["pct_chg"] = show["pct_chg"].round(2)
    show["amount_yi"] = show["amount_yi"].round(2)
    show["pe"] = show["pe"].round(2)
    show["pb"] = show["pb"].round(2)
    show["total_mv_yi"] = show["total_mv_yi"].round(1)
    for c in key_metric_cols:
        show[c] = show[c].round(2)
    show["score"] = show["score"].round(2)
    return show


COLS_MAP = {
    "strat1": ["ret_20", "vol_20"],
    "strat2": ["ret_20", "vol_20", "turnover", "ma_bull"],
    "strat3": ["dv_ttm", "turnover", "ret_20", "ma_bull"],
    "strat4": ["vol_20", "dist_ma20", "ret_20", "ma_bull"],
    "strat5": ["ret_20", "rsi14", "vol_ratio", "ma5"],
}


def to_records(sub, key_cols):
    recs = []
    for _, r in sub.iterrows():
        recs.append({
            "代码": r["ts_code"],
            "名称": r["name"],
            "行业": r["industry"],
            "收盘": round(float(r["close"]), 2),
            "涨跌幅%": round(float(r["pct_chg"]), 2),
            "成交额亿": round(float(r["amount_yi"]), 2),
            "PE": round(float(r["pe"]), 2),
            "PB": round(float(r["pb"]), 2),
            "总市值亿": round(float(r["total_mv_yi"]), 1),
            "关键技术指标": {c: round(float(r[c]), 2) for c in key_cols},
            "得分": round(float(r["score"]), 2),
        })
    return recs


def main():
    print("========== A股5策略选股 - 启动 ==========")
    print(f"交易日: {TRADE_DATE}  Top活跃: {TOP_ACTIVE}  K线天数: {KLINE_DAYS}")

    filtered = prepare_data()
    df = fetch_klines(filtered)
    if df.empty:
        raise RuntimeError("无可用计算样本，请检查数据或权限")

    print("\n========== 策略评分 ==========")
    s1 = strat_dual_low(df)
    s2 = strat_quality_value(df)
    s3 = strat_blue_chip_income(df)
    s4 = strat_low_volatility_quality(df)
    s5 = strat_oversold_reversal(df)

    strat_map = [
        ("策略1: dual_low 双低选股(价值)", s1, COLS_MAP["strat1"]),
        ("策略2: quality_value 稳健价值(价值)", s2, COLS_MAP["strat2"]),
        ("策略3: blue_chip_income 蓝筹收益质量(收益)", s3, COLS_MAP["strat3"]),
        ("策略4: low_volatility_quality 低波质量(质量)", s4, COLS_MAP["strat4"]),
        ("策略5: oversold_reversal 超跌反转(反转)", s5, COLS_MAP["strat5"]),
    ]

    print("\n========== Top10 结果 ==========")
    json_out = {}
    for title, sub, key_cols in strat_map:
        print(f"\n--- {title} (命中样本={len(sub)}) ---")
        if sub.empty:
            print("  无命中样本")
            json_out[title] = []
            continue
        show = fmt_df(sub, key_cols)
        show.columns = ["代码", "名称", "行业", "收盘", "涨跌%", "成交额亿",
                        "PE", "PB", "总市值亿"] + key_cols + ["得分"]
        # 显示
        with pd.option_context("display.max_rows", None, "display.width", 200,
                               "display.unicode.east_asian", True,
                               "display.max_colwidth", 12):
            print(show.to_string(index=False))
        json_out[title] = to_records(sub, key_cols)

    with open(JSON_PATH, "w", encoding="utf-8") as f:
        json.dump(json_out, f, ensure_ascii=False, indent=2)
    print(f"\nJSON结果已保存: {JSON_PATH}")
    print("========== 完成 ==========")


if __name__ == "__main__":
    main()
