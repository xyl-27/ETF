"""
市场监控脚本
- 读取 backtest_state.json + etf_74.csv
- 以 HS300(510300.XSHG) 为基准划分市场状态（牛/熊/震荡）
- 对比模型在各市场状态下的表现
- 输出 JSON 和图表，可嵌入日报

市场状态划分方法:
  以 HS300 滚动 20 日收益 + 滚动波动率为依据：
    bull:     滚动收益 > +5%  且 波动 < 30%（趋势向上且稳定）
    bear:     滚动收益 < -5%（趋势向下）
    sideways: 其余（震荡或趋势不明）
"""

import os
import json
import sys
import base64
from pathlib import Path
from datetime import datetime, timedelta

import numpy as np
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
DATA_PATH = PROJECT_ROOT / "etf_data" / "etf_74.csv"
STATE_PATH = PROJECT_ROOT / "output" / "backtest_state.json"
OUTPUT_DIR = PROJECT_ROOT / "output"
HS300_CODE = "510300.XSHG"


def load_hs300_data():
    df = pd.read_csv(DATA_PATH)
    df["日期"] = pd.to_datetime(df["日期"])
    hs = df[df["股票代码"] == HS300_CODE].sort_values("日期").copy()
    hs["return_pct"] = hs["收盘"].pct_change() * 100
    return hs


def _load_pool_prices():
    """加载全部74只ETF的收盘价（宽表），用于计算等权组合收益"""
    df = pd.read_csv(DATA_PATH)
    df["日期"] = pd.to_datetime(df["日期"])
    pivot = df.pivot_table(index="日期", columns="股票代码", values="收盘")
    return pivot


def _load_pool_value_weighted_roll(window=20):
    """用平均成交额做权重计算全池加权滚动收益"""
    df = pd.read_csv(DATA_PATH)
    df["日期"] = pd.to_datetime(df["日期"])
    weights = df.groupby("股票代码")["成交额"].mean()
    weights = weights / weights.sum()
    pivot_close = df.pivot_table(index="日期", columns="股票代码", values="收盘")
    rets = pivot_close.pct_change().dropna()
    weighted_rets = rets.mul(weights, axis=1).sum(axis=1) * 100
    roll = weighted_rets.rolling(window).sum().dropna()
    return roll


def load_backtest_equity(seq_key=None):
    with open(STATE_PATH) as f:
        state = json.load(f)
    seqs = state.get("sequences", state)
    if seq_key is None:
        for k in seqs:
            if k != "hs300":
                seq_key = k
                break
    seq = seqs.get(seq_key, {})
    eq = seq.get("equity_curve", [])
    if not eq:
        return None, None
    dates = []
    vals = []
    for e in eq:
        d = e["date"]
        if hasattr(d, "strftime"):
            d = d.strftime("%Y-%m-%d")
        elif isinstance(d, datetime):
            d = d.strftime("%Y-%m-%d")
        dates.append(str(d)[:10])
        vals.append(e["total_value"])
    return dates, vals


def classify_market_regime(hs_data, window=20):
    """将市场分为 牛市/熊市/震荡。

    方法：
      1. 计算 HS300 滚动 20 日收益 rolling_ret（%）
      2. 计算滚动年化波动率 rolling_vol（%）
      3. 判定规则：
         牛市: rolling_ret > +5%  且 rolling_vol < 30%  → 趋势向上、波动可控
         熊市: rolling_ret < -5%                          → 趋势显著向下
         震荡: 其余                                       → 无明显趋势

    Returns:
      hs: 新增 regime 列（bull / bear / sideways / N/A）
    """
    hs = hs_data.copy()
    hs["rolling_ret"] = hs["收盘"].pct_change(window) * 100
    hs["rolling_vol"] = hs["return_pct"].rolling(window).std() * np.sqrt(252)

    def regime(row):
        ret = row["rolling_ret"]
        vol = row["rolling_vol"]
        if pd.isna(ret) or pd.isna(vol):
            return "N/A"
        if ret > 5 and vol < 30:
            return "bull"
        if ret < -5:
            return "bear"
        return "sideways"

    hs["regime"] = hs.apply(regime, axis=1)
    return hs


def get_current_regime(hs_data, window=20):
    """返回最新一天的市场状态和关键数值"""
    hs = classify_market_regime(hs_data, window)
    last = hs.iloc[-1]
    return {
        "regime": last["regime"],
        "rolling_20d_return": round(last["rolling_ret"], 2),
        "rolling_vol": round(last["rolling_vol"], 2),
        "date": str(last["日期"].strftime("%Y-%m-%d")),
    }


def compute_market_breadth(window=20):
    """计算全ETF池子的市场宽度。

    对每只ETF用同样的滚动收益方法判牛/熊/震荡，
    然后按日统计三种状态的ETF数量占比。
    这比只看HS300更能反映市场的真实宽度。
    """
    df = pd.read_csv(DATA_PATH)
    df["日期"] = pd.to_datetime(df["日期"])
    codes = df["股票代码"].unique()

    # 每只ETF判状态
    all_regimes = {}
    for code in codes:
        sub = df[df["股票代码"] == code].sort_values("日期").copy()
        sub["return_pct"] = sub["收盘"].pct_change() * 100
        sub["rolling_ret"] = sub["收盘"].pct_change(window) * 100
        sub["rolling_vol"] = sub["return_pct"].rolling(window).std() * np.sqrt(252)

        def regime(row):
            ret = row["rolling_ret"]
            vol = row["rolling_vol"]
            if pd.isna(ret) or pd.isna(vol):
                return "N/A"
            if ret > 5 and vol < 30:
                return "bull"
            if ret < -5:
                return "bear"
            return "sideways"

        sub["regime"] = sub.apply(regime, axis=1)
        for _, row in sub.iterrows():
            d = row["日期"].strftime("%Y-%m-%d")
            all_regimes.setdefault(d, []).append(row["regime"])

    # 汇总每日占比
    records = []
    for d in sorted(all_regimes):
        rs = all_regimes[d]
        total = len(rs)
        n_bull = sum(1 for r in rs if r == "bull")
        n_bear = sum(1 for r in rs if r == "bear")
        n_side = sum(1 for r in rs if r == "sideways")
        records.append({
            "date": d,
            "bull_pct": round(n_bull / total * 100, 1),
            "bear_pct": round(n_bear / total * 100, 1),
            "sideways_pct": round(n_side / total * 100, 1),
            "bull_count": n_bull,
            "bear_count": n_bear,
            "sideways_count": n_side,
            "total": total,
        })
    breadth_df = pd.DataFrame(records)
    breadth_df["date"] = pd.to_datetime(breadth_df["date"])

    # 当前快照
    last = records[-1] if records else {}
    return breadth_df, last


def _rank_map_at_date(df, target_date_str, window=5):
    """计算截止到 target_date（含）的 window 日收益率排名"""
    dates = sorted(df["日期"].unique())
    target_dt = pd.Timestamp(target_date_str)
    filtered = [d for d in dates if d <= target_dt]
    if len(filtered) < window + 1:
        return {}
    end_date = filtered[-1]
    start_date = filtered[-(window + 1)]
    period = df[df["日期"].between(start_date, end_date)].copy()
    pivot = period.pivot_table(index="股票代码", columns="日期", values="收盘")
    if len(pivot.columns) < 2:
        return {}
    ret = (pivot.iloc[:, -1] / pivot.iloc[:, 0] - 1) * 100
    ret = ret.dropna().sort_values(ascending=False)
    return {code: i + 1 for i, code in enumerate(ret.index)}


def _load_etf_names():
    names = {}
    p = PROJECT_ROOT / "etf_data" / "etf_list_before_2022_74.csv"
    if p.exists():
        import csv
        with open(p, encoding="utf-8-sig") as f:
            for row in csv.DictReader(f):
                code = row.get("代码", "").strip()
                name = row.get("名称", "").strip()
                if code and name:
                    names[code] = name
    return names


def compute_top_etf_rankings(window=5, current_holdings_set=None, prev_holdings_set=None,
                              current_rebalance_date=None, prev_rebalance_date=None):
    """计算近期涨幅最大/最小的ETF排行，标记持仓

    Parameters
    ----------
    current_holdings_set : set[str] or None
        当前持仓代码集。None 时从 backtest_state.json 读取。
    prev_holdings_set : set[str] or None
        上期持仓代码集。None 时从 backtest_state.json 推算。
    current_rebalance_date : str or None
        当前调仓日（YYYY-MM-DD），用于计算调仓时排名。
    prev_rebalance_date : str or None
        上期调仓日（YYYY-MM-DD），用于计算上期调仓时排名。
    """
    df = pd.read_csv(DATA_PATH)
    df["日期"] = pd.to_datetime(df["日期"])
    dates = sorted(df["日期"].unique())
    if len(dates) < window + 1:
        return [], [], [], [], "", ""
    end_date = dates[-1]
    start_date = dates[-(window + 1)]
    period = df[df["日期"].between(start_date, end_date)].copy()
    pivot = period.pivot_table(index="股票代码", columns="日期", values="收盘")
    ret = (pivot.iloc[:, -1] / pivot.iloc[:, 0] - 1) * 100
    ret = ret.dropna().sort_values(ascending=False)

    # 加载持仓
    holdings = current_holdings_set
    if holdings is None:
        try:
            with open(STATE_PATH) as f:
                state = json.load(f)
            for k in state.get("sequences", state):
                if k != "hs300":
                    holdings = set(state["sequences"][k].get("positions", {}).keys())
                    break
            else:
                holdings = set()
        except Exception:
            holdings = set()

    # 5日趋势: 归一化线性回归斜率 (%/day)
    _trend_map = {}
    if pivot.shape[1] >= 5:
        x_t = np.arange(5)
        for code_t in pivot.index:
            vals_t = pivot.loc[code_t].iloc[-5:].values
            if np.any(np.isnan(vals_t)) or np.any(vals_t <= 0):
                continue
            mean_t = np.mean(vals_t)
            slope_t = np.polyfit(x_t, vals_t, 1)[0]
            _trend_map[code_t] = round(slope_t / mean_t * 100, 2)

    etf_names = _load_etf_names()
    top = []
    for code, r in ret.head(10).items():
        top.append({"code": code, "name": etf_names.get(code, ""), "return": round(r, 2), "trend_5d": _trend_map.get(code), "held": code in holdings})
    bot = []
    for code, r in ret.tail(10).items():
        bot.append({"code": code, "name": etf_names.get(code, ""), "return": round(r, 2), "trend_5d": _trend_map.get(code), "held": code in holdings})

    rank_map = {code: i + 1 for i, code in enumerate(ret.index)}

    # 当前持仓排行（含调仓时排名）
    rebalance_rank_map = _rank_map_at_date(df, current_rebalance_date, window) if current_rebalance_date else {}
    holdings_data = []
    for code in sorted(holdings):
        r = ret.get(code)
        if r is not None:
            holdings_data.append({
                "code": code,
                "name": etf_names.get(code, ""),
                "return": round(r, 2),
                "trend_5d": _trend_map.get(code),
                "rank": rank_map.get(code, 0),
                "rank_at_rebalance": rebalance_rank_map.get(code, 0),
                "total": len(ret),
                "rebalance_date": current_rebalance_date or "",
            })
    holdings_data.sort(key=lambda x: x["rank"])

    # 上期持仓排行
    prev_holdings_data = []
    if prev_holdings_set is not None:
        prev_rank_map = _rank_map_at_date(df, prev_rebalance_date, window) if prev_rebalance_date else {}
        eq_data = pd.read_csv(DATA_PATH)
        eq_dates = sorted(eq_data["日期"].unique())
        prev_idx = next((i for i, d in enumerate(eq_dates) if d >= (prev_rebalance_date or "")), len(eq_dates))
        days_ago = len(eq_dates) - prev_idx if prev_rebalance_date else 0
        for code in sorted(prev_holdings_set):
            r = ret.get(code)
            if r is not None:
                prev_holdings_data.append({
                    "code": code,
                    "name": etf_names.get(code, ""),
                    "return": round(r, 2),
                    "trend_5d": _trend_map.get(code),
                    "rank": rank_map.get(code, 0),
                    "rank_at_rebalance": prev_rank_map.get(code, 0),
                    "total": len(ret),
                    "days_ago": days_ago,
                    "rebalance_date": prev_rebalance_date or "",
                })
        prev_holdings_data.sort(key=lambda x: x["rank"])
    else:
        # 从 state 推算
        try:
            with open(STATE_PATH) as f:
                state = json.load(f)
            for k in state.get("sequences", state):
                if k != "hs300":
                    seq = state["sequences"][k]
                    break
            else:
                seq = {}
            preds = seq.get("predictions_history", [])
            if len(preds) >= 2:
                prev_date = preds[-2]["date"]
                prev_rank_map = _rank_map_at_date(df, prev_date, window)
                held = set()
                for t in seq.get("trades", []):
                    if t["date"] > prev_date:
                        break
                    if t["action"] == "买入":
                        held.add(t["stock"])
                    elif t["action"] == "卖出":
                        held.discard(t["stock"])
                eq_dates = sorted(set(e["date"] for e in seq.get("equity_curve", [])))
                days_ago = sum(1 for d in eq_dates if d >= prev_date)
                for code in sorted(held):
                    r = ret.get(code)
                    if r is not None:
                        pr = prev_rank_map.get(code, 0)
                        prev_holdings_data.append({
                            "code": code,
                            "name": etf_names.get(code, ""),
                            "return": round(r, 2),
                            "trend_5d": _trend_map.get(code),
                            "rank": rank_map[code],
                            "rank_at_rebalance": pr,
                            "total": len(ret),
                            "days_ago": days_ago,
                            "rebalance_date": prev_date,
                        })
                prev_holdings_data.sort(key=lambda x: x["rank"])
        except Exception:
            prev_holdings_data = []

    date_range = f"{pivot.columns[0].strftime('%Y-%m-%d')}~{pivot.columns[-1].strftime('%Y-%m-%d')}"
    return top, bot, holdings_data, prev_holdings_data, str(pivot.columns[0].date()), str(pivot.columns[-1].date())


def compute_model_vs_market(dates, values, hs_data):
    """对齐模型和HS300日期，按市场状态分段统计"""
    hs_dict = dict(zip(hs_data["日期"].dt.strftime("%Y-%m-%d"), hs_data["return_pct"]))
    regime_dict = dict(zip(hs_data["日期"].dt.strftime("%Y-%m-%d"), hs_data["regime"]))

    records = []
    for i, d in enumerate(dates):
        if d in hs_dict and d in regime_dict:
            model_ret = (values[i] / values[i - 1] - 1) * 100 if i > 0 else 0
            records.append({
                "date": d,
                "model_return": model_ret,
                "hs300_return": hs_dict[d],
                "regime": regime_dict[d],
                "model_value": values[i],
            })
    df = pd.DataFrame(records)

    stats = {}
    for regime in ["bull", "bear", "sideways"]:
        sub = df[df["regime"] == regime]
        if len(sub) < 3:
            continue
        model_total = ((1 + sub["model_return"] / 100).prod() - 1) * 100
        hs_total = ((1 + sub["hs300_return"] / 100).prod() - 1) * 100
        excess = model_total - hs_total
        beat_rate = (sub["model_return"] > sub["hs300_return"]).mean()
        model_win_rate = (sub["model_return"] > 0).mean()
        stats[regime] = {
            "days": len(sub),
            "model_return": round(model_total, 2),
            "hs300_return": round(hs_total, 2),
            "excess_return": round(excess, 2),
            "beat_rate": round(beat_rate, 4),
            "model_win_rate": round(model_win_rate, 4),
            "avg_model_return": round(sub["model_return"].mean(), 4),
            "avg_hs300_return": round(sub["hs300_return"].mean(), 4),
        }

    model_total = (values[-1] / values[0] - 1) * 100
    hs_total = (1 + df["hs300_return"] / 100).prod() - 1
    hs_total = hs_total * 100
    beat_all = (df["model_return"] > df["hs300_return"]).mean()
    model_win_all = (df["model_return"] > 0).mean()
    stats["all"] = {
        "days": len(df),
        "model_return": round(model_total, 2),
        "hs300_return": round(hs_total, 2),
        "excess_return": round(model_total - hs_total, 2),
        "beat_rate": round(beat_all, 4),
        "model_win_rate": round(model_win_all, 4),
        "avg_model_return": round(df["model_return"].mean(), 4),
        "avg_hs300_return": round(df["hs300_return"].mean(), 4),
    }

    return df, stats


def compute_recent_performance(dates, values, hs_data, windows=[5, 10, 20]):
    """计算近期滚动表现"""
    hs_prices = hs_data.set_index("日期")["收盘"]
    results = {}
    for w in windows:
        if len(dates) < w + 1:
            continue
        model_ret = (values[-1] / values[-w - 1] - 1) * 100
        hs_val = hs_prices.loc[:dates[-1]].iloc[-w - 1:]
        if len(hs_val) >= 2:
            hs_ret = (hs_val.iloc[-1] / hs_val.iloc[0] - 1) * 100
        else:
            hs_ret = 0
        excess = model_ret - hs_ret
        sub_dates = dates[-w:]
        wins = 0
        for i in range(1, len(sub_dates)):
            if values[dates.index(sub_dates[i])] > values[dates.index(sub_dates[i - 1])]:
                wins += 1
        win_rate = wins / (len(sub_dates) - 1) if len(sub_dates) > 1 else 0
        results[f"{w}d"] = {
            "model_return": round(model_ret, 2),
            "hs300_return": round(hs_ret, 2),
            "excess_return": round(excess, 2),
            "daily_win_rate": round(win_rate, 4),
        }
    return results


def plot_market_analysis(dates, values, df_regime, breadth_df, output_path):
    """生成市场监控图表 - 5 面板：
    0. 市场宽度（全ETF池：牛/熊/震荡占比）
    1. HS300 市场状态色条
    2. 净值曲线（策略 vs HS300）
    3. 日超额收益 + 累计超额
    4. 滚动 20 日收益对比
    """
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import matplotlib.dates as mdates
    from matplotlib.patches import Patch
    plt.rcParams["font.sans-serif"] = ["SimHei", "DejaVu Sans", "Arial Unicode MS"]
    plt.rcParams["axes.unicode_minus"] = False

    hs_data = load_hs300_data()
    hs_prices = hs_data.set_index("日期")["收盘"]

    plot_dates = pd.to_datetime(dates)
    model_vals = np.array(values)
    hs_aligned = hs_prices.reindex(plot_dates, method="ffill")
    hs_norm = hs_aligned.values / hs_aligned.values[0] * values[0]

    regime_colors = {"bull": "#e74c3c", "bear": "#27ae60", "sideways": "#f39c12", "N/A": "#bdc3c7"}
    regime_labels = {"bull": "牛市", "bear": "熊市", "sideways": "震荡"}
    regime_dates = pd.to_datetime(df_regime["date"])
    unique_regimes = [r for r in ["bull", "bear", "sideways"] if r in df_regime["regime"].values]

    fig = plt.figure(figsize=(14, 16))
    gs = fig.add_gridspec(6, 1, height_ratios=[0.8, 0.7, 1.2, 0.7, 0.7, 0.7], hspace=0.18)
    ax_breadth = fig.add_subplot(gs[0])
    ax_breadth_bt = fig.add_subplot(gs[1])
    ax_curve = fig.add_subplot(gs[2])
    ax_excess = fig.add_subplot(gs[3])
    ax_roll = fig.add_subplot(gs[4])

    # ===== 面板 0: 市场宽度（全历史） =====
    bdates = breadth_df["date"]
    ax_breadth.fill_between(bdates, 0, breadth_df["bull_pct"].values,
                            color="#e74c3c", alpha=0.5, label="bull%")
    ax_breadth.fill_between(bdates, breadth_df["bull_pct"].values,
                            breadth_df["bull_pct"].values + breadth_df["sideways_pct"].values,
                            color="#f39c12", alpha=0.5, label="sideways%")
    ax_breadth.fill_between(bdates,
                            breadth_df["bull_pct"].values + breadth_df["sideways_pct"].values,
                            100,
                            color="#27ae60", alpha=0.5, label="bear%")
    ax_breadth.axhline(50, color="gray", ls=":", lw=0.5, alpha=0.6)
    ax_breadth.set_ylim(-5, 105)
    ax_breadth.set_xlim(bdates.iloc[0], bdates.iloc[-1])
    ax_breadth.set_ylabel("ETF占比 (%)", fontsize=10)
    ax_breadth.set_title("市场宽度 — ETF池子牛/熊/震荡占比", fontsize=11, fontweight="bold")
    ax_breadth.text(0.01, 0.95, f"{bdates.iloc[0].strftime('%Y-%m')}~{bdates.iloc[-1].strftime('%Y-%m')}",
                    transform=ax_breadth.transAxes, fontsize=8, color="#888",
                    va="top", ha="left", alpha=0.7)
    ax_breadth.legend(loc="upper left", fontsize=8, ncol=3)
    ax_breadth.grid(True, alpha=0.2)

    # ===== 面板 1: 市场宽度（回测起始日至今） =====
    bt_start = plot_dates[0]
    bd_bt = breadth_df[breadth_df["date"] >= bt_start].copy()
    if len(bd_bt) > 5:
        ax_breadth_bt.fill_between(bd_bt["date"], 0, bd_bt["bull_pct"].values,
                                   color="#e74c3c", alpha=0.5, label="bull%")
        ax_breadth_bt.fill_between(bd_bt["date"], bd_bt["bull_pct"].values,
                                   bd_bt["bull_pct"].values + bd_bt["sideways_pct"].values,
                                   color="#f39c12", alpha=0.5, label="sideways%")
        ax_breadth_bt.fill_between(bd_bt["date"],
                                   bd_bt["bull_pct"].values + bd_bt["sideways_pct"].values,
                                   100,
                                   color="#27ae60", alpha=0.5, label="bear%")
    ax_breadth_bt.axhline(50, color="gray", ls=":", lw=0.5, alpha=0.6)
    ax_breadth_bt.set_ylim(-5, 105)
    ax_breadth_bt.set_ylabel("ETF占比 (%)", fontsize=10)
    ax_breadth_bt.set_title("市场宽度（回测期）", fontsize=11, fontweight="bold")
    ax_breadth_bt.legend(loc="upper left", fontsize=8, ncol=3)
    ax_breadth_bt.grid(True, alpha=0.2)

    # ===== 面板 2: 净值曲线 =====
    ax_curve.plot(plot_dates, model_vals, color="#2980b9", lw=2, label="策略")
    ax_curve.plot(plot_dates, hs_norm, color="#7f8c8d", lw=1.5, ls="--", label="沪深300(归一化)")
    ax_curve.set_ylabel("账户净值 (CNY)", fontsize=11)
    ax_curve.grid(True, alpha=0.2)

    model_total_ret = (model_vals[-1] / model_vals[0] - 1) * 100
    hs_total_ret = (hs_norm[-1] / hs_norm[0] - 1) * 100
    ax_curve.annotate(f"策略: {model_total_ret:+.2f}%\nHS300: {hs_total_ret:+.2f}%",
                      xy=(0.98, 0.05), xycoords="axes fraction",
                      ha="right", va="bottom", fontsize=10,
                      bbox=dict(boxstyle="round,pad=0.4", facecolor="white", alpha=0.8))

    # ===== 面板 3: 日超额收益 =====
    excess = df_regime["model_return"].values - df_regime["hs300_return"].values
    bar_colors = [regime_colors.get(r, "#bdc3c7") for r in df_regime["regime"].values]
    ax_excess.bar(regime_dates, excess, color=bar_colors, width=0.8, alpha=0.6, edgecolor="none")
    ax_excess.axhline(0, color="gray", lw=0.5)
    ax_excess.set_ylabel("日超额收益 (%)", fontsize=11)
    ax_excess.grid(True, alpha=0.2)
    cum_excess = np.cumprod(1 + df_regime["model_return"].values / 100) / np.cumprod(1 + df_regime["hs300_return"].values / 100) - 1
    cum_excess = cum_excess * 100
    ax2 = ax_excess.twinx()
    ax2.plot(regime_dates, cum_excess, color="#8e44ad", lw=1.5, label=f"累计超额 {cum_excess[-1]:+.2f}%")
    ax2.legend(loc="upper right", fontsize=9)
    ax2.set_ylabel("累计超额 (%)", fontsize=10, color="#8e44ad")

    # ===== 面板 4: 滚动 20 日收益对比 =====
    window = 20
    model_prices = pd.Series(values, index=plot_dates)
    model_roll = model_prices.pct_change(window).dropna() * 100
    hs_full = load_hs300_data().set_index("日期")["收盘"]
    hs_roll = hs_full.pct_change(window).dropna() * 100
    hs_roll = hs_roll.reindex(plot_dates)
    hs_roll_5 = hs_full.pct_change(5).dropna() * 100
    hs_roll_5 = hs_roll_5.reindex(plot_dates)
    pool_df = _load_pool_prices()
    pool_has_data = len(pool_df) > window
    if pool_has_data:
        pool_full_rets = pool_df.pct_change().dropna()
        pool_eq = pool_full_rets.mean(axis=1) * 100
        pool_roll = pool_eq.rolling(window).sum()
        pool_roll = pool_roll.reindex(plot_dates)
    pool_vw = _load_pool_value_weighted_roll(window)
    pool_vw = pool_vw.reindex(plot_dates)
    ax_roll.plot(model_roll.index, model_roll.values, color="#2980b9", lw=1.5, label=f"策略 {window}d 滚动收益")
    ax_roll.plot(plot_dates, hs_roll.values, color="#7f8c8d", lw=1.5, ls="--", label=f"HS300 {window}d 滚动收益")
    ax_roll.plot(plot_dates, hs_roll_5.values, color="#95a5a6", lw=1.5, ls="--", label="HS300 5d 滚动收益")
    if pool_has_data:
        ax_roll.plot(plot_dates, pool_roll.values, color="#e67e22", lw=1.5, ls=":", label=f"全池等权 {window}d 滚动收益")
    ax_roll.plot(plot_dates, pool_vw.values, color="#2ecc71", lw=1.5, ls="-.", label=f"全池加权 {window}d 滚动收益")
    ax_roll.axhline(0, color="gray", ls=":", lw=0.5)
    ax_roll.set_ylabel(f"{window}d 滚动收益 (%)", fontsize=11)
    ax_roll.legend(loc="upper left", fontsize=7)
    ax_roll.grid(True, alpha=0.2)

    # 各面板独立 X 轴：广度全历史面板显示x标签(无旋转)，回测期面板隐藏避免遮挡
    ax_breadth.xaxis.set_major_formatter(mdates.DateFormatter("%Y-%m"))
    ax_breadth.xaxis.set_major_locator(mdates.MonthLocator(interval=4))
    ax_breadth.tick_params(axis="x", labelsize=6, pad=0)
    ax_breadth_bt.set_title("")
    bt_dates = plot_dates
    bt_dates = plot_dates
    tick_step = 5
    bt_ticks = bt_dates[::tick_step]
    ax_breadth_bt.set_xticks(bt_ticks)
    ax_breadth_bt.tick_params(axis="x", labelbottom=False)
    for a in [ax_curve, ax_excess, ax_roll]:
        a.set_xticks(bt_ticks)
        a.set_xticklabels([d.strftime("%m/%d") for d in bt_ticks], fontsize=7)
    for a in [ax_curve, ax_excess]:
        a.tick_params(axis="x", labelbottom=False)

    fig.text(0.5, 0.005, "日期", ha="center", fontsize=11)
    fig.suptitle("市场监控 — 市场宽度 · HS300 · 策略表现", fontsize=14, fontweight="bold", y=0.99)
    fig.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    return str(output_path)


def plot_breadth_backtest(dates, breadth_df, output_path):
    """保存市场宽度（回测期）为独立 PNG"""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    bt_start = pd.Timestamp(dates[0])
    bd = breadth_df[breadth_df["date"] >= bt_start].copy()
    if len(bd) < 3:
        return None

    fig, ax = plt.subplots(figsize=(12, 3))
    ax.fill_between(bd["date"], 0, bd["bull_pct"].values,
                    color="#e74c3c", alpha=0.5, label="bull%")
    ax.fill_between(bd["date"], bd["bull_pct"].values,
                    bd["bull_pct"].values + bd["sideways_pct"].values,
                    color="#f39c12", alpha=0.5, label="sideways%")
    ax.fill_between(bd["date"],
                    bd["bull_pct"].values + bd["sideways_pct"].values, 100,
                    color="#27ae60", alpha=0.5, label="bear%")
    ax.axhline(50, color="gray", ls=":", lw=0.5, alpha=0.6)
    ax.set_ylim(-5, 105)
    ax.set_ylabel("ETF占比 (%)", fontsize=10)
    ax.set_title("市场宽度（回测期）", fontsize=11, fontweight="bold")
    ax.legend(loc="upper left", fontsize=8, ncol=3)
    ax.grid(True, alpha=0.2)
    fig.tight_layout()
    fig.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    return str(output_path)


def plot_rolling_20d(dates, values, output_path):
    """保存滚动 20 日收益对比（去掉策略曲线）为独立 PNG"""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    plot_dates = pd.to_datetime(dates)
    window = 20
    hs_full = load_hs300_data().set_index("日期")["收盘"]
    hs_roll = hs_full.pct_change(window).dropna() * 100
    hs_roll = hs_roll.reindex(plot_dates)
    hs_roll_5 = hs_full.pct_change(5).dropna() * 100
    hs_roll_5 = hs_roll_5.reindex(plot_dates)
    pool_df = _load_pool_prices()
    pool_has_data = len(pool_df) > window
    if pool_has_data:
        pool_full_rets = pool_df.pct_change().dropna()
        pool_eq = pool_full_rets.mean(axis=1) * 100
        pool_roll = pool_eq.rolling(window).sum()
        pool_roll = pool_roll.reindex(plot_dates)
    pool_vw = _load_pool_value_weighted_roll(window)
    pool_vw = pool_vw.reindex(plot_dates)

    fig, ax = plt.subplots(figsize=(12, 3))
    ax.plot(plot_dates, hs_roll.values, color="#7f8c8d", lw=1.5, ls="--", label=f"HS300 {window}d 滚动收益")
    ax.plot(plot_dates, hs_roll_5.values, color="#95a5a6", lw=1.5, ls="--", label="HS300 5d 滚动收益")
    if pool_has_data:
        ax.plot(plot_dates, pool_roll.values, color="#e67e22", lw=1.5, ls=":", label=f"全池等权 {window}d 滚动收益")
    ax.plot(plot_dates, pool_vw.values, color="#2ecc71", lw=1.5, ls="-.", label=f"全池加权 {window}d 滚动收益")
    ax.axhline(0, color="gray", ls=":", lw=0.5)
    ax.set_ylabel(f"{window}d 滚动收益 (%)", fontsize=11)
    ax.set_title("滚动 20 日收益对比", fontsize=11, fontweight="bold")
    ax.legend(loc="upper left", fontsize=8)
    ax.grid(True, alpha=0.2)
    fig.tight_layout()
    fig.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    return str(output_path)


def _img_to_b64(path):
    """PNG 转 base64 data URI"""
    with open(path, "rb") as f:
        data = base64.b64encode(f.read()).decode()
    return f"data:image/png;base64,{data}"


# ============================================================
# HTML 生成
# ============================================================

def build_regime_table_html(stats, current_regime=None, breadth_last=None):
    """生成市场状态对比 + 市场宽度 HTML 表格"""
    regime_labels = {"all": "全周期", "bull": "牛市 ↑", "bear": "熊市 ↓", "sideways": "震荡 →"}
    rows = ""
    for regime in ["all", "bull", "bear", "sideways"]:
        s = stats.get(regime)
        if not s:
            continue
        m_ret = s["model_return"]
        h_ret = s["hs300_return"]
        e_ret = s["excess_return"]
        mwr = s.get("model_win_rate", s.get("beat_rate", 0))
        m_clr = "#cc0000" if m_ret >= 0 else "#009900"
        h_clr = "#cc0000" if h_ret >= 0 else "#009900"
        e_clr = "#cc0000" if e_ret >= 0 else "#009900"
        cur_tag = " ★ 当前" if (current_regime and regime == current_regime["regime"]) else ""
        rows += f"""<tr>
            <td style="font-weight:bold;">{regime_labels.get(regime, regime)}{cur_tag}</td>
            <td style="text-align:right;">{s['days']}天</td>
            <td style="text-align:right;color:{m_clr};">{m_ret:+.2f}%</td>
            <td style="text-align:right;color:{h_clr};">{h_ret:+.2f}%</td>
            <td style="text-align:right;color:{e_clr};font-weight:bold;">{e_ret:+.2f}%</td>
            <td style="text-align:right;">{mwr*100:.1f}%</td>
        </tr>"""

    header_lines = []
    if current_regime:
        header_lines.append(
            f"HS300状态: <b>{regime_labels.get(current_regime['regime'], current_regime['regime'])}</b> "
            f"(20日滚动收益: {current_regime['rolling_20d_return']:+.2f}%, 波动率: {current_regime['rolling_vol']:.1f}%)"
        )
    if breadth_last:
        header_lines.append(
            f"全池宽度: 🔴牛市 {breadth_last['bull_pct']:.0f}% / 🟡震荡 {breadth_last['sideways_pct']:.0f}% / 🟢熊市 {breadth_last['bear_pct']:.0f}% "
            f"(共{breadth_last['total']}只ETF)"
        )
    header_info = "<br>".join(header_lines)

    return f"""<h3>市场状态</h3>
    <p style="font-size:11px;color:#666;">{header_info}</p>
    <table>
        <thead><tr><th>市场状态</th><th style="text-align:right;">天数</th><th style="text-align:right;">模型收益</th><th style="text-align:right;">HS300</th><th style="text-align:right;">超额收益</th><th style="text-align:right;">调仓胜率</th></tr></thead>
        <tbody>{rows}</tbody>
    </table>
    <p style="font-size:10px;color:#999;">市场状态判定: 以HS300滚动20日收益+波动率为基准。牛: >+5%且波动&lt;30%; 熊: &lt;-5%; 震荡: 其余。市场宽度: 同方法对全ETF池子逐只判定后统计占比。</p>"""


def compute_virtual_equity_curve(equity_curve):
    """回算虚拟无风控组合的权益曲线。

    原理：实际日收益 = stock_exposure × stock_component_return (现金收益≈0)。
    若未使用风控，stock_exposure 应为 actual_stock_exposure / risk_mult (上限100%)。
    故 virtual_return = min(actual_stock_exposure / prev_risk_mult, 1.0) × (actual_return / actual_stock_exposure)。

    Parameters
    ----------
    equity_curve : list[dict]
        每项含 date, total_value, risk_multiplier, stock_exposure。

    Returns
    -------
    (dates, virtual_vals) or None
    """
    if not equity_curve or len(equity_curve) < 2:
        return None

    actual_vals = [e["total_value"] for e in equity_curve]
    dates = [str(e["date"])[:10] for e in equity_curve]

    virtual_vals = [actual_vals[0]]

    for i in range(1, len(equity_curve)):
        prev_entry = equity_curve[i - 1]
        prev_val = actual_vals[i - 1]
        cur_val = actual_vals[i]
        actual_return = (cur_val / prev_val - 1)

        prev_stock_exposure = prev_entry.get("stock_exposure", 1.0)
        prev_risk_mult = prev_entry.get("risk_multiplier", 1.0)

        if prev_stock_exposure > 1e-8 and prev_risk_mult > 1e-8:
            stock_return = actual_return / prev_stock_exposure
            virtual_exposure = min(prev_stock_exposure / prev_risk_mult, 1.0)
            virtual_return = virtual_exposure * stock_return
        else:
            virtual_return = actual_return

        virtual_vals.append(virtual_vals[-1] * (1 + virtual_return))

    return dates, virtual_vals


def build_risk_state_html(risk_state):
    """生成风控状态 HTML 片段"""
    if risk_state is None:
        return ""
    rs = risk_state
    pct = rs["pos_ratio"] * 100
    mult = rs["multiplier"]
    pnl = rs.get("risk_pnl")
    pnl_line = ""
    if pnl is not None:
        actual_ret = rs.get("actual_return", 0)
        virtual_ret = rs.get("virtual_return", 0)
        pnl_clr = "#cc0000" if pnl >= 0 else "#009900"
        pnl_line = f"""<br>
        风控盈亏: <b style="color:{pnl_clr};">{pnl:+.2f}%</b>
        (实际 {actual_ret:+.2f}% vs 虚拟 {virtual_ret:+.2f}%)"""

    return f"""<div style="margin:8px 0;padding:10px;background:#f8f9fa;border:1px solid #dee2e6;border-radius:5px;font-size:12px;">
        <b>风控策略</b>: {rs['label']} ({rs['params_str']}) &nbsp;|&nbsp;
        当前正收益ETF占比: <b>{pct:.1f}%</b> &nbsp;|&nbsp;
        仓位乘数: <b>{mult:.2f}</b> &nbsp;|&nbsp;
        状态: {rs['status']}{pnl_line}
    </div>"""


def compute_breadth_risk_state(risk_config=None):
    """计算当前市场广度风控状态

    Returns
    -------
    dict with keys: pos_ratio, multiplier, status, label, params_str
    """
    if not risk_config or not risk_config.get("enabled"):
        return None
    params = risk_config.get("params", {})
    lb = params.get("lookback_days", 20)
    ht = params.get("high_threshold", 0.30)
    lt = params.get("low_threshold", 0.10)

    df = pd.read_csv(DATA_PATH)
    df["日期"] = pd.to_datetime(df["日期"])
    all_dates = sorted(df["日期"].unique())
    if len(all_dates) < lb + 1:
        return None
    end_date = all_dates[-1]
    start_date = all_dates[-(lb + 1)]
    start_prices = df[df["日期"] == start_date][["股票代码", "收盘"]]
    end_prices = df[df["日期"] == end_date][["股票代码", "收盘"]]
    merged = pd.merge(start_prices, end_prices, on="股票代码", suffixes=("_start", "_end"))
    if len(merged) < 5:
        return None
    returns = (merged["收盘_end"] - merged["收盘_start"]) / merged["收盘_start"]
    pos_ratio = (returns > 0).mean()

    if pos_ratio >= ht:
        mult = 1.0
    elif pos_ratio <= lt:
        mult = 0.0
    else:
        mult = (pos_ratio - lt) / (ht - lt)

    if mult >= 1.0:
        status = "🟢 正常"
    elif mult > 0:
        status = "🟡 缩仓"
    else:
        status = "🔴 空仓"

    label = {"market_breadth": "市场广度止损",
             "volatility_target": "波动率目标",
             "trend_filter": "趋势过滤",
             "drawdown_stop": "回撤止损",
             "none": "无"}.get(risk_config.get("strategy", ""), risk_config.get("strategy", ""))

    params_str = f"回看{lb}天, 阈值{ht:.0%}/{lt:.0%}"

    return {
        "pos_ratio": pos_ratio,
        "multiplier": mult,
        "status": status,
        "label": label,
        "params_str": params_str,
    }


def build_recent_table_html(recent):
    """生成近期表现 HTML 表格"""
    rows = ""
    for label, s in recent.items():
        m_ret = s["model_return"]
        h_ret = s["hs300_return"]
        e_ret = s["excess_return"]
        m_clr = "#cc0000" if m_ret >= 0 else "#009900"
        h_clr = "#cc0000" if h_ret >= 0 else "#009900"
        e_clr = "#cc0000" if e_ret >= 0 else "#009900"
        rows += f"""<tr>
            <td>{label}</td>
            <td style="text-align:right;color:{m_clr};">{m_ret:+.2f}%</td>
            <td style="text-align:right;color:{h_clr};">{h_ret:+.2f}%</td>
            <td style="text-align:right;color:{e_clr};font-weight:bold;">{e_ret:+.2f}%</td>
        </tr>"""
    return f"""<h3>近期表现</h3>
    <table>
        <thead><tr><th>区间</th><th style="text-align:right;">模型收益</th><th style="text-align:right;">HS300</th><th style="text-align:right;">超额收益</th></tr></thead>
        <tbody>{rows}</tbody>
    </table>"""


def build_etf_rankings_html(top, bot, holdings_data, prev_holdings_data, date_range):
    """生成ETF涨跌排行 + 持仓排行 HTML"""
    def _trend_cell(tv):
        if tv is None:
            return '<td style="text-align:right;color:#999;font-size:10px;">-</td>'
        clr = "#cc0000" if tv >= 0 else "#009900"
        return f'<td style="text-align:right;color:{clr};font-weight:bold;font-size:10px;">{tv:+.2f}%</td>'

    def rank_rows(items):
        rows = ""
        for i, item in enumerate(items):
            clr = "#cc0000" if item["return"] >= 0 else "#009900"
            held_tag = ' 🟢持仓' if item.get("held") else ''
            name_display = f'<br><span style="font-size:11px;color:#666;">{item["name"]}</span>' if item.get("name") else ''
            rows += f"""<tr>
                <td style="text-align:center;">{i+1}</td>
                <td><code>{item['code']}</code>{name_display}{held_tag}</td>
                <td style="text-align:right;color:{clr};font-weight:bold;">{item['return']:+.2f}%</td>
                {_trend_cell(item.get('trend_5d'))}
            </tr>"""
        return rows

    def holding_rows(items):
        rows = ""
        for item in items:
            clr = "#cc0000" if item["return"] >= 0 else "#009900"
            name_display = f'<br><span style="font-size:11px;color:#666;">{item["name"]}</span>' if item.get("name") else ''
            rr = item.get("rank_at_rebalance", 0)
            rr_display = str(rr) if rr else "-"
            rebalance_str = f"<br><span style='font-size:10px;color:#999;'>调仓时第{rr_display}名</span>"
            rows += f"""<tr>
                <td><code>{item['code']}</code>{name_display}</td>
                <td style="text-align:right;color:{clr};font-weight:bold;">{item['return']:+.2f}%</td>
                {_trend_cell(item.get('trend_5d'))}
                <td style="text-align:center;">第{item['rank']}/{item['total']}名{rebalance_str}</td>
            </tr>"""
        return rows

    cur_date = holdings_data[0].get("rebalance_date", "") if holdings_data else ""
    cur_date_display = cur_date if cur_date else "-"
    cur_label = f"📦 当前持仓（调仓日: {cur_date_display}）"
    cur_sec = f"""<tr style="background:#fffde7;"><td colspan="4" style="font-weight:bold;color:#f57f17;">{cur_label}</td></tr>
{holding_rows(holdings_data)}""" if holdings_data else ""

    prev_sec = ""
    if prev_holdings_data:
        prev_date = prev_holdings_data[0].get("rebalance_date", "")
        prev_date_display = prev_date if prev_date else "-"
        prev_label = f"📋 上期持仓（{prev_date_display}）"
        prev_sec = f"""<tr style="background:#f3e5f5;"><td colspan="4" style="font-weight:bold;color:#7b1fa2;">{prev_label}</td></tr>
{holding_rows(prev_holdings_data)}"""

    return f"""<h3>ETF 排行榜 {date_range}</h3>
    <table>
        <thead><tr><th style="text-align:center;">#</th><th>代码</th><th style="text-align:right;">收益率</th><th style="text-align:right;font-size:11px;">涨跌趋势<br>(5日)</th></tr></thead>
        <tbody>
            <tr style="background:#fff5f5;"><td colspan="4" style="font-weight:bold;color:#cc0000;">📈 涨幅前10</td></tr>
            {rank_rows(top)}
            <tr style="background:#f0fff0;"><td colspan="4" style="font-weight:bold;color:#009900;">📉 跌幅前10</td></tr>
            {rank_rows(bot)}
            {cur_sec}
            {prev_sec}
        </tbody>
    </table>"""


def run_market_monitor(seq_key=None, verbose=False, current_holdings_set=None, prev_holdings_set=None,
                       current_rebalance_date=None, prev_rebalance_date=None,
                       ext_dates=None, ext_values=None, risk_config=None, ext_equity_curve=None):
    """主函数：运行市场监控，返回 stats dict + chart path + HTML section

    Parameters
    ----------
    current_holdings_set : set[str] or None
        当前持仓代码集，传给 compute_top_etf_rankings。
    prev_holdings_set : set[str] or None
        上期持仓代码集，传给 compute_top_etf_rankings。
    current_rebalance_date : str or None
        当前调仓日，传给 compute_top_etf_rankings。
    prev_rebalance_date : str or None
        上期调仓日，传给 compute_top_etf_rankings。
    ext_dates : list[str] or None
        外部 eq curve dates（来自 juejin 等），替代本地 backtest_state.json。
    ext_values : list[float] or None
        外部 eq curve values。
    ext_equity_curve : list[dict] or None
        完整 equity curve（含 total_value, risk_multiplier, stock_exposure），
        用于虚拟无风控组合回算。若提供则替代 ext_dates/ext_values。
    """
    if verbose:
        print("=" * 50)
        print("Market Monitor")
        print("=" * 50)

    hs_data = load_hs300_data()
    if verbose:
        print(f"  HS300 data: {hs_data['日期'].min().date()} ~ {hs_data['日期'].max().date()} ({len(hs_data)} days)")

    hs_regime = classify_market_regime(hs_data)
    if verbose:
        regime_counts = hs_regime["regime"].value_counts()
        print(f"  Market regimes: {dict(regime_counts)}")

    current_regime = get_current_regime(hs_data)
    if verbose:
        print(f"  Current: {current_regime['regime']} (rolling_ret={current_regime['rolling_20d_return']:+.2f}%, vol={current_regime['rolling_vol']:.1f}%)")

    breadth_df, breadth_last = compute_market_breadth()
    if verbose:
        print(f"  Market breadth: bull={breadth_last['bull_pct']:.0f}%  sideways={breadth_last['sideways_pct']:.0f}%  bear={breadth_last['bear_pct']:.0f}%")

    if ext_equity_curve is not None:
        dates = [e["date"] for e in ext_equity_curve]
        values = [e["total_value"] for e in ext_equity_curve]
    elif ext_dates is not None and ext_values is not None:
        dates, values = list(ext_dates), list(ext_values)
    else:
        dates, values = load_backtest_equity(seq_key)
    if dates is None or len(dates) < 10:
        if verbose:
            print("  Warning: backtest equity data too short or missing")
        return {}, "", ""

    if verbose:
        print(f"  Backtest equity: {dates[0]} ~ {dates[-1]} ({len(dates)} days)")

    df_regime, stats = compute_model_vs_market(dates, values, hs_regime)

    chart_path = OUTPUT_DIR / "market_monitor.png"
    plot_market_analysis(dates, values, df_regime, breadth_df, chart_path)
    if verbose:
        print(f"  Chart saved: {chart_path}")

    regime_html = build_regime_table_html(stats, current_regime, breadth_last)

    # 风控状态
    risk_state = compute_breadth_risk_state(risk_config)

    # 风控盈亏: 虚拟无风控组合 vs 实际组合
    if risk_state is not None and ext_equity_curve is not None and len(ext_equity_curve) > 1:
        virtual_result = compute_virtual_equity_curve(ext_equity_curve)
        if virtual_result is not None:
            virtual_dates, virtual_vals = virtual_result
            actual_total_return = (values[-1] / values[0] - 1) * 100
            virtual_total_return = (virtual_vals[-1] / virtual_vals[0] - 1) * 100
            risk_state["risk_pnl"] = round(actual_total_return - virtual_total_return, 2)
            risk_state["actual_return"] = round(actual_total_return, 2)
            risk_state["virtual_return"] = round(virtual_total_return, 2)
            if verbose:
                print(f"  风控盈亏: 实际 {actual_total_return:+.2f}% / 虚拟 {virtual_total_return:+.2f}% / 差 {risk_state['risk_pnl']:+.2f}%")

    risk_state_html = build_risk_state_html(risk_state)

    top, bot, holdings_data, prev_holdings_data, rank_start, rank_end = compute_top_etf_rankings(
        current_holdings_set=current_holdings_set, prev_holdings_set=prev_holdings_set,
        current_rebalance_date=current_rebalance_date, prev_rebalance_date=prev_rebalance_date,
    )
    rank_date = f"{rank_start}~{rank_end}" if rank_start else ""
    rank_html = build_etf_rankings_html(top, bot, holdings_data, prev_holdings_data, rank_date) if top else ""
    if verbose:
        print("  Top 10 ETFs (5d):")
        for item in top:
            tag = " [持仓]" if item.get("held") else ""
            print(f"    {item['code']}: {item['return']:+.2f}%{tag}")
        print("  Bottom 10 ETFs (5d):")
        for item in bot:
            tag = " [持仓]" if item.get("held") else ""
            print(f"    {item['code']}: {item['return']:+.2f}%{tag}")
        print("  Holdings ranking:")
        for item in holdings_data:
            print(f"    {item['code']}: {item['return']:+.2f}% (rank {item['rank']}/{item['total']})")
        if prev_holdings_data:
            d = prev_holdings_data[0]["days_ago"]
            print(f"  Previous holdings ({d}d ago):")
            for item in prev_holdings_data:
                print(f"    {item['code']}: {item['return']:+.2f}% (rank {item['rank']}/{item['total']})")

    # 生成子图（市场宽度回测期 + 20d滚动收益）嵌入日报
    subplot_html = ""
    try:
        breadth_chart = OUTPUT_DIR / "market_breadth.png"
        plot_breadth_backtest(dates, breadth_df, breadth_chart)
        rolling_chart = OUTPUT_DIR / "market_rolling_20d.png"
        plot_rolling_20d(dates, values, rolling_chart)
        subplot_html = f"""
        <div style="margin-top:10px;">
            <div style="margin-bottom:10px;"><img src="{_img_to_b64(breadth_chart)}" style="width:100%;border:1px solid #ddd;border-radius:5px;"></div>
            <div><img src="{_img_to_b64(rolling_chart)}" style="width:100%;border:1px solid #ddd;border-radius:5px;"></div>
        </div>"""
    except Exception as e:
        print(f"  [子图] 生成失败: {e}")

    html_section = regime_html + risk_state_html + subplot_html + "<br>" + (rank_html if rank_html else "")

    json_path = OUTPUT_DIR / "market_monitor.json"
    with open(json_path, "w") as f:
        json.dump({
            "stats": stats,
            "current_regime": current_regime,
            "breadth": breadth_last,
            "risk_manager": risk_state,
            "etf_rankings": {
                "period": rank_date,
                "top": top,
                "bottom": bot,
                "holdings": holdings_data,
                "prev_holdings": prev_holdings_data,
            },
            "last_updated": datetime.now().strftime("%Y-%m-%d %H:%M"),
        }, f, indent=2)
        print(f"  JSON saved: {json_path}")

    if verbose:
        print("  Done.")
    return stats, str(chart_path), html_section


def main():
    stats, chart_path, html_section = run_market_monitor()
    if stats:
        print(f"\nSummary:")
        for regime, s in stats.items():
            print(f"  {regime:10s}: model={s['model_return']:+.2f}%  hs300={s['hs300_return']:+.2f}%  excess={s['excess_return']:+.2f}%  beat_rate={s['beat_rate']*100:.1f}%")
    print(f"\nChart: {chart_path}")


if __name__ == "__main__":
    main()
