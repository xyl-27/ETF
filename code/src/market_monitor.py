"""
市场监控脚本
- 读取 backtest_state.json + etf_74.csv
- 以 HS300(510300.XSHG) 为基准划分市场状态（牛/熊/震荡）
- 对比模型在各市场状态下的表现
- 输出 JSON 和图表，可嵌入日报
"""

import os
import json
import sys
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
    # normalise date format
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
    使用滚动收益率 + 滚动波动率判断。
    """
    hs = hs_data.copy()
    hs["rolling_ret"] = hs["收盘"].pct_change(window) * 100
    hs["rolling_vol"] = hs["return_pct"].rolling(window).std() * np.sqrt(252)

    def regime(row):
        ret = row["rolling_ret"]
        vol = row["rolling_vol"]
        if pd.isna(ret) or pd.isna(vol):
            return "N/A"
        # 牛: 收益 > 5% 且 波动 < 30%
        if ret > 5 and vol < 30:
            return "bull"
        # 熊: 收益 < -5%
        if ret < -5:
            return "bear"
        # 震荡: 其它
        return "sideways"

    hs["regime"] = hs.apply(regime, axis=1)
    return hs


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
        model_total = (sub["model_value"].iloc[-1] / sub["model_value"].iloc[0] - 1) * 100
        hs_total = (1 + sub["hs300_return"] / 100).prod() - 1
        hs_total = hs_total * 100
        excess = model_total - hs_total
        beat_rate = (sub["model_return"] > sub["hs300_return"]).mean()
        stats[regime] = {
            "days": len(sub),
            "model_return": round(model_total, 2),
            "hs300_return": round(hs_total, 2),
            "excess_return": round(excess, 2),
            "beat_rate": round(beat_rate, 4),
            "avg_model_return": round(sub["model_return"].mean(), 4),
            "avg_hs300_return": round(sub["hs300_return"].mean(), 4),
        }

    # 全周期
    model_total = (values[-1] / values[0] - 1) * 100
    hs_total = (1 + df["hs300_return"] / 100).prod() - 1
    hs_total = hs_total * 100
    beat_all = (df["model_return"] > df["hs300_return"]).mean()
    stats["all"] = {
        "days": len(df),
        "model_return": round(model_total, 2),
        "hs300_return": round(hs_total, 2),
        "excess_return": round(model_total - hs_total, 2),
        "beat_rate": round(beat_all, 4),
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
        # 最近窗口的日胜率
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


def plot_market_analysis(dates, values, df_regime, output_path):
    """生成市场监控图表"""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import matplotlib.dates as mdates
    plt.rcParams["font.sans-serif"] = ["SimHei", "DejaVu Sans", "Arial Unicode MS"]
    plt.rcParams["axes.unicode_minus"] = False

    hs_data = load_hs300_data()
    hs_prices = hs_data.set_index("日期")["收盘"]

    fig, axes = plt.subplots(3, 1, figsize=(14, 10), sharex=True)
    fig.suptitle("Market Monitor — Model vs HS300", fontsize=14, fontweight="bold")

    plot_dates = pd.to_datetime(dates)
    model_vals = np.array(values)
    hs_aligned = hs_prices.reindex(plot_dates, method="ffill")
    hs_norm = hs_aligned.values / hs_aligned.values[0] * values[0]

    # 1. Equity curve with regime background
    ax = axes[0]
    colors = {"bull": "#d4edda", "bear": "#f8d7da", "sideways": "#fff3cd", "N/A": "#f8f9fa"}
    regime_dates = pd.to_datetime(df_regime["date"])
    unique_regimes = df_regime["regime"].unique()
    for r in unique_regimes:
        mask = df_regime["regime"].values == r
        if mask.sum() < 2:
            continue
        sub_dates = regime_dates[mask]
        if len(sub_dates) < 2:
            continue
        ax.axvspan(sub_dates.iloc[0], sub_dates.iloc[-1],
                    alpha=0.15, color=colors.get(r, "#ccc"))
    ax.plot(plot_dates, model_vals, color="#e74c3c", lw=2, label="Strategy")
    ax.plot(plot_dates, hs_norm, color="#7f8c8d", lw=1.5, ls="--", label="HS300(norm)")
    ax.set_ylabel("Portfolio Value (CNY)")
    ax.legend(loc="upper left", fontsize=9)
    ax.grid(True, alpha=0.3)
    from matplotlib.patches import Patch
    legend_patches = [Patch(color=colors[r], alpha=0.4, label=r) for r in ["bull", "bear", "sideways"] if r in unique_regimes]
    ax.legend(handles=[plt.Line2D([0],[0],color="#e74c3c",lw=2,label="Strategy"),
                       plt.Line2D([0],[0],color="#7f8c8d",lw=1.5,ls="--",label="HS300(norm)"),
                       *legend_patches], loc="upper left", fontsize=8)

    # 2. Daily excess return
    ax = axes[1]
    excess = df_regime["model_return"].values - df_regime["hs300_return"].values
    bar_colors = ["#cc0000" if v >= 0 else "#009900" for v in excess]
    ax.bar(regime_dates, excess, color=bar_colors, width=0.8, alpha=0.7)
    ax.axhline(0, color="gray", ls="-", lw=0.5)
    ax.set_ylabel("Daily Excess Return (%)")
    ax.grid(True, alpha=0.3)

    # 3. Rolling 20-day return comparison
    ax = axes[2]
    window = 20
    model_prices = pd.Series(values, index=plot_dates)
    model_roll = model_prices.pct_change(window).dropna() * 100
    hs_roll = hs_prices.reindex(model_roll.index).pct_change(window) * 100
    # Reindex to model_roll index
    hs_roll = hs_roll.reindex(model_roll.index)
    ax.plot(model_roll.index, model_roll.values, color="#e74c3c", lw=1.5, label=f"Strategy {window}d return")
    ax.plot(hs_roll.index, hs_roll.values, color="#7f8c8d", lw=1.5, ls="--", label=f"HS300 {window}d return")
    ax.axhline(0, color="gray", ls=":", lw=0.5)
    ax.set_ylabel(f"{window}-day Return (%)")
    ax.legend(loc="upper left", fontsize=9)
    ax.grid(True, alpha=0.3)

    plt.xlabel("Date")
    plt.tight_layout()
    fig.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    return str(output_path)


def build_regime_table_html(stats):
    """生成市场状态对比的 HTML 表格"""
    rows = ""
    for regime, label in [("all", "全周期"), ("bull", "牛市"), ("bear", "熊市"), ("sideways", "震荡")]:
        s = stats.get(regime)
        if not s:
            continue
        m_ret = s["model_return"]
        h_ret = s["hs300_return"]
        e_ret = s["excess_return"]
        b_rate = s["beat_rate"]
        m_clr = "#cc0000" if m_ret >= 0 else "#009900"
        h_clr = "#cc0000" if h_ret >= 0 else "#009900"
        e_clr = "#cc0000" if e_ret >= 0 else "#009900"
        rows += f"""<tr>
            <td style="font-weight:bold;">{label}</td>
            <td style="text-align:right;">{s['days']}天</td>
            <td style="text-align:right;color:{m_clr};">{m_ret:+.2f}%</td>
            <td style="text-align:right;color:{h_clr};">{h_ret:+.2f}%</td>
            <td style="text-align:right;color:{e_clr};font-weight:bold;">{e_ret:+.2f}%</td>
            <td style="text-align:right;">{b_rate*100:.1f}%</td>
        </tr>"""
    return f"""<h3>市场状态 vs 模型表现</h3>
    <table>
        <thead><tr><th>市场状态</th><th style="text-align:right;">天数</th><th style="text-align:right;">模型收益</th><th style="text-align:right;">HS300</th><th style="text-align:right;">超额收益</th><th style="text-align:right;">日胜率</th></tr></thead>
        <tbody>{rows}</tbody>
    </table>"""


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
            <td style="font-weight:bold;">{label}</td>
            <td style="text-align:right;color:{m_clr};">{m_ret:+.2f}%</td>
            <td style="text-align:right;color:{h_clr};">{h_ret:+.2f}%</td>
            <td style="text-align:right;color:{e_clr};font-weight:bold;">{e_ret:+.2f}%</td>
            <td style="text-align:right;">{s['daily_win_rate']*100:.1f}%</td>
        </tr>"""
    return f"""<h3>近期表现</h3>
    <table>
        <thead><tr><th>区间</th><th style="text-align:right;">模型收益</th><th style="text-align:right;">HS300</th><th style="text-align:right;">超额收益</th><th style="text-align:right;">日胜率</th></tr></thead>
        <tbody>{rows}</tbody>
    </table>"""


def run_market_monitor(seq_key=None):
    """主函数：运行市场监控，返回 stats dict + chart path + HTML section"""
    print("=" * 50)
    print("Market Monitor")
    print("=" * 50)

    hs_data = load_hs300_data()
    print(f"  HS300 data: {hs_data['日期'].min().date()} ~ {hs_data['日期'].max().date()} ({len(hs_data)} days)")

    hs_regime = classify_market_regime(hs_data)
    regime_counts = hs_regime["regime"].value_counts()
    print(f"  Market regimes: {dict(regime_counts)}")

    dates, values = load_backtest_equity(seq_key)
    if dates is None or len(dates) < 10:
        print("  Warning: backtest equity data too short or missing")
        return {}, "", ""

    print(f"  Backtest equity: {dates[0]} ~ {dates[-1]} ({len(dates)} days)")

    df_regime, stats = compute_model_vs_market(dates, values, hs_regime)
    recent = compute_recent_performance(dates, values, hs_data)

    chart_path = OUTPUT_DIR / "market_monitor.png"
    plot_market_analysis(dates, values, df_regime, chart_path)
    print(f"  Chart saved: {chart_path}")

    regime_html = build_regime_table_html(stats)
    recent_html = build_recent_table_html(recent)
    html_section = regime_html + "<br>" + recent_html

    # 保存 JSON 供日报使用
    json_path = OUTPUT_DIR / "market_monitor.json"
    with open(json_path, "w") as f:
        json.dump({"stats": stats, "recent": recent, "last_updated": datetime.now().strftime("%Y-%m-%d %H:%M")}, f, indent=2)
    print(f"  JSON saved: {json_path}")

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
