"""
每日定时测评脚本 - ETF (实盘模拟)
支持多模型回测 (单模型 + 融合)，持久化状态，生成日报
"""

import os
import sys
import json
import re
import io
import base64
import traceback
import subprocess
from datetime import datetime
from typing import Optional, Dict, List, Any, Tuple
from pathlib import Path

import torch
import numpy as np
import pandas as pd


class NumpyEncoder(json.JSONEncoder):
    """将 numpy 类型转为 JSON 可序列化的 Python 原生类型"""
    def default(self, obj):
        if isinstance(obj, (np.integer,)):
            return int(obj)
        if isinstance(obj, (np.floating,)):
            return float(obj)
        if isinstance(obj, (np.ndarray,)):
            return obj.tolist()
        return super().default(obj)

# 项目根目录
PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
os.chdir(PROJECT_ROOT)
sys.path.insert(0, str(PROJECT_ROOT / "code" / "src"))

from backtest import BacktestEngine, ETFBacktester
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from matplotlib.ticker import FuncFormatter


# ============================================================
# 绘图
# ============================================================

def plot_equity_curves(sequences: Dict[str, Any], data_file: str, initial_capital: float, save_path: str):
    """绘制所有序列的收益曲线图 + 沪深300基准 (对齐首个交易日)"""
    plt.rcParams["font.sans-serif"] = ["SimHei", "DejaVu Sans", "Arial Unicode MS"]
    plt.rcParams["axes.unicode_minus"] = False

    fig, ax = plt.subplots(figsize=(14, 8))

    colors = {
        "average": "#e74c3c",
        "voting": "#e67e22",
        "exp_28": "#2ecc71",
        "exp_42": "#3498db",
        "exp_66": "#9b59b6",
        "exp_67": "#f39c12",
        "exp_48": "#1abc9c",
    }

    # 1. 确定策略实际开始交易的日期 (所有序列中最早的一笔交易)
    first_trade_date = None
    for seq in sequences.values():
        if seq["trades"]:
            t_date = pd.Timestamp(seq["trades"][0]["date"])
            if first_trade_date is None or t_date < first_trade_date:
                first_trade_date = t_date

    if first_trade_date is None:
        first_trade_date = pd.Timestamp("2026-04-01")

    # 2. 加载 HS300 数据，从首个交易日开始归一化
    raw_df = pd.read_csv(data_file, dtype={"股票代码": str})
    raw_df["股票代码"] = raw_df["股票代码"].astype(str).str.zfill(6)
    raw_df["日期"] = pd.to_datetime(raw_df["日期"])
    hs300_df = raw_df[raw_df["股票代码"] == "510300.XSHG"].sort_values("日期").copy()

    hs300_start_row = hs300_df[hs300_df["日期"] >= first_trade_date]
    if not hs300_start_row.empty:
        start_val = hs300_start_row["收盘"].iloc[0]
        hs300_plot_df = hs300_df[hs300_df["日期"] >= first_trade_date].copy()
        hs300_plot_df["value_wan"] = (hs300_plot_df["收盘"] / start_val) * (initial_capital / 10000)

        hs300_return_pct = (hs300_plot_df["收盘"].iloc[-1] / hs300_plot_df["收盘"].iloc[0] - 1) * 100

        ax.plot(
            hs300_plot_df["日期"], hs300_plot_df["value_wan"],
            label=f"沪深 300 ({hs300_return_pct:+.2f}%)",
            color="#7f8c8d", linewidth=2, linestyle=":",
        )

    # 3. 绘制各模型序列 (过滤掉交易前的缓存期)
    all_dates = []

    for key, seq in sequences.items():
        eq = seq["equity_curve"]
        if not eq:
            continue

        # 过滤 equity_curve，只显示从 first_trade_date 开始的数据
        filtered_eq = [e for e in eq if pd.Timestamp(e["date"]) >= first_trade_date]
        if not filtered_eq:
            continue

        dates = pd.to_datetime([e["date"] for e in filtered_eq])
        values = [e["total_value"] / 10000 for e in filtered_eq]
        all_dates.extend(dates)

        color = colors.get(key, None)
        linewidth = 3 if key in ("average", "voting") else 1.5
        linestyle = "--" if key in ("average", "voting") else "-"

        ax.plot(dates, values, label=f"{key} ({seq['metrics']['strategy_return_pct']:+.2f}%)",
                color=color, linewidth=linewidth, linestyle=linestyle)

    # 4. 设置坐标轴
    if all_dates:
        ax.set_xlim(min(all_dates), max(all_dates))
        ax.xaxis.set_major_formatter(mdates.DateFormatter("%m-%d"))
        fig.autofmt_xdate()

    ax.set_title("各模型序列收益曲线 (vs 沪深300)", fontsize=16, fontweight="bold", pad=20)
    ax.set_xlabel("日期", fontsize=12)
    ax.set_ylabel("账户总值 (万元)", fontsize=12)
    ax.yaxis.set_major_formatter(FuncFormatter(lambda x, _: f"{x:.1f}"))
    ax.grid(True, alpha=0.3, linestyle="--", linewidth=0.8)
    ax.legend(fontsize=10, framealpha=0.9, edgecolor="gray", loc="upper left")

    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close()


# ============================================================
OUTPUT_DIR = PROJECT_ROOT / "output"
STATE_PATH = OUTPUT_DIR / "backtest_state.json"
REPORT_PATH = OUTPUT_DIR / "latest_report.json"
PORTFOLIO_PATH = OUTPUT_DIR / "portfolio.json"
MODEL_SELECTION_PATH = OUTPUT_DIR / "model_selection.yaml"
DATA_FILE = PROJECT_ROOT / "etf_data" / "etf_74.csv"


# ============================================================
# 数据更新
# ============================================================

def update_etf_data(verbose: bool = True) -> bool:
    script_path = str(PROJECT_ROOT / "get_etf_data.py")
    if not os.path.exists(script_path):
        if verbose:
            print("[数据更新] 未找到 get_etf_data.py，跳过")
        return False

    if verbose:
        print("[数据更新] 运行 get_etf_data.py 获取最新数据...")

    try:
        result = subprocess.run(
            [sys.executable, script_path],
            cwd=str(PROJECT_ROOT),
            capture_output=False,
            timeout=600,
        )
        if result.returncode == 0:
            if verbose:
                print("[数据更新] ETF数据获取成功")
            return True
        else:
            if verbose:
                print(f"[数据更新] ETF数据获取失败 (exit code: {result.returncode})")
            return False
    except subprocess.TimeoutExpired:
        if verbose:
            print("[数据更新] 超时 (10分钟)，跳过")
        return False
    except Exception as e:
        if verbose:
            print(f"[数据更新] 异常: {e}")
        return False


# ============================================================
# 模型加载
# ============================================================

def load_model_selection(path: str) -> Tuple[List[Dict], str, bool, bool]:
    import yaml
    with open(path, "r", encoding="utf-8") as f:
        data = yaml.safe_load(f)
    models = []
    master = data.get("master", "")
    average_enabled = data.get("average", data.get("fusion", False))
    voting_enabled = data.get("voting", False)
    for m in data.get("models", []):
        if isinstance(m, dict):
            enabled = m.get("enabled", True)
            models.append({
                "exp_dir": m.get("dir", ""),
                "model_file": m.get("file", ""),
                "enabled": enabled,
            })
    return models, master, average_enabled, voting_enabled


def find_best_model(output_dir: str) -> Optional[Tuple[str, str, float]]:
    search_results_path = os.path.join(output_dir, "search_results.json")
    if not os.path.exists(search_results_path):
        return None
    with open(search_results_path, "r") as f:
        results = json.load(f)
    if not results:
        return None
    best = max(results, key=lambda x: x.get("score", 0))
    exp_idx = best["exp_idx"]
    exp_dir = os.path.join(output_dir, f"exp_{exp_idx}")
    if not os.path.exists(exp_dir):
        return None
    model_file = "best_model_sliding.pth"
    if not os.path.exists(os.path.join(exp_dir, model_file)):
        model_file = "best_model.pth"
        if not os.path.exists(os.path.join(exp_dir, model_file)):
            return None
    return exp_dir, model_file, best.get("score", 0)


# ============================================================
# 回测执行
# ============================================================

def _extract_drawdowns(vals, dates, max_periods=5):
    """提取前 N 大回撤区间 (开始→谷底→恢复, 深度, 持续时间)"""
    running_max = -np.inf
    peak_idx = 0
    in_dd = False
    dd_start = None
    dd_start_idx = None
    trough_date = None
    trough_idx = None
    trough_depth = 0
    periods = []

    for i, v in enumerate(vals):
        if v > running_max:
            running_max = v
            if in_dd:
                periods.append({
                    "start": str(dd_start),
                    "trough": str(trough_date),
                    "recovery": str(dates[i]),
                    "depth_pct": round(float(trough_depth), 2),
                    "duration_days": int(i - dd_start_idx),
                    "recovery_days": int(i - trough_idx),
                })
                in_dd = False
            peak_idx = i
        else:
            dd_pct = (running_max - v) / running_max * 100
            if not in_dd:
                in_dd = True
                dd_start = dates[peak_idx]
                dd_start_idx = peak_idx
                trough_date = dates[i]
                trough_idx = i
                trough_depth = dd_pct
            else:
                if dd_pct > trough_depth:
                    trough_date = dates[i]
                    trough_idx = i
                    trough_depth = dd_pct

    if in_dd:
        periods.append({
            "start": str(dd_start),
            "trough": str(trough_date),
            "recovery": None,
            "depth_pct": round(float(trough_depth), 2),
            "duration_days": int(len(vals) - dd_start_idx),
            "recovery_days": None,
        })

    periods.sort(key=lambda x: x["depth_pct"], reverse=True)
    return periods[:max_periods]


def run_backtest_sequence(
    predictions_func,
    data_file: str,
    start_date: str,
    end_date: str,
    top_k: int,
    rebalance_days: int,
    position_pct: float,
    initial_capital: float = 1000000,
) -> Dict[str, Any]:
    raw_df = pd.read_csv(data_file, dtype={"股票代码": str})
    raw_df["股票代码"] = raw_df["股票代码"].astype(str).str.zfill(6)
    raw_df["日期"] = pd.to_datetime(raw_df["日期"])
    price_data = raw_df.copy()

    start_ts = pd.Timestamp(start_date)
    end_ts = pd.Timestamp(end_date)
    all_dates = sorted(raw_df["日期"].unique())
    backtest_dates = [d for d in all_dates if start_ts <= d < end_ts]

    engine = BacktestEngine(
        initial_capital=initial_capital,
        commission=0.0003,
        top_k=top_k,
        position_pct=position_pct,
        log=False,
    )

    engine.run(
        dates=backtest_dates,
        price_data=price_data,
        predictions_func=predictions_func,
        rebalance_days=rebalance_days,
        first_rebalance_date=start_ts,
    )

    equity_curve = []
    for ec in engine.equity_curve:
        equity_curve.append({
            "date": ec["date"].strftime("%Y-%m-%d") if isinstance(ec["date"], pd.Timestamp) else str(ec["date"]),
            "total_value": round(ec["total_value"], 2),
        })

    trades = []
    for t in engine.trades:
        trade = {
            "date": t["date"].strftime("%Y-%m-%d") if isinstance(t["date"], pd.Timestamp) else str(t["date"]),
            "action": t["action"],
            "stock": t["stock"],
            "price": round(t["price"], 4),
            "shares": t["shares"],
            "amount": round(t["amount"], 2),
        }
        if "score" in t:
            trade["score"] = t["score"]
        if "advantage" in t:
            trade["advantage"] = t["advantage"]
        trades.append(trade)

    positions = {}
    for stock_id, pos in engine.positions.items():
        positions[stock_id] = {
            "shares": pos["shares"],
            "cost": round(pos["cost"], 2),
        }

    def _compute_window_metrics(ec_segment, init_cap):
        """Helper: compute metrics for a segment of equity curve."""
        if len(ec_segment) < 2:
            return {}
        vals = [e["total_value"] for e in ec_segment]
        dates = [e["date"] for e in ec_segment]
        total_ret = (vals[-1] / init_cap - 1) * 100
        cum = np.array(vals) / init_cap
        daily_rets = np.diff(vals) / np.array(vals[:-1])
        n_days = len(daily_rets)
        # 胜率
        win_rate = float(np.mean(daily_rets > 0)) if n_days > 0 else 0.0
        # 日均收益率
        daily_avg = float(np.mean(daily_rets)) * 100
        # 年化收益率
        ann_ret = (1 + total_ret / 100) ** (252 / n_days) - 1 if n_days > 0 else 0.0
        annualized_ret_pct = ann_ret * 100
        # 波动率
        daily_std = float(np.std(daily_rets)) if n_days > 0 else 0.0
        annualized_vol = daily_std * np.sqrt(252) * 100
        # 夏普
        sharpe = float((np.mean(daily_rets) / daily_std) * np.sqrt(252)) if daily_std != 0 else 0.0
        # 最大回撤
        running_max = np.maximum.accumulate(cum)
        dd = (cum - running_max) / running_max * 100
        max_dd = float(abs(dd.min())) if len(dd) > 0 else 0.0
        # 最大回撤区间
        if max_dd > 0:
            dd_end_idx = np.argmin(dd)
            dd_series = dd[:dd_end_idx + 1]
            dd_start_idx = np.argmax(running_max[:dd_end_idx + 1])
            mdd_start = dates[int(dd_start_idx)]
            mdd_end = dates[int(dd_end_idx)]
            mdd_duration = int(dd_end_idx - dd_start_idx)
        else:
            mdd_start = mdd_end = ""
            mdd_duration = 0
        # 卡玛比率
        calmar = ann_ret / (max_dd / 100) if max_dd > 0 else 0.0
        # 索提诺比率
        downside = daily_rets[daily_rets < 0]
        downside_std = float(np.std(downside)) if len(downside) > 1 else daily_std
        sortino = float((np.mean(daily_rets) / downside_std) * np.sqrt(252)) if downside_std != 0 else 0.0
        # 计算多段回撤（前5大）
        dd_periods = _extract_drawdowns(vals, dates)
        return {
            "strategy_return_pct": round(total_ret, 4),
            "annualized_return_pct": round(annualized_ret_pct, 4),
            "daily_avg_return_pct": round(daily_avg, 4),
            "daily_win_rate": round(win_rate, 4),
            "max_drawdown_pct": round(max_dd, 4),
            "max_drawdown_details": {
                "start_date": mdd_start,
                "end_date": mdd_end,
                "duration_days": mdd_duration,
            },
            "drawdown_periods": dd_periods,
            "sharpe_ratio": round(sharpe, 4),
            "calmar_ratio": round(calmar, 4),
            "sortino_ratio": round(sortino, 4),
            "annualized_volatility_pct": round(annualized_vol, 4),
            "total_days": n_days,
            "latest_value": round(vals[-1], 2),
        }

    # -- 整体指标 --
    overall = _compute_window_metrics(equity_curve, initial_capital)
    if not overall:
        return {
            "equity_curve": equity_curve,
            "trades": trades,
            "positions": positions,
            "metrics": {},
            "cash": round(engine.cash, 2),
        }

    # -- HS300 对比 --
    hs300_code = "510300.XSHG"
    hs300_df = raw_df[raw_df["股票代码"] == hs300_code].copy()
    hs300_df = hs300_df[(hs300_df["日期"] >= start_ts) & (hs300_df["日期"] < end_ts)]
    if len(hs300_df) >= 1:
        hs300_return = (hs300_df["收盘"].iloc[-1] / hs300_df["收盘"].iloc[0] - 1) * 100
    else:
        hs300_return = 0.0
    excess_return = overall["strategy_return_pct"] - hs300_return

    # -- 近期窗口 --
    ec_by_date = {}
    for ec in equity_curve:
        ec_by_date[ec["date"]] = ec
    sorted_dates = sorted(ec_by_date.keys())
    today = pd.Timestamp(sorted_dates[-1])

    windows = {}
    for label, delta in [("5d", 5), ("1m", 30)]:
        cutoff = (today - pd.Timedelta(days=delta)).strftime("%Y-%m-%d")
        seg = [ec_by_date[d] for d in sorted_dates if d >= cutoff]
        if seg:
            win_metrics = _compute_window_metrics(seg, seg[0]["total_value"])
            if win_metrics:
                windows[f"window_{label}"] = win_metrics

    # -- 交易统计 --
    trade_actions = [t for t in trades if t["action"] in ("buy", "sell")]
    n_trades = len(trade_actions)

    # -- 下一个调仓日 --
    try:
        import pandas_market_calendars as mcal
        xshg = mcal.get_calendar("XSHG")
        look_end = backtest_dates[-1] + pd.Timedelta(days=365)
        all_cal_dates = xshg.valid_days(start_date=start_ts, end_date=look_end, tz=None)
        start_pos = all_cal_dates.get_loc(pd.Timestamp(start_ts).normalize())
        current_pos = all_cal_dates.get_loc(pd.Timestamp(backtest_dates[-1]).normalize())
        n_periods = (current_pos - start_pos) // rebalance_days
        next_pos = start_pos + (n_periods + 1) * rebalance_days
        next_rebalance_date = all_cal_dates[next_pos].strftime("%Y-%m-%d") if next_pos < len(all_cal_dates) else ""
    except Exception:
        next_rebalance_date = ""

    # -- 今日盈亏 --
    today_pnl = {}
    if len(equity_curve) >= 2:
        ec_today = equity_curve[-1]
        ec_yesterday = equity_curve[-2]
        today_date_str = ec_today["date"]
        yesterday_date_str = ec_yesterday["date"]
        today_total = ec_today["total_value"]
        yesterday_total = ec_yesterday["total_value"]
        today_pnl_total = round(today_total - yesterday_total, 2)

        today_ts = pd.Timestamp(today_date_str)
        yesterday_ts = pd.Timestamp(yesterday_date_str)

        per_position_pnl = []
        for stock_id, pos in engine.positions.items():
            shares = pos["shares"]
            sub = raw_df[raw_df["股票代码"] == stock_id]
            today_close_s = sub.loc[sub["日期"] == today_ts, "收盘"]
            yesterday_close_s = sub.loc[sub["日期"] == yesterday_ts, "收盘"]
            if not today_close_s.empty and not yesterday_close_s.empty:
                tc = today_close_s.values[0]
                yc = yesterday_close_s.values[0]
                pnl = round(shares * (tc - yc), 2)
                pnl_pct = round((tc / yc - 1) * 100, 4) if yc > 0 else 0.0
                per_position_pnl.append({
                    "stock_id": stock_id,
                    "shares": shares,
                    "today_close": round(tc, 4),
                    "yesterday_close": round(yc, 4),
                    "pnl": pnl,
                    "pnl_pct": pnl_pct,
                })

        today_pnl = {
            "total_pnl": today_pnl_total,
            "positions": per_position_pnl,
        }

    # -- NDCG & KS 检验 --
    ndcg_list = []
    ks_stat_list = []
    ks_p_list = []
    all_dates_sorted = sorted(raw_df["日期"].unique())
    for ph in engine.predictions_history:
        rb_date = ph["date"]
        preds = ph["predictions"]
        stock_ids = [p["stock_id"] for p in preds]
        scores = [p["score"] for p in preds]

        rb_ts = pd.Timestamp(rb_date)
        next_rb = None
        for ph2 in engine.predictions_history:
            if ph2["date"] > rb_date:
                next_rb = ph2["date"]
                break

        forward_rets = []
        for sid in stock_ids:
            sub = raw_df[raw_df["股票代码"] == sid]
            close_rb = sub.loc[sub["日期"] == rb_ts, "收盘"]
            if close_rb.empty:
                forward_rets.append(0.0)
                continue
            close_rb_val = float(close_rb.values[0])
            if next_rb:
                next_ts = pd.Timestamp(next_rb)
                close_next = sub.loc[sub["日期"] == next_ts, "收盘"]

                if not close_next.empty:
                    fwd = (float(close_next.values[0]) / close_rb_val - 1) * 100
                else:
                    fwd = 0.0
            else:
                idx = all_dates_sorted.index(rb_ts) if rb_ts in all_dates_sorted else -1
                if idx >= 0 and idx + 5 < len(all_dates_sorted):
                    end_ts = all_dates_sorted[idx + 5]
                    close_end = sub.loc[sub["日期"] == end_ts, "收盘"]
                    if not close_end.empty:
                        fwd = (float(close_end.values[0]) / close_rb_val - 1) * 100
                    else:
                        fwd = 0.0
                else:
                    fwd = 0.0
            forward_rets.append(fwd)

        median_ret = float(np.median(forward_rets)) if forward_rets else 0.0
        rel = [1 if r > median_ret else 0 for r in forward_rets]

        k = min(top_k, len(preds))
        if k > 0:
            dcg = sum(rel[i] / np.log2(i + 2) for i in range(k))
            ideal_rel = sorted(rel, reverse=True)
            idcg = sum(ideal_rel[i] / np.log2(i + 2) for i in range(k))
            ndcg_list.append(dcg / idcg if idcg > 0 else 0.0)

        good_scores = [scores[i] for i in range(len(scores)) if forward_rets[i] > median_ret]
        bad_scores = [scores[i] for i in range(len(scores)) if forward_rets[i] <= median_ret]
        if len(good_scores) > 1 and len(bad_scores) > 1:
            from scipy.stats import ks_2samp
            ks_stat, ks_p = ks_2samp(good_scores, bad_scores)
            ks_stat_list.append(ks_stat)
            ks_p_list.append(ks_p)

    avg_ndcg = round(float(np.mean(ndcg_list)), 4) if ndcg_list else None
    avg_ks_stat = round(float(np.mean(ks_stat_list)), 4) if ks_stat_list else None
    avg_ks_p = round(float(np.mean(ks_p_list)), 4) if ks_p_list else None

    # -- 整合 --
    metrics = {
        **overall,
        "hs300_return_pct": round(hs300_return, 4),
        "excess_return_pct": round(excess_return, 4),
        "start_date": start_date,
        "end_date": end_date,
        "next_rebalance_date": next_rebalance_date,
        "total_trades": n_trades,
        "ndcg": avg_ndcg,
        "ks_stat": avg_ks_stat,
        "ks_p": avg_ks_p,
        **windows,
    }

    return {
        "equity_curve": equity_curve,
        "trades": trades,
        "positions": positions,
        "metrics": metrics,
        "cash": round(engine.cash, 2),
        "today_pnl": today_pnl,
    }


# ============================================================
# 主流程
# ============================================================

def _rebuild_positions(trades, up_to_date):
    """从交易记录重建截至某日的持仓"""
    pos = {}
    sorted_trades = sorted(trades, key=lambda x: x["date"])
    for t in sorted_trades:
        if t["date"] > up_to_date:
            break
        stock = t["stock"]
        if stock not in pos:
            pos[stock] = {"shares": 0, "cost": 0.0}
        if t["action"] == "买入":
            pos[stock]["shares"] += t["shares"]
            pos[stock]["cost"] += t.get("amount", t["shares"] * t["price"])
        elif t["action"] == "卖出":
            if pos[stock]["shares"] == 0:
                continue
            ratio = t["shares"] / pos[stock]["shares"]
            pos[stock]["cost"] -= pos[stock]["cost"] * min(ratio, 1.0)
            pos[stock]["shares"] -= t["shares"]
            if pos[stock]["shares"] <= 0:
                del pos[stock]
    # 保留 cost 作为持仓成本总金额
    return pos


def _history_chart_b64(all_sequences, hs300_raw, rb_date, first_date, initial_capital):
    """生成截止到rb_date的多序列收益曲线图, 返回base64 data URL"""
    colors = ["#2ecc71", "#3498db", "#9b59b6", "#f39c12", "#1abc9c"]
    fig, ax = plt.subplots(figsize=(10, 5))
    fig.patch.set_facecolor("white")
    ax.set_facecolor("#f5f5f5")

    all_dates = []
    for idx, (key, seq) in enumerate(all_sequences.items()):
        eq = seq.get("equity_curve", [])
        eq_seg = [e for e in eq if e["date"] <= rb_date]
        if len(eq_seg) < 2:
            continue
        dates = [pd.Timestamp(e["date"]) for e in eq_seg]
        vals = [e["total_value"] / 10000 for e in eq_seg]
        all_dates.extend(dates)
        ret_pct = (eq_seg[-1]["total_value"] / eq_seg[0]["total_value"] - 1) * 100
        c = colors[idx % len(colors)]
        lw = 3 if key in ("average", "voting") else 1.5
        ls = "--" if key in ("average", "voting") else "-"
        ax.plot(dates, vals, label=f"{key} ({ret_pct:+.2f}%)", color=c, linewidth=lw, linestyle=ls)

    hs300_plot = hs300_raw[(hs300_raw["date"] >= pd.Timestamp(first_date)) & (hs300_raw["date"] <= pd.Timestamp(rb_date))].copy()
    if len(hs300_plot) >= 2:
        hs300_init = hs300_plot["close"].iloc[0]
        hs300_val_wan = [(c / hs300_init) * (initial_capital / 10000) for c in hs300_plot["close"]]
        hs300_ret = (hs300_plot["close"].iloc[-1] / hs300_init - 1) * 100
        ax.plot(hs300_plot["date"], hs300_val_wan, label=f"沪深300 ({hs300_ret:+.2f}%)",
                color="#7f8c8d", linewidth=2, linestyle=":")

    if all_dates:
        ax.set_xlim(min(all_dates), max(all_dates))
    ax.axhline(y=initial_capital / 10000, color="gray", linewidth=0.8, linestyle=":")
    ax.set_title(f"收益曲线 (截至 {rb_date})", fontsize=14, fontweight="bold", pad=15)
    ax.set_ylabel("账户总值 (万元)", fontsize=11)
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%m-%d"))
    ax.grid(True, alpha=0.3, linestyle="--", linewidth=0.8)
    ax.legend(fontsize=9, framealpha=0.9, edgecolor="gray", loc="upper left")
    fig.tight_layout()
    buf = io.BytesIO()
    fig.savefig(buf, format="png", dpi=120)
    plt.close(fig)
    b64 = base64.b64encode(buf.getvalue()).decode()
    return f"data:image/png;base64,{b64}"


def _save_history_reports(seq, all_sequences, data_file, initial_capital, etf_names, model_key="", rebalance_days=5):
    """为序列的每一个调仓日保存历史报告HTML"""
    trades = seq.get("trades", [])
    equity_curve = seq.get("equity_curve", [])
    if not trades or not equity_curve:
        return

    raw_df = pd.read_csv(data_file, dtype={"股票代码": str})
    raw_df["股票代码"] = raw_df["股票代码"].astype(str).str.zfill(6)
    raw_df["日期"] = pd.to_datetime(raw_df["日期"])
    hs300_raw = raw_df[raw_df["股票代码"] == "510300.XSHG"][["日期", "收盘"]].copy()
    hs300_raw = hs300_raw.rename(columns={"日期": "date", "收盘": "close"})
    hs300_raw = hs300_raw.sort_values("date").reset_index(drop=True)
    ec_by_date = {e["date"]: e["total_value"] for e in equity_curve}
    sorted_dates = sorted(ec_by_date.keys())

    rebalance_dates = sorted(set(t["date"] for t in trades))
    history_dir = PROJECT_ROOT / "output" / "history_report"
    history_dir.mkdir(parents=True, exist_ok=True)

    for rb_date in rebalance_dates:
        ec_seg = [e for e in equity_curve if e["date"] <= rb_date]
        if len(ec_seg) < 1:
            continue

        today_total = ec_seg[-1]["total_value"]
        yesterday_total = ec_seg[-2]["total_value"] if len(ec_seg) >= 2 else today_total

        today_ts = pd.Timestamp(rb_date)
        # 用调仓日后第一个交易日的收盘价作为当前价
        next_dates = raw_df[raw_df["日期"] > today_ts]["日期"].unique()
        price_date = min(next_dates) if len(next_dates) > 0 else today_ts

        today_trades_list = []
        for key, s in all_sequences.items():
            seq_actual = [t for t in s.get("trades", []) if t["date"] == rb_date and t["action"] in ("买入", "卖出")]
            for t in seq_actual:
                t = {**t, "model_key": key, "name": etf_names.get(t["stock"], "")}
                today_trades_list.append(t)
            if seq_actual:
                seq_buys = {t["stock"] for t in seq_actual if t["action"] == "买入"}
                hist_positions = _rebuild_positions(s.get("trades", []), rb_date)
                for sid, sp in hist_positions.items():
                    if sid not in seq_buys:
                        sub = raw_df[raw_df["股票代码"] == sid]
                        tc_s = sub.loc[sub["日期"] == price_date, "收盘"]
                        price = tc_s.values[0] if not tc_s.empty else 0
                        today_trades_list.append({
                            "action": "保持",
                            "stock": sid,
                            "shares": sp["shares"],
                            "price": round(price, 4),
                            "model_key": key,
                            "name": etf_names.get(sid, ""),
                        })
        model_order = {key: i for i, key in enumerate(all_sequences.keys())}
        today_trades_list.sort(key=lambda x: (model_order.get(x.get("model_key", ""), 999), {"买入": 0, "卖出": 1}.get(x["action"], 2)))

        # 前一次调仓日
        prev_rb_date = None
        idx = rebalance_dates.index(rb_date)
        if idx > 0:
            prev_rb_date = rebalance_dates[idx - 1]

        # 当前持仓 = 调仓前的持仓（日报在收盘后运行，展示调仓前仓位供次日决策）
        positions = _rebuild_positions(trades, prev_rb_date) if prev_rb_date else {}

        # 调仓盈亏：从上次调仓日到本次调仓日的个股盈亏
        prev_close_prices = {}
        if prev_rb_date:
            prev_ts = pd.Timestamp(prev_rb_date)
            for sid in set(t["stock"] for t in today_trades_list):
                sub = raw_df[raw_df["股票代码"] == sid]
                pc_s = sub.loc[sub["日期"] == prev_ts, "收盘"]
                prev_close_prices[sid] = pc_s.values[0] if not pc_s.empty else 0
        for t in today_trades_list:
            if t["action"] == "买入":
                t["reb_pnl"] = None
            else:
                pc = prev_close_prices.get(t["stock"], 0)
                if pc > 0 and t.get("price", 0) > 0:
                    t["reb_pnl_pct"] = round((t["price"] / pc - 1) * 100, 2)
                    t["reb_pnl_amount"] = round(t.get("shares", 0) * (t["price"] - pc), 2)
                else:
                    t["reb_pnl"] = None

        holdings = []
        # 找前一个交易日：优先用 equity curve 的前一天，否则从行情数据取
        if rb_date in sorted_dates and sorted_dates.index(rb_date) > 0:
            yesterday_ts = pd.Timestamp(sorted_dates[sorted_dates.index(rb_date) - 1])
        else:
            prev_dates = raw_df[raw_df["日期"] < today_ts]["日期"].unique()
            yesterday_ts = pd.Timestamp(max(prev_dates)) if len(prev_dates) > 0 else today_ts

        today_buys = {t["stock"] for t in trades if t["date"] == rb_date and t["action"] == "买入"}

        # 每只持仓股的最新调仓价（优先取主序列今日买入/保持价）
        buy_prices = {}
        for t in today_trades_list:
            if t.get("model_key") == model_key and t["action"] in ("买入", "保持") and t.get("price", 0) > 0:
                buy_prices[t["stock"]] = t["price"]
        # 今日未交易的持仓（已被卖出或无操作），从历史交易中取最近一次买入/保持价
        for sid in list(positions.keys()):
            if sid not in buy_prices:
                hist_trades = sorted(
                    [t for t in trades if t["stock"] == sid and t["action"] == "买入" and t["date"] < rb_date],
                    key=lambda x: x["date"], reverse=True
                )
                if hist_trades:
                    buy_prices[sid] = hist_trades[0]["price"]

        for stock_id, p in positions.items():
            sub = raw_df[raw_df["股票代码"] == stock_id]
            tc_s = sub.loc[sub["日期"] == today_ts, "收盘"]
            yc_s = sub.loc[sub["日期"] == yesterday_ts, "收盘"]
            price = tc_s.values[0] if not tc_s.empty else 0
            yc = yc_s.values[0] if not yc_s.empty else price
            if stock_id in today_buys:
                pnl = 0
            else:
                pnl = round(p["shares"] * (price - yc), 2) if price and yc else 0
            holdings.append({
                "stock_id": stock_id,
                "name": etf_names.get(stock_id, ""),
                "price": round(price, 4) if price else 0,
                "buy_price": round(buy_prices.get(stock_id, 0), 4),
                "shares": p["shares"],
                "cost": round(p["cost"], 2),
                "today_pnl": pnl,
            })

        def _compute_metrics(ec_segment, init_cap):
            if len(ec_segment) < 2:
                return {"strategy_return_pct": 0, "total_days": 0, "latest_value": today_total}
            vals = [e["total_value"] for e in ec_segment]
            total_ret = (vals[-1] / init_cap - 1) * 100
            cum = np.array(vals) / init_cap
            daily_rets = np.diff(vals) / np.array(vals[:-1])
            n_days = len(daily_rets)
            win_rate = float(np.mean(daily_rets > 0)) if n_days > 0 else 0
            ann_ret = (1 + total_ret / 100) ** (252 / n_days) - 1 if n_days > 0 else 0
            daily_std = float(np.std(daily_rets)) if n_days > 0 else 0
            ann_vol = daily_std * np.sqrt(252) * 100
            sharpe = float((np.mean(daily_rets) / daily_std) * np.sqrt(252)) if daily_std != 0 else 0
            running_max = np.maximum.accumulate(cum)
            dd = (cum - running_max) / running_max * 100
            max_dd = float(abs(dd.min())) if len(dd) > 0 else 0
            if max_dd > 0:
                dd_end = np.argmin(dd)
                dd_start = np.argmax(running_max[:dd_end + 1])
                mdd_info = {"start_date": ec_segment[int(dd_start)]["date"], "end_date": ec_segment[int(dd_end)]["date"], "duration_days": int(dd_end - dd_start)}
            else:
                mdd_info = {}
            calmar = ann_ret / (max_dd / 100) if max_dd > 0 else 0
            downside = daily_rets[daily_rets < 0]
            ds_std = float(np.std(downside)) if len(downside) > 1 else daily_std
            sortino = float((np.mean(daily_rets) / ds_std) * np.sqrt(252)) if ds_std != 0 else 0
            dd_periods = _extract_drawdowns(vals, [e["date"] for e in ec_segment])
            return {
                "strategy_return_pct": round(total_ret, 4),
                "annualized_return_pct": round(ann_ret * 100, 4),
                "daily_win_rate": round(win_rate, 4),
                "max_drawdown_pct": round(max_dd, 4),
                "max_drawdown_details": mdd_info,
                "drawdown_periods": dd_periods,
                "sharpe_ratio": round(sharpe, 4),
                "calmar_ratio": round(calmar, 4),
                "sortino_ratio": round(sortino, 4),
                "annualized_volatility_pct": round(ann_vol, 4),
                "total_days": n_days,
                "latest_value": round(vals[-1], 2),
            }

        metrics = _compute_metrics(ec_seg, initial_capital)

        # 窗口指标 (5d, 1m)
        ec_by_dict = {e["date"]: e for e in ec_seg}
        ec_sorted = sorted(ec_by_dict.keys())
        today_dt = pd.Timestamp(rb_date)
        for wlabel, wdays in [("5d", 5), ("1m", 30)]:
            wcut = (today_dt - pd.Timedelta(days=wdays)).strftime("%Y-%m-%d")
            wseg = [ec_by_dict[d] for d in ec_sorted if d >= wcut]
            if len(wseg) >= 2:
                base = initial_capital if wseg[0]["date"] == ec_seg[0]["date"] else wseg[0]["total_value"]
                wm = _compute_metrics(wseg, base)
                if wm:
                    metrics[f"window_{wlabel}"] = wm

        # 沪深300收益
        hs300_seg = hs300_raw[(hs300_raw["date"] >= pd.Timestamp(sorted_dates[0])) & (hs300_raw["date"] <= today_ts)]
        if len(hs300_seg) >= 2:
            hs300_ret = (hs300_seg["close"].iloc[-1] / hs300_seg["close"].iloc[0] - 1) * 100
        else:
            hs300_ret = 0.0
        metrics["hs300_return_pct"] = round(hs300_ret, 4)
        metrics["excess_return_pct"] = round(metrics["strategy_return_pct"] - hs300_ret, 4)

        # 下一个调仓日（非末尾直接用列表下一个，末尾用交易日历计算）
        rb_idx = rebalance_dates.index(rb_date)
        if rb_idx + 1 < len(rebalance_dates):
            next_rb = rebalance_dates[rb_idx + 1]
        else:
            next_rb = seq.get("metrics", {}).get("next_rebalance_date", "")
            if not next_rb:
                try:
                    import pandas_market_calendars as mcal
                    xshg = mcal.get_calendar("XSHG")
                    start_ts = pd.Timestamp(rebalance_dates[0])
                    look_end = pd.Timestamp(rb_date) + pd.Timedelta(days=365)
                    cal_dates = xshg.valid_days(start_date=start_ts, end_date=look_end, tz=None)
                    start_pos = cal_dates.get_loc(start_ts.normalize())
                    current_pos = cal_dates.get_loc(pd.Timestamp(rb_date).normalize())
                    n_periods = (current_pos - start_pos) // rebalance_days
                    next_pos = start_pos + (n_periods + 1) * rebalance_days
                    next_rb = cal_dates[next_pos].strftime("%Y-%m-%d") if next_pos < len(cal_dates) else ""
                except Exception:
                    next_rb = ""

        # 构建简化版的报告HTML
        cash = today_total - sum(h["cost"] for h in holdings)

        today_pnl_total = round(sum(h["today_pnl"] for h in holdings), 2)

        today_pnl_positions = [
            {"stock_id": h["stock_id"], "shares": h["shares"], "pnl": h["today_pnl"]}
            for h in holdings
        ]

        # 生成截止到该调仓日的收益曲线（绘制所有序列）
        chart_data_url = _history_chart_b64(all_sequences, hs300_raw, rb_date, sorted_dates[0], initial_capital)

        from send_report import build_report_html, _build_model_stats_table
        hist_sequences = {}
        for hkey, hseq in all_sequences.items():
            hist_trades = [t for t in hseq.get("trades", []) if t["date"] <= rb_date]
            hist_positions = _rebuild_positions(hseq.get("trades", []), rb_date)
            # 当日买入的用当日收盘价（浮盈0%），此前买入的用次日收盘价（避免浮盈0%）
            next_dates = raw_df[raw_df["日期"] > today_ts]["日期"].unique()
            price_date = min(next_dates) if len(next_dates) > 0 else today_ts
            today_buys = {t["stock"] for t in hist_trades if t["date"] == rb_date and t["action"] == "买入"}
            hist_current_prices = {}
            for sid, sp in hist_positions.items():
                sub = raw_df[raw_df["股票代码"] == sid]
                if sid in today_buys:
                    tc_s = sub.loc[sub["日期"] == today_ts, "收盘"]
                else:
                    tc_s = sub.loc[sub["日期"] == price_date, "收盘"]
                if not tc_s.empty:
                    hist_current_prices[sid] = tc_s.values[0]
            hist_ec = [e for e in hseq.get("equity_curve", []) if e["date"] <= rb_date]
            if len(hist_ec) >= 2:
                hist_metrics = _compute_metrics(hist_ec, initial_capital)
                hs300_seg = hs300_raw[(hs300_raw["date"] >= pd.Timestamp(sorted_dates[0])) & (hs300_raw["date"] <= today_ts)]
                hs300_ret = (hs300_seg["close"].iloc[-1] / hs300_seg["close"].iloc[0] - 1) * 100 if len(hs300_seg) >= 2 else 0.0
                hist_metrics["hs300_return_pct"] = round(hs300_ret, 4)
                hist_metrics["excess_return_pct"] = round(hist_metrics["strategy_return_pct"] - hs300_ret, 4)
            else:
                hist_metrics = {"strategy_return_pct": 0, "sharpe_ratio": 0, "max_drawdown_pct": 0, "hs300_return_pct": 0, "excess_return_pct": 0}
            model_stats = _compute_model_stats(hist_trades, hist_current_prices, report_date=rb_date)
            ec_dict = {e["date"]: e["total_value"] for e in hseq.get("equity_curve", [])}
            # 所有调仓期收益率（含当前期），从 equity curve 计算
            reb_period_rets = []
            for i, rd in enumerate(rebalance_dates):
                if rd > rb_date:
                    break
                cur_v = ec_dict.get(rd, 0)
                if i == 0:
                    prev_v = initial_capital
                else:
                    prev_v = ec_dict.get(rebalance_dates[i - 1], 0)
                if prev_v > 0 and cur_v > 0:
                    reb_period_rets.append((cur_v / prev_v - 1) * 100)
            model_stats["reb_pnl_pct"] = round(reb_period_rets[-1], 2) if reb_period_rets else 0.0
            last_3 = reb_period_rets[-3:]
            model_stats["last_3_reb_avg_pct"] = round(sum(last_3) / len(last_3), 2) if last_3 else 0.0
            hist_sequences[hkey] = {
                "model_stats": model_stats,
                "metrics": hist_metrics,
            }
        model_stats_section = _build_model_stats_table(hist_sequences)
        hist_equity = {}
        for hkey, hseq in all_sequences.items():
            ec = hseq.get("equity_curve", [])
            ec_seg = [e for e in ec if e["date"] <= rb_date]
            if ec_seg:
                disp = hkey.replace("search_", "").replace("_exp_", " ")
                hist_equity[disp] = ec_seg
        try:
            html = build_report_html(
                date=rb_date,
                model_display="历史调仓" if not model_key else model_key.replace("search_", "").replace("_exp_", " "),
                total_value=today_total,
                cash=cash,
                holdings=holdings,
                trades_list=today_trades_list,
                metrics=metrics,
                next_rebalance=next_rb,
                is_rebalance=True,
                today_pnl_total=today_pnl_total,
                today_pnl_positions=today_pnl_positions,
                chart_data_url=chart_data_url,
                model_stats_section=model_stats_section,
                equity_data=hist_equity,
            )
            history_path = history_dir / f"{rb_date}.html"
            history_path.write_text(html, encoding="utf-8")
        except Exception as e:
            print(f"  [历史报告] {rb_date} 保存失败: {e}")


def _compute_model_stats(trades, current_prices=None, report_date=None):
    """从trades列表计算模型级交易统计。
    current_prices: {stock: price} 持仓股的当前价格，用于计算浮盈。
    report_date: 报告日期（YYYY-MM-DD），当天买入的浮盈不计入交易统计。"""
    buy_trades = [t for t in trades if t['action'] == '买入']
    sell_trades = [t for t in trades if t['action'] == '卖出']

    sells_by_stock = {}
    for t in sell_trades:
        stock = t['stock']
        d = t['date']
        if stock not in sells_by_stock:
            sells_by_stock[stock] = []
        sells_by_stock[stock].append({'date': d, 'price': t['price']})

    trade_returns = []       # 全部交易（含浮盈）
    completed_returns = []   # 已完成的 buy→sell
    for buy in buy_trades:
        stock = buy['stock']
        buy_date = buy['date']
        buy_price = buy['price']
        sells = sells_by_stock.get(stock, [])
        future_sells = [s for s in sells if s['date'] > buy_date]
        if future_sells:
            future_sells.sort(key=lambda x: x['date'])
            sell = future_sells[0]
            ret = (sell['price'] - buy_price) / buy_price
            trade_returns.append({'return': ret, 'success': ret > 0, 'date': sell['date']})
            completed_returns.append({'return': ret, 'success': ret > 0, 'date': sell['date']})
        elif current_prices and stock in current_prices and current_prices[stock] > 0:
            if report_date is not None and buy_date == report_date:
                continue  # 当日买入的浮盈(0%)不计入统计
            ret = (current_prices[stock] - buy_price) / buy_price
            trade_returns.append({'return': ret, 'success': ret > 0, 'date': buy_date})
        else:
            continue

    if not trade_returns:
        return {
            "total_trades": 0,
            "total_win_rate_pct": 0,
            "total_avg_return_pct": 0,
            "last_trade_return_pct": 0,
            "last_3_avg_return_pct": 0,
            "last_3_win_rate_pct": 0,
        }

    total_trades = len(trade_returns)
    total_wins = sum(1 for t in trade_returns if t['success'])
    total_win_rate_pct = round(total_wins / total_trades * 100, 1)
    total_avg_return_pct = round(sum(t['return'] for t in trade_returns) / total_trades * 100, 2)

    trade_returns.sort(key=lambda x: x['date'])
    last_trade_return_pct = round(trade_returns[-1]['return'] * 100, 2)

    last_3 = trade_returns[-3:]
    last_3_avg_return_pct = round(sum(t['return'] for t in last_3) / len(last_3) * 100, 2)
    last_3_wins = sum(1 for t in last_3 if t['success'])
    last_3_win_rate_pct = round(last_3_wins / len(last_3) * 100, 1)

    return {
        "total_trades": total_trades,
        "total_win_rate_pct": total_win_rate_pct,
        "total_avg_return_pct": total_avg_return_pct,
        "last_trade_return_pct": last_trade_return_pct,
        "last_3_avg_return_pct": last_3_avg_return_pct,
        "last_3_win_rate_pct": last_3_win_rate_pct,
    }


def _make_model_key(m):
    """生成 {search_type}_{model_type}_exp_X 格式的模型标识"""
    if isinstance(m, dict):
        exp_dir = m["exp_dir"]
    elif isinstance(m, (list, tuple)):
        exp_dir = m[0]
    else:
        exp_dir = m
    parent = os.path.basename(os.path.dirname(exp_dir))
    parent = re.sub(r'_\d+_\d+', '', parent)
    name = os.path.basename(exp_dir)
    return f"{parent}_{name}"


def daily_eval(
    config_name: str = "config",
    update_data: bool = True,
    top_k: int = 3,
    verbose: bool = True,
    start_date: str = "2026-04-01",
    rebalance_days: int = 5,
    position_pct: float = 0.95,
    initial_capital: float = 100000,
):
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    try:
        if update_data:
            if verbose:
                print(f"\n{'='*60}")
                print(f"[{timestamp}] 每日测评开始")
                print(f"{'='*60}")
                print("\n[1/4] 获取最新ETF数据...")
            update_etf_data(verbose=verbose)

        raw_df = pd.read_csv(DATA_FILE, dtype={"股票代码": str})
        raw_df["股票代码"] = raw_df["股票代码"].astype(str).str.zfill(6)
        raw_df["日期"] = pd.to_datetime(raw_df["日期"])
        latest_date = raw_df["日期"].max()
        latest_date_str = latest_date.strftime("%Y-%m-%d")
        end_date = (latest_date + pd.Timedelta(days=1)).strftime("%Y-%m-%d")

        if verbose:
            print(f"\n[数据] 最新交易日: {latest_date_str}")

        if verbose:
            print(f"\n[2/4] 加载模型...")

        single_models = []
        average_enabled = False
        voting_enabled = False
        if os.path.exists(str(MODEL_SELECTION_PATH)):
            single_models, master, average_enabled, voting_enabled = load_model_selection(str(MODEL_SELECTION_PATH))
            enabled_models = [m for m in single_models if m.get("enabled", True)]
            if verbose:
                print(f"  平均: {'开' if average_enabled else '关'}, 投票: {'开' if voting_enabled else '关'}, 主序列: {master or 'auto'}, 模型数: {len(enabled_models)}")
                for m in single_models:
                    status = "启用" if m.get("enabled", True) else "禁用"
                    print(f"    {status}: {_make_model_key(m)} ({m['model_file']})")
            single_models = enabled_models
        else:
            config_module = __import__(config_name, fromlist=["config"])
            config = config_module.config.copy()
            output_dir = config.get("output_dir", "./model/default")
            model_info = find_best_model(output_dir)
            if not model_info:
                print("错误: 未找到可用模型")
                return
            exp_dir, model_file, score = model_info
            single_models = [{"exp_dir": exp_dir, "model_file": model_file, "enabled": True}]
            master = ""
            average_enabled = False
            voting_enabled = False
            if verbose:
                print(f"  单模型: {_make_model_key((exp_dir,))} (score={score:.4f})")

        if not single_models:
            print("错误: 无可用模型")
            return

        first_model = single_models[0]
        scaler_path = os.path.join(first_model["exp_dir"], "scaler.pkl")
        config_path = os.path.join(first_model["exp_dir"], "config.json")
        with open(config_path, "r") as f:
            first_config = json.load(f)
        feature_num = first_config["feature_num"]

        if verbose:
            print(f"\n[3/4] 加载并缓存数据...")
        cached_data, cached_features = ETFBacktester.load_data_once(
            data_path=str(DATA_FILE),
            scaler_path=scaler_path,
            feature_num=feature_num,
            verbose=False,
        )

        single_backtesters = []
        for m in single_models:
            bt = ETFBacktester.from_cached_data(
                model_dir=m["exp_dir"],
                cached_data=cached_data,
                cached_features=cached_features,
                device=device,
                model_file=m["model_file"],
                verbose=False,
            )
            single_backtesters.append((m, bt))

        if verbose:
            print(f"\n[4/4] 运行回测序列...")

        sequences = {}

        for m, bt in single_backtesters:
            model_key = _make_model_key(m)
            if verbose:
                print(f"  回测单模型: {model_key}...")
            pred_func = lambda date, _bt=bt: _bt._get_predictions(date)
            result = run_backtest_sequence(
                predictions_func=pred_func,
                data_file=str(DATA_FILE),
                start_date=start_date,
                end_date=end_date,
                top_k=top_k,
                rebalance_days=rebalance_days,
                position_pct=position_pct,
                initial_capital=initial_capital,
            )
            sequences[model_key] = result

        if average_enabled and len(single_backtesters) >= 2:
            if verbose:
                print(f"  回测平均模型 ({len(single_backtesters)}个模型)...")

            def avg_pred_func(date):
                all_score_dicts = []
                for _, bt in single_backtesters:
                    preds = bt._get_predictions(date)
                    if preds is None:
                        return None
                    all_score_dicts.append({p["stock_id"]: p["score"] for p in preds})
                avg_scores = {}
                for stock_id in all_score_dicts[0].keys():
                    scores = [sd[stock_id] for sd in all_score_dicts]
                    avg_scores[stock_id] = float(np.mean(scores))
                sorted_stocks = sorted(avg_scores.items(), key=lambda x: x[1], reverse=True)
                return [{"rank": i+1, "stock_id": sid, "score": sc} for i, (sid, sc) in enumerate(sorted_stocks)]

            result = run_backtest_sequence(
                predictions_func=avg_pred_func,
                data_file=str(DATA_FILE),
                start_date=start_date,
                end_date=end_date,
                top_k=top_k,
                rebalance_days=rebalance_days,
                position_pct=position_pct,
                initial_capital=initial_capital,
            )
            sequences["average"] = result

        if voting_enabled and len(single_backtesters) >= 2:
            if verbose:
                print(f"  回测投票模型 ({len(single_backtesters)}个模型)...")

            first_model_key = _make_model_key(single_backtesters[0][0])
            voting_pred_cache = {}

            def voting_pred_func(date):
                nonlocal voting_pred_cache
                all_preds = []
                for _, bt in single_backtesters:
                    preds = bt._get_predictions(date)
                    if preds is None:
                        return None
                    all_preds.append(preds)
                freq = {}
                avg_score = {}
                for preds in all_preds:
                    for i, p in enumerate(preds):
                        if i >= top_k:
                            break
                        sid = p["stock_id"]
                        freq[sid] = freq.get(sid, 0) + 1
                        avg_score[sid] = avg_score.get(sid, 0) + p["score"]
                for sid in avg_score:
                    avg_score[sid] /= freq.get(sid, 1)
                ranked = sorted(freq.items(), key=lambda x: (-x[1], -avg_score.get(x[0], 0)))
                result = [{"rank": i+1, "stock_id": sid, "score": float(freq)} for i, (sid, freq) in enumerate(ranked)]
                voting_pred_cache[date] = {"ranked": ranked, "top_k": top_k}
                if len(result) < top_k * 2:
                    first_picks = [p["stock_id"] for p in all_preds[0]]
                    existing = {r["stock_id"] for r in result}
                    for sid in first_picks:
                        if sid not in existing:
                            result.append({"rank": len(result)+1, "stock_id": sid, "score": 0.0})
                            existing.add(sid)
                        if len(result) >= top_k * 2:
                            break
                return result

            result = run_backtest_sequence(
                predictions_func=voting_pred_func,
                data_file=str(DATA_FILE),
                start_date=start_date,
                end_date=end_date,
                top_k=top_k,
                rebalance_days=rebalance_days,
                position_pct=position_pct,
                initial_capital=initial_capital,
            )
            for t in result.get("trades", []):
                if t.get("action") == "买入" and t["date"] in voting_pred_cache:
                    cache = voting_pred_cache[t["date"]]
                    full_ranked = cache["ranked"]
                    k = cache["top_k"]
                    cutoff_votes = full_ranked[k][1] if len(full_ranked) > k else 0
                    stock_votes = next((f for sid, f in full_ranked if sid == t["stock"]), 0)
                    t["score"] = float(stock_votes)
                    t["advantage"] = int(stock_votes - cutoff_votes)
            sequences["voting"] = result


        # 绘制收益曲线图
        plot_path = OUTPUT_DIR / "equity_curves.png"
        try:
            plot_equity_curves(sequences, str(DATA_FILE), initial_capital, str(plot_path))
            if verbose:
                print(f"\n  [图表] 收益曲线已保存: {plot_path}")
        except Exception as e:
            if verbose:
                print(f"\n  [图表] 保存失败: {e}")

        if master == "first":
            report_key = list(sequences.keys())[0]
        else:
            for fallback in ["average", "voting", list(sequences.keys())[0]]:
                if fallback in sequences:
                    report_key = fallback
                    break
            else:
                report_key = list(sequences.keys())[0]
        report_data = sequences[report_key]

        # 当前价格（从今日盈亏中取）
        pnl_positions = {p["stock_id"]: p for p in report_data.get("today_pnl", {}).get("positions", [])}

        today_actual_trades = [t for t in report_data["trades"] if t["date"] == latest_date_str and t["action"] in ("买入", "卖出")]
        is_rebalance_day = len(today_actual_trades) > 0

        today_trades = list(today_actual_trades)
        if is_rebalance_day:
            today_report_buys = {t["stock"] for t in today_actual_trades if t["action"] == "买入"}
            for sid, pos in report_data["positions"].items():
                if sid not in today_report_buys:
                    price = pnl_positions.get(sid, {}).get("today_close", 0)
                    today_trades.append({
                        "action": "保持",
                        "stock": sid,
                        "shares": pos["shares"],
                        "price": price,
                    })

        # 收集所有序列的今日调仓（含保持）
        all_today_trades = []
        latest_date_ts = pd.Timestamp(latest_date_str)
        # 用调仓日后第一个交易日的收盘价作为保持价格
        next_dates = raw_df[raw_df["日期"] > latest_date_ts]["日期"].unique()
        hold_price_date = min(next_dates) if len(next_dates) > 0 else latest_date_ts

        for key, seq in sequences.items():
            seq_actual_trades = [t for t in seq.get("trades", []) if t["date"] == latest_date_str and t["action"] in ("买入", "卖出")]
            for t in seq_actual_trades:
                t["model_key"] = key
                all_today_trades.append(t)
            if seq_actual_trades:
                seq_today_buys = {t["stock"] for t in seq_actual_trades if t["action"] == "买入"}
                for sid, pos in seq.get("positions", {}).items():
                    if sid not in seq_today_buys:
                        sub = raw_df[raw_df["股票代码"] == sid]
                        tc_s = sub.loc[sub["日期"] == hold_price_date, "收盘"]
                        price = round(tc_s.values[0], 4) if not tc_s.empty else 0
                        all_today_trades.append({
                            "action": "保持",
                            "stock": sid,
                            "shares": pos["shares"],
                            "price": price,
                            "model_key": key,
                        })

        # 持久化保存最新调仓价（仅取主序列的调仓价格）
        last_trade_prices = {}
        for t in all_today_trades:
            if t.get("model_key") == report_key and t.get("price", 0) > 0:
                last_trade_prices[t["stock"]] = t["price"]
        # 非调仓日从最近一次调仓日推算保持价格
        if not last_trade_prices and STATE_PATH.exists():
            try:
                prev_state = json.loads(STATE_PATH.read_text(encoding="utf-8"))
                for seq in prev_state.get("sequences", {}).values():
                    ltp = seq.get("metrics", {}).get("last_trade_prices", {})
                    if ltp:
                        last_trade_prices = ltp
                        break
            except Exception:
                pass
        # 兜底：从最近一次调仓日推算（买入取调仓价，保持取下个交易日收盘）
        if not last_trade_prices:
            rb_dates = sorted(set(t["date"] for t in report_data["trades"]), reverse=True)
            if rb_dates:
                last_rb = rb_dates[0]
                last_rb_ts = pd.Timestamp(last_rb)
                rb_buys = {}
                for t in report_data["trades"]:
                    if t["date"] == last_rb and t["action"] == "买入" and t.get("price", 0) > 0:
                        rb_buys[t["stock"]] = t["price"]
                hold_dates = raw_df[raw_df["日期"] > last_rb_ts]["日期"].unique()
                hold_date = min(hold_dates) if len(hold_dates) > 0 else last_rb_ts
                for sid in report_data.get("positions", {}):
                    if sid in rb_buys:
                        last_trade_prices[sid] = rb_buys[sid]
                    else:
                        sub = raw_df[raw_df["股票代码"] == sid]
                        tc_s = sub.loc[sub["日期"] == hold_date, "收盘"]
                        price = tc_s.values[0] if not tc_s.empty else 0
                        if price > 0:
                            last_trade_prices[sid] = round(price, 4)
        report_data["metrics"]["last_trade_prices"] = last_trade_prices

        # 重写 state（此时 latest_trade_prices 已写入 metrics）
        state = {
            "sequences": sequences,
            "last_updated": timestamp,
        }
        os.makedirs(OUTPUT_DIR, exist_ok=True)
        with open(STATE_PATH, "w", encoding="utf-8") as f:
            json.dump(state, f, ensure_ascii=False, indent=2, cls=NumpyEncoder)

        # 调仓盈亏：从上次调仓日到本次调仓日的个股盈亏
        trade_dates = sorted(set(t["date"] for t in report_data["trades"]))
        prev_rb_date = None
        if latest_date_str in trade_dates:
            idx = trade_dates.index(latest_date_str)
            if idx > 0:
                prev_rb_date = trade_dates[idx - 1]
        if prev_rb_date:
            m_raw = pd.read_csv(DATA_FILE, dtype={"股票代码": str})
            m_raw["股票代码"] = m_raw["股票代码"].astype(str).str.zfill(6)
            m_raw["日期"] = pd.to_datetime(m_raw["日期"])
            prev_ts = pd.Timestamp(prev_rb_date)
            all_stocks = set(t["stock"] for t in today_trades + all_today_trades)
            prev_close_prices = {}
            for sid in all_stocks:
                sub = m_raw[m_raw["股票代码"] == sid]
                pc_s = sub.loc[sub["日期"] == prev_ts, "收盘"]
                prev_close_prices[sid] = pc_s.values[0] if not pc_s.empty else 0
            for t in today_trades + all_today_trades:
                if t["action"] == "买入":
                    t["reb_pnl"] = None
                else:
                    pc = prev_close_prices.get(t["stock"], 0)
                    if pc > 0 and t.get("price", 0) > 0:
                        t["reb_pnl_pct"] = round((t["price"] / pc - 1) * 100, 2)
                        t["reb_pnl_amount"] = round(t.get("shares", 0) * (t["price"] - pc), 2)
                    else:
                        t["reb_pnl"] = None

        # 加载ETF名称映射
        etf_names = {}
        etf_list_path = PROJECT_ROOT / "etf_data" / "etf_list_before_2022_74.csv"
        if etf_list_path.exists():
            import csv
            with open(etf_list_path, "r", encoding="utf-8-sig") as f:
                for row in csv.DictReader(f):
                    code = row.get("代码", "").strip()
                    name = row.get("名称", "").strip()
                    if code and name:
                        etf_names[code] = name

        holdings = []
        # 从持久化的调仓价取成交价，非调仓日加载上次保存的价格
        latest_trade_prices = report_data.get("metrics", {}).get("last_trade_prices", {})
        if not latest_trade_prices and STATE_PATH.exists():
            try:
                prev_state = json.loads(STATE_PATH.read_text(encoding="utf-8"))
                for seq in prev_state.get("sequences", {}).values():
                    ltp = seq.get("metrics", {}).get("last_trade_prices", {})
                    if ltp:
                        latest_trade_prices = ltp
                        break
            except Exception:
                pass
        for stock_id, pos in report_data["positions"].items():
            price = pnl_positions.get(stock_id, {}).get("today_close", 0)
            holdings.append({
                "stock_id": stock_id,
                "name": etf_names.get(stock_id, ""),
                "price": price,
                "buy_price": round(latest_trade_prices.get(stock_id, 0), 4),
                "shares": pos["shares"],
                "cost": pos["cost"],
            })

        for t in today_trades:
            t["name"] = etf_names.get(t["stock"], "")

        for t in all_today_trades:
            if "name" not in t:
                t["name"] = etf_names.get(t["stock"], "")

        # 收集所有序列的信息
        sequences_summary = {}
        for key, seq in sequences.items():
            seq_pnl = {p["stock_id"]: p for p in seq.get("today_pnl", {}).get("positions", [])}
            seq_current_prices = {sid: p["today_close"] for sid, p in seq_pnl.items() if p.get("today_close", 0) > 0}
            model_stats = _compute_model_stats(seq["trades"], seq_current_prices, report_date=latest_date_str)
            seq["model_stats"] = model_stats
            sequences_summary[key] = {
                "metrics": seq["metrics"],
                "cash": seq["cash"],
                "positions_count": len(seq["positions"]),
                "trades_count": len(seq["trades"]),
                "trades": seq["trades"],
                "today_pnl": seq.get("today_pnl", {}),
                "model_stats": model_stats,
                "equity_curve": seq.get("equity_curve", []),
            }

        next_rebalance = report_data["metrics"].get("next_rebalance_date", "")

        report = {
            "date": latest_date_str,
            "is_rebalance_day": is_rebalance_day,
            "next_rebalance_date": next_rebalance,
            "today_trades": today_trades,
            "all_today_trades": all_today_trades,
            "metrics": report_data["metrics"],
            "holdings": holdings,
            "cash": report_data["cash"],
            "total_value": report_data["metrics"]["latest_value"],
            "sequences": sequences_summary,
        }

        # 计算持仓变动（相对上一次调仓）
        with open(REPORT_PATH, "w", encoding="utf-8") as f:
            json.dump(report, f, ensure_ascii=False, indent=2, cls=NumpyEncoder)

        portfolio = {
            "last_updated": timestamp,
            "predict_date": latest_date_str,
            "holdings": holdings,
            "cash": report_data["cash"],
            "total_value": report_data["metrics"]["latest_value"],
        }
        with open(PORTFOLIO_PATH, "w", encoding="utf-8") as f:
            json.dump(portfolio, f, ensure_ascii=False, indent=2, cls=NumpyEncoder)

        # 发送邮件报告
        try:
            from send_report import send_report
            if master == "first":
                email_key = list(sequences.keys())[0]
            else:
                for fallback in ["average", "voting", list(sequences.keys())[0]]:
                    if fallback in sequences:
                        email_key = fallback
                        break
                else:
                    email_key = list(sequences.keys())[0]
            if "holdings" not in sequences[email_key]:
                sequences[email_key]["holdings"] = holdings
            if verbose:
                print(f"\n[邮件] 正在发送报告: {email_key}...")
            send_report(model_key=email_key)
        except Exception as e:
            if verbose:
                print(f"\n[邮件] 发送失败: {e}")

        # 生成历史调仓日报告（从回测数据重建）
        try:
            if verbose:
                print(f"\n[历史] 生成各调仓日报告...")
            _save_history_reports(
                sequences[report_key], sequences, str(DATA_FILE), initial_capital, etf_names, model_key=report_key, rebalance_days=rebalance_days
            )
        except Exception as e:
            if verbose:
                print(f"\n[历史] 生成失败: {e}")

        for fn in ["equity_curve.csv", "daily_metrics.json", "trades_log.csv"]:
            fp = OUTPUT_DIR / fn
            if fp.exists():
                fp.unlink()

        if verbose:
            m = report_data["metrics"]
            today_pnl_data = report_data.get("today_pnl", {})
            pnl_total = today_pnl_data.get("total_pnl", 0)
            print(f"\n{'='*60}")
            print(f"  每日报告 ({latest_date_str}) [序列: {report_key}]")
            print(f"{'='*60}")
            print(f"  今日盈亏: {pnl_total:+.2f}")
            print(f"  累计收益: {m['strategy_return_pct']:+.2f}%")
            print(f"  年化收益: {m.get('annualized_return_pct', 0):+.2f}%")
            print(f"  日胜率:   {m.get('daily_win_rate', 0)*100:.1f}%")
            print(f"  沪深300:  {m['hs300_return_pct']:+.2f}%")
            print(f"  超额收益: {m['excess_return_pct']:+.2f}%")
            print(f"  最大回撤: {m['max_drawdown_pct']:.2f}%")
            print(f"  夏普:     {m['sharpe_ratio']:.2f}")
            print(f"  卡玛:     {m.get('calmar_ratio', 0):.2f}")
            print(f"  索提诺:   {m.get('sortino_ratio', 0):.2f}")
            print(f"  账户总值: {m['latest_value']:,.2f}")
            print(f"  调仓日:   {'是' if is_rebalance_day else '否'}")
            print(f"  下个调仓: {m.get('next_rebalance_date', '')}")
            print(f"{'='*60}")

            print(f"\n  各序列收益:")
            for key, seq in sequences.items():
                sr = seq["metrics"]["strategy_return_pct"]
                print(f"    {key}: {sr:+.2f}%")

            if all_today_trades:
                print(f"\n  今日调仓 (全部序列):")
                for t in all_today_trades:
                    print(f"    [{t.get('model_key', '?')}] {t['action']} {t['stock']} x {t['shares']}股 @ {t['price']:.4f}")

            print(f"\n  当前持仓:")
            for h in holdings:
                price_str = f" @ {h['price']:.4f}" if h.get("price") else ""
                print(f"    {h['stock_id']}: {h['shares']}股{price_str} (成本: {h['cost']:.2f})")

            if today_pnl_data.get("positions"):
                print(f"\n  今日持仓盈亏:")
                for pos in today_pnl_data["positions"]:
                    print(f"    {pos['stock_id']}: {pos['pnl']:+.2f} ({pos['pnl_pct']:+.2f}%)")

            print(f"\n{'='*60}")
            print(f"[{timestamp}] 每日测评完成")
            print(f"{'='*60}")

        return state

    except Exception as e:
        print(f"\n[错误] 每日测评失败: {e}")
        traceback.print_exc()
        return None


if __name__ == "__main__":
    import argparse
    import multiprocessing as mp

    parser = argparse.ArgumentParser(description="每日定时测评")
    parser.add_argument("--start-date", type=str, default="2026-04-01", help="测评起始日期")
    parser.add_argument("--config", type=str, default="config", help="配置模块名")
    parser.add_argument("--no-update", action="store_true", help="跳过数据更新")
    parser.add_argument("--topk", type=int, default=3, help="Top-K推荐数量")
    parser.add_argument("--rebalance-days", type=int, default=5, help="调仓频率(天)")
    parser.add_argument("--position-pct", type=float, default=0.95, help="仓位比例")
    parser.add_argument("--quiet", action="store_true", help="静默模式")
    args = parser.parse_args()

    mp.set_start_method("spawn", force=True)

    daily_eval(
        config_name=args.config,
        update_data=not args.no_update,
        top_k=args.topk,
        verbose=not args.quiet,
        start_date=args.start_date,
        rebalance_days=args.rebalance_days,
        position_pct=args.position_pct,
    )
