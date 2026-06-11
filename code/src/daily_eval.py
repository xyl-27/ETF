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
import time
import traceback
import subprocess
from datetime import datetime
from typing import Optional, Dict, List, Any, Tuple, Callable
from pathlib import Path

import torch
import numpy as np
import pandas as pd


# 项目根目录
PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
os.chdir(PROJECT_ROOT)
sys.path.insert(0, str(PROJECT_ROOT / "code" / "src"))

from backtest import BacktestEngine, ETFBacktester, compute_volatility
from ml_backtester import MLBacktester
from metrics import NumpyEncoder, compute_window_metrics, extract_drawdowns, compute_longterm_risk_metrics

# GM Python 路径（用于数据下载，该 Python 装有 gm SDK）
# 可通过环境变量 GM_PYTHON 覆盖，默认 D:\opt\python3.12.4\python.exe
_GM_PYTHON_DEFAULT = "D:\\opt\\python3.12.4\\python.exe"
GM_PYTHON = os.environ.get("GM_PYTHON", _GM_PYTHON_DEFAULT)
# WSL 路径转换
if os.name == "posix" and ":" in GM_PYTHON:
    drive = GM_PYTHON[0].lower()
    rest = GM_PYTHON[2:].replace("\\", "/")
    GM_PYTHON = f"/mnt/{drive}{rest}"
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
    raw_df = load_etf_data(data_file, dtype={"股票代码": str})
    raw_df["股票代码"] = raw_df["股票代码"].astype(object).str.zfill(6)
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
JUEJIN_STATE_PATH = OUTPUT_DIR / "juejin_state.json"
REPORT_PATH = OUTPUT_DIR / "latest_report.json"
PORTFOLIO_PATH = OUTPUT_DIR / "portfolio.json"
PREDICTIONS_PATH = OUTPUT_DIR / "predictions.json"
MODEL_SELECTION_PATH = OUTPUT_DIR / "model_selection.yaml"
CONFIG_PATH = PROJECT_ROOT / "config.yaml"
DATA_FILE = PROJECT_ROOT / "etf_data" / "etf_74.csv"


_raw_df_cache = None
_raw_df_cache_key = None

def load_etf_data(path=None, dtype=None):
    global _raw_df_cache, _raw_df_cache_key
    p = path or DATA_FILE
    key = (str(p), str(dtype))
    if _raw_df_cache is not None and _raw_df_cache_key == key:
        return _raw_df_cache.copy()
    df = pd.read_csv(p, dtype=dtype)
    _raw_df_cache = df
    _raw_df_cache_key = key
    return df


def _build_pivots(raw_df):
    close_pivot = raw_df.pivot_table(index="日期", columns="股票代码", values="收盘")
    hl_pivot = raw_df.pivot_table(index="日期", columns="股票代码", values="涨停价")
    ll_pivot = raw_df.pivot_table(index="日期", columns="股票代码", values="跌停价")
    return close_pivot, hl_pivot, ll_pivot


# ============================================================
# 数据更新
# ============================================================

def _check_data_integrity(verbose: bool = True) -> bool:
    """检查每只股票数据是否足够（>= seq_length 天），不足则触发全量下载。"""
    data_file = PROJECT_ROOT / "etf_data" / "etf_74.csv"
    if not data_file.exists():
        return True  # 文件不存在，后续流程会处理

    try:
        df = pd.read_csv(data_file)
        df["股票代码"] = df["股票代码"].astype(object).str.zfill(6)

        # 从模型 config 读取 seq_length
        cfg = load_full_config()
        models = cfg.get("models", [])
        if not models:
            return True
        first_exp = models[0]["dir"]
        config_path = PROJECT_ROOT / first_exp / "config.json"
        if not config_path.exists():
            return True
        with open(config_path) as f:
            model_cfg = json.load(f)
        seq_length = int(model_cfg.get("sequence_length", 60))

        missing = []
        for sid in sorted(df["股票代码"].unique()):
            sub = df[df["股票代码"] == sid]
            if len(sub) < seq_length:
                missing.append((sid, len(sub), sub["日期"].min(), sub["日期"].max()))

        if not missing:
            return True

        if verbose:
            print(f"\n[数据完整性] 发现 {len(missing)} 只股票数据不足（需要 >= {seq_length} 天）:")
            for sid, n, mn, mx in missing[:10]:
                print(f"  {sid}: 仅 {n} 天 ({mn} ~ {mx})")
            if len(missing) > 10:
                print(f"  ... 共 {len(missing)} 只")

        # 触发全量下载
        script_path = str(PROJECT_ROOT / "juejin" / "download_etf_data.py")
        gm_python = GM_PYTHON
        if not os.path.exists(script_path) or not os.path.exists(gm_python):
            print("[数据完整性] 无法执行全量下载（脚本或 GM Python 不可用）")
            return False

        print(f"[数据完整性] 触发全量下载...")
        cmd = [gm_python, script_path, "--start-date", "2022-01-01"]
        result = subprocess.run(cmd, cwd=str(PROJECT_ROOT), capture_output=True, timeout=600)
        stdout = result.stdout.decode("utf-8", errors="replace")
        stderr = result.stderr.decode("utf-8", errors="replace")
        if verbose:
            if stdout.strip():
                print(stdout.strip())
            if stderr.strip():
                print(stderr.strip())

        has_fail = "FAIL" in stdout or result.returncode != 0
        if has_fail:
            fail_lines = [l for l in stdout.split("\n") if "FAIL" in l]
            print(f"[数据完整性] 全量下载完成，{len(fail_lines)} 只失败")
            return False

        print(f"[数据完整性] 全量下载成功")
        return True
    except Exception as e:
        if verbose:
            print(f"[数据完整性] 检查异常: {e}")
        return False


def update_etf_data(verbose: bool = True) -> bool:
    script_path = str(PROJECT_ROOT / "juejin" / "download_etf_data.py")
    if not os.path.exists(script_path):
        print("[数据更新] 未找到 juejin/download_etf_data.py，跳过")
        return False

    gm_python = GM_PYTHON
    if not os.path.exists(gm_python):
        print(f"[数据更新] GM Python 不可用: {gm_python}")
        print("[数据更新] 请设置 GM_PYTHON 环境变量指向带有 gm SDK 的 Python 路径")
        return False

    data_file = PROJECT_ROOT / "etf_data" / "etf_74.csv"
    if data_file.exists():
        cmd = [gm_python, script_path, "--update"]
        print("[数据更新] 运行 download_etf_data.py --update 增量更新...")
    else:
        cmd = [gm_python, script_path, "--start-date", "2022-01-01"]
        print("[数据更新] 运行 download_etf_data.py 全量下载 (2022-01-01 起)...")

    try:
        result = subprocess.run(
            cmd,
            cwd=str(PROJECT_ROOT),
            capture_output=True,
            timeout=600,
        )
        stdout = result.stdout.decode("utf-8", errors="replace")
        stderr = result.stderr.decode("utf-8", errors="replace")
        if verbose:
            if stdout.strip():
                print(stdout.strip())
            if stderr.strip():
                print(stderr.strip())

        has_fail = "FAIL" in stdout or "调用失败" in stderr or result.returncode != 0
        if has_fail:
            print(f"[数据更新] 部分ETF下载失败（未登录掘金终端？）exit code: {result.returncode}")
            fail_lines = [l for l in stdout.split("\n") if "FAIL" in l]
            if fail_lines:
                print(f"[数据更新] 失败: {len(fail_lines)} 只ETF")
                for fl in fail_lines[:5]:
                    print(f"  {fl.strip()}")
                if len(fail_lines) > 5:
                    print(f"  ... 共 {len(fail_lines)} 条失败")
            return False
        else:
            print("[数据更新] ETF数据获取成功")
            if data_file.exists():
                try:
                    tmp = pd.read_csv(data_file)
                    last_date = pd.to_datetime(tmp["日期"]).max().strftime("%Y-%m-%d")
                    print(f"[数据更新] 最新日期: {last_date}")
                except Exception:
                    pass
            return True
    except subprocess.TimeoutExpired:
        print("[数据更新] 超时 (10分钟)，跳过")
        return False
    except Exception as e:
        print(f"[数据更新] 异常: {e}")
        return False


# ============================================================
# 模型加载
# ============================================================

def load_model_selection(path: str = None, cfg_dict: dict = None) -> Tuple[List[Dict], str, bool, bool]:
    """从 config.yaml 或旧 model_selection.yaml 加载模型选择。

    优先使用 cfg_dict（来自 load_full_config），否则读取 path。
    """
    if cfg_dict:
        data = cfg_dict
    else:
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
                "type": m.get("type", "dl"),
                "enabled": enabled,
                "name": m.get("name", ""),
            })
    return models, master, average_enabled, voting_enabled


def _format_strategy_info(weight_strategy, strategy_params, top_k, position_pct, commission, slippage, rebalance_days):
    sp = strategy_params or {}
    extra = ""
    if weight_strategy == "softmax":
        extra = f" T={sp.get('temperature', 1.0)}"
    elif weight_strategy in ("risk_parity", "score_risk", "score_risk_v1", "kelly"):
        extra = f" w={sp.get('vol_window', 20)}"
    sname = {"equal": "等权", "softmax": "Softmax", "rank_linear": "线性排名",
             "risk_parity": "风险平价", "score_risk": "评分风险",
             "score_risk_v1": "评分风险V1",
             "kelly": "Kelly", "liquidity": "流动性优先"}.get(weight_strategy, weight_strategy)
    return (
        f"策略: {sname}{extra}"
        f" | Top-K: {top_k}"
        f" | 仓位: {position_pct:.0%}"
        f" | 费率: {commission*100:.2f}%"
        f" | 滑点: {slippage*100:.1f}%"
        f" | 调仓: {rebalance_days}天"
    )


def load_full_config(config_path=None) -> dict:
    """读取 config.yaml，返回完整配置字典（含默认值）。"""
    defaults = {
        "mode": "full",
        "update_data": True,
        "start_date": "2026-04-01",
        "rebalance_days": 5,
        "trade_mode": "open",
        "initial_capital": 100000,
        "position_pct": 0.95,
        "commission": 0.0003,
        "slippage": 0.001,
        "top_k": 3,
        "weight_strategy": "equal",
        "strategy_params": {},
        "models": [],
        "average": True,
        "voting": True,
        "master": "first",
        "juejin": {},
        "risk_manager": {"enabled": False, "strategy": "none", "params": {}},
    }
    path = Path(config_path) if config_path else CONFIG_PATH
    if not path.exists():
        return defaults
    try:
        import yaml
        with open(path, "r", encoding="utf-8") as f:
            data = yaml.safe_load(f) or {}
        for k, v in data.items():
            if k in defaults and v is not None:
                defaults[k] = v
    except Exception as e:
        print(f"[配置] 读取 {path} 失败: {e}")
    return defaults


def find_best_model(output_dir: str) -> Optional[Tuple[str, str, float]]:
    search_results_path = os.path.join(output_dir, "search_results.json")
    if not os.path.exists(search_results_path):
        return None
    with open(search_results_path, "r") as f:
        results = json.load(f)
    if not results:
        return None
    best = max(results, key=lambda x: x.get("sharpe", x.get("score", 0)))
    exp_idx = best["exp_idx"]
    exp_dir = os.path.join(output_dir, f"exp_{exp_idx}")
    if not os.path.exists(exp_dir):
        return None
    model_file = "best_model_sliding.pth"
    if not os.path.exists(os.path.join(exp_dir, model_file)):
        model_file = "best_model.pth"
        if not os.path.exists(os.path.join(exp_dir, model_file)):
            return None
    return exp_dir, model_file, best.get("sharpe", best.get("score", 0))


# ============================================================
# 回测执行
# ============================================================


def run_backtest_sequence(
    data_file: str,
    start_date: str,
    end_date: str,
    predictions_func: Callable,
    top_k: int,
    rebalance_days: int,
    position_pct: float,
    weight_strategy: str = "equal",
    strategy_params: dict = None,
    initial_capital: float = 1000000,
    commission: float = 0.0003,
    slippage: float = 0.001,
    trade_mode: str = "open",
    risk_manager_config: dict = None,
) -> Dict[str, Any]:
    raw_df = load_etf_data(data_file, dtype={"股票代码": str})
    raw_df["股票代码"] = raw_df["股票代码"].astype(object).str.zfill(6)
    raw_df["日期"] = pd.to_datetime(raw_df["日期"])
    price_data = raw_df.copy()
    close_pivot = raw_df.pivot_table(index="日期", columns="股票代码", values="收盘")

    start_ts = pd.Timestamp(start_date)
    end_ts = pd.Timestamp(end_date)
    all_dates = sorted(raw_df["日期"].unique())
    backtest_dates = [d for d in all_dates if start_ts <= d < end_ts]

    engine = BacktestEngine(
        initial_capital=initial_capital,
        commission=commission,
        slippage=slippage,
        top_k=top_k,
        position_pct=position_pct,
        weight_strategy=weight_strategy,
        strategy_params=strategy_params,
        log=False,
        risk_manager_config=risk_manager_config,
    )

    engine.run(
        dates=backtest_dates,
        price_data=price_data,
        predictions_func=predictions_func,
        rebalance_days=rebalance_days,
        first_rebalance_date=start_ts,
        trade_mode=trade_mode,
    )

    equity_curve = []
    for ec in engine.equity_curve:
        entry = {
            "date": ec["date"].strftime("%Y-%m-%d") if isinstance(ec["date"], pd.Timestamp) else str(ec["date"]),
            "total_value": round(ec["total_value"], 2),
        }
        if "risk_multiplier" in ec:
            entry["risk_multiplier"] = ec["risk_multiplier"]
        if "stock_exposure" in ec:
            entry["stock_exposure"] = ec["stock_exposure"]
        equity_curve.append(entry)

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
        if "trade_cost" in t:
            trade["trade_cost"] = t["trade_cost"]
        trades.append(trade)

    positions = {}
    for stock_id, pos in engine.positions.items():
        positions[stock_id] = {
            "shares": pos["shares"],
            "cost": round(pos["cost"], 2),
        }

    # -- 整体指标 --
    overall = compute_window_metrics(equity_curve, initial_capital)
    if not overall:
        return {
            "equity_curve": equity_curve,
            "trades": trades,
            "positions": positions,
            "metrics": {},
            "cash": round(engine.cash, 2),
            "skipped_trades": engine.skipped_trades,
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
            win_metrics = compute_window_metrics(seg, seg[0]["total_value"])
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
            if stock_id not in close_pivot.columns:
                continue
            tc = close_pivot.loc[today_ts, stock_id]
            yc = close_pivot.loc[yesterday_ts, stock_id]
            if pd.notna(tc) and pd.notna(yc):
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

        # -- NDCG & KS 检验 & Rank IC --
    ndcg_list = []
    mrr_list = []
    ks_stat_list = []
    ks_p_list = []
    rank_ic_raw = []
    all_dates_sorted = sorted(raw_df["日期"].unique())
    scores_by_rb = {}

    for ph in engine.predictions_history:
        rb_date = ph["date"]
        preds = ph["predictions"]
        stock_ids = [p["stock_id"] for p in preds]
        scores = [p["score"] for p in preds]
        scores_by_rb[rb_date] = (stock_ids, scores)

    active_stock_ids = None
    active_scores = None
    active_rb_str = None
    from scipy.stats import spearmanr, ks_2samp
    for d in all_dates_sorted:
        d_str = d.strftime("%Y-%m-%d")
        if d_str in scores_by_rb:
            active_stock_ids, active_scores = scores_by_rb[d_str]
            active_rb_str = d_str
        if active_scores is None or len(set(active_scores)) < 5:
            continue
        cum_rets = []
        for sid in active_stock_ids:
            if sid not in close_pivot.columns:
                cum_rets.append(0.0)
                continue
            close_rb = close_pivot.loc[pd.Timestamp(active_rb_str), sid]
            close_d = close_pivot.loc[d, sid]
            if pd.notna(close_rb) and pd.notna(close_d):
                cum_rets.append((float(close_d) / float(close_rb) - 1) * 100)
            else:
                cum_rets.append(0.0)

        unique_cum = len(set(cum_rets))
        if unique_cum > 1:
            ic, _ = spearmanr(active_scores, cum_rets)
            if not np.isnan(ic):
                rank_ic_raw.append({"date": d_str, "value": ic})

            k = min(top_k, len(active_scores))
            if k > 0:
                sorted_idx = np.argsort(active_scores)[::-1][:k]
                sorted_cum = [cum_rets[i] for i in sorted_idx]
                dcg = sum(sorted_cum[i] / np.log2(i + 2) for i in range(k))
                ideal_cum = sorted(cum_rets, reverse=True)[:k]
                idcg = sum(ideal_cum[i] / np.log2(i + 2) for i in range(k))
                ndcg_list.append({"date": d_str, "value": dcg / (idcg + 1e-12)})

                mrr_d = sum(sorted_cum[i] / (i + 1) for i in range(k))
                mrr_id = sum(ideal_cum[i] / (i + 1) for i in range(k))
                mrr_list.append({"date": d_str, "value": mrr_d / (mrr_id + 1e-12)})

            median_cum = float(np.median(cum_rets))
            good_scores = [active_scores[i] for i in range(len(active_scores)) if cum_rets[i] > median_cum]
            bad_scores = [active_scores[i] for i in range(len(active_scores)) if cum_rets[i] <= median_cum]
            if len(good_scores) > 1 and len(bad_scores) > 1:
                ks_stat, ks_p = ks_2samp(good_scores, bad_scores)
                ks_stat_list.append({"date": d_str, "value": ks_stat})
                ks_p_list.append({"date": d_str, "value": ks_p})

    def _avg_vals(lst):
        vals = [e["value"] for e in lst]
        return round(float(np.mean(vals)), 4) if vals else None

    avg_ndcg = _avg_vals(ndcg_list)
    avg_mrr = _avg_vals(mrr_list)
    avg_ks_stat = _avg_vals(ks_stat_list)
    avg_ks_p = _avg_vals(ks_p_list)
    avg_rank_ic = _avg_vals(rank_ic_raw)

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
        "mrr": avg_mrr,
        "ks_stat": avg_ks_stat,
        "ks_p": avg_ks_p,
        "rank_ic": avg_rank_ic,
        "_rank_ic_raw": rank_ic_raw,
        "_ndcg_raw": ndcg_list,
        "_mrr_raw": mrr_list,
        "_ks_stat_raw": ks_stat_list,
        "_ks_p_raw": ks_p_list,
        **windows,
    }

    pre_rebalance_positions = {}
    for stock_id, pos in engine.pre_rebalance_positions.items():
        pre_rebalance_positions[stock_id] = {
            "shares": pos["shares"],
            "cost": round(pos["cost"], 2),
        }

    return {
        "equity_curve": equity_curve,
        "trades": trades,
        "positions": positions,
        "pre_rebalance_positions": pre_rebalance_positions,
        "metrics": metrics,
        "cash": round(engine.cash, 2),
        "today_pnl": today_pnl,
        "skipped_trades": engine.skipped_trades,
        "predictions_history": engine.predictions_history,
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
    if ax.get_legend_handles_labels()[0]:
        ax.legend(fontsize=9, framealpha=0.9, edgecolor="gray", loc="upper left")
    fig.tight_layout()
    buf = io.BytesIO()
    fig.savefig(buf, format="png", dpi=120)
    plt.close(fig)
    b64 = base64.b64encode(buf.getvalue()).decode()
    return f"data:image/png;base64,{b64}"


def _build_scatter_section(scatter_data, cur_date):
    """从历史数据生成 KS-p/Rank IC/NDCG/MRR vs 策略收益 散点图，返回带 base64 图片的 HTML 片段"""
    import io
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import numpy as np
    from scipy.stats import pearsonr, spearmanr, t as t_dist

    all_models = sorted(set(
        m for date_data in scatter_data.values() for m in date_data
    ))
    models = []
    labels = []
    colors_list = []
    cmap = plt.cm.tab10
    for i, m in enumerate(all_models):
        valid_dates = [d for d in scatter_data if scatter_data[d].get(m, {}).get("ret") is not None]
        if len(valid_dates) >= 3:
            models.append(m)
            labels.append(m.replace("search_", "").replace("_exp_", " exp "))
            colors_list.append(cmap(i % 10))
    n_models = len(models)
    if n_models == 0:
        return '<h3>相关性分析</h3><p>无有效模型数据</p>'

    metrics_cfg = [
        ("ksp", "KS-p"),
        ("ic", "Rank IC"),
        ("ndcg", "NDCG"),
        ("mrr", "MRR"),
    ]

    pairs = {}
    for date in sorted(scatter_data.keys()):
        if date > cur_date:
            break
        for m in models:
            d = scatter_data[date].get(m, {})
            ret = d.get("ret")
            if ret is None:
                continue
            pairs.setdefault(m, {"ret": [], "ksp": [], "ic": [], "ndcg": [], "mrr": [], "dates": []})
            pairs[m]["ret"].append(ret)
            pairs[m]["dates"].append(date)
            for key, _ in metrics_cfg:
                v = d.get(key)
                pairs[m][key].append(v if v is not None else np.nan)

    fig, axes = plt.subplots(4, n_models, figsize=(5 * n_models, 16))
    fig.suptitle(f'相关性分析（截至 {cur_date}）', fontsize=14, fontweight='bold')

    def _fmt_p(pv):
        return f'{pv:.3f}' if pv >= 0.001 else f'{pv:.1e}'

    def _plot_one(ax, xvals, yvals, xlabel, title, color):
        mask = ~(np.isnan(xvals) | np.isnan(yvals))
        xv = np.array(xvals)[mask]
        yv = np.array(yvals)[mask]
        n = len(xv)
        ax.scatter(xv, yv, c=color, s=50, zorder=3, edgecolors='k', linewidths=0.5, alpha=0.5, label='历史')
        if n >= 3:
            try:
                p_val, p_p = pearsonr(yv, xv)
                s_val, s_p = spearmanr(yv, xv)
                xa, ya = xv, yv
                z = np.polyfit(xa, ya, 1)
                y_pred = np.polyval(z, xa)
                mse = np.sum((ya - y_pred) ** 2) / (n - 2)
                x_mean = np.mean(xa)
                xx = np.linspace(xa.min(), xa.max(), 100)
                se = np.sqrt(mse * (1 / n + (xx - x_mean) ** 2 / np.sum((xa - x_mean) ** 2)))
                t_val = t_dist.ppf(0.975, n - 2)
                ci = t_val * se
                ax.fill_between(xx, np.polyval(z, xx) - ci, np.polyval(z, xx) + ci,
                                color=color, alpha=0.15, label='95% CI')
                ax.plot(xx, np.polyval(z, xx), '--', color=color, alpha=0.7, linewidth=1.5)
                label = f'n={n}  r={p_val:.3f}(p={_fmt_p(p_p)})  ρ={s_val:.3f}(p={_fmt_p(s_p)})'
            except Exception:
                label = f'n={n}'
        else:
            label = f'n={n}'
        if n >= 1:
            ax.scatter(xv[-1], yv[-1], c='#cc0000', s=120, zorder=4, edgecolors='k', linewidths=1, marker='*', label='最新')
        ax.set_xlabel(xlabel)
        ax.set_ylabel('策略收益 (%)')
        ax.set_title(title)
        ax.legend(fontsize=7, loc='lower right')
        ax.text(0.05, 0.95, label, transform=ax.transAxes, fontsize=8, va='top',
                bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))
        ax.grid(True, alpha=0.3)

    for ri, (key, xlabel) in enumerate(metrics_cfg):
        for ai, m in enumerate(models):
            ax = axes[ri][ai]
            p = pairs.get(m)
            if p and len(p["ret"]) > 0:
                _plot_one(ax, p[key], p["ret"], xlabel, labels[ai], colors_list[ai])
            else:
                ax.text(0.5, 0.5, '无数据', ha='center', va='center', transform=ax.transAxes, fontsize=10, color='gray')
                ax.set_title(labels[ai])

    plt.tight_layout()
    buf = io.BytesIO()
    fig.savefig(buf, format='png', dpi=120, bbox_inches='tight')
    plt.close(fig)
    b64 = base64.b64encode(buf.getvalue()).decode()

    return f'<h3>相关性分析</h3><div class="scatter-section"><img src="data:image/png;base64,{b64}" style="max-width:100%;height:auto;border:1px solid #ddd;border-radius:5px;" /></div>'


HEALTH_CONFIG = [
    ('avgret', 3, 1.5, (-3.0, 3.0)),
    ('wr',    10, 1.5, (0.2, 0.8)),
    ('dd',    5,  1.2, (-15.0, 0.0)),
    ('vol',   3,  0.8, (5.0, 0.0)),
]


def _compute_health_score(model_data):
    """Compute universal health score (0-100) with details and correlation."""
    from scipy.stats import pearsonr
    ec = model_data.get("equity_curve", [])
    if len(ec) < 2:
        return {"score": 50.0, "details": {}, "corr": None}
    values = [e["total_value"] for e in ec]
    daily_rets = [(values[i] / values[i - 1] - 1) * 100 for i in range(1, len(values))]
    df = pd.DataFrame({"ret": daily_rets})
    df["cummax"] = values[1:]
    raw = {}
    for name, window, weight, (worst, best) in HEALTH_CONFIG:
        if name == "avgret":
            s = df["ret"].rolling(window).mean()
            raw[name] = s.fillna(0)
        elif name == "dd":
            running_max = np.maximum.accumulate(df["cummax"].values)
            dd_vals = (df["cummax"].values - running_max) / running_max * 100
            raw[name] = pd.Series(dd_vals).rolling(window).min().fillna(0)
        elif name == "wr":
            raw[name] = (df["ret"] > 0).rolling(window).mean().fillna(0.5)
        elif name == "vol":
            raw[name] = df["ret"].rolling(window).std().fillna(0)
    latest = {k: float(v.iloc[-1]) for k, v in raw.items()}
    scores = []
    details = {}
    for name, window, weight, (worst, best) in HEALTH_CONFIG:
        v = latest[name]
        v_clipped = np.clip(v, worst, best)
        if best - worst < 1e-12:
            norm = 0.5
        else:
            norm = (v_clipped - worst) / (best - worst)
        scores.append(norm * weight)
        if name == "wr":
            details["wr"] = f"{v*100:.0f}%"
        elif name == "avgret":
            details["avgret"] = f"{v:+.2f}%"
        elif name == "dd":
            details["dd"] = f"{v:.1f}%"
        elif name == "vol":
            details["vol"] = f"{v:.2f}%"
    raw_score = sum(scores)
    total_weight = sum(w for _, _, w, _ in HEALTH_CONFIG)
    score_01 = raw_score / total_weight if total_weight > 0 else 0.5

    all_scores = []
    for i in range(len(daily_rets)):
        s = []
        for name, window, weight, (worst, best) in HEALTH_CONFIG:
            v_i = float(raw[name].iloc[i])
            v_clipped = np.clip(v_i, worst, best)
            if best - worst < 1e-12:
                n = 0.5
            else:
                n = (v_clipped - worst) / (best - worst)
            s.append(n * weight)
        rs = sum(s) / total_weight if total_weight > 0 else 0.5
        all_scores.append(rs)
    all_scores = np.array(all_scores)
    fwd_ret = []
    for i in range(len(daily_rets)):
        end = min(i + 5, len(daily_rets))
        fwd_ret.append(sum(daily_rets[i:end]))
    fwd_ret = np.array(fwd_ret)
    corr_val = None
    window_corr = 14
    if len(all_scores) >= window_corr and np.std(all_scores[-window_corr:]) > 0 and np.std(fwd_ret[-window_corr:]) > 0:
        r, p = pearsonr(all_scores[-window_corr:], fwd_ret[-window_corr:])
        corr_val = {"r": round(r, 3), "p": round(p, 4), "n": window_corr}
        multiplier = 1.0 + 0.15 * r
        multiplier = max(0.85, min(1.15, multiplier))
        score_01 = max(0.0, min(1.0, score_01 * multiplier))
    return {
        "score": round(max(0.0, min(100.0, score_01 * 100)), 1),
        "details": details,
        "corr": corr_val,
    }


def _health_color(score):
    if score >= 80:
        return "green"
    if score >= 60:
        return "goldenrod"
    if score >= 40:
        return "darkorange"
    return "red"


def _save_history_reports(seq, all_sequences, data_file, initial_capital, etf_names, model_key="", rebalance_days=5, trade_mode="open", top_k=3, position_pct=0.95, weight_strategy="equal", strategy_params=None):
    """为序列的每一个调仓日保存历史报告HTML"""
    trades = seq.get("trades", [])
    equity_curve = seq.get("equity_curve", [])
    if not trades or not equity_curve:
        return

    raw_df = load_etf_data(data_file, dtype={"股票代码": str})
    raw_df["股票代码"] = raw_df["股票代码"].astype(object).str.zfill(6)
    raw_df["日期"] = pd.to_datetime(raw_df["日期"])
    close_pivot, hl_pivot, ll_pivot = _build_pivots(raw_df)
    hs300_raw = raw_df[raw_df["股票代码"] == "510300.XSHG"][["日期", "收盘"]].copy()
    hs300_raw = hs300_raw.rename(columns={"日期": "date", "收盘": "close"})
    hs300_raw = hs300_raw.sort_values("date").reset_index(drop=True)
    ec_by_date = {e["date"]: e["total_value"] for e in equity_curve}
    sorted_dates = sorted(ec_by_date.keys())

    rebalance_dates = sorted(set(t["date"] for t in trades))
    history_dir = PROJECT_ROOT / "output" / "history_report"
    history_dir.mkdir(parents=True, exist_ok=True)
    for cur_date in sorted_dates:
        ec_seg = [e for e in equity_curve if e["date"] <= cur_date]
        if len(ec_seg) < 1:
            continue

        is_rebalance = cur_date in rebalance_dates
        today_total = ec_seg[-1]["total_value"]
        yesterday_total = ec_seg[-2]["total_value"] if len(ec_seg) >= 2 else today_total
        today_ts = pd.Timestamp(cur_date)

        # rank map from close_pivot (avoids CSV read + pivot rebuild)
        # close_pivot: index=日期 (Timestamp), columns=股票代码 (str)
        _p = close_pivot[close_pivot.index <= today_ts]
        _rank_map = {}
        if len(_p) >= 6:
            _r5 = (_p.iloc[-1] / _p.iloc[-6] - 1) * 100
            _rk5 = _r5.dropna().sort_values(ascending=False)
            for i, code in enumerate(_rk5.index):
                _rank_map.setdefault(code, {})["rank_5d"] = i + 1
                _rank_map[code]["ret_5d"] = round(float(_rk5.iloc[i]), 2)
        if len(_p) >= 2:
            _r1 = (_p.iloc[-1] / _p.iloc[-2] - 1) * 100
            _rk1 = _r1.dropna().sort_values(ascending=False)
            for i, code in enumerate(_rk1.index):
                _rank_map.setdefault(code, {})["rank_1d"] = i + 1
                _rank_map[code]["ret_1d"] = round(float(_rk1.iloc[i]), 2)
        if len(_p) >= 5:
            _x = np.arange(5)
            for code in _p.columns:
                _vals = _p[code].iloc[-5:].values
                if np.any(np.isnan(_vals)) or np.any(_vals <= 0):
                    continue
                _slope = np.polyfit(_x, _vals, 1)[0]
                _trend = _slope / np.mean(_vals) * 100
                _rank_map.setdefault(code, {})["trend_5d"] = round(float(_trend), 2)

        today_trades_list = []
        if is_rebalance:
            _all_trading_dates = close_pivot.index
            _date_idx = _all_trading_dates.get_loc(today_ts)
            if isinstance(_date_idx, slice):
                _date_idx = _date_idx.stop - 1
            price_date = _all_trading_dates[_date_idx + 1] if _date_idx + 1 < len(_all_trading_dates) else today_ts
            for key, s in all_sequences.items():
                seq_actual = [t for t in s.get("trades", []) if t["date"] == cur_date and t["action"] in ("买入", "卖出")]
                for t in seq_actual:
                    t = {**t, "model_key": key, "name": etf_names.get(t["stock"], "")}
                    today_trades_list.append(t)
                if seq_actual:
                    seq_buys = {t["stock"] for t in seq_actual if t["action"] == "买入"}
                    hist_positions = _rebuild_positions(s.get("trades", []), cur_date)
                    for sid, sp in hist_positions.items():
                        if sid not in seq_buys:
                            price = close_pivot.loc[price_date, sid] if sid in close_pivot.columns else 0
                            price = 0 if pd.isna(price) else price
                            today_trades_list.append({
                                "action": "保持",
                                "stock": sid,
                                "shares": sp["shares"],
                                "price": round(price, 4),
                                "model_key": key,
                                "name": etf_names.get(sid, ""),
                            })
            model_order = {key: i for i, key in enumerate(all_sequences.keys())}
            today_trades_list.sort(key=lambda x: (model_order.get(x.get("model_key", ""), 999), {"买入": 0, "卖出": 1, "跳过": 2}.get(x["action"], 3)))
            # 追加因涨跌停/停牌跳过的操作
            for key, s in all_sequences.items():
                for st in s.get("skipped_trades", []):
                    if st["date"] == cur_date:
                        today_trades_list.append({
                            "action": "跳过",
                            "stock": st["stock"],
                            "reason": st["reason"],
                            "model_key": key,
                            "name": etf_names.get(st["stock"], ""),
                        })

            # 涨停价/跌停价
            for t in today_trades_list:
                sid = t["stock"]
                hl = hl_pivot.loc[today_ts, sid] if sid in hl_pivot.columns else 0
                ll = ll_pivot.loc[today_ts, sid] if sid in ll_pivot.columns else 0
                t["high_limit"] = 0 if pd.isna(hl) else round(float(hl), 4)
                t["low_limit"] = 0 if pd.isna(ll) else round(float(ll), 4)

        # 前一次调仓日
        prev_rb_date = None
        for rd in reversed(rebalance_dates):
            if rd < cur_date:
                prev_rb_date = rd
                break

        positions = _rebuild_positions(trades, prev_rb_date) if prev_rb_date else {}

        prev_close_prices = {}
        if prev_rb_date and is_rebalance:
            prev_ts = pd.Timestamp(prev_rb_date)
            for sid in set(t["stock"] for t in today_trades_list):
                pc = close_pivot.loc[prev_ts, sid] if sid in close_pivot.columns else 0
                prev_close_prices[sid] = 0 if pd.isna(pc) else pc
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
        if cur_date in sorted_dates and sorted_dates.index(cur_date) > 0:
            yesterday_ts = pd.Timestamp(sorted_dates[sorted_dates.index(cur_date) - 1])
        else:
            _prev_dates = close_pivot.index[close_pivot.index < today_ts]
            yesterday_ts = _prev_dates[-1] if len(_prev_dates) > 0 else today_ts

        today_buys = {t["stock"] for t in trades if t["date"] == cur_date and t["action"] == "买入"}

        buy_prices = {}
        buy_dates = {}
        for t in today_trades_list:
            if t.get("model_key") == model_key and t["action"] in ("买入", "保持") and t.get("price", 0) > 0:
                buy_prices[t["stock"]] = t["price"]
                buy_dates[t["stock"]] = t.get("date", cur_date)
        for sid in list(positions.keys()):
            if sid not in buy_prices:
                hist_trades = sorted(
                    [t for t in trades if t["stock"] == sid and t["action"] == "买入" and t["date"] < cur_date],
                    key=lambda x: x["date"], reverse=True
                )
                if hist_trades:
                    buy_prices[sid] = hist_trades[0]["price"]
                    buy_dates[sid] = hist_trades[0]["date"]

        for stock_id, p in positions.items():
            if stock_id in close_pivot.columns:
                price = close_pivot.loc[today_ts, stock_id]
                price = 0 if pd.isna(price) else price
                yc = close_pivot.loc[yesterday_ts, stock_id]
                yc = yc if pd.notna(yc) else price
            else:
                price = 0
                yc = 0
            hl = hl_pivot.loc[today_ts, stock_id] if stock_id in hl_pivot.columns else 0
            ll = ll_pivot.loc[today_ts, stock_id] if stock_id in ll_pivot.columns else 0
            if stock_id in today_buys:
                pnl = 0
            else:
                pnl = round(p["shares"] * (price - yc), 2) if price and yc else 0
            holdings.append({
                "stock_id": stock_id,
                "name": etf_names.get(stock_id, ""),
                "price": round(price, 4) if price else 0,
                "buy_price": round(buy_prices.get(stock_id, 0), 4),
                "buy_date": buy_dates.get(stock_id, ""),
                "shares": p["shares"],
                "cost": round(p["cost"], 2),
                "today_pnl": pnl,
                "high_limit": 0 if pd.isna(hl) else round(float(hl), 4),
                "low_limit": 0 if pd.isna(ll) else round(float(ll), 4),
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
            dd_periods = extract_drawdowns(vals, [e["date"] for e in ec_segment])
            risk_metrics = compute_longterm_risk_metrics(daily_rets, cum, [e["date"] for e in ec_segment], dd_periods)
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
                **risk_metrics,
            }

        metrics = _compute_metrics(ec_seg, initial_capital)

        ec_by_dict = {e["date"]: e for e in ec_seg}
        ec_sorted_dates = sorted(ec_by_dict.keys())
        today_dt = pd.Timestamp(cur_date)
        for wlabel, wdays in [("5d", 5), ("1m", 30)]:
            wcut = (today_dt - pd.Timedelta(days=wdays)).strftime("%Y-%m-%d")
            wseg = [ec_by_dict[d] for d in ec_sorted_dates if d >= wcut]
            if len(wseg) >= 2:
                base = initial_capital if wseg[0]["date"] == ec_seg[0]["date"] else wseg[0]["total_value"]
                wm = _compute_metrics(wseg, base)
                if wm:
                    metrics[f"window_{wlabel}"] = wm

        hs300_seg = hs300_raw[(hs300_raw["date"] >= pd.Timestamp(sorted_dates[0])) & (hs300_raw["date"] <= today_ts)]
        if len(hs300_seg) >= 2:
            hs300_ret = (hs300_seg["close"].iloc[-1] / hs300_seg["close"].iloc[0] - 1) * 100
        else:
            hs300_ret = 0.0
        metrics["hs300_return_pct"] = round(hs300_ret, 4)
        metrics["excess_return_pct"] = round(metrics["strategy_return_pct"] - hs300_ret, 4)

        next_rb = ""
        for rd in rebalance_dates:
            if rd > cur_date:
                next_rb = rd
                break
        if not next_rb:
            next_rb = seq.get("metrics", {}).get("next_rebalance_date", "")
            if not next_rb:
                try:
                    import pandas_market_calendars as mcal
                    xshg = mcal.get_calendar("XSHG")
                    start_ts = pd.Timestamp(rebalance_dates[0])
                    look_end = pd.Timestamp(cur_date) + pd.Timedelta(days=365)
                    cal_dates = xshg.valid_days(start_date=start_ts, end_date=look_end, tz=None)
                    start_pos = cal_dates.get_loc(start_ts.normalize())
                    current_pos = cal_dates.get_loc(pd.Timestamp(cur_date).normalize())
                    n_periods = (current_pos - start_pos) // rebalance_days
                    next_pos = start_pos + (n_periods + 1) * rebalance_days
                    next_rb = cal_dates[next_pos].strftime("%Y-%m-%d") if next_pos < len(cal_dates) else ""
                except Exception:
                    next_rb = ""

        cash = today_total - sum(h["cost"] for h in holdings)
        today_pnl_total = round(sum(h["today_pnl"] for h in holdings), 2)
        today_pnl_positions = [
            {"stock_id": h["stock_id"], "shares": h["shares"], "pnl": h["today_pnl"]}
            for h in holdings
        ]

        chart_data_url = _history_chart_b64(all_sequences, hs300_raw, cur_date, sorted_dates[0], initial_capital)

        from send_report import build_report_html, _build_model_stats_table, _build_health_table, _build_pred_signals_table
        from regenerate_history import build_market_monitor_section
        hist_sequences = {}
        for hkey, hseq in all_sequences.items():
            hist_trades = [t for t in hseq.get("trades", []) if t["date"] <= cur_date]
            hist_positions = _rebuild_positions(hseq.get("trades", []), cur_date)
            _all_trading_dates = close_pivot.index
            _date_idx = _all_trading_dates.get_loc(today_ts)
            if isinstance(_date_idx, slice):
                _date_idx = _date_idx.stop - 1
            price_date = _all_trading_dates[_date_idx + 1] if _date_idx + 1 < len(_all_trading_dates) else today_ts
            today_buys_hist = {t["stock"] for t in hist_trades if t["date"] == cur_date and t["action"] == "买入"}
            hist_current_prices = {}
            for sid, sp in hist_positions.items():
                if sid not in close_pivot.columns:
                    continue
                price_date_lookup = today_ts if sid in today_buys_hist else price_date
                pc = close_pivot.loc[price_date_lookup, sid]
                if pd.notna(pc):
                    hist_current_prices[sid] = pc
            hist_ec = [e for e in hseq.get("equity_curve", []) if e["date"] <= cur_date]
            if len(hist_ec) >= 2:
                hist_metrics = _compute_metrics(hist_ec, initial_capital)
                hs300_seg = hs300_raw[(hs300_raw["date"] >= pd.Timestamp(sorted_dates[0])) & (hs300_raw["date"] <= today_ts)]
                hs300_ret = (hs300_seg["close"].iloc[-1] / hs300_seg["close"].iloc[0] - 1) * 100 if len(hs300_seg) >= 2 else 0.0
                hist_metrics["hs300_return_pct"] = round(hs300_ret, 4)
                hist_metrics["excess_return_pct"] = round(hist_metrics["strategy_return_pct"] - hs300_ret, 4)
            else:
                hist_metrics = {"strategy_return_pct": 0, "sharpe_ratio": 0, "calmar_ratio": 0, "sortino_ratio": 0, "max_drawdown_pct": 0, "hs300_return_pct": 0, "excess_return_pct": 0}
            model_stats = _compute_model_stats(hist_trades, hist_current_prices, report_date=cur_date)
            ec_dict = {e["date"]: e["total_value"] for e in hseq.get("equity_curve", [])}
            reb_period_rets = []
            for i, rd in enumerate(rebalance_dates):
                if rd > cur_date:
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

            n_periods = sum(1 for d in rebalance_dates if d <= cur_date)
            if n_periods > 0:
                raw_ic = hseq.get("metrics", {}).get("_rank_ic_raw", [])
                ic_vals = [e["value"] for e in raw_ic if e["date"] <= cur_date]
                if ic_vals:
                    hist_metrics["rank_ic"] = round(float(np.mean(ic_vals)), 4)
                for key, lst_name in [("ndcg", "_ndcg_raw"), ("mrr", "_mrr_raw"), ("ks_stat", "_ks_stat_raw"), ("ks_p", "_ks_p_raw")]:
                    raw = hseq.get("metrics", {}).get(lst_name, [])
                    vals = [e["value"] for e in raw if e["date"] <= cur_date]
                    if vals:
                        hist_metrics[key] = round(float(np.mean(vals)), 4)
            hist_sequences[hkey] = {
                "model_stats": model_stats,
                "metrics": hist_metrics,
            }
        health_scores = {}
        for hkey in all_sequences:
            truncated = {"equity_curve": [e for e in all_sequences[hkey].get("equity_curve", []) if e["date"] <= cur_date]}
            health_scores[hkey] = _compute_health_score(truncated)
        health_section = _build_health_table(health_scores)

        model_stats_section = _build_model_stats_table(hist_sequences)

        hist_equity = {}
        hs300_period = hs300_raw[(hs300_raw["date"] >= pd.Timestamp(sorted_dates[0])) & (hs300_raw["date"] <= pd.Timestamp(cur_date))]
        if not hs300_period.empty:
            hs300_init = float(hs300_period["close"].iloc[0])
            hist_equity["沪深300"] = [
                {"date": row["date"].strftime("%Y-%m-%d"), "total_value": round(float(row["close"]) / hs300_init * initial_capital, 2)}
                for _, row in hs300_period.iterrows()
            ]
        for hkey, hseq in all_sequences.items():
            ec = hseq.get("equity_curve", [])
            ec_seg_f = [e for e in ec if e["date"] <= cur_date]
            if ec_seg_f:
                disp = hkey.replace("search_", "").replace("_exp_", " ").replace("_full", " (full)")
                hist_equity[disp] = ec_seg_f
        # 上期持仓（最近一次调仓前的持仓）
        pre_holdings = []
        if prev_rb_date:
            prev_rb_ts = pd.Timestamp(prev_rb_date)
            all_trading_dates = sorted(raw_df["日期"].unique())
            trading_before = [d for d in all_trading_dates if d < prev_rb_ts]
            day_before_rb = trading_before[-1].strftime("%Y-%m-%d") if trading_before else None
            if day_before_rb:
                pre_positions = _rebuild_positions(trades, day_before_rb)
                for stock_id, p in pre_positions.items():
                    price = close_pivot.loc[today_ts, stock_id] if stock_id in close_pivot.columns else 0
                    price = 0 if pd.isna(price) else price
                    pre_holdings.append({
                        "stock_id": stock_id,
                        "name": etf_names.get(stock_id, ""),
                        "price": price,
                        "price_display": price,
                        "shares": p["shares"],
                        "cost": p.get("cost", 0),
                    })
        # 主序列的调仓胜率
        master_stats = hist_sequences.get(model_key, {}).get("model_stats", {}) if model_key else {}
        hist_rebalance_win_rate = master_stats.get("total_win_rate_pct")
        # 预测信号（截至当前日期）
        hist_ph = [p for p in seq.get("predictions_history", []) if p.get("date", "") <= cur_date]
        hist_seq_data = {**seq, "predictions_history": hist_ph}
        pred_signals_section = _build_pred_signals_table(hist_seq_data, cur_date, weight_strategy=weight_strategy, strategy_params=strategy_params, top_k=top_k, position_pct=position_pct, rank_map=_rank_map)
        holdings_at_date = {h["stock_id"] for h in holdings}
        market_monitor_section = build_market_monitor_section(raw_df, seq, cur_date, holdings_at_date, etf_names, close_pivot=close_pivot)
        try:
            html = build_report_html(
                date=cur_date,
                model_display="历史调仓" if not model_key else model_key.replace("search_", "").replace("_exp_", " ").replace("_full", " (full)"),
                total_value=today_total,
                cash=cash,
                holdings=holdings,
                pre_holdings=pre_holdings,
                trades_list=today_trades_list,
                metrics=metrics,
                next_rebalance=next_rb,
                is_rebalance=is_rebalance,
                today_pnl_total=today_pnl_total,
                today_pnl_positions=today_pnl_positions,
                chart_data_url=chart_data_url,
                model_stats_section=model_stats_section,
                equity_data=hist_equity,
                scatter_section="",
                health_section=health_section,
                pred_signals_section=pred_signals_section,
                market_monitor_section=market_monitor_section,
                source="本地回测",
                trade_mode=trade_mode,
                rebalance_win_rate=hist_rebalance_win_rate,
                rank_map=_rank_map,
            )
            suffix = "(调仓日)" if is_rebalance else ""
            history_path = history_dir / f"{cur_date}{suffix}.html"
            history_path.write_text(html, encoding="utf-8")
        except Exception as e:
            print(f"  [历史报告] {cur_date} 保存失败: {e}")


def _compute_distinct_topk(predictions_history, top_k):
    all_stocks = set()
    for ph in predictions_history:
        for p in ph.get("predictions", [])[:top_k]:
            all_stocks.add(p["stock_id"])
    return len(all_stocks)


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
        name_override = m.get("name", "")
        if name_override:
            return name_override
        exp_dir = m.get("exp_dir", m.get("dir", ""))
    elif isinstance(m, (list, tuple)):
        exp_dir = m[0]
    else:
        exp_dir = m
    parent = os.path.basename(os.path.dirname(exp_dir))
    parent = re.sub(r'_\d+_\d+', '', parent)
    name = os.path.basename(exp_dir)
    if name == "full":
        grandparent = os.path.basename(os.path.dirname(os.path.dirname(exp_dir)))
        grandparent = re.sub(r'_\d+_\d+', '', grandparent)
        return f"{grandparent}_{parent}_{name}"
    return f"{parent}_{name}"


def _resolve_report_key(sequences):
    """从 config.yaml（优先）或 model_selection.yaml 确定主序列。

    优先级:
      1. master 显式指定且在 sequences 中存在 → 使用
      2. master 显式指定但不存在 → 取第一个真实模型（非 average/voting/juejin）
      3. master 未指定 → 兜底: juejin → average → voting → 第一个
    """
    master = ""
    for cfg_path in (CONFIG_PATH, MODEL_SELECTION_PATH):
        if cfg_path.exists():
            try:
                import yaml
                with open(cfg_path, "r", encoding="utf-8") as f:
                    sel = yaml.safe_load(f)
                master = sel.get("master", "")
                if master == "first":
                    models = sel.get("models", [])
                    for m in models:
                        if m.get("enabled", True):
                            master = _make_model_key(m)
                            break
                if master:
                    break
            except Exception:
                pass
    if master:
        if master in sequences:
            return master
        # master 指定了但不存在（如 juejin 但回测尚未运行），取第一个真实模型
        for key in sequences:
            if key not in ("juejin", "average", "voting"):
                return key
    # 未指定 master: 兜底优先级 juejin → average → voting → 第一个
    for pref in ["juejin", "average", "voting"]:
        if pref in sequences:
            return pref
    return list(sequences.keys())[0] if sequences else None


def _save_predictions(sequences, path=None):
    """Save per-model predictions_history to JSON.

    Keys are pred_date (the date for which features were computed),
    NOT the rebalance date. This ensures compatibility with
    _make_predictions_func_from_saved() used by --from-predictions mode.

    Preserves _meta (backtest_dates calendar) from existing file so that
    juejin/main.py can compute correct rebalance dates.
    """
    path = path or PREDICTIONS_PATH
    # preserve _meta from existing file (full calendar for juejin)
    meta = {}
    if os.path.exists(str(path)):
        try:
            with open(str(path), "r") as f:
                existing = json.load(f)
            meta = existing.get("_meta", {})
        except Exception:
            pass
    preds = {}
    for key, seq in sequences.items():
        ph = seq.get("predictions_history", [])
        preds[key] = {entry.get("pred_date", entry["date"]): entry["predictions"] for entry in ph}
    if meta:
        preds["_meta"] = meta
    with open(str(path), "w", encoding="utf-8") as f:
        json.dump(preds, f, ensure_ascii=False, indent=2, cls=NumpyEncoder)
    return preds


def _load_predictions(path=None):
    """Load predictions from JSON, return {model_key: {date_str: [preds]}}."""
    path = path or PREDICTIONS_PATH
    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)
    data.pop("_meta", None)
    return data


def _make_predictions_func_from_saved(preds_dict):
    """Build predictions_func callable from a saved {date_str: [preds]} dict."""
    def pred_func(date):
        d_str = date.strftime("%Y-%m-%d") if hasattr(date, "strftime") else str(date)
        return preds_dict.get(d_str)
    return pred_func


def daily_eval(
    config_name: str = "config",
    update_data: bool = True,
    top_k: int = 3,
    verbose: bool = True,
    start_date: str = "2026-04-01",
    rebalance_days: int = 5,
    position_pct: float = 0.95,
    weight_strategy: str = "equal",
    strategy_params: dict = None,
    initial_capital: float = 100000,
    trade_mode: str = "open",
):
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    _t0 = time.time()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    try:
        if update_data:
            print(f"\n[1/4] 更新数据...", end=" ", flush=True)
            ok = update_etf_data(verbose=verbose)
            print(f"{'✅' if ok else '❌'}  ({time.time()-_t0:.0f}s)")
            _check_data_integrity(verbose=verbose)

        raw_df = load_etf_data(DATA_FILE, dtype={"股票代码": str})
        raw_df["股票代码"] = raw_df["股票代码"].astype(object).str.zfill(6)
        raw_df["日期"] = pd.to_datetime(raw_df["日期"])
        close_pivot, hl_pivot, ll_pivot = _build_pivots(raw_df)
        latest_date = raw_df["日期"].max()
        latest_date_str = latest_date.strftime("%Y-%m-%d")
        end_date = (latest_date + pd.Timedelta(days=1)).strftime("%Y-%m-%d")

        if verbose:
            print(f"[数据] 最新交易日: {latest_date_str}")

        print(f"[2/4] 加载模型...", end=" ", flush=True)

        single_models = []
        average_enabled = False
        voting_enabled = False
        config = {"slippage": 0.001, "commission": 0.0003}
        if CONFIG_PATH.exists():
            cfg_dict = load_full_config()
            single_models, master, average_enabled, voting_enabled = load_model_selection(cfg_dict=cfg_dict)
        elif os.path.exists(str(MODEL_SELECTION_PATH)):
            single_models, master, average_enabled, voting_enabled = load_model_selection(path=str(MODEL_SELECTION_PATH))
        if single_models:
            enabled_models = [m for m in single_models if m.get("enabled", True)]
            if verbose:
                print()
                print(f"  平均: {'开' if average_enabled else '关'}, 投票: {'开' if voting_enabled else '关'}, 主序列: {master or 'auto'}, 模型数: {len(enabled_models)}")
                for m in single_models:
                    status = "启用" if m.get("enabled", True) else "禁用"
                    print(f"    {status}: {_make_model_key(m)} ({m['model_file']})")
            single_models = enabled_models
            if not verbose:
                print(f"{len(enabled_models)}个模型 (最新: {latest_date_str})")
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
            if not verbose:
                print(f"{_make_model_key((exp_dir,))}")
            elif verbose:
                print(f"  单模型: {_make_model_key((exp_dir,))} (score={score:.4f})")

        if not single_models:
            print("错误: 无可用模型")
            return

        first_model = single_models[0]
        scaler_path = os.path.join(first_model["exp_dir"], "scaler.pkl")
        if not os.path.exists(scaler_path):
            parent = os.path.dirname(os.path.normpath(first_model["exp_dir"]))
            parent_scaler = os.path.join(parent, "scaler.pkl")
            if os.path.exists(parent_scaler):
                scaler_path = parent_scaler
        config_path = os.path.join(first_model["exp_dir"], "config.json")
        if not os.path.exists(config_path):
            parent = os.path.dirname(os.path.normpath(first_model["exp_dir"]))
            parent_cfg = os.path.join(parent, "config.json")
            if os.path.exists(parent_cfg):
                config_path = parent_cfg
        with open(config_path, "r") as f:
            first_config = json.load(f)
        feature_num = first_config["feature_num"]

        if verbose:
            print(f"\n[3/4] 加载并缓存数据...")

        model_types = set(m.get("type", "dl") for m in single_models)
        has_ml = bool(model_types & {"xgb", "lightgbm", "catboost"})
        has_dl = bool(model_types & {"dl", "ensemble_folds"})

        cached_data = cached_features = None
        ml_cached_data = None

        if has_dl:
            if verbose:
                print(f"  加载 DL 数据...")
            cached_data, cached_features = ETFBacktester.load_data_once(
                data_path=str(DATA_FILE),
                scaler_path=scaler_path,
                feature_num=feature_num,
                verbose=False,
            )
        if has_ml:
            if verbose:
                print(f"  加载 ML 数据...")
            ml_cached_data = MLBacktester.load_data_once(
                data_path=str(DATA_FILE),
                add_cs_features=True,
            )

        single_backtesters = []
        ensemble_models = []
        for m in single_models:
            mtype = m.get("type", "dl")
            if mtype == "ensemble_folds":
                ensemble_models.append(m)
                continue
            if mtype in ("xgb", "lightgbm", "catboost"):
                bt = MLBacktester.from_cached_data(
                    model_dir=m["exp_dir"],
                    cached_data=ml_cached_data,
                    model_file=m["model_file"],
                    verbose=False,
                )
            else:
                bt = ETFBacktester.from_cached_data(
                    model_dir=m["exp_dir"],
                    cached_data=cached_data,
                    cached_features=cached_features,
                    device=device,
                    model_file=m["model_file"],
                    verbose=False,
                )
            single_backtesters.append((m, bt))

        print(f"[3/4] 运行回测... ({time.time()-_t0:.0f}s)")

        sequences = {}

        for m, bt in single_backtesters:
            model_key = _make_model_key(m)
            _t_model = time.time()
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
                weight_strategy=weight_strategy,
                strategy_params=strategy_params,
                initial_capital=initial_capital,
                commission=config.get("commission", 0.0003),
                slippage=config.get("slippage", 0.001),
                trade_mode=trade_mode,
                risk_manager_config=config.get("risk_manager", {}),
            )
            sequences[model_key] = result
            if verbose:
                print(f"    ✓ ({time.time()-_t_model:.0f}s)")

        if average_enabled and len(single_backtesters) >= 2:
            _t_avg = time.time()
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
                weight_strategy=weight_strategy,
                strategy_params=strategy_params,
                initial_capital=initial_capital,
                commission=config.get("commission", 0.0003),
                slippage=config.get("slippage", 0.001),
                trade_mode=trade_mode,
                risk_manager_config=config.get("risk_manager", {}),
            )
            sequences["average"] = result
            if verbose:
                print(f"    ✓ ({time.time()-_t_avg:.0f}s)")

        if voting_enabled and len(single_backtesters) >= 2:
            _t_vote = time.time()
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
                        if i >= top_k * 3:
                            break
                        sid = p["stock_id"]
                        freq[sid] = freq.get(sid, 0) + 1
                        avg_score[sid] = avg_score.get(sid, 0) + p["score"]
                for sid in avg_score:
                    avg_score[sid] /= freq.get(sid, 1)
                ranked = sorted(freq.items(), key=lambda x: (-x[1], -avg_score.get(x[0], 0)))[:max(top_k, 10)]
                result = [{"rank": i+1, "stock_id": sid, "score": float(freq)} for i, (sid, freq) in enumerate(ranked)]
                voting_pred_cache[date] = {"ranked": ranked, "top_k": top_k}
                return result

            result = run_backtest_sequence(
                predictions_func=voting_pred_func,
                data_file=str(DATA_FILE),
                start_date=start_date,
                end_date=end_date,
                top_k=top_k,
                rebalance_days=rebalance_days,
                position_pct=position_pct,
                weight_strategy=weight_strategy,
                strategy_params=strategy_params,
                initial_capital=initial_capital,
                commission=config.get("commission", 0.0003),
                slippage=config.get("slippage", 0.001),
                trade_mode=trade_mode,
                risk_manager_config=config.get("risk_manager", {}),
            )
            voting_total_models = len(single_backtesters)
            for ph in result.get("predictions_history", []):
                ph["voting_total_models"] = voting_total_models
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
            if verbose:
                print(f"    ✓ ({time.time()-_t_vote:.0f}s)")

        if ensemble_models:
            for m in ensemble_models:
                _t_ens = time.time()
                model_key = _make_model_key(m) + "_ensemble"
                if verbose:
                    print(f"  回测 fold 集成: {model_key}...")
                fold_dirs = sorted([
                    d for d in os.listdir(m["exp_dir"])
                    if d.startswith("fold_") and os.path.isdir(os.path.join(m["exp_dir"], d))
                ])
                fold_bts = []
                for fd in fold_dirs:
                    fold_dir = os.path.join(m["exp_dir"], fd)
                    bt = ETFBacktester.from_cached_data(
                        model_dir=fold_dir,
                        cached_data=cached_data,
                        cached_features=cached_features,
                        device=device,
                        model_file=m["model_file"],
                        verbose=False,
                    )
                    fold_bts.append(bt)

                def _make_ensemble_pred_func(bts):
                    def ensemble_pred(date):
                        all_score_dicts = []
                        for bti in bts:
                            preds = bti._get_predictions(date)
                            if preds is None:
                                return None
                            all_score_dicts.append({p["stock_id"]: p["score"] for p in preds})
                        avg_scores = {}
                        for sid in all_score_dicts[0].keys():
                            scores = [sd[sid] for sd in all_score_dicts]
                            avg_scores[sid] = float(np.mean(scores))
                        sorted_stocks = sorted(avg_scores.items(), key=lambda x: x[1], reverse=True)
                        return [{"rank": i+1, "stock_id": sid, "score": sc} for i, (sid, sc) in enumerate(sorted_stocks)]
                    return ensemble_pred

            result = run_backtest_sequence(
                predictions_func=pred_func,
                data_file=str(DATA_FILE),
                start_date=start_date,
                end_date=end_date,
                top_k=top_k,
                rebalance_days=rebalance_days,
                position_pct=position_pct,
                weight_strategy=weight_strategy,
                strategy_params=strategy_params,
                initial_capital=initial_capital,
                commission=config.get("commission", 0.0003),
                slippage=config.get("slippage", 0.001),
                trade_mode=trade_mode,
                risk_manager_config=config.get("risk_manager", {}),
            )
            sequences[model_key] = result
            if verbose:
                print(f"    ✓ ({time.time()-_t_ens:.0f}s)")

        print(f"  ✓ 回测完成 ({time.time()-_t0:.0f}s)")

        # 绘制收益曲线图
        plot_path = OUTPUT_DIR / "equity_curves.png"
        try:
            plot_equity_curves(sequences, str(DATA_FILE), initial_capital, str(plot_path))
            if verbose:
                print(f"\n  [图表] 收益曲线已保存: {plot_path}")
        except Exception as e:
            if verbose:
                print(f"\n  [图表] 保存失败: {e}")

        report_key = _resolve_report_key(sequences)
        if not report_key:
            report_key = list(sequences.keys())[0]
        report_data = sequences[report_key]

        # 当前价格（从今日盈亏中取）
        pnl_positions = {p["stock_id"]: p for p in report_data.get("today_pnl", {}).get("positions", [])}

        today_actual_trades = [t for t in report_data["trades"] if t["date"] == latest_date_str and t["action"] in ("买入", "卖出")]
        is_rebalance_day = len(today_actual_trades) > 0
        if not is_rebalance_day:
            all_dates = sorted(raw_df["日期"].unique())
            start_idx = next((i for i, d in enumerate(all_dates) if d >= pd.Timestamp(start_date)), None)
            today_idx = next((i for i, d in enumerate(all_dates) if d >= pd.Timestamp(latest_date_str)), None)
            if start_idx is not None and today_idx is not None and (today_idx - start_idx) % rebalance_days == 0:
                is_rebalance_day = True

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
                        price = close_pivot.loc[hold_price_date, sid] if sid in close_pivot.columns else 0
                        price = 0 if pd.isna(price) else round(price, 4)
                        all_today_trades.append({
                            "action": "保持",
                            "stock": sid,
                            "shares": pos["shares"],
                            "price": price,
                            "model_key": key,
                        })
            # 追加今日因涨跌停/停牌跳过的操作
            for st in seq.get("skipped_trades", []):
                if st["date"] == latest_date_str:
                    all_today_trades.append({
                        "action": "跳过",
                        "stock": st["stock"],
                        "reason": st["reason"],
                        "model_key": key,
                    })

        # 涨停价/跌停价
        for t in all_today_trades:
            sid = t["stock"]
            hl = hl_pivot.loc[latest_date, sid] if sid in hl_pivot.columns else 0
            ll = ll_pivot.loc[latest_date, sid] if sid in ll_pivot.columns else 0
            t["high_limit"] = 0 if pd.isna(hl) else round(float(hl), 4)
            t["low_limit"] = 0 if pd.isna(ll) else round(float(ll), 4)

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
                        price = close_pivot.loc[hold_date, sid] if sid in close_pivot.columns else 0
                        if pd.notna(price) and price > 0:
                            last_trade_prices[sid] = round(price, 4)
        report_data["metrics"]["last_trade_prices"] = last_trade_prices

        # 重写 state（此时 latest_trade_prices 已写入 metrics）
        state = {
            "sequences": sequences,
            "last_updated": timestamp,
        }

        # 保存全部交易日预测信号（供日报复用，无需重跑模型）
        try:
            # 生成全部日期的预测（不只是调仓日），覆盖 _save_predictions 仅存调仓日的问题
            _all_dates_full = sorted(raw_df["日期"].unique())
            _start_ts_full = pd.Timestamp(start_date)
            _end_ts_full = pd.Timestamp(end_date)
            _backtest_dates_full = [d for d in _all_dates_full if _start_ts_full <= d < _end_ts_full]
            _seed_dates_full = set(_backtest_dates_full)
            _prev_idx_full = _all_dates_full.index(_backtest_dates_full[0]) - 1
            if _prev_idx_full >= 0:
                _seed_dates_full.add(_all_dates_full[_prev_idx_full])
            _t_pred = time.time()
            if verbose:
                print(f"  [预测] 生成全部交易日预测信号...")
            _full_preds = {}
            for _m, _bt in single_backtesters:
                _mk = _make_model_key(_m)
                _pm = {}
                for _d in sorted(_seed_dates_full):
                    _p = _bt._get_predictions(_d)
                    if _p:
                        _pm[_d.strftime("%Y-%m-%d")] = _p
                _full_preds[_mk] = _pm
            if average_enabled and len(single_backtesters) >= 2:
                _avg_p = {}
                for _d in sorted(_seed_dates_full):
                    _d_str = _d.strftime("%Y-%m-%d")
                    _all_ranks = []
                    for _, _bt2 in single_backtesters:
                        _p2 = _bt2._get_predictions(_d)
                        if _p2 is None:
                            _all_ranks = None
                            break
                        _all_ranks.append({_p["stock_id"]: _p["rank"] for _p in _p2})
                    if _all_ranks:
                        _avg_map = {}
                        for _sid in _all_ranks[0].keys():
                            _rks = [_sd[_sid] for _sd in _all_ranks]
                            _avg_map[_sid] = float(np.mean(_rks))
                        _sorted = sorted(_avg_map.items(), key=lambda x: x[1])
                        _avg_p[_d_str] = [{"rank": i+1, "stock_id": sid, "score": -sc} for i, (sid, sc) in enumerate(_sorted)]
                _full_preds["average"] = _avg_p
            if voting_enabled and len(single_backtesters) >= 2:
                _vote_p = {}
                for _d in sorted(_seed_dates_full):
                    _d_str = _d.strftime("%Y-%m-%d")
                    _all_preds = []
                    for _, _bt2 in single_backtesters:
                        _p2 = _bt2._get_predictions(_d)
                        if _p2 is None:
                            _all_preds = None
                            break
                        _all_preds.append(_p2)
                    if _all_preds:
                        _freq = {}
                        _avg_sc = {}
                        for _preds in _all_preds:
                            for _i, _p in enumerate(_preds):
                                if _i >= top_k * 3:
                                    break
                                _sid = _p["stock_id"]
                                _freq[_sid] = _freq.get(_sid, 0) + 1
                                _avg_sc[_sid] = _avg_sc.get(_sid, 0) + _p["score"]
                        for _sid in _avg_sc:
                            _avg_sc[_sid] /= _freq.get(_sid, 1)
                        _ranked = sorted(_freq.items(), key=lambda x: (-x[1], -_avg_sc.get(x[0], 0)))[:top_k]
                        _result = [{"rank": i+1, "stock_id": sid, "score": float(freq)} for i, (sid, freq) in enumerate(_ranked)]
                        _vote_p[_d_str] = _result
                _full_preds["voting"] = _vote_p
            _full_preds["_meta"] = {
                "start_date": start_date,
                "backtest_dates": [d.strftime("%Y-%m-%d") for d in _backtest_dates_full],
            }
            with open(PREDICTIONS_PATH, "w", encoding="utf-8") as _f:
                json.dump(_full_preds, _f, ensure_ascii=False, indent=2, cls=NumpyEncoder)
            if verbose:
                print(f"  [预测] 已保存 ({time.time()-_t_pred:.0f}s, {len(_backtest_dates_full)} 日)")
        except Exception as e:
            if verbose:
                print(f"  [预测] 保存失败: {e}")
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
            m_raw = load_etf_data(DATA_FILE, dtype={"股票代码": str})
            m_raw["股票代码"] = m_raw["股票代码"].astype(object).str.zfill(6)
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

        # 收盘交易模式下，调仓日显示调仓前持仓（新持仓明天再出现）
        if trade_mode == "close" and is_rebalance_day:
            display_positions = report_data.get("pre_rebalance_positions", {})
            if not display_positions:
                display_positions = report_data["positions"]
        else:
            display_positions = report_data["positions"]

        # 本期调仓参考价：从最近一次调仓日收盘价计算所有持仓的调仓以来收益
        rb_ref_date = prev_rb_date
        if not rb_ref_date and trade_dates:
            for d in reversed(trade_dates):
                if d < latest_date_str:
                    rb_ref_date = d
                    break
        rb_close_prices = {}
        if rb_ref_date:
            rb_ts = pd.Timestamp(rb_ref_date)
            for stock_id in display_positions:
                pc = close_pivot.loc[rb_ts, stock_id] if stock_id in close_pivot.columns else 0
                if pd.notna(pc):
                    rb_close_prices[stock_id] = float(pc)

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
        for stock_id, pos in display_positions.items():
            price = pnl_positions.get(stock_id, {}).get("today_close", 0)
            hl = hl_pivot.loc[latest_date, stock_id] if stock_id in hl_pivot.columns else 0
            ll = ll_pivot.loc[latest_date, stock_id] if stock_id in ll_pivot.columns else 0
            if stock_id in rb_close_prices:
                buy_price = rb_close_prices[stock_id]
                buy_date = rb_ref_date
            else:
                buy_price = 0
                buy_date = ""
                for t in report_data.get("trades", []):
                    if t["action"] in ("买入", "卖出") and t["stock"] == stock_id and t["date"] > buy_date:
                        buy_price = t["price"]
                        buy_date = t["date"]
                if not buy_price:
                    buy_price = round(pos.get("cost", 0) / pos.get("shares", 1), 4) if pos.get("shares", 0) > 0 else 0
            holdings.append({
                "stock_id": stock_id,
                "name": etf_names.get(stock_id, ""),
                "price": price,
                "buy_price": round(buy_price, 4),
                "buy_date": buy_date,
                "shares": pos["shares"],
                "cost": pos["cost"],
                "high_limit": 0 if pd.isna(hl) else round(float(hl), 4),
                "low_limit": 0 if pd.isna(ll) else round(float(ll), 4),
            })

        for t in today_trades:
            t["name"] = etf_names.get(t["stock"], "")

        for t in all_today_trades:
            if "name" not in t:
                t["name"] = etf_names.get(t["stock"], "")

        # 注入最新预测到 predictions_history（日报预测信号表需要最新数据，不是调仓日数据）
        try:
            with open(PREDICTIONS_PATH, "r") as _f_inj:
                _all_preds_inj = json.load(_f_inj)
            _all_preds_inj.pop("_meta", None)
            for _mk_inj, _pd_inj in _all_preds_inj.items():
                _seq_inj = sequences.get(_mk_inj)
                if not _seq_inj:
                    continue
                _ph_inj = _seq_inj.get("predictions_history", [])
                _latest_d_inj = latest_date_str
                _match_inj = _pd_inj.get(_latest_d_inj)
                if not _match_inj:
                    for _d_inj in reversed(sorted(_pd_inj.keys())):
                        if _d_inj <= _latest_d_inj and _pd_inj[_d_inj]:
                            _match_inj = _pd_inj[_d_inj]
                            _latest_d_inj = _d_inj
                            break
                if _match_inj:
                    _sp_snap_inj = dict(strategy_params) if strategy_params else {}
                    _sp_snap_inj.pop("vol_dict", None)
                    if weight_strategy in ("risk_parity", "score_risk", "score_risk_v1", "kelly"):
                        _top_ids_inj = [p["stock_id"] for p in _match_inj[:top_k]]
                        _vol_win_inj = _sp_snap_inj.get("vol_window", 20)
                        _vd_inj = compute_volatility(raw_df, _top_ids_inj, _latest_d_inj, _vol_win_inj)
                        if _vd_inj:
                            _sp_snap_inj["vol_dict"] = _vd_inj
                    _entry_inj = {
                        "date": _latest_d_inj,
                        "predictions": _match_inj[:10],
                        "strategy_params": dict(_sp_snap_inj),
                    }
                    if not _ph_inj or _ph_inj[-1].get("date") != _latest_d_inj:
                        _ph_inj.append(_entry_inj)
                    else:
                        _ph_inj[-1].update({k: v for k, v in _entry_inj.items() if k != "date"})
        except Exception:
            pass

        # 收集所有序列的信息
        sequences_summary = {}
        for key, seq in sequences.items():
            seq_pnl = {p["stock_id"]: p for p in seq.get("today_pnl", {}).get("positions", [])}
            seq_current_prices = {sid: p["today_close"] for sid, p in seq_pnl.items() if p.get("today_close", 0) > 0}
            model_stats = _compute_model_stats(seq["trades"], seq_current_prices, report_date=latest_date_str)
            model_stats["distinct_topk"] = _compute_distinct_topk(seq.get("predictions_history", []), top_k)
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
                "skipped_trades": seq.get("skipped_trades", []),
                "predictions_history": seq.get("predictions_history", []),
            }

        next_rebalance = report_data["metrics"].get("next_rebalance_date", "")

        hs300_curve = []
        hs300_raw = raw_df[raw_df["股票代码"] == "510300.XSHG"].sort_values("日期")
        if not hs300_raw.empty:
            hs300_period = hs300_raw[(hs300_raw["日期"] >= pd.Timestamp(start_date)) & (hs300_raw["日期"] < pd.Timestamp(end_date))]
            if not hs300_period.empty:
                hs300_start_price = float(hs300_period["收盘"].iloc[0])
                for _, row in hs300_period.iterrows():
                    hs300_curve.append({
                        "date": row["日期"].strftime("%Y-%m-%d"),
                        "total_value": round(float(row["收盘"]) / hs300_start_price * initial_capital, 2),
                    })

        pre_holdings = []
        pre_positions = report_data.get("pre_rebalance_positions", {})
        for stock_id, pos in pre_positions.items():
            sub = raw_df[raw_df["股票代码"] == stock_id]
            tc_s = sub.loc[sub["日期"] == latest_date, "收盘"]
            price = float(tc_s.values[0]) if not tc_s.empty else 0
            pre_holdings.append({
                "stock_id": stock_id,
                "name": etf_names.get(stock_id, ""),
                "price": price,
                "price_display": price,
                "shares": pos["shares"],
                "cost": pos.get("cost", 0),
            })

        report = {
            "date": latest_date_str,
            "is_rebalance_day": is_rebalance_day,
            "next_rebalance_date": next_rebalance,
            "today_trades": today_trades,
            "all_today_trades": all_today_trades,
            "metrics": report_data["metrics"],
            "holdings": holdings,
            "pre_holdings": pre_holdings,
            "cash": report_data["cash"],
            "total_value": report_data["metrics"]["latest_value"],
            "sequences": sequences_summary,
            "hs300_curve": hs300_curve,
            "trade_mode": trade_mode,
            "weight_strategy": weight_strategy,
            "strategy_params": strategy_params,
            "top_k": top_k,
            "position_pct": position_pct,
            "voting_total_models": len(single_backtesters),
            "strategy_info": _format_strategy_info(
                weight_strategy, strategy_params, top_k, position_pct,
                config.get("commission", 0.0003), config.get("slippage", 0.001),
                rebalance_days,
            ),
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
        print(f"[4/4] 生成并发送邮件...", end=" ", flush=True)
        try:
            from send_report import send_report
            email_key = _resolve_report_key(sequences)
            if not email_key:
                email_key = list(sequences.keys())[0]
            if "holdings" not in sequences[email_key]:
                sequences[email_key]["holdings"] = holdings
            send_report(model_key=email_key)
            print(f"✅  ({time.time()-_t0:.0f}s)")
        except Exception as e:
            print(f"❌  ({time.time()-_t0:.0f}s)")
            if verbose:
                print(f"\n[邮件] 发送失败: {e}")

        # 生成历史调仓日报告（从回测数据重建）
        try:
            if verbose:
                print(f"\n[历史] 生成各调仓日报告...")
            _save_history_reports(
                sequences[report_key], sequences, str(DATA_FILE), initial_capital, etf_names, model_key=report_key, rebalance_days=rebalance_days, trade_mode=trade_mode, top_k=top_k, position_pct=position_pct, weight_strategy=weight_strategy, strategy_params=strategy_params
            )
        except Exception as e:
            print(f"\n[历史] 生成失败: {e}")
            traceback.print_exc()

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
            print(f"[{timestamp}] 每日测评完成 (总用时 {time.time()-_t0:.0f}s)")
            print(f"{'='*60}")

        return state

    except Exception as e:
        print(f"\n[错误] 每日测评失败: {e}")
        traceback.print_exc()
        return None


# ============================================================
# 模式1: 仅生成预测信号（不执行回测）
# ============================================================

def generate_predictions_only(
    config_name: str = "config",
    update_data: bool = False,
    top_k: int = 3,
    verbose: bool = True,
    start_date: str = "2026-04-01",
    rebalance_days: int = 5,
    position_pct: float = 0.95,
    initial_capital: float = 100000,
):
    """模式1: 仅保存模型预测信号，不执行回测和生成日报"""
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    if verbose:
        print(f"\n{'='*60}")
        print(f"[{timestamp}] 仅生成预测信号")
        print(f"{'='*60}")

    try:
        if update_data:
            if verbose:
                print("[1/4] 获取最新ETF数据...")
            update_etf_data(verbose=verbose)

        raw_df = load_etf_data(DATA_FILE, dtype={"股票代码": str})
        raw_df["股票代码"] = raw_df["股票代码"].astype(object).str.zfill(6)
        raw_df["日期"] = pd.to_datetime(raw_df["日期"])
        latest_date = raw_df["日期"].max()
        end_date = (latest_date + pd.Timedelta(days=1)).strftime("%Y-%m-%d")

        start_ts = pd.Timestamp(start_date)
        end_ts = pd.Timestamp(end_date)
        all_dates = sorted(raw_df["日期"].unique())
        backtest_dates = [d for d in all_dates if start_ts <= d < end_ts]

        if verbose:
            print(f"\n[数据] 交易日: {len(backtest_dates)}天 ({start_date} ~ {end_date})")
            print(f"\n[2/4] 加载模型...")

        single_models = []
        average_enabled = False
        voting_enabled = False
        if CONFIG_PATH.exists():
            cfg_dict = load_full_config()
            single_models, master, average_enabled, voting_enabled = load_model_selection(cfg_dict=cfg_dict)
        elif os.path.exists(str(MODEL_SELECTION_PATH)):
            single_models, master, average_enabled, voting_enabled = load_model_selection(path=str(MODEL_SELECTION_PATH))
        if single_models:
            enabled_models = [m for m in single_models if m.get("enabled", True)]
            if verbose:
                print(f"  平均: {'开' if average_enabled else '关'}, 投票: {'开' if voting_enabled else '关'}, 模型数: {len(enabled_models)}")
            single_models = enabled_models
        if not single_models:
            config_module = __import__(config_name, fromlist=["config"])
            config = config_module.config.copy()
            output_dir = config.get("output_dir", "./model/default")
            model_info = find_best_model(output_dir)
            if not model_info:
                print("错误: 未找到可用模型")
                return None
            exp_dir, model_file, score = model_info
            single_models = [{"exp_dir": exp_dir, "model_file": model_file, "enabled": True}]
            average_enabled = False
            voting_enabled = False
            if verbose:
                print(f"  单模型: {_make_model_key((exp_dir,))} (score={score:.4f})")

        if not single_models:
            print("错误: 无可用模型")
            return None

        if verbose:
            print(f"\n[3/4] 加载数据并缓存...")

        model_types = set(m.get("type", "dl") for m in single_models)
        has_ml = bool(model_types & {"xgb", "lightgbm", "catboost"})
        has_dl = bool(model_types & {"dl", "ensemble_folds"})

        cached_data = cached_features = None
        ml_cached_data = None

        if has_dl:
            first_model = single_models[0]
            scaler_path = os.path.join(first_model["exp_dir"], "scaler.pkl")
            config_path = os.path.join(first_model["exp_dir"], "config.json")
            with open(config_path, "r") as f:
                first_config = json.load(f)
            feature_num = first_config["feature_num"]

            if verbose:
                print(f"  加载 DL 数据...")
            cached_data, cached_features = ETFBacktester.load_data_once(
                data_path=str(DATA_FILE), scaler_path=scaler_path, feature_num=feature_num, verbose=False,
            )

        if has_ml:
            if verbose:
                print(f"  加载 ML 数据...")
            ml_cached_data = MLBacktester.load_data_once(
                data_path=str(DATA_FILE),
                add_cs_features=True,
            )

        single_backtesters = []
        ensemble_models = []
        for m in single_models:
            mtype = m.get("type", "dl")
            if mtype == "ensemble_folds":
                ensemble_models.append(m)
                continue
            if mtype in ("xgb", "lightgbm", "catboost"):
                bt = MLBacktester.from_cached_data(
                    model_dir=m["exp_dir"],
                    cached_data=ml_cached_data,
                    model_file=m["model_file"],
                    verbose=False,
                )
            else:
                bt = ETFBacktester.from_cached_data(
                    model_dir=m["exp_dir"], cached_data=cached_data, cached_features=cached_features,
                    device=device, model_file=m["model_file"], verbose=False,
                )
            single_backtesters.append((m, bt))

        if verbose:
            print(f"\n[4/4] 生成预测信号...")

        all_predictions = {}

        # 生成所有特征日的预测，按特征日日期存储
        # BacktestEngine 根据 trade_mode 决定查哪天的预测:
        #   "open": 调仓日 D 查 D-1（前日特征）; "close": 调仓日 D 查 D（当日特征）
        backtest_dates_str_set = {d.strftime("%Y-%m-%d") for d in backtest_dates}
        seed_dates = set(backtest_dates)
        prev_idx = all_dates.index(backtest_dates[0]) - 1
        if prev_idx >= 0:
            seed_dates.add(all_dates[prev_idx])

        # 单模型预测
        for m, bt in single_backtesters:
            model_key = _make_model_key(m)
            if verbose:
                print(f"  预测: {model_key}...")
            preds_for_model = {}
            for d in sorted(seed_dates):
                preds = bt._get_predictions(d)
                if preds:
                    preds_for_model[d.strftime("%Y-%m-%d")] = preds
            all_predictions[model_key] = preds_for_model
            if verbose:
                only_rebalance = [k for k in preds_for_model if k in backtest_dates_str_set]
                print(f"    → {len(preds_for_model)} 日预测 ({len(only_rebalance)} 调仓日)")

        # 平均模型预测
        if average_enabled and len(single_backtesters) >= 2:
            if verbose:
                print(f"  预测: average ({len(single_backtesters)}个模型)...")
            avg_preds = {}
            for d in sorted(seed_dates):
                d_str = d.strftime("%Y-%m-%d")
                all_scores = []
                for _, bt in single_backtesters:
                    preds = bt._get_predictions(d)
                    if preds is None:
                        all_scores = None
                        break
                    all_scores.append({p["stock_id"]: p["rank"] for p in preds})
                if all_scores:
                    avg_map = {}
                    for sid in all_scores[0].keys():
                        ranks = [sd[sid] for sd in all_scores]
                        avg_map[sid] = float(np.mean(ranks))
                    sorted_stocks = sorted(avg_map.items(), key=lambda x: x[1])
                    avg_preds[d_str] = [{"rank": i+1, "stock_id": sid, "score": -sc} for i, (sid, sc) in enumerate(sorted_stocks)]
            all_predictions["average"] = avg_preds
            if verbose:
                print(f"    → {len(avg_preds)} 个交易日")

        # 投票模型预测
        if voting_enabled and len(single_backtesters) >= 2:
            if verbose:
                print(f"  预测: voting ({len(single_backtesters)}个模型)...")
            vote_preds = {}
            for d in sorted(seed_dates):
                d_str = d.strftime("%Y-%m-%d")
                all_preds = []
                for _, bt in single_backtesters:
                    preds = bt._get_predictions(d)
                    if preds is None:
                        all_preds = None
                        break
                    all_preds.append(preds)
                if all_preds:
                    freq = {}
                    avg_score = {}
                    for preds in all_preds:
                        for i, p in enumerate(preds):
                            if i >= top_k * 3:
                                break
                            sid = p["stock_id"]
                            freq[sid] = freq.get(sid, 0) + 1
                            avg_score[sid] = avg_score.get(sid, 0) + p["score"]
                    for sid in avg_score:
                        avg_score[sid] /= freq.get(sid, 1)
                    ranked = sorted(freq.items(), key=lambda x: (-x[1], -avg_score.get(x[0], 0)))[:max(top_k, 10)]
                    result = [{"rank": i+1, "stock_id": sid, "score": float(freq)} for i, (sid, freq) in enumerate(ranked)]
                    vote_preds[d_str] = result
            all_predictions["voting"] = vote_preds
            if verbose:
                print(f"    → {len(vote_preds)} 个交易日")

        # Fold 集成平均预测
        if ensemble_models:
            for m in ensemble_models:
                model_key = _make_model_key(m) + "_ensemble"
                if verbose:
                    print(f"  预测: {model_key} ({len(ensemble_models)} 个 fold)...")
                fold_dirs = sorted([
                    d for d in os.listdir(m["exp_dir"])
                    if d.startswith("fold_") and os.path.isdir(os.path.join(m["exp_dir"], d))
                ])
                fold_bts = []
                for fd in fold_dirs:
                    fold_dir = os.path.join(m["exp_dir"], fd)
                    bt = ETFBacktester.from_cached_data(
                        model_dir=fold_dir, cached_data=cached_data, cached_features=cached_features,
                        device=device, model_file=m["model_file"], verbose=False,
                    )
                    fold_bts.append(bt)
                ens_preds = {}
                for d in sorted(seed_dates):
                    d_str = d.strftime("%Y-%m-%d")
                    all_scores = []
                    for bt in fold_bts:
                        preds = bt._get_predictions(d)
                        if preds is None:
                            all_scores = None
                            break
                        all_scores.append({p["stock_id"]: p["score"] for p in preds})
                    if all_scores:
                        avg_map = {}
                        for sid in all_scores[0].keys():
                            scores = [sd[sid] for sd in all_scores]
                            avg_map[sid] = float(np.mean(scores))
                        sorted_stocks = sorted(avg_map.items(), key=lambda x: x[1], reverse=True)
                        ens_preds[d_str] = [{"rank": i+1, "stock_id": sid, "score": sc} for i, (sid, sc) in enumerate(sorted_stocks)]
                all_predictions[model_key] = ens_preds
                if verbose:
                    print(f"    → {len(ens_preds)} 个交易日")

        # 保存（附带交易日历元信息，供掘金策略对齐调仓日）
        all_predictions["_meta"] = {
            "start_date": start_date,
            "backtest_dates": [d.strftime("%Y-%m-%d") for d in backtest_dates],
        }
        with open(PREDICTIONS_PATH, "w", encoding="utf-8") as f:
            json.dump(all_predictions, f, ensure_ascii=False, indent=2, cls=NumpyEncoder)

        # 同步保存一份到实盘文件夹
        live_pred_path = Path("juejin/live/predictions.json")
        try:
            live_pred_path.parent.mkdir(parents=True, exist_ok=True)
            with open(live_pred_path, "w", encoding="utf-8") as f:
                json.dump(all_predictions, f, ensure_ascii=False, indent=2, cls=NumpyEncoder)
            if verbose:
                print(f"  → 同步至 {live_pred_path}")
        except Exception as e:
            print(f"  [警告] 同步至实盘文件夹失败: {e}")

        if verbose:
            total = sum(len(v) for v in all_predictions.values())
            print(f"\n[完成] 预测信号已保存至 {PREDICTIONS_PATH}")
            print(f"  模型数: {len(all_predictions)}, 总预测条目: {total}")
            print(f"{'='*60}")

        return all_predictions

    except Exception as e:
        print(f"\n[错误] 生成预测失败: {e}")
        traceback.print_exc()
        return None


# ============================================================
# 模式2: 从已保存的预测信号运行回测并生成日报（无需加载模型）
# ============================================================

def run_from_predictions(
    update_data: bool = False,
    top_k: int = 3,
    verbose: bool = True,
    start_date: str = "2026-04-01",
    rebalance_days: int = 5,
    position_pct: float = 0.95,
    weight_strategy: str = "equal",
    strategy_params: dict = None,
    initial_capital: float = 100000,
    config_name: str = "config",
    trade_mode: str = "open",
):
    """模式2: 读取已保存的 predictions.json，跳过模型推理直接回测+日报"""
    from send_report import build_report_html, send_report
    import time as _time

    _t_start = _time.time()
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

    if verbose:
        print(f"\n{'='*60}")
        print(f"[{timestamp}] 从已保存预测信号生成日报")
        print(f"{'='*60}")

    if not PREDICTIONS_PATH.exists():
        print(f"错误: 未找到预测信号文件 {PREDICTIONS_PATH}")
        print("请先运行 daily_eval --predictions-only 或完整的 daily_eval")
        return None

    try:
        # 1. 加载预测信号
        if verbose:
            print(f"[1/4] 加载预测信号...")
        all_predictions = _load_predictions()
        _t_load = _time.time()
        if verbose:
            print(f"  [TIMING] 加载预测信号: {_t_load - _t_start:.2f}s")
        for key in all_predictions:
            if verbose:
                print(f"  {key}: {len(all_predictions[key])} 个日期")

        # 2. 更新数据
        if update_data:
            if verbose:
                print(f"[2/4] 获取最新ETF数据...")
            update_etf_data(verbose=verbose)

        raw_df = load_etf_data(DATA_FILE, dtype={"股票代码": str})
        raw_df["股票代码"] = raw_df["股票代码"].astype(object).str.zfill(6)
        raw_df["日期"] = pd.to_datetime(raw_df["日期"])
        close_pivot, hl_pivot, ll_pivot = _build_pivots(raw_df)
        latest_date = raw_df["日期"].max()
        end_date = (latest_date + pd.Timedelta(days=1)).strftime("%Y-%m-%d")
        _t_data = _time.time()
        if verbose:
            print(f"  [TIMING] 数据加载+pivot: {_t_data - _t_load:.2f}s")

        if verbose:
            print(f"[3/4] 运行回测...")

        sequences = {}
        config = load_full_config()

        for model_key, preds_dict in all_predictions.items():
            _t_model = _time.time()
            if verbose:
                print(f"  回测: {model_key}...")
            pred_func = _make_predictions_func_from_saved(preds_dict)
            result = run_backtest_sequence(
                predictions_func=pred_func,
                data_file=str(DATA_FILE),
                start_date=start_date,
                end_date=end_date,
                top_k=top_k,
                rebalance_days=rebalance_days,
                position_pct=position_pct,
                weight_strategy=weight_strategy,
                strategy_params=strategy_params,
                initial_capital=initial_capital,
                commission=config.get("commission", 0.0003),
                slippage=config.get("slippage", 0.001),
                trade_mode=trade_mode,
                risk_manager_config=config.get("risk_manager", {}),
            )
            sequences[model_key] = result
            _t_model_end = _time.time()
            if verbose:
                print(f"    [TIMING] 回测 {model_key}: {_t_model_end - _t_model:.2f}s")
        _t_backtest = _time.time()
        if verbose:
            print(f"  [TIMING] 全部回测完成: {_t_backtest - _t_data:.2f}s")

        if not sequences:
            print("错误: 回测未产生任何结果")
            return None

        # 注入最新预测到 predictions_history（日报预测信号表需要最新数据，不是调仓日数据）
        _latest_pred_date = latest_date.strftime("%Y-%m-%d")
        for model_key, preds_dict in all_predictions.items():
            seq = sequences.get(model_key)
            if not seq:
                continue
            _found = _latest_pred_date if preds_dict.get(_latest_pred_date) else None
            if not _found:
                _all_dates = sorted(preds_dict.keys())
                for d in reversed(_all_dates):
                    if d <= _latest_pred_date:
                        _found = d
                        break
            if _found:
                latest_preds = preds_dict[_found]
                ph = seq.get("predictions_history", [])
                _sp_snapshot = dict(strategy_params) if strategy_params else {}
                _sp_snapshot.pop("vol_dict", None)
                if weight_strategy in ("risk_parity", "score_risk", "score_risk_v1", "kelly"):
                    top_ids = [p["stock_id"] for p in latest_preds[:top_k]]
                    _vol_window = _sp_snapshot.get("vol_window", 20)
                    _vd = compute_volatility(raw_df, top_ids, _found, _vol_window)
                    if _vd:
                        _sp_snapshot["vol_dict"] = _vd
                _entry = {
                    "date": _found,
                    "predictions": latest_preds[:10],
                    "strategy_params": dict(_sp_snapshot),
                }
                if not ph or ph[-1]["date"] != _found:
                    ph.append(_entry)
                else:
                    # 更新已有 entry 的 strategy_params（补 vol_dict）
                    ph[-1].update({k: v for k, v in _entry.items() if k != "date"})

        # 在从预测信号模式中计算投票总模型数
        _voting_n = None
        if "voting" in sequences:
            _special = {"average", "voting", "juejin"}
            _voting_n = len([k for k in all_predictions if k not in _special])
            for ph in sequences["voting"].get("predictions_history", []):
                ph["voting_total_models"] = _voting_n

        # 选取主序列（遵循 model_selection.yaml 的 master 设置）
        report_key = _resolve_report_key(sequences)
        if not report_key:
            print("错误: 无法确定主序列")
            return None
        report_data = sequences[report_key]
        latest_date_str = latest_date.strftime("%Y-%m-%d")

        if verbose:
            print(f"[4/4] 生成并发送日报...")

        # 绘图
        plot_path = OUTPUT_DIR / "equity_curves.png"
        try:
            plot_equity_curves(sequences, str(DATA_FILE), initial_capital, str(plot_path))
        except Exception as e:
            if verbose:
                print(f"  [图表] 保存失败: {e}")

        # 当前价格
        today_actual_trades = [t for t in report_data["trades"] if t["date"] == latest_date_str and t["action"] in ("买入", "卖出")]
        is_rebalance_day = len(today_actual_trades) > 0
        if not is_rebalance_day:
            all_dates = sorted(raw_df["日期"].unique())
            start_idx = next((i for i, d in enumerate(all_dates) if d >= pd.Timestamp(start_date)), None)
            today_idx = next((i for i, d in enumerate(all_dates) if d >= pd.Timestamp(latest_date_str)), None)
            if start_idx is not None and today_idx is not None and (today_idx - start_idx) % rebalance_days == 0:
                is_rebalance_day = True
        pnl_positions = {p["stock_id"]: p for p in report_data.get("today_pnl", {}).get("positions", [])}

        # 加载 ETF 名称
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

        # 收盘交易模式下，调仓日显示调仓前持仓（新持仓明天再出现）
        if trade_mode == "close" and is_rebalance_day:
            display_positions = report_data.get("pre_rebalance_positions", {})
            if not display_positions:
                display_positions = report_data["positions"]
        else:
            display_positions = report_data["positions"]

        # 构造持仓
        trade_dates = sorted(set(t["date"] for t in report_data.get("trades", [])))
        rb_ref_date = None
        if latest_date_str in trade_dates:
            idx = trade_dates.index(latest_date_str)
            if idx > 0:
                rb_ref_date = trade_dates[idx - 1]
        else:
            for d in reversed(trade_dates):
                if d < latest_date_str:
                    rb_ref_date = d
                    break
        rb_close_prices = {}
        if rb_ref_date:
            rb_ts = pd.Timestamp(rb_ref_date)
            for stock_id in display_positions:
                pc = close_pivot.loc[rb_ts, stock_id] if stock_id in close_pivot.columns else 0
                if pd.notna(pc):
                    rb_close_prices[stock_id] = float(pc)
        holdings = []
        for stock_id, pos in display_positions.items():
            price = pnl_positions.get(stock_id, {}).get("today_close", 0)
            hl = hl_pivot.loc[latest_date, stock_id] if stock_id in hl_pivot.columns else 0
            ll = ll_pivot.loc[latest_date, stock_id] if stock_id in ll_pivot.columns else 0
            if stock_id in rb_close_prices:
                buy_price = rb_close_prices[stock_id]
                buy_date = rb_ref_date
            else:
                buy_price = 0
                buy_date = ""
                for t in report_data.get("trades", []):
                    if t["action"] == "买入" and t["stock"] == stock_id and t["date"] > buy_date:
                        buy_price = t["price"]
                        buy_date = t["date"]
                if not buy_price:
                    buy_price = (pos.get("cost", 0) / pos.get("shares", 1)) if pos.get("shares", 0) > 0 else 0
            holdings.append({
                "stock_id": stock_id,
                "name": etf_names.get(stock_id, ""),
                "price": price,
                "buy_price": round(buy_price, 4),
                "buy_date": buy_date,
                "shares": pos["shares"],
                "cost": pos["cost"],
                "high_limit": 0 if pd.isna(hl) else round(float(hl), 4),
                "low_limit": 0 if pd.isna(ll) else round(float(ll), 4),
            })

        # 今日调仓（全部序列）
        all_today_trades = []
        for key, seq in sequences.items():
            for t in seq.get("trades", []):
                if t["date"] == latest_date_str and t["action"] in ("买入", "卖出"):
                    t["model_key"] = key
                    all_today_trades.append(t)

        # 构造 report JSON
        hs300_raw = raw_df[raw_df["股票代码"] == "510300.XSHG"].sort_values("日期")
        hs300_curve = []
        if not hs300_raw.empty:
            hs300_period = hs300_raw[(hs300_raw["日期"] >= pd.Timestamp(start_date)) & (hs300_raw["日期"] < pd.Timestamp(end_date))]
            if not hs300_period.empty:
                hs300_start_price = float(hs300_period["收盘"].iloc[0])
                for _, row in hs300_period.iterrows():
                    hs300_curve.append({
                        "date": row["日期"].strftime("%Y-%m-%d"),
                        "total_value": round(float(row["收盘"]) / hs300_start_price * initial_capital, 2),
                    })

        sequences_summary = {}
        for key, seq in sequences.items():
            seq_pnl = {p["stock_id"]: p for p in seq.get("today_pnl", {}).get("positions", [])}
            seq_current_prices = {sid: p["today_close"] for sid, p in seq_pnl.items() if p.get("today_close", 0) > 0}
            model_stats = _compute_model_stats(seq["trades"], seq_current_prices, report_date=latest_date_str)
            model_stats["distinct_topk"] = _compute_distinct_topk(seq.get("predictions_history", []), top_k)
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
                "skipped_trades": seq.get("skipped_trades", []),
                "predictions_history": seq.get("predictions_history", []),
            }

        today_pnl_data = report_data.get("today_pnl", {})
        pre_holdings = []
        pre_positions = report_data.get("pre_rebalance_positions", {})
        for stock_id, pos in pre_positions.items():
            price = close_pivot.loc[latest_date, stock_id] if stock_id in close_pivot.columns else 0
            price = 0 if pd.isna(price) else float(price)
            pre_holdings.append({
                "stock_id": stock_id,
                "name": etf_names.get(stock_id, ""),
                "price": price,
                "price_display": price,
                "shares": pos["shares"],
                "cost": pos.get("cost", 0),
            })
        report = {
            "date": latest_date_str,
            "is_rebalance_day": is_rebalance_day,
            "next_rebalance_date": report_data["metrics"].get("next_rebalance_date", ""),
            "today_trades": today_actual_trades,
            "all_today_trades": all_today_trades,
            "metrics": report_data["metrics"],
            "holdings": holdings,
            "pre_holdings": pre_holdings,
            "cash": report_data["cash"],
            "total_value": report_data["metrics"]["latest_value"],
            "sequences": sequences_summary,
            "hs300_curve": hs300_curve,
            "trade_mode": trade_mode,
            "weight_strategy": weight_strategy,
            "strategy_params": strategy_params,
            "top_k": top_k,
            "position_pct": position_pct,
            "voting_total_models": _voting_n if "voting" in sequences else None,
            "strategy_info": _format_strategy_info(
                weight_strategy, strategy_params, top_k, position_pct,
                config.get("commission", 0.0003), config.get("slippage", 0.001),
                rebalance_days,
            ),
        }

        with open(REPORT_PATH, "w", encoding="utf-8") as f:
            json.dump(report, f, ensure_ascii=False, indent=2, cls=NumpyEncoder)

        # 保存 state
        state = {
            "sequences": sequences,
            "last_updated": timestamp,
        }
        with open(STATE_PATH, "w", encoding="utf-8") as f:
            json.dump(state, f, ensure_ascii=False, indent=2, cls=NumpyEncoder)
        _t_report = _time.time()
        if verbose:
            print(f"  [TIMING] 报表构建+序列化: {_t_report - _t_backtest:.2f}s")

        # 发送邮件
        try:
            if verbose:
                _dn = {"average": "平均", "voting": "投票"}
                if report_key == "juejin":
                    _rk = next((k for k in sequences if k not in ("juejin", "average", "voting")), report_key)
                    model_display = _dn.get(_rk, _rk.replace("search_", "").replace("_exp_", " ").replace("_full", " (full)"))
                else:
                    model_display = _dn.get(report_key, report_key.replace("search_", "").replace("_exp_", " ").replace("_full", " (full)"))
                print(f"\n[邮件] 发送报告: {model_display}...")
            send_report(model_key=report_key)
        except Exception as e:
            if verbose:
                print(f"\n[邮件] 发送失败: {e}")
        _t_email = _time.time()
        if verbose:
            print(f"  [TIMING] 发送邮件: {_t_email - _t_report:.2f}s")

        # 历史报告
        try:
            _save_history_reports(sequences[report_key], sequences, str(DATA_FILE), initial_capital, etf_names, model_key=report_key, rebalance_days=rebalance_days, trade_mode=trade_mode, top_k=top_k, position_pct=position_pct, weight_strategy=weight_strategy, strategy_params=strategy_params)
        except Exception as e:
            print(f"\n[历史] 生成失败: {e}")
            traceback.print_exc()
        _t_history = _time.time()
        if verbose:
            print(f"  [TIMING] 历史报告: {_t_history - _t_email:.2f}s")

        if verbose:
            m = report_data["metrics"]
            print(f"\n{'='*60}")
            print(f"  日报 ({latest_date_str}) [序列: {report_key}]")
            print(f"  今日盈亏: {today_pnl_data.get('total_pnl', 0):+.2f}")
            print(f"  累计收益: {m['strategy_return_pct']:+.2f}%")
            print(f"  账户总值: {m['latest_value']:,.2f}")
            print(f"{'='*60}")
            print(f"  [TIMING] 总计: {_time.time() - _t_start:.2f}s")

        return state

    except Exception as e:
        print(f"\n[错误] 从预测信号生成日报失败: {e}")
        traceback.print_exc()
        return None


# ============================================================
# 模式3: 从已保存的 juejin_state.json 生成日报（无模型无回测）
# ============================================================

def run_from_juejin(verbose=True, start_date="2026-04-01", initial_capital=100000, trade_mode="open", rebalance_days=5):
    """读取 juejin_state.json 直接生成日报，不执行任何模型或回测。"""
    from send_report import send_report

    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    cfg = load_full_config()

    if not JUEJIN_STATE_PATH.exists():
        print(f"错误: 未找到 {JUEJIN_STATE_PATH}")
        print("请先运行 juejin/main.py 生成掘金回测状态")
        return None

    if verbose:
        print(f"\n{'='*60}")
        print(f"[{timestamp}] 从 {JUEJIN_STATE_PATH.name} 生成日报")
        print(f"{'='*60}")

    try:
        with open(JUEJIN_STATE_PATH, "r", encoding="utf-8") as f:
            state = json.load(f)

        sequences = state.get("sequences", {})
        if not sequences:
            print("错误: juejin_state.json 中无序列数据")
            return None

        if verbose:
            print(f"  加载掘金回测结果: {JUEJIN_STATE_PATH}")

        # 选主序列（遵循 model_selection.yaml 的 master 设置）
        report_key = _resolve_report_key(sequences)
        if not report_key:
            print("错误: 无法确定主序列")
            return None
        report_data = sequences[report_key]

        # 加载价格数据
        raw_df = load_etf_data(DATA_FILE, dtype={"股票代码": str})
        raw_df["股票代码"] = raw_df["股票代码"].astype(object).str.zfill(6)
        raw_df["日期"] = pd.to_datetime(raw_df["日期"])
        close_pivot, hl_pivot, ll_pivot = _build_pivots(raw_df)
        latest_date = raw_df["日期"].max()
        latest_date_str = latest_date.strftime("%Y-%m-%d")
        end_date = (latest_date + pd.Timedelta(days=1)).strftime("%Y-%m-%d")

        # 加载 ETF 名称
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

        # 今日调仓
        all_today_trades = []
        for key, seq in sequences.items():
            for t in seq.get("trades", []):
                if t["date"] == latest_date_str and t["action"] in ("买入", "卖出"):
                    t["model_key"] = key
                    all_today_trades.append(t)
        is_rebalance_day = len(all_today_trades) > 0
        if not is_rebalance_day:
            all_dates = sorted(raw_df["日期"].unique())
            start_idx = next((i for i, d in enumerate(all_dates) if d >= pd.Timestamp(start_date)), None)
            today_idx = next((i for i, d in enumerate(all_dates) if d >= pd.Timestamp(latest_date_str)), None)
            if start_idx is not None and today_idx is not None and (today_idx - start_idx) % rebalance_days == 0:
                is_rebalance_day = True

        # 收盘交易模式下，调仓日显示调仓前持仓（新持仓明天再出现）
        if trade_mode == "close" and is_rebalance_day:
            display_positions = report_data.get("pre_rebalance_positions", {})
            if not display_positions:
                display_positions = report_data.get("positions", {})
        else:
            display_positions = report_data.get("positions", {})

        # 构造当前持仓
        pnl_positions = {p["stock_id"]: p for p in report_data.get("today_pnl", {}).get("positions", [])}
        # 取最近一次调仓日收盘价作为本期调仓参考
        trade_dates = sorted(set(t["date"] for t in report_data.get("trades", [])))
        rb_ref_date = None
        if latest_date_str in trade_dates:
            idx = trade_dates.index(latest_date_str)
            if idx > 0:
                rb_ref_date = trade_dates[idx - 1]
        else:
            for d in reversed(trade_dates):
                if d < latest_date_str:
                    rb_ref_date = d
                    break
        rb_close_prices = {}
        if rb_ref_date:
            rb_ts = pd.Timestamp(rb_ref_date)
            for stock_id in display_positions:
                sub = raw_df[raw_df["股票代码"] == stock_id]
                pc_s = sub.loc[sub["日期"] == rb_ts, "收盘"]
                if not pc_s.empty:
                    rb_close_prices[stock_id] = float(pc_s.values[0])
        # 有持仓但不在rb_close_prices中→用平均成本
        for stock_id, pos in display_positions.items():
            if stock_id not in rb_close_prices:
                avg_cost = round(pos.get("cost", 0) / pos.get("shares", 1), 4) if pos.get("shares", 0) > 0 else 0
                if avg_cost > 0:
                    rb_close_prices[stock_id] = avg_cost
        holdings = []
        for stock_id, pos in display_positions.items():
            price = pnl_positions.get(stock_id, {}).get("today_close", 0)
            sub = raw_df[raw_df["股票代码"] == stock_id]
            tc_s = sub.loc[sub["日期"] == latest_date, "收盘"]
            if price == 0 and not tc_s.empty:
                price = float(tc_s.values[0])
            price_display_s = sub.loc[sub["日期"] == latest_date, "收盘"]
            price_display = float(price_display_s.values[0]) if not price_display_s.empty else price
            hl_s = sub.loc[sub["日期"] == latest_date, "涨停价"]
            ll_s = sub.loc[sub["日期"] == latest_date, "跌停价"]
            holdings.append({
                "stock_id": stock_id,
                "name": etf_names.get(stock_id, ""),
                "price": price,
                "buy_price": round(rb_close_prices.get(stock_id, 0), 4),
                "buy_date": rb_ref_date or "",
                "price_display": price_display,
                "buy_price_display": round(rb_close_prices.get(stock_id, 0), 4),
                "shares": pos["shares"],
                "cost": pos["cost"],
                "high_limit": round(float(hl_s.values[0]), 4) if not hl_s.empty else 0,
                "low_limit": round(float(ll_s.values[0]), 4) if not ll_s.empty else 0,
            })

        # 上期持仓（最近一次调仓前的持仓）
        pre_holdings = []
        pre_positions = report_data.get("pre_rebalance_positions", {})
        for stock_id, pos in pre_positions.items():
            price = close_pivot.loc[latest_date, stock_id] if stock_id in close_pivot.columns else 0
            price = 0 if pd.isna(price) else float(price)
            pre_holdings.append({
                "stock_id": stock_id,
                "name": etf_names.get(stock_id, ""),
                "price": price,
                "price_display": price,
                "shares": pos["shares"],
                "cost": pos.get("cost", 0),
            })

        # 构建 sequences_summary
        sequences_summary = {}
        _jj_top_k = cfg.get("top_k", 3)
        for key, seq in sequences.items():
            seq_pnl = {p["stock_id"]: p for p in seq.get("today_pnl", {}).get("positions", [])}
            seq_current_prices = {sid: p["today_close"] for sid, p in seq_pnl.items() if p.get("today_close", 0) > 0}
            model_stats = _compute_model_stats(seq.get("trades", []), seq_current_prices, report_date=latest_date_str)
            model_stats["distinct_topk"] = _compute_distinct_topk(seq.get("predictions_history", []), _jj_top_k)
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
                "skipped_trades": seq.get("skipped_trades", []),
                "predictions_history": seq.get("predictions_history", []),
            }

        # 注入最新预测到 predictions_history（掘金 state 可能没有最新日期的数据）
        try:
            with open(PREDICTIONS_PATH, "r") as _f_jj:
                _all_preds_jj = json.load(_f_jj)
            _all_preds_jj.pop("_meta", None)
            for _sk_jj, _sd_jj in sequences_summary.items():
                _ph_jj = _sd_jj.get("predictions_history", [])
                if not _ph_jj:
                    continue
                # 从 predictions.json 中找有最新日期的模型
                _latest_jj = latest_date_str
                _match_preds_jj = None
                for _mk_jj, _pd_jj in _all_preds_jj.items():
                    _mp = _pd_jj.get(_latest_jj)
                    if _mp:
                        _match_preds_jj = _mp
                        break
                if not _match_preds_jj:
                    for _mk_jj, _pd_jj in _all_preds_jj.items():
                        for _d_jj in reversed(sorted(_pd_jj.keys())):
                            if _d_jj <= _latest_jj and _pd_jj[_d_jj]:
                                _match_preds_jj = _pd_jj[_d_jj]
                                _latest_jj = _d_jj
                                break
                        if _match_preds_jj:
                            break
                if _match_preds_jj and (not _ph_jj or _ph_jj[-1].get("date") != _latest_jj):
                    _ph_jj.append({"date": _latest_jj, "predictions": _match_preds_jj[:10]})
        except Exception:
            pass

        # HS300 基准曲线
        hs300_curve = []
        hs300_raw = raw_df[raw_df["股票代码"] == "510300.XSHG"].sort_values("日期")
        if not hs300_raw.empty:
            hs300_period = hs300_raw[(hs300_raw["日期"] >= pd.Timestamp(start_date)) & (hs300_raw["日期"] < pd.Timestamp(end_date))]
            if not hs300_period.empty:
                hs300_start_price = float(hs300_period["收盘"].iloc[0])
                for _, row in hs300_period.iterrows():
                    hs300_curve.append({
                        "date": row["日期"].strftime("%Y-%m-%d"),
                        "total_value": round(float(row["收盘"]) / hs300_start_price * initial_capital, 2),
                    })

        # 补齐 report_key 序列的缺失数据（today_pnl / cash / 完整 metrics）
        rk_seq = sequences_summary[report_key]
        if not rk_seq.get("today_pnl", {}).get("total_pnl"):
            _total_pnl = 0
            _positions = []
            yesterday = raw_df[raw_df["日期"] < pd.Timestamp(latest_date_str)]["日期"].max()
            for h in holdings:
                code = h["stock_id"]
                sub = raw_df[raw_df["股票代码"] == code]
                tc = sub.loc[sub["日期"] == pd.Timestamp(latest_date_str), "收盘"]
                yc = sub.loc[sub["日期"] == yesterday, "收盘"]
                if not tc.empty and not yc.empty:
                    _pnl = round(h["shares"] * (float(tc.values[0]) - float(yc.values[0])), 2)
                    _total_pnl += _pnl
                    _positions.append({
                        "stock_id": code,
                        "today_close": float(tc.values[0]),
                        "pnl": _pnl,
                        "return_pct": round((float(tc.values[0]) / float(yc.values[0]) - 1) * 100, 2),
                    })
            rk_seq["today_pnl"] = {"total_pnl": _total_pnl, "positions": _positions}

        if not rk_seq.get("cash"):
            _total_mkt_val = sum(h["shares"] * h["price"] for h in holdings if h["price"])
            _total_value = sequences_summary[report_key].get("metrics", {}).get("latest_value", 0)
            rk_seq["cash"] = round(_total_value - _total_mkt_val, 2)

        _rk_metrics = rk_seq.get("metrics", {})
        _ec = rk_seq.get("equity_curve", [])
        # 始终从 equity curve 重算策略收益（state 中的值可能来自旧版本）
        if _ec and len(_ec) >= 2:
            _rk_metrics["strategy_return_pct"] = round((_ec[-1]["total_value"] / _ec[0]["total_value"] - 1) * 100, 2)
        if not _rk_metrics.get("total_days") and _ec:
            _rk_metrics["total_days"] = len(_ec)
            _daily_rets = []
            for i in range(1, len(_ec)):
                _r = (_ec[i]["total_value"] / _ec[i - 1]["total_value"] - 1)
                _daily_rets.append(_r)
            if _daily_rets:
                _rk_metrics["daily_win_rate"] = round(sum(1 for r in _daily_rets if r > 0) / len(_daily_rets), 4)
                _rk_metrics["annualized_volatility_pct"] = round(np.std(_daily_rets) * np.sqrt(252) * 100, 4)
                _rk_metrics["calmar_ratio"] = round(_rk_metrics.get("annualized_return_pct", 0) / 100 / max(abs(_rk_metrics.get("max_drawdown_pct", 1) / 100), 0.01), 2) if _rk_metrics.get("annualized_return_pct") else 0
        if not _rk_metrics.get("hs300_return_pct") and _ec:
            _hs_r = raw_df[raw_df["股票代码"] == "510300.XSHG"].sort_values("日期")
            if not _hs_r.empty:
                _hs_start = _hs_r[_hs_r["日期"] == pd.Timestamp(_ec[0]["date"])]
                _hs_end = _hs_r[_hs_r["日期"] == pd.Timestamp(_ec[-1]["date"])]
                if not _hs_start.empty and not _hs_end.empty:
                    _hs_sp = float(_hs_start["收盘"].iloc[0])
                    _hs_ep = float(_hs_end["收盘"].iloc[0])
                    if _hs_sp > 0:
                        _rk_metrics["hs300_return_pct"] = round((_hs_ep / _hs_sp - 1) * 100, 2)
                if "hs300_return_pct" not in _rk_metrics:
                    _rk_metrics["hs300_return_pct"] = 0
                _sr = _rk_metrics.get("strategy_return_pct", 0)
                _rk_metrics["excess_return_pct"] = round(_sr - _rk_metrics["hs300_return_pct"], 2)
        # 补齐长期风险指标（从 equity curve 重算）
        if _ec and len(_ec) >= 5:
            _ec_vals = [e["total_value"] for e in _ec]
            _ec_dates = [e["date"] for e in _ec]
            _r_cum = np.array(_ec_vals) / _ec_vals[0]
            _r_rets = np.diff(_ec_vals) / np.array(_ec_vals[:-1])
            _r_dd_periods = extract_drawdowns(_ec_vals, _ec_dates)
            _r_risk = compute_longterm_risk_metrics(_r_rets, _r_cum, _ec_dates, _r_dd_periods)
            for _rk, _rv in _r_risk.items():
                if _rk not in _rk_metrics:
                    _rk_metrics[_rk] = _rv
        rk_seq["metrics"] = _rk_metrics
        # 同步到 report 根层
        report_data["metrics"] = _rk_metrics

        # 补齐 next_rebalance_date（掘金序列可能没有）
        next_rebalance_date = report_data.get("metrics", {}).get("next_rebalance_date", "")
        if not next_rebalance_date:
            all_dates = sorted(raw_df["日期"].unique())
            rb_dates = state.get("rebalance_dates", [])
            if rb_dates:
                for d in rb_dates:
                    if d >= latest_date_str:
                        next_rebalance_date = d
                        break
                if not next_rebalance_date:
                    last_rb = rb_dates[-1]
                    last_idx = next((i for i, d in enumerate(all_dates) if d >= pd.Timestamp(last_rb)), None)
                    if last_idx is not None:
                        next_idx = last_idx + rebalance_days
                        if next_idx < len(all_dates):
                            next_rebalance_date = all_dates[next_idx].strftime("%Y-%m-%d")
                        else:
                            # 数据不足，用 pandas_market_calendars 推算
                            try:
                                import pandas_market_calendars as mcal
                                xshg = mcal.get_calendar("XSHG")
                                look_end = pd.Timestamp(last_rb) + pd.Timedelta(days=rebalance_days * 7)
                                cal_dates = xshg.valid_days(start_date=pd.Timestamp(last_rb), end_date=look_end, tz=None)
                                rb_pos = cal_dates.get_loc(pd.Timestamp(last_rb).normalize())
                                next_pos = rb_pos + rebalance_days
                                if next_pos < len(cal_dates):
                                    next_rebalance_date = cal_dates[next_pos].strftime("%Y-%m-%d")
                            except Exception:
                                pass
            if not next_rebalance_date:
                start_idx = next((i for i, d in enumerate(all_dates) if d >= pd.Timestamp(start_date)), None)
                today_idx = next((i for i, d in enumerate(all_dates) if d >= pd.Timestamp(latest_date_str)), None)
                if start_idx is not None and today_idx is not None:
                    for offset in range(0, len(all_dates) - start_idx + rebalance_days, rebalance_days):
                        rb_idx = start_idx + offset
                        if rb_idx >= len(all_dates):
                            from datetime import timedelta
                            next_rebalance_date = (all_dates[-1] + timedelta(days=1)).strftime("%Y-%m-%d")
                            break
                        if rb_idx > today_idx:
                            next_rebalance_date = all_dates[rb_idx].strftime("%Y-%m-%d")
                            break

        # 补齐 vol_dict（掘金 predictions_history 可能没有 strategy_params.vol_dict）
        _ws = cfg.get("weight_strategy", "equal")
        _tk = cfg.get("top_k", 3)
        _sp = dict(cfg.get("strategy_params", {}))
        if _ws in ("risk_parity", "score_risk", "score_risk_v1", "kelly") and "vol_dict" not in _sp:
            latest_ph_seq = None
            for entry in reversed(rk_seq.get("predictions_history", [])):
                if entry.get("predictions"):
                    latest_ph_seq = entry
                    break
            if latest_ph_seq:
                top_ids = [p["stock_id"] for p in latest_ph_seq["predictions"][:_tk]]
                _vol_window = _sp.get("vol_window", 20)
                vol_dict = compute_volatility(raw_df, top_ids, latest_date_str, _vol_window)
                if vol_dict:
                    _sp["vol_dict"] = vol_dict
        _pp = cfg.get("position_pct", 0.95)

        # 构建 latest_report.json
        source = "掘金" if report_key == "juejin" else "本地回测"
        report = {
            "date": latest_date_str,
            "is_rebalance_day": is_rebalance_day,
            "next_rebalance_date": next_rebalance_date,
            "today_trades": all_today_trades,
            "all_today_trades": all_today_trades,
            "metrics": report_data.get("metrics", {}),
            "holdings": holdings,
            "pre_holdings": pre_holdings,
            "cash": rk_seq.get("cash", report_data.get("cash", 0)),
            "total_value": report_data.get("metrics", {}).get("latest_value", 0),
            "sequences": sequences_summary,
            "hs300_curve": hs300_curve,
            "trade_mode": trade_mode,
            "source": source,
            "weight_strategy": _ws,
            "strategy_params": _sp,
            "top_k": _tk,
            "position_pct": _pp,
            "rebalance_dates": state.get("rebalance_dates", []),
            "strategy_info": _format_strategy_info(
                _ws, _sp, _tk, _pp,
                cfg.get("commission", 0.0003), cfg.get("slippage", 0.001),
                cfg.get("rebalance_days", 5),
            ),
        }

        with open(REPORT_PATH, "w", encoding="utf-8") as f:
            json.dump(report, f, ensure_ascii=False, indent=2, cls=NumpyEncoder)

        if verbose:
            print(f"\n[日报] latest_report.json 已保存")
            print(f"  日期: {latest_date_str}, 持仓: {len(holdings)}, 交易: {len(all_today_trades)}, 上期持仓: {len(pre_holdings)}")

        # 生成收益曲线图
        plot_path = OUTPUT_DIR / "equity_curves.png"
        try:
            plot_equity_curves(sequences, str(DATA_FILE), initial_capital, str(plot_path))
            if verbose:
                print(f"  [图表] 收益曲线已保存: {plot_path}")
        except Exception as e:
            if verbose:
                print(f"  [图表] 保存失败: {e}")

        # 发送邮件
        try:
            _dn = {"average": "平均", "voting": "投票"}
            if report_key == "juejin":
                _rk = next((k for k in sequences if k not in ("juejin", "average", "voting")), report_key)
                model_display = _dn.get(_rk, _rk.replace("search_", "").replace("_exp_", " ").replace("_full", " (full)"))
            else:
                model_display = _dn.get(report_key, report_key.replace("search_", "").replace("_exp_", " ").replace("_full", " (full)"))
            print(f"\n[邮件] 发送报告: {model_display}...")
            send_report(model_key=report_key, verbose=verbose)
        except Exception as e:
            if verbose:
                print(f"\n[邮件] 发送失败: {e}")

        return report

    except Exception as e:
        print(f"\n[错误] 从 juejin_state 生成日报失败: {e}")
        traceback.print_exc()
        return None


def sync_models_to_live():
    """将 config.yaml 中的启用模型复制到 juejin/live/ 目录（供掘金使用）"""
    import shutil
    import yaml
    from pathlib import Path

    config_path = PROJECT_ROOT / "config.yaml"
    with open(config_path, "r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f) or {}

    models = cfg.get("models", [])
    if not models:
        return

    live_root = PROJECT_ROOT / "juejin" / "live"
    live_root.mkdir(parents=True, exist_ok=True)

    for m in models:
        if not m.get("enabled", True):
            continue
        if m.get("type") == "ensemble_folds":
            continue
        model_rel = m["dir"]
        model_file = m.get("file", "")

        if model_rel.startswith("juejin"):
            continue

        src = PROJECT_ROOT / model_rel / model_file
        if not src.exists():
            print(f"[模型同步] 源文件不存在，跳过: {src}")
            continue

        parts = Path(model_rel).parts
        dst_rel = Path(*parts[1:]) if len(parts) > 1 else Path(model_rel)
        dst = live_root / dst_rel / model_file
        dst.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(src, dst)
        print(f"[模型同步] -> {dst}")


if __name__ == "__main__":
    import argparse
    import multiprocessing as mp

    parser = argparse.ArgumentParser(description="每日定时测评")
    # --- 模式选择（CLI 优先于 config.yaml）---
    parser.add_argument("--predictions-only", action="store_true", help="仅保存预测信号，不执行回测")
    parser.add_argument("--from-predictions", action="store_true", help="从已保存的预测信号生成日报（跳过模型加载）")
    parser.add_argument("--update-only", action="store_true", help="仅更新ETF数据，不执行任何其他操作")
    parser.add_argument("--from-juejin", action="store_true", help="从已保存的 juejin_state.json 生成日报（跳过模型和回测）")
    # --- 参数覆盖（默认 None，使用 config.yaml 的值）---
    parser.add_argument("--start-date", type=str, default=None, help="测评起始日期")
    parser.add_argument("--config", type=str, default=None, help="配置模块名")
    parser.add_argument("--no-update", action="store_true", help="跳过数据更新")
    parser.add_argument("--topk", type=int, default=None, help="Top-K推荐数量")
    parser.add_argument("--rebalance-days", type=int, default=None, help="调仓频率(天)")
    parser.add_argument("--position-pct", type=float, default=None, help="仓位比例")
    parser.add_argument("--debug", action="store_true", help="打印详细调试日志")
    parser.add_argument("--trade-mode", type=str, default=None, choices=["open", "close"], help="交易模式: open（开盘交易，用前日收盘特征）或 close（收盘交易，用当日收盘特征）")
    parser.add_argument("--weight-strategy", type=str, default=None, choices=["equal", "softmax", "rank_linear", "risk_parity", "score_risk", "score_risk_v1", "kelly", "liquidity"], help="加权策略")
    parser.add_argument("--weight-temperature", type=float, default=None, help="softmax 温度参数，存入 strategy_params['temperature']")
    parser.add_argument("--volatility-window", type=int, default=None, help="波动率计算回看天数，存入 strategy_params['vol_window']")
    parser.add_argument("--clear", action="store_true", help="先清除旧输出文件")
    args = parser.parse_args()

    # --- 加载 config.yaml（主配置源）---
    cfg = load_full_config()

    # --- CLI 参数覆盖 config.yaml ---
    def _cli(key, cfg_key=None):
        v = getattr(args, key, None)
        return v if v is not None else cfg.get(cfg_key or key)

    start_date = _cli("start_date")
    top_k = _cli("topk", "top_k")
    rebalance_days = _cli("rebalance_days")
    position_pct = _cli("position_pct")
    trade_mode = _cli("trade_mode")
    weight_strategy = _cli("weight_strategy")
    config_name = _cli("config") or "config"
    update_data = cfg.get("update_data", True) and not args.no_update
    initial_capital = cfg.get("initial_capital", 100000)

    # 策略参数
    strategy_params = dict(cfg.get("strategy_params", {}))
    if args.weight_temperature is not None:
        strategy_params["temperature"] = args.weight_temperature
    if args.volatility_window is not None:
        strategy_params["vol_window"] = args.volatility_window

    # --- 确定运行模式 ---
    mode = cfg.get("mode", "full")
    if args.update_only:
        mode = "update_only"
    elif args.predictions_only:
        mode = "predictions_only"
    elif args.from_predictions:
        mode = "from_predictions"
    elif args.from_juejin:
        mode = "from_juejin"

    if args.clear:
        import glob
        kept = {"config.yaml", "model_selection.yaml"}
        for fp in glob.glob(str(OUTPUT_DIR / "*")):
            if os.path.basename(fp) not in kept:
                if os.path.isdir(fp):
                    import shutil
                    shutil.rmtree(fp)
                else:
                    os.remove(fp)
        print(f"已清除 output/ 下旧文件（保留: {', '.join(sorted(kept))}）")

    mp.set_start_method("spawn", force=True)

    if mode == "update_only":
        update_etf_data(verbose=True)
        try:
            import pandas as pd
            _df = pd.read_csv(str(DATA_FILE))
            _dates = sorted(_df["日期"].unique())
            _last = _dates[-1]
            _prev = _dates[-2] if len(_dates) >= 2 else None
            _last_dt = pd.to_datetime(_last)
            print(f"\n数据文件: {DATA_FILE.name}")
            print(f"  股票数量: {_df['股票代码'].nunique()}")
            print(f"  总交易日: {len(_dates)}")
            print(f"  日期范围: {_dates[0]} ~ {_last}")
            _issues = []
            if _last_dt.weekday() >= 5:
                _issues.append(f"最新日期 {_last} 是{['周六','周日'][_last_dt.weekday()-5]}，非交易日")
            if _prev:
                _cur = _df[_df["日期"] == _last]
                _prv = _df[_df["日期"] == _prev]
                _merged = _cur[["股票代码","开盘","收盘","最高","最低"]].merge(
                    _prv[["股票代码","收盘"]].rename(columns={"收盘":"prev_close"}), on="股票代码")
                _all_flat = (_merged["收盘"] == _merged["开盘"]).all()
                _all_vs_prev = (_merged["收盘"] == _merged["prev_close"]).all()
                if _all_flat and _all_vs_prev:
                    _issues.append(f"最新日期 {_last} 所有 {len(_merged)} 只股票 open=close=high=low=昨收，数据无效")
                elif _all_flat:
                    _flat_pct = (_merged["收盘"] == _merged["开盘"]).mean() * 100
                    _issues.append(f"最新日期 {_last} {_flat_pct:.0f}% 的股票 open=close，数据可能不完整")
                if len(_merged) > 0:
                    _total_chg = (_merged["收盘"] - _merged["prev_close"]).abs().sum()
                    if _total_chg == 0:
                        _issues.append(f"最新日期 {_last} 所有股票收盘价和前一交易日完全相同，数据疑似无效")
            if _issues:
                print(f"  ⚠️ 数据警告:")
                for _msg in _issues:
                    print(f"    - {_msg}")
                print(f"  💡 建议等盘中或收盘后重新运行 --update-only")
        except Exception as _e:
            print(f"\n无法读取数据文件: {_e}")
        print("\n数据更新完成。")
    elif mode == "from_juejin":
        run_from_juejin(verbose=args.debug, start_date=start_date, trade_mode=trade_mode, initial_capital=initial_capital, rebalance_days=rebalance_days)
        sync_models_to_live()
    elif mode == "predictions_only":
        generate_predictions_only(
            config_name=config_name,
            top_k=top_k,
            verbose=args.debug,
            start_date=start_date,
            rebalance_days=rebalance_days,
            position_pct=position_pct,
        )
    elif mode == "from_predictions":
        run_from_predictions(
            top_k=top_k,
            verbose=args.debug,
            start_date=start_date,
            rebalance_days=rebalance_days,
            position_pct=position_pct,
            weight_strategy=weight_strategy,
            strategy_params=strategy_params,
            config_name=config_name,
            trade_mode=trade_mode,
        )
        sync_models_to_live()
    else:
        daily_eval(
            config_name=config_name,
            update_data=update_data,
            top_k=top_k,
            verbose=args.debug,
            start_date=start_date,
            rebalance_days=rebalance_days,
            position_pct=position_pct,
            weight_strategy=weight_strategy,
            strategy_params=strategy_params,
            trade_mode=trade_mode,
        )
        sync_models_to_live()
