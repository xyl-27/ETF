"""
每日定时测评脚本 - ETF (实盘模拟)
支持多模型回测 (单模型 + 融合)，持久化状态，生成日报
"""

import os
import sys
import json
import traceback
import subprocess
from datetime import datetime
from typing import Optional, Dict, List, Any, Tuple
from pathlib import Path

import numpy as np
import pandas as pd
import torch

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
        "fusion": "#e74c3c",
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
        linewidth = 3 if key == "fusion" else 1.5
        linestyle = "--" if key == "fusion" else "-"

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
MODEL_SELECTION_PATH = OUTPUT_DIR / "model_selection.txt"
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

def load_model_selection(path: str) -> Tuple[str, List[Dict]]:
    models = []
    mode = "single"
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith("#"):
                continue
            if line.startswith("mode:"):
                mode = line.split(":", 1)[1].strip()
            elif line.startswith("models:") or line.startswith("count:"):
                continue
            else:
                parts = line.split(",")
                if len(parts) == 3:
                    models.append({
                        "exp_dir": parts[0],
                        "model_file": parts[1],
                        "score": float(parts[2]),
                    })
    return mode, models


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
        trades.append({
            "date": t["date"].strftime("%Y-%m-%d") if isinstance(t["date"], pd.Timestamp) else str(t["date"]),
            "action": t["action"],
            "stock": t["stock"],
            "price": round(t["price"], 4),
            "shares": t["shares"],
            "amount": round(t["amount"], 2),
        })

    positions = {}
    for stock_id, pos in engine.positions.items():
        positions[stock_id] = {
            "shares": pos["shares"],
            "cost": round(pos["cost"], 2),
        }

    if equity_curve:
        ev = [ec["total_value"] for ec in equity_curve]
        strategy_return = (ev[-1] / initial_capital - 1) * 100
    else:
        strategy_return = 0.0

    hs300_code = "510300.XSHG"
    hs300_df = raw_df[raw_df["股票代码"] == hs300_code].copy()
    hs300_df = hs300_df[(hs300_df["日期"] >= start_ts) & (hs300_df["日期"] < end_ts)]
    if len(hs300_df) >= 1:
        hs300_return = (hs300_df["收盘"].iloc[-1] / hs300_df["收盘"].iloc[0] - 1) * 100
    else:
        hs300_return = 0.0
    excess_return = strategy_return - hs300_return

    if equity_curve:
        cumulative = np.array(ev) / initial_capital
        running_max = np.maximum.accumulate(cumulative)
        drawdown = (cumulative - running_max) / running_max * 100
        max_drawdown = float(abs(drawdown.min())) if len(drawdown) > 0 else 0.0
    else:
        max_drawdown = 0.0

    if len(equity_curve) >= 2:
        ev_arr = np.array(ev)
        daily_returns = np.diff(ev_arr) / ev_arr[:-1]
        sharpe = float((np.mean(daily_returns) / np.std(daily_returns)) * np.sqrt(252)) if np.std(daily_returns) != 0 else 0.0
    else:
        sharpe = 0.0

    metrics = {
        "strategy_return_pct": round(strategy_return, 4),
        "hs300_return_pct": round(hs300_return, 4),
        "excess_return_pct": round(excess_return, 4),
        "max_drawdown_pct": round(max_drawdown, 4),
        "sharpe_ratio": round(sharpe, 4),
        "start_date": start_date,
        "end_date": end_date,
        "latest_value": equity_curve[-1]["total_value"] if equity_curve else initial_capital,
        "total_days": len(equity_curve),
    }

    return {
        "equity_curve": equity_curve,
        "trades": trades,
        "positions": positions,
        "metrics": metrics,
        "cash": round(engine.cash, 2),
    }


# ============================================================
# 主流程
# ============================================================

def _make_model_key(m):
    """生成 model_type_exp_X 格式的模型标识"""
    if isinstance(m, dict):
        exp_dir = m["exp_dir"]
    elif isinstance(m, (list, tuple)):
        exp_dir = m[0]
    else:
        exp_dir = m
    parent = os.path.basename(os.path.dirname(exp_dir))
    name = os.path.basename(exp_dir)
    return f"{parent}_{name}"


def daily_eval(
    config_name: str = "config",
    update_data: bool = True,
    top_k: int = 3,
    verbose: bool = True,
    start_date: str = "2026-04-02",
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
        mode = "single"
        if os.path.exists(str(MODEL_SELECTION_PATH)):
            mode, single_models = load_model_selection(str(MODEL_SELECTION_PATH))
            if verbose:
                print(f"  模式: {mode}, 单模型数: {len(single_models)}")
                for m in single_models:
                    print(f"    {_make_model_key(m)} ({m['model_file']}, score={m['score']:.4f})")
        else:
            config_module = __import__(config_name, fromlist=["config"])
            config = config_module.config.copy()
            output_dir = config.get("output_dir", "./model/default")
            model_info = find_best_model(output_dir)
            if not model_info:
                print("错误: 未找到可用模型")
                return
            exp_dir, model_file, score = model_info
            single_models = [{"exp_dir": exp_dir, "model_file": model_file, "score": score}]
            mode = "single"
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

        if mode == "fusion" and len(single_backtesters) >= 2:
            if verbose:
                print(f"  回测融合模型 ({len(single_backtesters)}个模型)...")

            def fusion_pred_func(date):
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
                predictions_func=fusion_pred_func,
                data_file=str(DATA_FILE),
                start_date=start_date,
                end_date=end_date,
                top_k=top_k,
                rebalance_days=rebalance_days,
                position_pct=position_pct,
                initial_capital=initial_capital,
            )
            sequences["fusion"] = result

        state = {
            "sequences": sequences,
            "last_updated": timestamp,
        }
        os.makedirs(OUTPUT_DIR, exist_ok=True)
        with open(STATE_PATH, "w", encoding="utf-8") as f:
            json.dump(state, f, ensure_ascii=False, indent=2)

        # 绘制收益曲线图
        plot_path = OUTPUT_DIR / "equity_curves.png"
        try:
            plot_equity_curves(sequences, str(DATA_FILE), initial_capital, str(plot_path))
            if verbose:
                print(f"\n  [图表] 收益曲线已保存: {plot_path}")
        except Exception as e:
            if verbose:
                print(f"\n  [图表] 保存失败: {e}")

        report_key = "fusion" if "fusion" in sequences else list(sequences.keys())[0]
        report_data = sequences[report_key]
        today_trades = [t for t in report_data["trades"] if t["date"] == latest_date_str]
        is_rebalance_day = len(today_trades) > 0

        holdings = []
        for stock_id, pos in report_data["positions"].items():
            holdings.append({
                "stock_id": stock_id,
                "shares": pos["shares"],
                "cost": pos["cost"],
            })

        # 收集所有序列的信息
        sequences_summary = {}
        for key, seq in sequences.items():
            sequences_summary[key] = {
                "metrics": seq["metrics"],
                "cash": seq["cash"],
                "positions_count": len(seq["positions"]),
                "trades_count": len(seq["trades"]),
            }

        report = {
            "date": latest_date_str,
            "is_rebalance_day": is_rebalance_day,
            "today_trades": today_trades,
            "metrics": report_data["metrics"],
            "holdings": holdings,
            "cash": report_data["cash"],
            "total_value": report_data["metrics"]["latest_value"],
            "sequences": sequences_summary,
        }
        with open(REPORT_PATH, "w", encoding="utf-8") as f:
            json.dump(report, f, ensure_ascii=False, indent=2)

        portfolio = {
            "last_updated": timestamp,
            "predict_date": latest_date_str,
            "holdings": holdings,
            "cash": report_data["cash"],
            "total_value": report_data["metrics"]["latest_value"],
        }
        with open(PORTFOLIO_PATH, "w", encoding="utf-8") as f:
            json.dump(portfolio, f, ensure_ascii=False, indent=2)

        for fn in ["equity_curve.csv", "daily_metrics.json", "trades_log.csv"]:
            fp = OUTPUT_DIR / fn
            if fp.exists():
                fp.unlink()

        if verbose:
            m = report_data["metrics"]
            print(f"\n{'='*60}")
            print(f"  每日报告 ({latest_date_str}) [序列: {report_key}]")
            print(f"{'='*60}")
            print(f"  累计收益: {m['strategy_return_pct']:+.2f}%")
            print(f"  沪深300:  {m['hs300_return_pct']:+.2f}%")
            print(f"  超额收益: {m['excess_return_pct']:+.2f}%")
            print(f"  最大回撤: {m['max_drawdown_pct']:.2f}%")
            print(f"  夏普比率: {m['sharpe_ratio']:.2f}")
            print(f"  账户总值: {m['latest_value']:,.2f}")
            print(f"  调仓日:   {'是' if is_rebalance_day else '否'}")
            print(f"{'='*60}")

            print(f"\n  各序列收益:")
            for key, seq in sequences.items():
                sr = seq["metrics"]["strategy_return_pct"]
                print(f"    {key}: {sr:+.2f}%")

            if today_trades:
                print(f"\n  今日调仓:")
                for t in today_trades:
                    print(f"    {t['action']} {t['stock']} x {t['shares']}股 @ {t['price']:.4f}")

            print(f"\n  当前持仓:")
            for h in holdings:
                print(f"    {h['stock_id']}: {h['shares']}股 (成本: {h['cost']:.2f})")

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
