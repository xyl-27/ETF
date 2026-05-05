"""
每日定时测评脚本 - ETF (实盘模拟)
- 获取最新ETF数据
- 运行预测（Top-K推荐）
- 维护持仓组合：对比新旧持仓，生成买卖指令
- 运行近期回测（滚动评估）
- 记录交易历史和当前持仓
"""

import os
import sys
import json
import traceback
import subprocess
from datetime import datetime
from typing import Optional
from pathlib import Path

import numpy as np
import pandas as pd

# 项目根目录
PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
os.chdir(PROJECT_ROOT)
sys.path.insert(0, str(PROJECT_ROOT / "code" / "src"))

import torch

from predict import (
    preprocess_predict_data,
    build_inference_sequences,
)
from models import create_model
import joblib
import multiprocessing as mp


PORTFOLIO_PATH = str(PROJECT_ROOT / "output" / "portfolio.json")
HISTORY_PATH = str(PROJECT_ROOT / "output" / "daily_eval_history.json")


# ============================================================
# 数据更新
# ============================================================

def update_etf_data(verbose: bool = True) -> bool:
    """运行 get_etf_data.py 获取最新ETF数据"""
    script_path = str(PROJECT_ROOT / "get_etf_data.py")
    if not os.path.exists(script_path):
        if verbose:
            print("[数据更新] 未找到 get_etf_data.py，跳过")
        return False

    if verbose:
        print("[数据更新] 运行 get_etf_data.py 获取最新数据...")

    try:
        result = subprocess.run(
            ["python", script_path],
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
# 模型查找
# ============================================================

def find_best_model(output_dir: str) -> Optional[tuple]:
    """从搜索目录中查找最佳模型 (exp_N)"""
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
# 预测
# ============================================================

def run_prediction(
    exp_dir: str,
    model_file: str,
    data_file: str,
    config_module,
    top_k: int = 5,
    verbose: bool = True,
) -> Optional[dict]:
    """运行预测，返回Top-K推荐"""
    config = config_module.config.copy()

    with open(os.path.join(exp_dir, "config.json"), "r") as f:
        exp_config = json.load(f)
    config.update(exp_config)

    scaler_path = os.path.join(exp_dir, "scaler.pkl")
    if not os.path.exists(scaler_path) or not os.path.exists(os.path.join(exp_dir, model_file)):
        print(f"[预测] 模型或scaler文件缺失")
        return None

    raw_df = pd.read_csv(data_file, dtype={"股票代码": str})
    raw_df["股票代码"] = raw_df["股票代码"].astype(str).str.zfill(6)
    raw_df["日期"] = pd.to_datetime(raw_df["日期"])
    latest_date = raw_df["日期"].max()
    stock_ids = sorted(raw_df["股票代码"].unique())
    stockid2idx = {sid: idx for idx, sid in enumerate(stock_ids)}

    processed, features = preprocess_predict_data(raw_df, stockid2idx)
    processed[features] = processed[features].replace([np.inf, -np.inf], np.nan).fillna(0.0)

    scaler = joblib.load(scaler_path)
    processed[features] = scaler.transform(processed[features])

    sequence_length = config["sequence_length"]
    sequences_np, sequence_stock_ids = build_inference_sequences(
        processed, features, sequence_length, stock_ids, latest_date
    )

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = create_model(config["model_type"], input_dim=len(features), config=config, num_stocks=len(stock_ids))
    model.load_state_dict(torch.load(os.path.join(exp_dir, model_file), map_location=device))
    model.to(device)
    model.eval()

    with torch.no_grad():
        x = torch.from_numpy(sequences_np).unsqueeze(0).to(device)
        scores = model(x).squeeze(0).detach().cpu().numpy()

    order = np.argsort(scores)[::-1]
    ranked = [sequence_stock_ids[i] for i in order]
    top_k = min(top_k, len(ranked))
    top_stocks = ranked[:top_k]
    top_scores = [float(scores[order[i]]) for i in range(top_k)]

    result = {
        "predict_date": str(latest_date.date()),
        "total_stocks": len(ranked),
        "top_k": top_k,
        "top_stocks": top_stocks,
        "top_scores": top_scores,
    }

    if verbose:
        print(f"\n[预测] 日期: {latest_date.date()}, 排序股票: {len(ranked)}只")
        print(f"[预测] Top-{top_k} 推荐:")
        for i, stock in enumerate(top_stocks):
            print(f"  {i+1}. {stock} (score: {top_scores[i]:.4f})")

    del model
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    return result


# ============================================================
# 持仓管理
# ============================================================

def load_portfolio() -> Optional[dict]:
    """加载当前持仓"""
    if os.path.exists(PORTFOLIO_PATH):
        with open(PORTFOLIO_PATH, "r", encoding="utf-8") as f:
            return json.load(f)
    return None


def save_portfolio(portfolio: dict):
    """保存当前持仓"""
    os.makedirs(os.path.dirname(PORTFOLIO_PATH), exist_ok=True)
    with open(PORTFOLIO_PATH, "w", encoding="utf-8") as f:
        json.dump(portfolio, f, ensure_ascii=False, indent=2)


def generate_trades(current_portfolio: Optional[dict], new_prediction: dict, data_file: str, verbose: bool = True) -> list:
    """
    对比当前持仓和新预测，生成买卖指令
    返回交易列表
    """
    trades = []
    date_str = new_prediction["predict_date"]
    new_top = set(new_prediction["top_stocks"])

    if current_portfolio is None:
        # 首次建仓
        for stock in new_prediction["top_stocks"]:
            trades.append({
                "action": "买入",
                "stock_id": stock,
                "reason": "首次建仓",
                "date": date_str,
            })
        if verbose:
            print(f"\n[交易] 首次建仓，买入 {len(new_prediction['top_stocks'])} 只:")
            for t in trades:
                print(f"  {t['action']} {t['stock_id']} ({t['reason']})")
        return trades

    # 现有持仓
    current_holdings = set()
    for h in current_portfolio.get("holdings", []):
        current_holdings.add(h["stock_id"])

    # 卖出：不在新Top-K中的持仓
    to_sell = current_holdings - new_top
    for stock in to_sell:
        # 找到卖出原因
        reason = "不在新Top-K中"
        if new_prediction["total_stocks"] == 0:
            reason = "无预测数据，清仓"
        trades.append({
            "action": "卖出",
            "stock_id": stock,
            "reason": reason,
            "date": date_str,
        })

    # 买入：新Top-K中不在当前持仓的
    to_buy = new_top - current_holdings
    for stock in new_prediction["top_stocks"]:
        if stock in to_buy:
            idx = new_prediction["top_stocks"].index(stock)
            score = new_prediction["top_scores"][idx]
            trades.append({
                "action": "买入",
                "stock_id": stock,
                "reason": f"新入选Top-K (score={score:.4f})",
                "date": date_str,
            })

    if verbose and trades:
        print(f"\n[交易] {len(trades)} 笔操作:")
        for t in trades:
            action_color = "买入" if t["action"] == "买入" else "卖出"
            print(f"  {t['action']} {t['stock_id']} ({t['reason']})")
    elif verbose:
        print(f"\n[交易] 无变化，维持持仓")

    return trades


def update_portfolio(current_portfolio: Optional[dict], trades: list, new_prediction: dict) -> dict:
    """根据交易更新持仓"""
    if current_portfolio is None:
        holdings = []
    else:
        holdings = list(current_portfolio.get("holdings", []))

    # 执行卖出
    sold_ids = set()
    for t in trades:
        if t["action"] == "卖出":
            sold_ids.add(t["stock_id"])

    holdings = [h for h in holdings if h["stock_id"] not in sold_ids]

    # 执行买入
    for t in trades:
        if t["action"] == "买入":
            if t["stock_id"] not in [h["stock_id"] for h in holdings]:
                idx = new_prediction["top_stocks"].index(t["stock_id"])
                score = new_prediction["top_scores"][idx]
                holdings.append({
                    "stock_id": t["stock_id"],
                    "buy_date": t["date"],
                    "buy_score": score,
                })

    new_portfolio = {
        "last_updated": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "predict_date": new_prediction["predict_date"],
        "holdings": holdings,
        "model_used": new_prediction.get("model", ""),
        "total_value_pct": 1.0,  # 等权分配
    }
    return new_portfolio


# ============================================================
# 回测
# ============================================================

# ============================================================
# 日志管理
# ============================================================

def load_history(history_path: str) -> list:
    if os.path.exists(history_path):
        with open(history_path, "r", encoding="utf-8") as f:
            return json.load(f)
    return []


def save_history(history: list, history_path: str):
    os.makedirs(os.path.dirname(history_path), exist_ok=True)
    with open(history_path, "w", encoding="utf-8") as f:
        json.dump(history, f, ensure_ascii=False, indent=2)


# ============================================================
# 主流程
# ============================================================

def daily_eval(
    config_name: str = "config",
    update_data: bool = True,
    top_k: int = 5,
    verbose: bool = True,
):
    """每日测评主流程"""
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    log_entry = {"timestamp": timestamp, "status": "success"}

    try:
        config_module = __import__(config_name, fromlist=["config"])
        config = config_module.config.copy()
        output_dir = config.get("output_dir", "./model/default")
        data_path = config.get("data_path", "./etf_data")
        data_file = os.path.join(data_path, config.get("data_file", "etf_data_74_new.csv"))

        # 1. 更新数据
        if update_data:
            if verbose:
                print(f"\n{'='*60}")
                print(f"[{timestamp}] 每日测评开始")
                print(f"{'='*60}")
                print("\n[1/3] 获取最新ETF数据...")
            data_success = update_etf_data(verbose=verbose)
            log_entry["data_update"] = data_success
            if not data_success:
                print("[数据更新] 失败，使用现有数据继续")

        step_num = 2 if update_data else 1
        total_steps = 3 if update_data else 2

        # 2. 查找最佳模型
        if verbose:
            print(f"\n[{step_num}/{total_steps}] 查找最佳模型并预测...")

        model_info = find_best_model(output_dir)
        if model_info is None:
            print(f"[模型] 未在 {output_dir} 找到有效模型")
            log_entry["status"] = "failed"
            log_entry["error"] = "no_model_found"
            return log_entry

        exp_dir, model_file, best_score = model_info
        log_entry["model"] = {
            "exp_dir": os.path.basename(exp_dir),
            "model_file": model_file,
            "search_score": round(best_score, 4),
        }
        if verbose:
            print(f"[模型] 使用 {os.path.basename(exp_dir)} ({model_file}), 搜索分数: {best_score:.4f}")

        # 运行预测
        pred_result = run_prediction(
            exp_dir=exp_dir,
            model_file=model_file,
            data_file=data_file,
            config_module=config_module,
            top_k=top_k,
            verbose=verbose,
        )
        log_entry["prediction"] = pred_result

        # 3. 维护持仓（对比新旧，生成交易）
        if verbose:
            print(f"\n[{step_num+1}/{total_steps}] 维护持仓...")

        current_portfolio = load_portfolio()
        has_previous = current_portfolio is not None

        trades = generate_trades(current_portfolio, pred_result, data_file, verbose)
        new_portfolio = update_portfolio(current_portfolio, trades, pred_result)

        save_portfolio(new_portfolio)

        log_entry["trades"] = trades
        log_entry["portfolio"] = new_portfolio
        log_entry["has_previous_portfolio"] = has_previous

        if verbose:
            print(f"\n[持仓] 当前持有 {len(new_portfolio['holdings'])} 只:")
            for h in new_portfolio["holdings"]:
                print(f"  {h['stock_id']} (买入日期: {h['buy_date']})")

        # 保存日志
        history = load_history(HISTORY_PATH)
        history.append(log_entry)
        save_history(history, HISTORY_PATH)

        if verbose:
            print(f"\n{'='*60}")
            print(f"[{timestamp}] 每日测评完成")
            print(f"日志已保存: {HISTORY_PATH}")
            print(f"持仓已保存: {PORTFOLIO_PATH}")
            print(f"{'='*60}")

        return log_entry

    except Exception as e:
        log_entry["status"] = "error"
        log_entry["error"] = str(e)
        log_entry["traceback"] = traceback.format_exc()

        history = load_history(HISTORY_PATH)
        history.append(log_entry)
        save_history(history, HISTORY_PATH)

        print(f"\n[错误] 每日测评失败: {e}")
        traceback.print_exc()
        return log_entry


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="每日定时测评")
    parser.add_argument("--config", type=str, default="config", help="配置模块名")
    parser.add_argument("--no-update", action="store_true", help="跳过数据更新")
    parser.add_argument("--topk", type=int, default=5, help="Top-K推荐数量")
    parser.add_argument("--quiet", action="store_true", help="静默模式")
    args = parser.parse_args()

    mp.set_start_method("spawn", force=True)

    daily_eval(
        config_name=args.config,
        update_data=not args.no_update,
        top_k=args.topk,
        verbose=not args.quiet,
    )
