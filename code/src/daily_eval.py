"""
每日定时测评脚本
- 更新股票数据
- 合并JQ因子
- 运行预测（Top-K推荐）
- 运行近期回测（滚动评估）
- 记录结果到历史日志
"""

import os
import sys
import json
import glob
import shutil
import traceback
from datetime import datetime, timedelta
from dataclasses import dataclass, field, asdict
from typing import Optional
from pathlib import Path

import numpy as np
import pandas as pd

# 项目根目录
PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
os.chdir(PROJECT_ROOT)
sys.path.insert(0, str(PROJECT_ROOT / "code" / "src"))

import baostock as bs
import torch

from backtest import run_etf_backtest
from predict import (
    preprocess_predict_data,
    build_inference_sequences,
    feature_cloums_map,
    feature_engineer_func_map,
)
from models import create_model
import joblib
import multiprocessing as mp


# ============================================================
# 数据更新
# ============================================================

def update_stock_data(start_date: str = "2024-01-01", verbose: bool = True) -> bool:
    """增量更新股票数据，返回是否成功"""
    try:
        lg = bs.login()
        if lg.error_code != "0":
            print(f"[数据更新] baostock登录失败: {lg.error_msg}")
            return False

        end_date = datetime.now().strftime("%Y-%m-%d")
        output_path = str(PROJECT_ROOT / "data" / "stock_data.csv")

        # 获取沪深300成分股
        rs = bs.query_hs300_stocks()
        if rs.error_code != "0":
            print(f"[数据更新] 获取成分股失败: {rs.error_msg}")
            bs.logout()
            return False

        hs300_stocks = []
        while (rs.error_code == "0") & rs.next():
            hs300_stocks.append(rs.get_row_data())
        hs300_df = pd.DataFrame(hs300_stocks, columns=rs.fields)

        # 保存成分股列表
        hs300_list_path = str(PROJECT_ROOT / "data" / "hs300_stock_list.csv")
        hs300_df.to_csv(hs300_list_path, index=False, encoding="utf-8-sig")

        # 读取现有数据
        existing_df = None
        existing_stocks = set()
        if os.path.exists(output_path):
            try:
                existing_df = pd.read_csv(output_path)
                existing_df["股票代码"] = existing_df["股票代码"].astype(str).str.zfill(6)
                existing_stocks = set(existing_df["股票代码"].unique())
            except Exception:
                existing_df = None

        hs300_df["纯代码"] = hs300_df["code"].str.replace("sh.", "").str.replace("sz.", "").str.zfill(6)
        success_count = 0
        total = len(hs300_df)

        for idx, row in hs300_df.iterrows():
            bs_code = row["code"]
            pure_code = row["纯代码"]

            # 检查是否需要增量
            if existing_df is not None and pure_code in existing_stocks:
                stock_df = existing_df[existing_df["股票代码"] == pure_code].copy()
                stock_df["日期_dt"] = pd.to_datetime(stock_df["日期"], format="%Y/%m/%d", errors="coerce")
                stock_df = stock_df.dropna(subset=["日期_dt"])
                if not stock_df.empty:
                    existing_max = stock_df["日期_dt"].max()
                    target_start = pd.to_datetime(start_date)
                    if existing_max >= target_start:
                        fetch_start = (existing_max + timedelta(days=1)).strftime("%Y-%m-%d")
                        fetch_end = end_date
                        if fetch_start > fetch_end:
                            if idx % 30 == 0 and verbose:
                                print(f"  [{idx+1}/{total}] {bs_code} 数据已完整，跳过")
                            continue
                    else:
                        fetch_start = start_date
                        fetch_end = end_date
                else:
                    fetch_start = start_date
                    fetch_end = end_date
            else:
                fetch_start = start_date
                fetch_end = end_date

            rs = bs.query_history_k_data_plus(
                bs_code,
                "date,code,open,high,low,close,preclose,volume,amount,turn,pctChg",
                start_date=fetch_start,
                end_date=fetch_end,
                frequency="d",
                adjustflag="1",
            )
            if rs.error_code != "0" or not rs.next():
                if idx % 30 == 0 and verbose:
                    print(f"  [{idx+1}/{total}] {bs_code} 无新数据")
                continue

            data_list = [rs.get_row_data()]
            while (rs.error_code == "0") & rs.next():
                data_list.append(rs.get_row_data())

            new_df = pd.DataFrame(data_list, columns=rs.fields)
            for col in ["open", "high", "low", "close", "preclose", "volume", "amount", "turn", "pctChg"]:
                new_df[col] = pd.to_numeric(new_df[col], errors="coerce")

            new_df["振幅"] = ((new_df["high"] - new_df["low"]) / new_df["preclose"] * 100).round(2)
            new_df["涨跌额"] = (new_df["close"] - new_df["preclose"]).round(2)
            new_df["日期"] = pd.to_datetime(new_df["date"]).dt.strftime("%Y/%m/%d")
            new_df["股票代码"] = new_df["code"].str.replace("sh.", "").str.replace("sz.", "").str.zfill(6)
            new_df = new_df.rename(columns={
                "code": "股票代码", "date": "日期", "open": "开盘", "close": "收盘",
                "high": "最高", "low": "最低", "volume": "成交量", "amount": "成交额",
                "turn": "换手率", "pctChg": "涨跌幅",
            })
            new_df = new_df[["股票代码", "日期", "开盘", "收盘", "最高", "最低",
                             "成交量", "成交额", "振幅", "涨跌额", "换手率", "涨跌幅"]]

            if existing_df is not None and not existing_df.empty:
                new_df["日期_dt"] = pd.to_datetime(new_df["日期"], format="%Y/%m/%d")
                combined = pd.concat([existing_df[existing_df["股票代码"] != pure_code], new_df], ignore_index=True)
                combined = combined.drop(columns=["日期_dt"], errors="ignore")
                existing_df = combined
            else:
                existing_df = new_df

            success_count += 1
            if verbose and (idx % 30 == 0 or success_count <= 3):
                print(f"  [{idx+1}/{total}] {bs_code} 更新成功 +{len(new_df)}条")

        if existing_df is not None:
            existing_df.to_csv(output_path, index=False, encoding="utf-8-sig")

        bs.logout()

        if verbose:
            print(f"[数据更新] 完成: {success_count}只股票更新, 共{len(existing_df) if existing_df is not None else 0}条记录")
        return True

    except Exception as e:
        print(f"[数据更新] 失败: {e}")
        traceback.print_exc()
        try:
            bs.logout()
        except Exception:
            pass
        return False


def merge_jq_factors(verbose: bool = True) -> bool:
    """合并JQ因子到股票数据"""
    try:
        data_dir = str(PROJECT_ROOT / "data")
        stock_data = pd.read_csv(os.path.join(data_dir, "stock_data.csv"))
        hs300_jq = pd.read_csv(os.path.join(data_dir, "hs300_jq.csv"))

        stock_data["股票代码"] = stock_data["股票代码"].astype(str).str.strip()
        hs300_jq["股票代码"] = hs300_jq["股票代码"].str.replace(".XSHG", "").str.replace(".XSHE", "").str.strip()
        stock_data["日期"] = pd.to_datetime(stock_data["日期"]).dt.strftime("%Y-%m-%d")
        hs300_jq["日期"] = pd.to_datetime(hs300_jq["日期"]).dt.strftime("%Y-%m-%d")

        jq_cols = hs300_jq.columns.tolist()[2:]
        merged = stock_data.merge(hs300_jq[["股票代码", "日期"] + jq_cols], on=["股票代码", "日期"], how="left")
        merged.to_csv(os.path.join(data_dir, "stock_data_with_jqfactors.csv"), index=False)

        if verbose:
            print(f"[JQ因子] 合并完成: {len(merged)}行, {len(merged.columns)}列")
        return True
    except Exception as e:
        print(f"[JQ因子] 失败: {e}")
        traceback.print_exc()
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

    # 保存结果
    output_path = str(PROJECT_ROOT / "output" / "result.csv")
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    output_df = pd.DataFrame({
        "stock_id": top_stocks,
        "weight": [1.0 / top_k] * len(top_stocks),
    })
    output_df.to_csv(output_path, index=False)

    result = {
        "predict_date": str(latest_date.date()),
        "total_stocks": len(ranked),
        "top_k": top_k,
        "top_stocks": top_stocks,
        "top_scores": [float(scores[order[i]]) for i in range(top_k)],
    }

    if verbose:
        print(f"\n[预测] 日期: {latest_date.date()}, 排序股票: {len(ranked)}只")
        print(f"[预测] Top-{top_k} 推荐:")
        for i, stock in enumerate(top_stocks):
            print(f"  {i+1}. {stock} (score: {result['top_scores'][i]:.4f})")
        print(f"[预测] 结果已保存: {output_path}")

    del model
    torch.cuda.empty_cache() if torch.cuda.is_available() else None

    return result


# ============================================================
# 回测
# ============================================================

def run_recent_backtest(
    exp_dir: str,
    model_file: str,
    data_path: str,
    config_module,
    months: int = 6,
    verbose: bool = True,
) -> Optional[dict]:
    """运行近期回测评估"""
    config = config_module.config.copy()
    with open(os.path.join(exp_dir, "config.json"), "r") as f:
        exp_config = json.load(f)
    config.update(exp_config)

    df = pd.read_csv(data_path)
    df["日期"] = pd.to_datetime(df["日期"])
    all_dates = sorted(df["日期"].unique())
    end_date = all_dates[-1]
    start_date = (end_date - timedelta(days=months * 30)).strftime("%Y-%m-%d")

    if verbose:
        print(f"\n[回测] 区间: {start_date} ~ {end_date}")

    try:
        result = run_etf_backtest(
            model_dir=exp_dir,
            data_path=data_path,
            start_date=start_date,
            end_date=end_date.strftime("%Y-%m-%d"),
            top_k=config.get("top_k", 5),
            rebalance_days=5,
            position_pct=0.95,
            model_file=model_file,
            verbose=verbose,
        )

        bt_result = {
            "start_date": start_date,
            "end_date": str(end_date.date()),
            "strategy_return": round(result.strategy_return, 2),
            "hs300_return": round(result.hs300_return, 2),
            "excess_return": round(result.excess_return, 2),
            "max_drawdown": round(result.max_drawdown, 2),
            "drawdown_days": result.drawdown_days,
            "recovered": result.recovered,
            "recovery_days": result.recovery_days,
        }

        if verbose:
            print(f"\n[回测] 策略收益: {bt_result['strategy_return']:+.2f}%")
            print(f"[回测] 基准收益: {bt_result['hs300_return']:+.2f}%")
            print(f"[回测] 超额收益: {bt_result['excess_return']:+.2f}%")
            print(f"[回测] 最大回撤: {bt_result['max_drawdown']:.2f}%")

        return bt_result

    except Exception as e:
        print(f"[回测] 失败: {e}")
        traceback.print_exc()
        return None


# ============================================================
# 日志管理
# ============================================================

def load_history(history_path: str) -> list:
    """加载历史记录"""
    if os.path.exists(history_path):
        with open(history_path, "r", encoding="utf-8") as f:
            return json.load(f)
    return []


def save_history(history: list, history_path: str):
    """保存历史记录"""
    os.makedirs(os.path.dirname(history_path), exist_ok=True)
    with open(history_path, "w", encoding="utf-8") as f:
        json.dump(history, f, ensure_ascii=False, indent=2)


# ============================================================
# 主流程
# ============================================================

def daily_eval(
    config_name: str = "config",
    update_data: bool = True,
    backtest_months: int = 6,
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
        data_file = os.path.join(config["data_path"], config.get("data_file", "train.csv"))
        data_path = config.get("data_path", "./data")

        # 1. 更新数据
        if update_data:
            if verbose:
                print(f"\n{'='*60}")
                print(f"[{timestamp}] 每日测评开始")
                print(f"{'='*60}")
                print("\n[1/4] 更新股票数据...")
            success = update_stock_data(verbose=verbose)
            log_entry["data_update"] = success
            if not success:
                print("[数据更新] 失败，使用现有数据继续")

        # 2. 合并JQ因子
        if verbose:
            print("\n[2/4] 合并JQ因子...")
        jq_success = merge_jq_factors(verbose=verbose)
        log_entry["jq_factors"] = jq_success
        if not jq_success:
            print("[JQ因子] 合并失败，使用现有合并数据")

        # 3. 查找最佳模型
        if verbose:
            print("\n[3/4] 查找最佳模型并预测...")
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

        # 4. 运行近期回测
        if verbose:
            print(f"\n[4/4] 运行近期回测 ({backtest_months}个月)...")
        bt_result = run_recent_backtest(
            exp_dir=exp_dir,
            model_file=model_file,
            data_path=data_file,
            config_module=config_module,
            months=backtest_months,
            verbose=verbose,
        )
        log_entry["backtest"] = bt_result

        # 保存日志
        history_path = str(PROJECT_ROOT / "output" / "daily_eval_history.json")
        history = load_history(history_path)
        history.append(log_entry)
        save_history(history, history_path)

        if verbose:
            print(f"\n{'='*60}")
            print(f"[{timestamp}] 每日测评完成")
            print(f"日志已保存: {history_path}")
            print(f"{'='*60}")

        return log_entry

    except Exception as e:
        log_entry["status"] = "error"
        log_entry["error"] = str(e)
        log_entry["traceback"] = traceback.format_exc()

        history_path = str(PROJECT_ROOT / "output" / "daily_eval_history.json")
        history = load_history(history_path)
        history.append(log_entry)
        save_history(history, history_path)

        print(f"\n[错误] 每日测评失败: {e}")
        traceback.print_exc()
        return log_entry


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="每日定时测评")
    parser.add_argument("--config", type=str, default="config", help="配置模块名")
    parser.add_argument("--no-update", action="store_true", help="跳过数据更新")
    parser.add_argument("--backtest-months", type=int, default=6, help="回测月数")
    parser.add_argument("--topk", type=int, default=5, help="Top-K推荐数量")
    parser.add_argument("--quiet", action="store_true", help="静默模式")
    args = parser.parse_args()

    mp.set_start_method("spawn", force=True)

    daily_eval(
        config_name=args.config,
        update_data=not args.no_update,
        backtest_months=args.backtest_months,
        top_k=args.topk,
        verbose=not args.quiet,
    )
