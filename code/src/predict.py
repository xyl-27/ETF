import os
import json
import multiprocessing as mp

import joblib
import numpy as np
import pandas as pd
import torch
from tqdm import tqdm

from config import config
from models import create_model
from utils import (
    engineer_features_39,
    engineer_features_158plus39,
    engineer_features_97,
    engineer_features_39plus97,
    engineer_features_158plus97,
    engineer_features_158plus39plus97,
    JQ_FACTORS,
)


feature_cloums_map = {
    "39": [
        "instrument",
        "开盘",
        "收盘",
        "最高",
        "最低",
        "成交量",
        "成交额",
        "振幅",
        "涨跌额",
        "换手率",
        "涨跌幅",
        "sma_5",
        "sma_20",
        "ema_12",
        "ema_26",
        "rsi",
        "macd",
        "macd_signal",
        "volume_change",
        "obv",
        "volume_ma_5",
        "volume_ma_20",
        "volume_ratio",
        "kdj_k",
        "kdj_d",
        "kdj_j",
        "boll_mid",
        "boll_std",
        "atr_14",
        "ema_60",
        "volatility_10",
        "volatility_20",
        "return_1",
        "return_5",
        "return_10",
        "high_low_spread",
        "open_close_spread",
        "high_close_spread",
        "low_close_spread",
    ],
    "158+39": [
        "instrument",
        "开盘",
        "收盘",
        "最高",
        "最低",
        "成交量",
        "成交额",
        "振幅",
        "涨跌额",
        "换手率",
        "涨跌幅",
        "KMID",
        "KLEN",
        "KMID2",
        "KUP",
        "KUP2",
        "KLOW",
        "KLOW2",
        "KSFT",
        "KSFT2",
        "OPEN0",
        "HIGH0",
        "LOW0",
        "VWAP0",
        "ROC5",
        "ROC10",
        "ROC20",
        "ROC30",
        "ROC60",
        "MA5",
        "MA10",
        "MA20",
        "MA30",
        "MA60",
        "STD5",
        "STD10",
        "STD20",
        "STD30",
        "STD60",
        "BETA5",
        "BETA10",
        "BETA20",
        "BETA30",
        "BETA60",
        "RSQR5",
        "RSQR10",
        "RSQR20",
        "RSQR30",
        "RSQR60",
        "RESI5",
        "RESI10",
        "RESI20",
        "RESI30",
        "RESI60",
        "MAX5",
        "MAX10",
        "MAX20",
        "MAX30",
        "MAX60",
        "MIN5",
        "MIN10",
        "MIN20",
        "MIN30",
        "MIN60",
        "QTLU5",
        "QTLU10",
        "QTLU20",
        "QTLU30",
        "QTLU60",
        "QTLD5",
        "QTLD10",
        "QTLD20",
        "QTLD30",
        "QTLD60",
        "RANK5",
        "RANK10",
        "RANK20",
        "RANK30",
        "RANK60",
        "RSV5",
        "RSV10",
        "RSV20",
        "RSV30",
        "RSV60",
        "IMAX5",
        "IMAX10",
        "IMAX20",
        "IMAX30",
        "IMAX60",
        "IMIN5",
        "IMIN10",
        "IMIN20",
        "IMIN30",
        "IMIN60",
        "IMXD5",
        "IMXD10",
        "IMXD20",
        "IMXD30",
        "IMXD60",
        "CORR5",
        "CORR10",
        "CORR20",
        "CORR30",
        "CORR60",
        "CORD5",
        "CORD10",
        "CORD20",
        "CORD30",
        "CORD60",
        "CNTP5",
        "CNTP10",
        "CNTP20",
        "CNTP30",
        "CNTP60",
        "CNTN5",
        "CNTN10",
        "CNTN20",
        "CNTN30",
        "CNTN60",
        "CNTD5",
        "CNTD10",
        "CNTD20",
        "CNTD30",
        "CNTD60",
        "SUMP5",
        "SUMP10",
        "SUMP20",
        "SUMP30",
        "SUMP60",
        "SUMN5",
        "SUMN10",
        "SUMN20",
        "SUMN30",
        "SUMN60",
        "SUMD5",
        "SUMD10",
        "SUMD20",
        "SUMD30",
        "SUMD60",
        "VMA5",
        "VMA10",
        "VMA20",
        "VMA30",
        "VMA60",
        "VSTD5",
        "VSTD10",
        "VSTD20",
        "VSTD30",
        "VSTD60",
        "WVMA5",
        "WVMA10",
        "WVMA20",
        "WVMA30",
        "WVMA60",
        "VSUMP5",
        "VSUMP10",
        "VSUMP20",
        "VSUMP30",
        "VSUMP60",
        "VSUMN5",
        "VSUMN10",
        "VSUMN20",
        "VSUMN30",
        "VSUMN60",
        "VSUMD5",
        "VSUMD10",
        "VSUMD20",
        "VSUMD30",
        "VSUMD60",
        "sma_5",
        "sma_20",
        "ema_12",
        "ema_26",
        "rsi",
        "macd",
        "macd_signal",
        "volume_change",
        "obv",
        "volume_ma_5",
        "volume_ma_20",
        "volume_ratio",
        "kdj_k",
        "kdj_d",
        "kdj_j",
        "boll_mid",
        "boll_std",
        "atr_14",
        "ema_60",
        "volatility_10",
        "volatility_20",
        "return_1",
        "return_5",
        "return_10",
        "high_low_spread",
        "open_close_spread",
        "high_close_spread",
        "low_close_spread",
    ],
    "97": ["instrument"] + JQ_FACTORS,
    "39+97": ["instrument"] + JQ_FACTORS,
    "158+97": ["instrument"] + JQ_FACTORS,
    "158+39+97": ["instrument"] + JQ_FACTORS,
}

feature_engineer_func_map = {
    "39": engineer_features_39,
    "158+39": engineer_features_158plus39,
    "97": engineer_features_97,
    "39+97": engineer_features_39plus97,
    "158+97": engineer_features_158plus97,
    "158+39+97": engineer_features_158plus39plus97,
}


def preprocess_predict_data(df, stockid2idx):
    assert config["feature_num"] in feature_engineer_func_map, (
        f"Unsupported feature_num: {config['feature_num']}"
    )
    feature_engineer = feature_engineer_func_map[config["feature_num"]]
    feature_columns = feature_cloums_map[config["feature_num"]]

    df = df.copy()
    df = df.sort_values(["股票代码", "日期"]).reset_index(drop=True)
    groups = [group for _, group in df.groupby("股票代码", sort=False)]
    if len(groups) == 0:
        raise ValueError("输入数据为空，无法预测")

    num_processes = min(10, mp.cpu_count())
    print("cpus!!!!!!!!!!!!!!!!!!", mp.cpu_count())
    with mp.Pool(processes=num_processes) as pool:
        processed_list = list(
            tqdm(
                pool.imap(feature_engineer, groups),
                total=len(groups),
                desc="预测集特征工程",
            )
        )

    processed = pd.concat(processed_list).reset_index(drop=True)
    processed["instrument"] = processed["股票代码"].map(stockid2idx)
    processed = processed.dropna(subset=["instrument"]).copy()
    processed["instrument"] = processed["instrument"].astype(np.int64)
    processed["日期"] = pd.to_datetime(processed["日期"])

    return processed, feature_columns


def build_inference_sequences(data, features, sequence_length, stock_ids, latest_date):
    sequences, sequence_stock_ids = [], []
    for stock_id in stock_ids:
        stock_history = (
            data[(data["股票代码"] == stock_id) & (data["日期"] <= latest_date)]
            .sort_values("日期")
            .tail(sequence_length)
        )

        if len(stock_history) == sequence_length:
            sequences.append(stock_history[features].values.astype(np.float32))
            sequence_stock_ids.append(stock_id)

    if len(sequences) == 0:
        raise ValueError("没有可用于预测的股票序列，请检查数据与 sequence_length")

    return np.asarray(sequences, dtype=np.float32), sequence_stock_ids


def load_model_selection(selection_path: str) -> tuple:
    """加载模型选择文件, 返回 (mode, models_list)"""
    models = []
    mode = "single"
    with open(selection_path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith("#"):
                continue
            if line.startswith("mode:"):
                mode = line.split(":", 1)[1].strip()
            elif line.startswith("models:"):
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


def predict_single(model, sequences_np, device):
    """单模型预测"""
    with torch.no_grad():
        x = torch.from_numpy(sequences_np).unsqueeze(0).to(device)
        scores = model(x).squeeze(0).detach().cpu().numpy()
    return scores


def predict_fusion(models_info, sequences_np, device, config, num_stocks, features, input_dim, stock_ids, top_k):
    """多模型融合预测 (分数平均)，同时输出各模型单独预测结果"""
    all_scores = []
    individual_preds = []

    for m in models_info:
        # 每个模型加载自己的config
        model_config = config.copy()
        config_json_path = os.path.join(m["exp_dir"], "config.json")
        if os.path.exists(config_json_path):
            with open(config_json_path, "r") as f:
                exp_config = json.load(f)
            model_config.update(exp_config)

        from config import get_model_config

        model_type = model_config.get("model_type", "transformer")
        model_defaults = get_model_config(model_type)
        model_defaults.update(model_config)
        model_config = model_defaults

        model = create_model(
            model_type,
            input_dim=input_dim,
            config=model_config,
            num_stocks=num_stocks,
        )
        model_path = os.path.join(m["exp_dir"], m["model_file"])
        model.load_state_dict(torch.load(model_path, map_location=device))
        model.to(device)
        model.eval()

        with torch.no_grad():
            x = torch.from_numpy(sequences_np).unsqueeze(0).to(device)
            scores = model(x).squeeze(0).detach().cpu().numpy()
        all_scores.append(scores)

        # 单模型排序
        order = np.argsort(scores)[::-1]
        top_stocks = [stock_ids[order[i]] for i in range(min(top_k, len(stock_ids)))]
        top_scores = [float(scores[order[i]]) for i in range(min(top_k, len(stock_ids)))]
        model_name = os.path.basename(m["exp_dir"])
        model_type = model_config.get("model_type", "unknown")
        individual_preds.append({
            "model": model_name,
            "model_type": model_type,
            "score": m["score"],
            "top_stocks": top_stocks,
            "top_scores": top_scores,
        })

        print(f"\n  [{model_name}] ({model_type}, search_score={m['score']:.4f})")
        for i, (stock, sc) in enumerate(zip(top_stocks, top_scores)):
            print(f"    {i+1}. {stock} (score: {sc:.4f})")

        del model
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    # 平均分数
    fused_scores = np.mean(all_scores, axis=0)
    return fused_scores, individual_preds


def main(args):
    config_name = args.config
    config_module = __import__(config_name, fromlist=["config"])
    config = config_module.config.copy()

    data_file = os.path.join(config["data_path"], config.get("data_file", "train.csv"))
    if args.data_file:
        data_file = args.data_file

    # 优先级: --selection > --exp > 默认
    use_selection = False
    selection_path = os.path.join("./output/", "model_selection.txt")

    if args.selection:
        selection_path = args.selection
        use_selection = os.path.exists(selection_path)
    elif os.path.exists(selection_path):
        use_selection = True

    if use_selection:
        mode, models_info = load_model_selection(selection_path)
        print(f"使用模型选择文件: {selection_path}")
        print(f"模式: {mode}, 模型数: {len(models_info)}")
        for m in models_info:
            print(f"  - {m['exp_dir']} ({m['model_file']}) score={m['score']:.4f}")

        # 加载第一个模型的scaler和config
        first_model = models_info[0]
        scaler_path = os.path.join(first_model["exp_dir"], "scaler.pkl")
        config_json_path = os.path.join(first_model["exp_dir"], "config.json")
        if os.path.exists(config_json_path):
            with open(config_json_path, "r") as f:
                exp_config = json.load(f)
            config.update(exp_config)
    elif args.exp:
        exp_dir = os.path.join(config["output_dir"], args.exp)
        model_path = os.path.join(exp_dir, "best_model.pth")
        scaler_path = os.path.join(exp_dir, "scaler.pkl")
        config_json_path = os.path.join(exp_dir, "config.json")
        if os.path.exists(config_json_path):
            with open(config_json_path, "r") as f:
                exp_config = json.load(f)
            config.update(exp_config)
        models_info = [{"exp_dir": exp_dir, "model_file": "best_model.pth", "score": 0}]
        mode = "single"
    else:
        model_path = os.path.join(config["output_dir"], "best_model.pth")
        scaler_path = os.path.join(config["output_dir"], "scaler.pkl")
        models_info = [{"exp_dir": config["output_dir"], "model_file": "best_model.pth", "score": 0}]
        mode = "single"

    output_path = os.path.join("./output/", "result.csv")

    raw_df = pd.read_csv(data_file, dtype={"股票代码": str})
    raw_df["股票代码"] = raw_df["股票代码"].astype(str).str.zfill(6)
    raw_df["日期"] = pd.to_datetime(raw_df["日期"])
    latest_date = raw_df["日期"].max()
    stock_ids = sorted(raw_df["股票代码"].unique())
    stockid2idx = {sid: idx for idx, sid in enumerate(stock_ids)}

    processed, features = preprocess_predict_data(raw_df, stockid2idx)
    processed[features] = (
        processed[features].replace([np.inf, -np.inf], np.nan).fillna(0.0)
    )

    scaler = joblib.load(scaler_path)
    processed[features] = scaler.transform(processed[features])

    sequence_length = config["sequence_length"]
    sequences_np, sequence_stock_ids = build_inference_sequences(
        processed,
        features,
        sequence_length,
        stock_ids,
        latest_date,
    )

    if torch.cuda.is_available():
        device = torch.device("cuda")
    elif torch.backends.mps.is_available():
        device = torch.device("mps")
    else:
        device = torch.device("cpu")

    top_k = config.get("top_k", 5)

    if mode == "fusion" and len(models_info) > 1:
        # 融合模式
        print(f"\n各模型单独预测结果:")
        print(f"{'='*50}")
        scores, individual_preds = predict_fusion(
            models_info, sequences_np, device, config, len(stock_ids), features, len(features),
            sequence_stock_ids, top_k,
        )
        # 保存汇总
        summary_path = os.path.join("./output/", "pred_summary.csv")
        summary_rows = []
        for pred in individual_preds:
            for rank, (stock, sc) in enumerate(zip(pred["top_stocks"], pred["top_scores"])):
                summary_rows.append({
                    "model": pred["model"],
                    "model_type": pred["model_type"],
                    "search_score": pred["score"],
                    "rank": rank + 1,
                    "stock_id": stock,
                    "model_score": sc,
                })
        summary_df = pd.DataFrame(summary_rows)
        summary_df.to_csv(summary_path, index=False)
        print(f"\n各模型预测汇总已保存: {summary_path}")
    else:
        # 单模型模式
        first = models_info[0]
        model = create_model(
            config["model_type"],
            input_dim=len(features),
            config=config,
            num_stocks=len(stock_ids),
        )
        model_path = os.path.join(first["exp_dir"], first["model_file"])
        model.load_state_dict(torch.load(model_path, map_location=device))
        model.to(device)
        model.eval()
        scores = predict_single(model, sequences_np, device)
        del model

    order = np.argsort(scores)[::-1]
    ranked_stock_ids = [sequence_stock_ids[i] for i in order]

    if len(ranked_stock_ids) < top_k:
        raise ValueError(
            f"可预测股票不足{top_k}只，当前仅有 {len(ranked_stock_ids)} 只"
        )
    top_k_stocks = ranked_stock_ids[:top_k]
    output_df = pd.DataFrame(
        {
            "stock_id": top_k_stocks,
            "weight": [1.0 / top_k] * len(top_k_stocks),
        }
    )
    output_df.to_csv(output_path, index=False)

    print(f"预测日期: {latest_date.date()}")
    print(f"参与排序股票数: {len(ranked_stock_ids)}")
    print(f"模式: {mode} ({len(models_info)}个模型)")
    print(f"结果已写入: {output_path}")


if __name__ == "__main__":
    import argparse
    import importlib
    import multiprocessing as mp

    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, default="config", help="Config module name")
    parser.add_argument("--exp", type=str, default=None, help="Experiment directory, e.g. exp_57")
    parser.add_argument("--selection", type=str, default=None, help="Model selection file path")
    parser.add_argument("--data-file", type=str, default=None, help="Override data file path")
    parser.add_argument("--topk", type=int, default=5)
    args = parser.parse_args()

    config_module = importlib.import_module(args.config)
    globals().update(config_module.config)
    get_model_config = config_module.get_model_config

    mp.set_start_method("spawn", force=True)
    main(args)
