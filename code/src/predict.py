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


def main(args):
    config_name = args.config
    config_module = __import__(config_name, fromlist=["config"])
    config = config_module.config.copy()

    data_file = os.path.join(config["data_path"], config.get("data_file", "train.csv"))
    
    # 支持指定实验目录
    if args.exp:
        exp_dir = os.path.join(config["output_dir"], args.exp)
        model_path = os.path.join(exp_dir, "best_model.pth")
        scaler_path = os.path.join(exp_dir, "scaler.pkl")
        # 从实验目录加载config.json
        config_json_path = os.path.join(exp_dir, "config.json")
        if os.path.exists(config_json_path):
            with open(config_json_path, 'r') as f:
                exp_config = json.load(f)
            config.update(exp_config)
    else:
        model_path = os.path.join(config["output_dir"], "best_model.pth")
        scaler_path = os.path.join(config["output_dir"], "scaler.pkl")
    
    output_path = os.path.join("./output/", "result.csv")

    if not os.path.exists(model_path):
        raise FileNotFoundError(f"未找到模型文件: {model_path}")
    if not os.path.exists(scaler_path):
        raise FileNotFoundError(f"未找到Scaler文件: {scaler_path}")

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

    model = create_model(
        config["model_type"],
        input_dim=len(features),
        config=config,
        num_stocks=len(stock_ids),
    )
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.to(device)
    model.eval()

    with torch.no_grad():
        x = torch.from_numpy(sequences_np).unsqueeze(0).to(device)  # [1, N, L, F]
        scores = model(x).squeeze(0).detach().cpu().numpy()  # [N]

    order = np.argsort(scores)[::-1]
    ranked_stock_ids = [sequence_stock_ids[i] for i in order]

    top_k = config.get("top_k", 5)

    # 仅输出前top_k，权重固定 1/top_k
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
    print(f"结果已写入: {output_path}")


if __name__ == "__main__":
    import argparse
    import importlib
    import multiprocessing as mp

    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, default="config", help="Config module name")
    parser.add_argument("--exp", type=str, default=None, help="Experiment directory, e.g. exp_57")
    parser.add_argument("--topk", type=int, default=5)
    args = parser.parse_args()

    config_module = importlib.import_module(args.config)
    globals().update(config_module.config)
    get_model_config = config_module.get_model_config

    mp.set_start_method("spawn", force=True)
    main(args)
