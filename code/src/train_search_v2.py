import os
import sys
import json
import time
import traceback
import joblib
import pandas as pd
import numpy as np
import torch
import gc
import weakref
import optuna
from sklearn.preprocessing import StandardScaler
from tqdm import tqdm
import multiprocessing as mp

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from model import create_model
from utils import (
    engineer_features_39,
    engineer_features_158plus39,
    engineer_features_97,
    engineer_features_39plus97,
    engineer_features_158plus97,
    engineer_features_158plus39plus97,
    JQ_FACTORS,
    create_ranking_dataset_vectorized,
)
from train import (
    RankingDataset,
    collate_fn,
    split_train_val_by_last_month,
    preprocess_data,
    preprocess_val_data,
    train_ranking_model,
    calculate_ranking_metrics,
    WeightedRankingLoss,
    set_seed,
    evaluate_ranking_model,
    config as train_config,
)


def preprocess_and_save(config, search_dir):
    """只执行一次特征工程，返回预处理后的数据
    
    如果 search_dir 下已有 preprocessed_data.pkl + scaler.pkl，直接加载缓存。
    """
    preprocessed_path = os.path.join(search_dir, "preprocessed_data.pkl").replace("\\", "/")
    scaler_path = os.path.join(search_dir, "scaler.pkl").replace("\\", "/")
    if os.path.exists(preprocessed_path) and os.path.exists(scaler_path):
        print(f"Loading cached preprocessed data from {preprocessed_path}...")
        return joblib.load(preprocessed_path), joblib.load(scaler_path)

    # Set global config for train.py functions that use it
    for key in config:
        if key in train_config:
            train_config[key] = config[key]

    feature_num = config["feature_num"]
    data_path = config["data_path"]
    data_file = config.get("data_file", "train.csv")
    data_file_path = os.path.join(data_path, data_file).replace("\\", "/")

    if not os.path.exists(data_file_path):
        raise FileNotFoundError(f"Data file not found: {data_file_path}")

    print(f"Loading data from {data_file_path}...")
    full_df = pd.read_csv(data_file_path)

    train_df, val_df, val_start, val_end = split_train_val_by_last_month(
        full_df, config["sequence_length"], config["val_months"],
        val_start_date=config.get("val_start_date"),
        val_end_date=config.get("val_end_date"),
    )

    all_stock_ids = full_df["股票代码"].unique()
    stockid2idx = {sid: idx for idx, sid in enumerate(sorted(all_stock_ids))}

    train_data, features = preprocess_data(
        train_df, is_train=True, stockid2idx=stockid2idx
    )
    val_data, _ = preprocess_val_data(val_df, stockid2idx=stockid2idx)

    scaler = StandardScaler()
    train_data[features] = train_data[features].replace([np.inf, -np.inf], np.nan)
    val_data[features] = val_data[features].replace([np.inf, -np.inf], np.nan)
    train_data = train_data.dropna(subset=features)
    val_data = val_data.dropna(subset=features)
    # Preserve original instrument (int 0-73) before scaling — "instrument" is in features
    train_instrument = train_data["instrument"].copy()
    val_instrument = val_data["instrument"].copy()
    train_data[features] = scaler.fit_transform(train_data[features])
    val_data[features] = scaler.transform(val_data[features])
    train_data["instrument"] = train_instrument
    val_data["instrument"] = val_instrument

    # 4.3 滑动验证集: 使用验证期内数据
    print("\n[验证集-滑动]")
    full_df_dates = pd.to_datetime(full_df["日期"])
    val_context_start = val_start - pd.tseries.offsets.BDay(
        config["sequence_length"] - 1
    )
    full_df["日期"] = full_df_dates
    val_sliding_df = full_df[
        (full_df["日期"] >= val_context_start) & (full_df["日期"] <= val_end)
    ]
    print(
        f"滑动验证取数范围: {val_context_start.strftime('%Y-%m-%d')} 到 {val_end.strftime('%Y-%m-%d')}"
    )

    val_sliding_data, _ = preprocess_val_data(val_sliding_df, stockid2idx=stockid2idx)
    print(
        f"滑动验证预处理后: {len(val_sliding_data)} 行, {val_sliding_data['日期'].nunique()} 唯一日期"
    )
    val_sliding_data[features] = val_sliding_data[features].replace(
        [np.inf, -np.inf], np.nan
    )
    val_sliding_data = val_sliding_data.dropna(subset=features)
    val_sliding_instrument = val_sliding_data["instrument"].copy()
    val_sliding_data[features] = scaler.transform(val_sliding_data[features])
    val_sliding_data["instrument"] = val_sliding_instrument

    # 提取 HS300 收益率用于计算超额收益
    # 注意：ETF标签使用5日开盘收益率，HS300也要用相同的计算方式
    hs300_code = "510300.XSHG"
    hs300_data = full_df[full_df["股票代码"] == hs300_code].sort_values("日期").copy()
    hs300_data["open_t1"] = hs300_data["开盘"].shift(-1)
    hs300_data["open_t5"] = hs300_data["开盘"].shift(-5)
    hs300_data["label"] = (hs300_data["open_t5"] - hs300_data["open_t1"]) / (hs300_data["open_t1"] + 1e-12)
    hs300_labels_map = {}
    for _, row in hs300_data.dropna(subset=["label"]).iterrows():
        hs300_labels_map[str(row["日期"])[:10]] = row["label"]

    train_sequences, train_targets, train_relevance, train_stock_indices, _, train_hs300_rets, _ = (
        create_ranking_dataset_vectorized(
            train_data, features, config["sequence_length"], ranking_data_path=None,
            hs300_labels_map=hs300_labels_map
        )
    )
    (
        val_sequences,
        val_targets,
        val_relevance,
        val_stock_indices,
        val_first_window_date,
        val_hs300_rets,
        _,
    ) = create_ranking_dataset_vectorized(
        val_data,
        features,
        config["sequence_length"],
        ranking_data_path=None,
        min_window_end_date=val_start.strftime("%Y-%m-%d"),
        hs300_labels_map=hs300_labels_map,
    )

    # 记录按周验证的第一个窗口结束日期，用于滑动验证对齐
    val_first_sample_date = (
        pd.to_datetime(val_first_window_date) if val_first_window_date else val_start
    )

    # 滑动验证使用val_first_sample_date作为min_window_end_date，与按周验证对齐
    min_date_for_sliding = val_first_sample_date.strftime("%Y-%m-%d")

    # 提取 HS300 收益率用于计算超额收益
    # 注意：ETF标签使用5日开盘收益率，HS300也要用相同的计算方式
    hs300_code = "510300.XSHG"
    hs300_data = full_df[full_df["股票代码"] == hs300_code].sort_values("日期").copy()
    hs300_data["open_t1"] = hs300_data["开盘"].shift(-1)
    hs300_data["open_t5"] = hs300_data["开盘"].shift(-5)
    hs300_data["label"] = (hs300_data["open_t5"] - hs300_data["open_t1"]) / (hs300_data["open_t1"] + 1e-12)
    hs300_labels_map = {}
    for _, row in hs300_data.dropna(subset=["label"]).iterrows():
        hs300_labels_map[str(row["日期"])[:10]] = row["label"]

    (
        val_sliding_sequences,
        val_sliding_targets,
        val_sliding_relevance,
        val_sliding_stock_indices,
        _,
        val_sliding_hs300_rets,
        val_sliding_dates,
    ) = create_ranking_dataset_vectorized(
        val_sliding_data,
        features,
        config["sequence_length"],
        ranking_data_path=None,
        min_window_end_date=min_date_for_sliding,
        require_natural_day_consecutive=False,
        hs300_labels_map=hs300_labels_map,
    )

    preprocessed_data = {
        "features": features,
        "stockid2idx": stockid2idx,
        "num_stocks": len(stockid2idx),
        "val_start": val_start.strftime("%Y-%m-%d"),
        "train_sequences": train_sequences,
        "train_targets": train_targets,
        "train_relevance": train_relevance,
        "train_stock_indices": train_stock_indices,
        "train_hs300_rets": train_hs300_rets,
        "val_sequences": val_sequences,
        "val_targets": val_targets,
        "val_relevance": val_relevance,
        "val_stock_indices": val_stock_indices,
        "val_sliding_sequences": val_sliding_sequences,
        "val_sliding_targets": val_sliding_targets,
        "val_sliding_relevance": val_sliding_relevance,
        "val_sliding_stock_indices": val_sliding_stock_indices,
        "val_hs300_rets": val_hs300_rets,
        "val_sliding_hs300_rets": val_sliding_hs300_rets,
        "val_sliding_dates": val_sliding_dates,
    }

    preprocessed_path = os.path.join(search_dir, "preprocessed_data.pkl").replace("\\", "/")
    joblib.dump(preprocessed_data, preprocessed_path)
    joblib.dump(scaler, os.path.join(search_dir, "scaler.pkl").replace("\\", "/"))

    print(f"Preprocessed data saved to {preprocessed_path}")
    print(f"Train samples: {len(train_sequences)}")
    print(f"Val samples (weekly): {len(val_sequences)}")
    print(f"Val samples (sliding): {len(val_sliding_sequences)}")

    return preprocessed_data, scaler


def run_experiment(
    params, config, preprocessed_data, scaler, search_dir, exp_idx, config_module
):
    """运行单个实验"""
    set_seed(42)

    output_dir = os.path.join(search_dir, f"exp_{exp_idx}")
    os.makedirs(output_dir, exist_ok=True)

    exp_config = config.copy()
    exp_config.update(config)  # 继承配置中的top_k等参数
    model_type = params.get("model_type", exp_config.get("model_type", "transformer"))
    model_defaults = config_module.get_model_config(model_type)
    exp_config.update(model_defaults)  # 先写入模型默认参数（如 d_ff）
    exp_config.update(params)  # 再用搜索参数覆盖
    exp_config["output_dir"] = output_dir
    exp_config["num_epochs"] = config.get("num_epochs", 15)

    # 保存实验参数到config.json
    with open(os.path.join(output_dir, "config.json"), "w") as f:
        json.dump(exp_config, f, indent=2)

    # 复制scaler到实验目录，方便回测
    import shutil

    scaler_path = os.path.join(search_dir, "scaler.pkl")
    if os.path.exists(scaler_path):
        shutil.copy(scaler_path, os.path.join(output_dir, "scaler.pkl"))

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    features = preprocessed_data["features"]
    num_stocks = preprocessed_data["num_stocks"]

    train_dataset = RankingDataset(
        preprocessed_data["train_sequences"],
        preprocessed_data["train_targets"],
        preprocessed_data["train_relevance"],
        preprocessed_data["train_stock_indices"],
        preprocessed_data.get("train_hs300_rets"),
    )
    val_dataset = RankingDataset(
        preprocessed_data["val_sequences"],
        preprocessed_data["val_targets"],
        preprocessed_data["val_relevance"],
        preprocessed_data["val_stock_indices"],
        preprocessed_data.get("val_hs300_rets"),
    )
    val_sliding_dataset = RankingDataset(
        preprocessed_data["val_sliding_sequences"],
        preprocessed_data["val_sliding_targets"],
        preprocessed_data["val_sliding_relevance"],
        preprocessed_data["val_sliding_stock_indices"],
        preprocessed_data.get("val_sliding_hs300_rets"),
    )

    train_loader = torch.utils.data.DataLoader(
        train_dataset,
        batch_size=exp_config["batch_size"],
        shuffle=True,
        collate_fn=collate_fn,
        num_workers=0,
    )
    val_loader = torch.utils.data.DataLoader(
        val_dataset,
        batch_size=exp_config["batch_size"],
        shuffle=False,
        collate_fn=collate_fn,
        num_workers=0,
    )
    val_sliding_loader = torch.utils.data.DataLoader(
        val_sliding_dataset,
        batch_size=exp_config["batch_size"],
        shuffle=False,
        collate_fn=collate_fn,
        num_workers=0,
    )

    model_type = exp_config.get("model_type", "transformer")
    model_config = config_module.get_model_config(model_type)
    model_config.update(exp_config)
    model = create_model(model_type, len(features), model_config, num_stocks).to(device)

    criterion = WeightedRankingLoss(
        temperature=1.0,
        k=exp_config.get("top_k", 5),
        weight_factor=exp_config.get("top5_weight", 2.0),
        pairwise_weight=exp_config.get("pairwise_weight", 1),
        base_weight=exp_config.get("base_weight", 1.0),
    )
    optimizer = torch.optim.AdamW(model.parameters(), lr=exp_config["learning_rate"])
    scheduler = torch.optim.lr_scheduler.LinearLR(
        optimizer,
        start_factor=1.0,
        end_factor=0.2,
        total_iters=exp_config["num_epochs"],
    )

    search_metric = exp_config.get("search_metric", "ndcg")

    best_score = -float("inf")
    best_sliding_score = -float("inf")
    best_sliding_score_all = -float("inf")
    best_ndcg = -float("inf")
    best_metric_val = -float("inf")
    best_epoch = -1
    best_sliding_epoch = -1
    best_ndcg_epoch = -1
    best_metric_epoch = -1

    epoch_scores_file = os.path.join(output_dir, "epoch_scores.txt")
    with open(epoch_scores_file, "w") as f:
        f.write(
            "epoch,weekly_score,sliding_score,train_loss,eval_loss,eval_sliding_loss,lr,epoch_time_sec,"
            "weekly_pred_return_sum,weekly_max_return_sum,weekly_random_return_sum,"
            "weekly_excess_return,weekly_hit_rate,weekly_proximity_score,weekly_rank_ic,"
            "weekly_precision,weekly_recall,weekly_mrr,weekly_ndcg,weekly_std_pred_return,weekly_std_final_score,"
            "sliding_pred_return_sum,sliding_max_return_sum,sliding_random_return_sum,"
            "sliding_excess_return,sliding_hit_rate,sliding_proximity_score,sliding_rank_ic,"
            "sliding_precision,sliding_recall,sliding_mrr,sliding_ndcg,sliding_std_pred_return,sliding_std_final_score\n"
        )

    sharpe_ratio = None

    for epoch in range(exp_config["num_epochs"]):
        epoch_start_time = time.time()

        train_loss, train_metrics = train_ranking_model(
            model, train_loader, criterion, optimizer, device, epoch, None
        )

        eval_loss, eval_metrics = evaluate_ranking_model(
            model, val_loader, criterion, device, None, epoch
        )

        eval_sliding_loss, eval_sliding_metrics = evaluate_ranking_model(
            model, val_sliding_loader, criterion, device, None, epoch
        )

        epoch_time = time.time() - epoch_start_time
        current_lr = optimizer.param_groups[0]["lr"]
        current_score = eval_metrics.get("final_score", 0.0)
        current_sliding_score = eval_sliding_metrics.get("final_score", 0.0)

        # Weekly metrics
        weekly_pred_sum = eval_metrics.get("pred_return_sum", 0.0)
        weekly_max_sum = eval_metrics.get("max_return_sum", 0.0)
        weekly_random_sum = eval_metrics.get("random_return_sum", 0.0)
        weekly_excess = eval_metrics.get("excess_return", 0.0)
        
        weekly_hit_rate = eval_metrics.get("hit_rate", 0.0)
        weekly_proximity = eval_metrics.get("proximity_score", 0.0)
        weekly_rank_ic = eval_metrics.get("rank_ic", 0.0)
        weekly_precision = eval_metrics.get("precision", 0.0)
        weekly_recall = eval_metrics.get("recall", 0.0)
        weekly_mrr = eval_metrics.get("mrr", 0.0)
        weekly_ndcg = eval_metrics.get("ndcg", 0.0)
        weekly_std_pred = eval_metrics.get("std_pred_return", 0.0)
        weekly_std_score = eval_metrics.get("std_final_score", 0.0)

        # Sliding metrics
        sliding_pred_sum = eval_sliding_metrics.get("pred_return_sum", 0.0)
        sliding_max_sum = eval_sliding_metrics.get("max_return_sum", 0.0)
        sliding_random_sum = eval_sliding_metrics.get("random_return_sum", 0.0)
        sliding_excess = eval_sliding_metrics.get("excess_return", 0.0)
        
        sliding_hit_rate = eval_sliding_metrics.get("hit_rate", 0.0)
        sliding_proximity = eval_sliding_metrics.get("proximity_score", 0.0)
        sliding_rank_ic = eval_sliding_metrics.get("rank_ic", 0.0)
        sliding_precision = eval_sliding_metrics.get("precision", 0.0)
        sliding_recall = eval_sliding_metrics.get("recall", 0.0)
        sliding_mrr = eval_sliding_metrics.get("mrr", 0.0)
        sliding_ndcg = eval_sliding_metrics.get("ndcg", 0.0)
        sliding_std_pred = eval_sliding_metrics.get("std_pred_return", 0.0)
        sliding_std_score = eval_sliding_metrics.get("std_final_score", 0.0)

        with open(epoch_scores_file, "a") as f:
            f.write(
                f"{epoch + 1},{current_score:.6f},{current_sliding_score:.6f},"
                f"{train_loss:.6f},{eval_loss:.6f},{eval_sliding_loss:.6f},"
                f"{current_lr:.2e},{epoch_time:.2f},"
                f"{weekly_pred_sum:.6f},{weekly_max_sum:.6f},{weekly_random_sum:.6f},"
                f"{weekly_excess:.6f},{weekly_hit_rate:.6f},{weekly_proximity:.6f},{weekly_rank_ic:.6f},"
                f"{weekly_precision:.6f},{weekly_recall:.6f},{weekly_mrr:.6f},{weekly_ndcg:.6f},"
                f"{weekly_std_pred:.6f},{weekly_std_score:.6f},"
                f"{sliding_pred_sum:.6f},{sliding_max_sum:.6f},{sliding_random_sum:.6f},"
                f"{sliding_excess:.6f},{sliding_hit_rate:.6f},{sliding_proximity:.6f},{sliding_rank_ic:.6f},"
                f"{sliding_precision:.6f},{sliding_recall:.6f},{sliding_mrr:.6f},{sliding_ndcg:.6f},"
                f"{sliding_std_pred:.6f},{sliding_std_score:.6f}\n"
            )

        if exp_config.get("save_predictions", False):
            model.eval()
            with torch.no_grad():
                # Weekly predictions
                batch_preds_weekly = []
                for batch in val_loader:
                    sequences = batch["sequences"].to(device)
                    outputs = model(sequences)
                    masks = batch["masks"].to(device)
                    masked_outputs = outputs * masks + (1 - masks) * (-1e9)
                    batch_preds_weekly.append(masked_outputs.cpu().numpy())
                epoch_preds_weekly = np.concatenate(batch_preds_weekly, axis=0)
                np.save(
                    os.path.join(output_dir, f"preds_weekly_epoch{epoch + 1}.npy"),
                    epoch_preds_weekly,
                )

                # Sliding predictions
                batch_preds_sliding = []
                for batch in val_sliding_loader:
                    sequences = batch["sequences"].to(device)
                    outputs = model(sequences)
                    masks = batch["masks"].to(device)
                    masked_outputs = outputs * masks + (1 - masks) * (-1e9)
                    batch_preds_sliding.append(masked_outputs.cpu().numpy())
                epoch_preds_sliding = np.concatenate(batch_preds_sliding, axis=0)
                np.save(
                    os.path.join(output_dir, f"preds_sliding_epoch{epoch + 1}.npy"),
                    epoch_preds_sliding,
                )

        current_ndcg = eval_sliding_metrics.get("ndcg", 0.0)

        if current_ndcg > best_metric_val:
            best_metric_val = current_ndcg
            best_metric_epoch = epoch + 1
            torch.save(model.state_dict(), os.path.join(output_dir, "best_model.pth"))

        if current_score > best_score:
            best_score = current_score
            best_sliding_score = current_sliding_score
            best_epoch = epoch + 1

        if current_sliding_score > best_sliding_score_all:
            best_sliding_score_all = current_sliding_score
            best_sliding_epoch = epoch + 1
            torch.save(
                model.state_dict(), os.path.join(output_dir, "best_model_sliding.pth")
            )

        if sliding_ndcg > best_ndcg:
            best_ndcg = sliding_ndcg
            best_ndcg_epoch = epoch + 1
            torch.save(
                model.state_dict(), os.path.join(output_dir, "best_model_ndcg.pth")
            )

    if not os.path.exists(os.path.join(output_dir, "best_model_sliding.pth")):
        torch.save(
            model.state_dict(), os.path.join(output_dir, "best_model_sliding.pth")
        )

    if not os.path.exists(os.path.join(output_dir, "best_model_ndcg.pth")):
        torch.save(
            model.state_dict(), os.path.join(output_dir, "best_model_ndcg.pth")
        )

    scheduler.step()

    # 保存验证集的 targets（只保存一次）
    def save_targets(dataloader, desc):
        """保存验证集的 targets"""
        all_targets = []
        all_masks = []
        all_stock_indices = []

        for batch in dataloader:
            targets = batch["targets"].cpu()
            masks = batch["masks"].cpu()
            stock_indices = batch["stock_indices"]
            all_targets.append(targets)
            all_masks.append(masks)
            all_stock_indices.extend(stock_indices.tolist())

        all_targets = torch.cat(all_targets, dim=0).numpy()
        all_masks = torch.cat(all_masks, dim=0).numpy()

        np.save(os.path.join(output_dir, f"targets_{desc}.npy"), all_targets)
        np.save(os.path.join(output_dir, f"masks_{desc}.npy"), all_masks)

        with open(os.path.join(output_dir, f"stock_indices_{desc}.txt"), "w") as f:
            f.write(str(all_stock_indices))

        print(f"Saved {desc} targets: {all_targets.shape}")
        return all_targets

    # 只保存 targets
    print("Saving targets...")
    save_targets(val_loader, "weekly")
    save_targets(val_sliding_loader, "sliding")

    # 可视化 epoch_scores
    try:
        plot_epoch_scores(output_dir, search_metric)
    except Exception as e:
        print(f"  Warning: failed to plot epoch scores: {e}")

    # === 验证集回测 (on all checkpoints, pick best by Sharpe) ===
    def _eval_checkpoint(ckpt_path, label=""):
        """Load ckpt, run inference, backtest, return 5-sub Sharpe."""
        if not os.path.exists(ckpt_path):
            return None
        model.load_state_dict(torch.load(ckpt_path, map_location=device))
        model.eval()
        idx2stockid = {v: k for k, v in preprocessed_data["stockid2idx"].items()}
        val_sliding_dates = preprocessed_data.get("val_sliding_dates", [])

        preds = {}
        sample_idx = 0
        with torch.no_grad():
            for batch in val_sliding_loader:
                sequences = batch["sequences"].to(device)
                outputs = model(sequences)
                masks = batch["masks"].cpu().numpy()
                stock_indices = batch["stock_indices"].cpu().numpy()

                for bi in range(outputs.size(0)):
                    date_str = val_sliding_dates[sample_idx + bi]
                    valid_mask = masks[bi] > 0
                    valid_scores = outputs[bi].cpu().numpy()[valid_mask]
                    valid_stocks = stock_indices[bi][valid_mask]

                    if date_str not in preds:
                        preds[date_str] = []
                    for score, sidx in zip(valid_scores, valid_stocks):
                        sid = int(sidx)
                        if sid not in idx2stockid:
                            continue
                        preds[date_str].append({"stock_id": idx2stockid[sid], "score": float(score)})
                sample_idx += outputs.size(0)

        for date_str in preds:
            preds[date_str].sort(key=lambda x: x["score"], reverse=True)
            for rank, item in enumerate(preds[date_str]):
                item["rank"] = rank + 1

        from backtest import run_backtest_from_predictions
        from metrics import compute_window_metrics as _cwm

        val_start = preprocessed_data.get("val_start", config.get("val_start_date", "2025-01-01"))
        val_end = config.get("val_end_date", "2025-12-31")
        bt_top_k = exp_config.get("top_k", 3)

        # Weekly
        bt = run_backtest_from_predictions(
            predictions_dict=preds, data_path=backtest_data_path,
            start_date=val_start, end_date=val_end, top_k=bt_top_k,
            rebalance_days=5, position_pct=0.95, initial_capital=100000,
            weight_strategy="equal", trade_mode="open", log=False, verbose=False,
        )
        bm = _cwm(bt.equity_curve, 100000)
        weekly_sharpe = bm.get("sharpe_ratio", 0.0)

        # 5-sub
        all_dates = sorted(preds.keys())
        sub_rets, sub_curves = [], []
        for offset in range(5):
            if offset >= len(all_dates):
                break
            try:
                sr = run_backtest_from_predictions(
                    predictions_dict=preds, data_path=backtest_data_path,
                    start_date=all_dates[offset], end_date=val_end, top_k=bt_top_k,
                    rebalance_days=5, position_pct=0.95, initial_capital=100000 / 5,
                    weight_strategy="equal", first_rebalance_date=all_dates[offset],
                    trade_mode="open", log=False, verbose=False,
                )
                sub_rets.append(sr.strategy_return)
                ec = sr.equity_curve[["date", "total_value"]].copy()
                ec = ec.rename(columns={"total_value": "v"})
                ec["v"] = ec["v"] * 5
                sub_curves.append(ec)
            except Exception:
                pass

        sub_sharpe = 0.0
        if sub_curves:
            all_dts = sorted(set().union(*[set(ec["date"]) for ec in sub_curves]))
            aligned = pd.DataFrame({"date": all_dts})
            for i, ec in enumerate(sub_curves):
                ec = ec.rename(columns={"v": f"v_{i}"})
                aligned = aligned.merge(ec[["date", f"v_{i}"]], on="date", how="left")
            vcols = [c for c in aligned.columns if c.startswith("v_")]
            aligned["total_value"] = aligned[vcols].mean(axis=1)
            aligned = aligned.dropna(subset=["total_value"]).sort_values("date").reset_index(drop=True)
            sub_sharpe = _cwm(aligned, 100000).get("sharpe_ratio", 0.0)

        print(f"  {label}: weekly_sharpe={weekly_sharpe:.2f}  5sub_sharpe={sub_sharpe:.2f}"
              f"  return={bt.strategy_return:.2f}%")
        return sub_sharpe, preds

    try:
        project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
        backtest_data_path = os.path.join(project_root, "etf_data", "etf_74.csv")

        if not os.path.exists(backtest_data_path):
            print(f"  Warning: backtest data not found at {backtest_data_path}, skipping backtest")
        else:
            ckpt_dir = output_dir
            ckpts = {
                "best_model.pth": "ndcg(best_model)",
                "best_model_sliding.pth": "sliding_score",
                "best_model_ndcg.pth": "ndcg(best_ndcg)",
            }
            best_sharpe, best_preds, best_label = -1e9, None, ""
            for ckpt_name, label in ckpts.items():
                ckpt_path = os.path.join(ckpt_dir, ckpt_name)
                result = _eval_checkpoint(ckpt_path, label)
                if result is None:
                    continue
                sub_sharpe, preds = result
                if sub_sharpe > best_sharpe:
                    best_sharpe = sub_sharpe
                    best_preds = preds
                    best_label = label

            if best_preds is not None:
                sharpe_ratio = best_sharpe
                predictions_dict = best_preds
                # Save val_predictions from best checkpoint
                val_pred_path = os.path.join(output_dir, "val_predictions.json")
                with open(val_pred_path, "w") as f:
                    json.dump(predictions_dict, f, indent=2)
                n_dates = len(predictions_dict)
                n_stocks = sum(len(v) for v in predictions_dict.values()) // max(n_dates, 1)
                print(f"  Saved val_predictions.json ({n_dates} dates, ~{n_stocks} stocks/date)")
                print(f"  Best checkpoint by 5-sub Sharpe: {best_label} ({best_sharpe:.4f})")

                # Overwrite best_model_sliding.pth with Sharpe-best checkpoint
                best_ckpt_name = {"ndcg(best_model)": "best_model.pth",
                                   "sliding_score": "best_model_sliding.pth",
                                   "ndcg(best_ndcg)": "best_model_ndcg.pth"}.get(best_label, "")
                if best_ckpt_name and best_ckpt_name != "best_model_sliding.pth":
                    src = os.path.join(ckpt_dir, best_ckpt_name)
                    dst = os.path.join(ckpt_dir, "best_model_sliding.pth")
                    import shutil
                    shutil.copy2(src, dst)
                    print(f"  Updated best_model_sliding.pth ← {best_ckpt_name} (Sharpe={best_sharpe:.4f})")

    except Exception as e:
        print(f"  Warning: Backtest evaluation failed: {e}")
        traceback.print_exc()

    trial_sharpe = sharpe_ratio if sharpe_ratio is not None else best_metric_val
    with open(os.path.join(output_dir, "final_score.txt"), "w") as f:
        f.write(
            f"Best epoch: {best_epoch}\nBest weekly_final_score: {best_score:.6f}\n"
            f"Best sliding_epoch: {best_sliding_epoch}\nBest sliding_final_score: {best_sliding_score_all:.6f}\n"
            f"Best ndcg_epoch: {best_ndcg_epoch}\nBest sliding_ndcg: {best_ndcg:.6f}\n"
            f"Best ndcg_epoch (best_model.pth): {best_metric_epoch}\nBest ndcg: {best_metric_val:.6f}\n"
            f"Sharpe (trial objective): {trial_sharpe:.6f}\n"
        )

    # Thorough cleanup
    del model
    del optimizer
    del criterion
    del scheduler

    # Delete data loaders to free memory
    del train_loader
    del val_loader
    del val_sliding_loader
    del train_dataset
    del val_dataset
    del val_sliding_dataset

    # Clear CUDA cache
    if torch.cuda.is_available():
        torch.cuda.synchronize()
        torch.cuda.empty_cache()
    gc.collect()

    return {
        "success": True,
        "score": trial_sharpe,
        "metric": "sharpe",
        "model_type": exp_config.get("model_type", ""),
        "sliding_score": best_sliding_score,
        "best_epoch": best_epoch,
        "params": params,
    }


def plot_epoch_scores(output_dir, search_metric="ndcg"):
    """Read epoch_scores.txt and save a multi-panel visualization."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import pandas as pd

    csv_path = os.path.join(output_dir, "epoch_scores.txt")
    df = pd.read_csv(csv_path)
    epochs = df["epoch"]

    fig, axes = plt.subplots(2, 3, figsize=(16, 8))
    fig.suptitle(f"Training History — {os.path.basename(output_dir)}", fontsize=14, fontweight="bold")

    # 1. Loss
    ax = axes[0, 0]
    for col, label, style in [("train_loss", "Train", "-"), ("eval_loss", "Val(weekly)", "--"), ("eval_sliding_loss", "Val(sliding)", ":")]:
        if col in df.columns:
            ax.plot(epochs, df[col], style, label=label)
    ax.set_title("Loss"); ax.set_xlabel("Epoch"); ax.legend(); ax.grid(True, alpha=0.3)

    # 2. Main Score
    ax = axes[0, 1]
    for col, label, style in [("weekly_score", "Weekly", "-"), ("sliding_score", "Sliding", "--")]:
        if col in df.columns:
            ax.plot(epochs, df[col], style, label=label)
    if search_metric in df.columns:
        ax.plot(epochs, df[search_metric], ":", label=f"Best({search_metric})", alpha=0.8)
    ax.set_title("Score"); ax.set_xlabel("Epoch"); ax.legend(); ax.grid(True, alpha=0.3)

    # 3. NDCG
    ax = axes[0, 2]
    for col, label, style in [("weekly_ndcg", "Weekly", "-"), ("sliding_ndcg", "Sliding", "--")]:
        if col in df.columns:
            ax.plot(epochs, df[col], style, label=label)
    ax.set_title("NDCG"); ax.set_xlabel("Epoch"); ax.legend(); ax.grid(True, alpha=0.3)

    # 4. Excess Return
    ax = axes[1, 0]
    for col, label, style in [("weekly_excess_return", "Weekly", "-"), ("sliding_excess_return", "Sliding", "--")]:
        if col in df.columns:
            ax.plot(epochs, df[col], style, label=label)
    ax.set_title("Excess Return"); ax.set_xlabel("Epoch"); ax.legend(); ax.grid(True, alpha=0.3)

    # 5. Hit Rate & MRR
    ax = axes[1, 1]
    for col, label, style in [("weekly_hit_rate", "Hit(weekly)", "-"), ("sliding_hit_rate", "Hit(sliding)", "--"),
                               ("weekly_mrr", "MRR(weekly)", "-."), ("sliding_mrr", "MRR(sliding)", ":")]:
        if col in df.columns:
            ax.plot(epochs, df[col], style, label=label)
    ax.set_title("Hit Rate / MRR"); ax.set_xlabel("Epoch"); ax.legend(fontsize=8); ax.grid(True, alpha=0.3)

    # 6. Rank IC
    ax = axes[1, 2]
    for col, label, style in [("weekly_rank_ic", "Weekly", "-"), ("sliding_rank_ic", "Sliding", "--")]:
        if col in df.columns:
            ax.plot(epochs, df[col], style, label=label)
    ax.axhline(0, color="gray", ls=":", alpha=0.5)
    ax.set_title("Rank IC"); ax.set_xlabel("Epoch"); ax.legend(); ax.grid(True, alpha=0.3)

    plt.tight_layout()
    out_path = os.path.join(output_dir, "epoch_scores.png")
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved epoch_scores.png ({os.path.getsize(out_path)/1024:.0f}KB)")


def main(args):
    total_start_time = time.time()

    config_name = args.config
    config_module = __import__(config_name, fromlist=["config"])
    config = config_module.config.copy()

    # CLI overrides
    if args.model_type is not None:
        config["model_type"] = args.model_type
    if args.feature_num is not None:
        config["feature_num"] = args.feature_num
    if args.data_file is not None:
        config["data_file"] = args.data_file
    if args.topk is not None:
        config["top_k"] = args.topk
    if args.sequence_length is not None:
        config["sequence_length"] = args.sequence_length
    if args.N is not None:
        config["N"] = args.N
    if args.search_metric is not None:
        config["search_metric"] = args.search_metric
    if args.save_predictions:
        config["save_predictions"] = True
    if args.val_start_date is not None:
        config["val_start_date"] = args.val_start_date
    if args.val_end_date is not None:
        config["val_end_date"] = args.val_end_date

    # Recompute output_dir after CLI overrides
    mt = config["model_type"]
    n = config.get("N", 74)
    tk = config.get("top_k", 3)
    date_tag = f"_{config['val_start_date']}_{config['val_end_date']}" if config.get("val_start_date") else ""
    config["output_dir"] = f"./model/search_{mt}_{n}_{tk}{date_tag}"

    # Clear any previous GPU memory state
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        # Enable memory allocator settings to reduce fragmentation
        torch.cuda.set_per_process_memory_fraction(0.8)
    gc.collect()

    data_file = config.get("data_file", "data.csv")
    file_prefix = data_file.split("_")[1].split(".")[0]
    search_method = args.search_method or config.get("search_method", "bayesian")
    method_prefix = "grid" if search_method == "grid" else "bayes"
    search_dir = config["output_dir"].replace("search_", f"{method_prefix}_")
    os.makedirs(search_dir, exist_ok=True)

    preprocessed_path = os.path.join(search_dir, "preprocessed_data.pkl")

    if args.resume and os.path.exists(preprocessed_path):
        print(f"Loading preprocessed data from {preprocessed_path}...")
        preprocessed_data = joblib.load(preprocessed_path)
        scaler = joblib.load(os.path.join(search_dir, "scaler.pkl"))
    else:
        preprocessed_data, scaler = preprocess_and_save(config, search_dir)

    search_metric_display = config.get("search_metric", "ndcg")
    print(f"Optimization metric: {search_metric_display}")

    if search_method == "grid":
        # ========== 网格搜索 ==========
        PARAM_GRID = config_module.get_param_grid(config["model_type"])

        results = []
        results_path = os.path.join(search_dir, "search_results.json")
        if args.resume and os.path.exists(results_path):
            with open(results_path, "r") as f:
                results = json.load(f)
            print(f"Resuming from {len(results)} existing results...")

            completed_indices = set()
            for r in results:
                completed_indices.add(r.get("exp_idx", -1))

            import glob
            exp_dirs = glob.glob(os.path.join(search_dir, "exp_*"))
            num_epochs = config.get("num_epochs", 30)
            for d in exp_dirs:
                dirname = os.path.basename(d)
                try:
                    idx = int(dirname.split("_")[1])
                except:
                    continue
                final_score_file = os.path.join(d, "final_score.txt")
                epoch_scores_file = os.path.join(d, "epoch_scores.txt")
                if os.path.exists(final_score_file):
                    completed_indices.add(idx)
                elif os.path.exists(epoch_scores_file):
                    try:
                        df = pd.read_csv(epoch_scores_file)
                        if len(df) >= num_epochs:
                            completed_indices.add(idx)
                    except:
                        pass

            start_idx = max(completed_indices) + 1 if completed_indices else 0
            print(f"Continuing from experiment {start_idx + 1}")
        else:
            start_idx = 0

        for i, params in enumerate(PARAM_GRID):
            if i < start_idx:
                continue

            output_dir = os.path.join(search_dir, f"exp_{i}")
            if args.resume and os.path.exists(output_dir):
                final_score_file = os.path.join(output_dir, "final_score.txt")
                epoch_scores_file = os.path.join(output_dir, "epoch_scores.txt")
                skip = False
                if os.path.exists(final_score_file):
                    skip = True
                elif os.path.exists(epoch_scores_file):
                    try:
                        df = pd.read_csv(epoch_scores_file)
                        if len(df) >= config.get("num_epochs", 30):
                            skip = True
                    except:
                        pass
                if skip:
                    print(f"\n{'=' * 50}")
                    print(f"Experiment {i + 1}/{len(PARAM_GRID)} - SKIPPED (already completed)")
                    print(f"Params: {params}")
                    print(f"{'=' * 50}")
                    continue

            print(f"\n{'=' * 50}")
            print(f"Experiment {i + 1}/{len(PARAM_GRID)}")
            print(f"Model: {config.get('model_type', '?')}")
            print(f"Params: {params}")
            print(f"{'=' * 50}")

            if torch.cuda.is_available():
                torch.cuda.reset_peak_memory_stats()
                print(f"GPU before: alloc={torch.cuda.memory_allocated()/1e9:.2f}GB  reserved={torch.cuda.memory_reserved()/1e9:.2f}GB")

            start_time = time.time()
            result = run_experiment(params, config, preprocessed_data, scaler, search_dir, i, config_module)
            result["exp_idx"] = i
            elapsed = time.time() - start_time

            if torch.cuda.is_available():
                print(f"GPU after:  alloc={torch.cuda.memory_allocated()/1e9:.2f}GB  reserved={torch.cuda.memory_reserved()/1e9:.2f}GB  "
                      f"peak={torch.cuda.max_memory_allocated()/1e9:.2f}GB")
                torch.cuda.synchronize()
                torch.cuda.empty_cache()
            gc.collect()
            elapsed = time.time() - start_time

            results.append(result)
            with open(results_path, "w") as f:
                json.dump(results, f, indent=2)

            completed = i + 1
            remaining = len(PARAM_GRID) - completed
            avg_time = results[-1].get("time", elapsed) if results else elapsed
            if "time" not in results[-1]:
                results[-1]["time"] = elapsed

            print(f"\n📊 Experiment {i + 1} result:")
            print(f"   Model: {result.get('model_type', '?')}")
            print(f"   Metric: {result.get('metric', config.get('search_metric', 'ndcg'))}")
            print(f"   Score:  {result['score']:.6f}")
            print(f"   Sliding final_score: {result.get('sliding_score', 0):.6f}")
            print(f"   Best epoch: {result.get('best_epoch', '?')}")
            print(f"   ⏱️  Time: {elapsed:.1f}s")
            best_sofar = max(r['score'] for r in results if r['success'])
            print(f"   ✅ Best score so far: {best_sofar:.6f}")
            print(f"\n📊 Progress: {completed}/{len(PARAM_GRID)} ({completed / len(PARAM_GRID) * 100:.1f}%)")
            if remaining > 0:
                print(f"⏳ Est. remaining: {remaining * avg_time / 60:.1f} min")

    else:
        # ========== 贝叶斯搜索 (Optuna) ==========
        n_trials = args.n_trials or config.get("n_trials", 50)

        search_space_fn = config_module.get_search_space(config["model_type"])

        results_path = os.path.join(search_dir, "search_results.json")

        # 加载已有的结果（兼容旧格点搜索的结果或之前贝叶斯的结果）
        results = []
        if args.resume and os.path.exists(results_path):
            with open(results_path, "r") as f:
                results = json.load(f)
            print(f"Loaded {len(results)} existing results for best-score tracking.")

        optuna_db_path = os.path.join(search_dir, "optuna_study.db")
        optuna_storage = f"sqlite:///{optuna_db_path}"

        if args.fresh:
            for f_path in [optuna_db_path, optuna_db_path + "-journal"]:
                if os.path.exists(f_path):
                    os.remove(f_path)
                    print(f"Deleted old study: {f_path}")
            results = []
            study = optuna.create_study(
                study_name=f"etf_{config['model_type']}_{config.get('top_k', 3)}",
                direction="maximize",
                sampler=optuna.samplers.TPESampler(seed=42),
            )
        else:
            study = optuna.create_study(
                storage=optuna_storage,
                study_name=f"etf_{config['model_type']}_{config.get('top_k', 3)}",
                load_if_exists=True,
                direction="maximize",
                sampler=optuna.samplers.TPESampler(seed=42),
            )

        search_metric = config.get("search_metric", "ndcg")
        n_completed = len([t for t in study.trials if t.state == optuna.trial.TrialState.COMPLETE])
        remaining_trials = max(0, n_trials - n_completed)
        if len(study.trials) > 0:
            print(f"Optuna study loaded with {len(study.trials)} existing trials.")
            print(f"  Completed: {n_completed}, remaining: {remaining_trials}")
        else:
            print(f"Starting new Optuna study ({n_trials} trials).")
        print(f"Optimization metric: {search_metric}")

        def objective(trial):
            try:
                params = search_space_fn(trial)
            except Exception as e:
                print(f"  Trial {trial.number} search space error: {e}")
                raise optuna.exceptions.TrialPruned()

            exp_idx = trial.number

            output_dir = os.path.join(search_dir, f"exp_{exp_idx}")
            if args.resume and os.path.exists(output_dir):
                final_score_file = os.path.join(output_dir, "final_score.txt")
                if os.path.exists(final_score_file):
                    with open(final_score_file) as f:
                        for line in f:
                            target_key = "Sharpe (trial objective):"
                            if target_key in line:
                                score = float(line.split(":")[-1].strip())
                                return score

            print(f"\n{'=' * 50}")
            print(f"Trial {trial.number + 1}/{min(n_trials, remaining_trials + n_completed)} (Bayesian)")
            print(f"Model: {config.get('model_type', '?')}")
            print(f"Params: {params}")
            print(f"{'=' * 50}")

            if torch.cuda.is_available():
                torch.cuda.reset_peak_memory_stats()
                print(f"GPU before: alloc={torch.cuda.memory_allocated()/1e9:.2f}GB  reserved={torch.cuda.memory_reserved()/1e9:.2f}GB")

            start_time = time.time()
            try:
                result = run_experiment(params, config, preprocessed_data, scaler, search_dir, exp_idx, config_module)
            except Exception as e:
                print(f"  Trial {trial.number} failed during training: {e}")
                traceback.print_exc()
                raise optuna.exceptions.TrialPruned()
            elapsed = time.time() - start_time
            result["exp_idx"] = exp_idx

            if torch.cuda.is_available():
                print(f"GPU after:  alloc={torch.cuda.memory_allocated()/1e9:.2f}GB  reserved={torch.cuda.memory_reserved()/1e9:.2f}GB  "
                      f"peak={torch.cuda.max_memory_allocated()/1e9:.2f}GB")
                torch.cuda.synchronize()
                torch.cuda.empty_cache()
            gc.collect()

            results.append(result)
            with open(results_path, "w") as f:
                json.dump(results, f, indent=2)

            current_completed = [t for t in study.trials if t.state == optuna.trial.TrialState.COMPLETE]
            best_sofar = max(t.value for t in current_completed) if current_completed else result["score"]
            print(f"\n📊 Trial {trial.number + 1} result:")
            print(f"   Model: {result.get('model_type', '?')}")
            print(f"   Metric: {result.get('metric', search_metric)}")
            print(f"   Score:  {result['score']:.6f}")
            print(f"   Sliding final_score: {result.get('sliding_score', 0):.6f}")
            print(f"   Best epoch: {result.get('best_epoch', '?')}")
            print(f"   ⏱️  Time: {elapsed:.1f}s")
            print(f"   ✅ Best score so far: {best_sofar:.6f}")

            return result["score"]

        study.optimize(objective, n_trials=remaining_trials, show_progress_bar=True)

        # 后处理：将所有完成的 Optuna trial 同步到 results，确保 search_results.json 完整
        completed_trials = [t for t in study.trials if t.state == optuna.trial.TrialState.COMPLETE]
        for t in completed_trials:
            exp_idx = t.number
            if not any(r.get("exp_idx") == exp_idx for r in results):
                entry = {
                    "success": True,
                    "score": t.value,
                    "params": t.params,
                    "exp_idx": exp_idx,
                }
                results.append(entry)

        results.sort(key=lambda x: x.get("exp_idx", -1))

        with open(results_path, "w") as f:
            json.dump(results, f, indent=2)

        best_trial = study.best_trial
        print(f"\n{'=' * 50}")
        print(f"Bayesian search completed!")
        print(f"Total completed: {len(completed_trials)} trials")
        print(f"Best trial: #{best_trial.number}")
        print(f"Best params: {best_trial.params}")
        print(f"Best score: {best_trial.value:.4f}")

    total_elapsed = time.time() - total_start_time
    hours = int(total_elapsed // 3600)
    minutes = int((total_elapsed % 3600) // 60)
    seconds = int(total_elapsed % 60)

    print(f"\n{'=' * 50}")
    print("Search completed!")
    print(f"总用时: {hours}h {minutes}m {seconds}s ({total_elapsed:.1f}秒)")

    successful = [r for r in results if r["success"]]
    if successful:
        best = max(successful, key=lambda x: x["score"])
        print(f"\nBest: {best['params']}, score: {best['score']:.4f}")


if __name__ == "__main__":
    import argparse

    # 默认搜索的模型类型（不传 --model-type 时全部搜索）
    # SEARCH_MODEL_TYPES = ["itransformer", "gru", "tcn", "dlinear", "lstm", "timesnet", "nlinear", "patchtst", "mamba"]
    SEARCH_MODEL_TYPES = ["timesnet","dlinear","tcn","gru", "patchtst","itransformer"]

    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, default="config")
    parser.add_argument("--resume", action="store_true", default=True)
    parser.add_argument("--model-type", type=str, default=None,
                        help=f"模型类型，不传则搜索全部: {', '.join(SEARCH_MODEL_TYPES)}")
    parser.add_argument("--feature-num", type=str, default="39")
    parser.add_argument("--data-file", type=str, default="etf_74_train.csv")
    parser.add_argument("--topk", type=int, default=3)
    parser.add_argument("--sequence-length", type=int, default=60)
    parser.add_argument("--N", type=int, default=74)
    parser.add_argument("--val-start-date", type=str, default=None, help="验证集开始日期，如 2025-01-01")
    parser.add_argument("--val-end-date", type=str, default=None, help="验证集结束日期，如 2025-12-31")
    parser.add_argument("--search-method", type=str, default="bayesian", choices=["grid", "bayesian"])
    parser.add_argument("--n-trials", type=int, default=80)
    parser.add_argument("--search-metric", type=str, default="sharpe",
                        help="优化目标 (sharpe=夏普比率, ndcg=NDCG@K)")
    parser.add_argument("--fresh", action="store_true", help="删除旧的 Optuna study，重新开始")
    parser.add_argument("--save-predictions", action="store_true", help="保存每 epoch 的预测结果 npy 文件（默认不保存）")
    args = parser.parse_args()

    model_types = [args.model_type] if args.model_type else SEARCH_MODEL_TYPES
    for mt in model_types:
        args.model_type = mt
        print(f"\n{'=' * 60}")
        print(f"  搜索模型: {mt}  ({model_types.index(mt) + 1}/{len(model_types)})")
        print(f"{'=' * 60}")
        main(args)
