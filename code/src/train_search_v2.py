import os
import sys
import json
import time
import joblib
import pandas as pd
import numpy as np
import torch
import gc
import weakref
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
    """只执行一次特征工程，返回预处理后的数据"""
    # Set global config for train.py functions that use it
    for key in config:
        if key in train_config:
            train_config[key] = config[key]

    feature_num = config["feature_num"]
    data_path = config["data_path"]
    data_file = config.get("data_file", "train.csv")
    data_file_path = os.path.join(data_path, data_file)

    if not os.path.exists(data_file_path):
        raise FileNotFoundError(f"Data file not found: {data_file_path}")

    print(f"Loading data from {data_file_path}...")
    full_df = pd.read_csv(data_file_path)

    train_df, val_df, val_start = split_train_val_by_last_month(
        full_df, config["sequence_length"], config["val_months"]
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
    train_data[features] = scaler.fit_transform(train_data[features])
    val_data[features] = scaler.transform(val_data[features])

    # 4.3 滑动验证集: 使用最后2个月数据，但过滤掉验证集起始日期之前的数据
    print("\n[验证集-滑动]")
    full_df_dates = pd.to_datetime(full_df["日期"])
    last_date = full_df_dates.max()
    val_context_start = val_start - pd.tseries.offsets.BDay(
        config["sequence_length"] - 1
    )
    full_df["日期"] = full_df_dates
    val_sliding_df = full_df[
        (full_df["日期"] >= val_context_start) & (full_df["日期"] <= last_date)
    ]
    print(
        f"滑动验证取数范围: {val_context_start.strftime('%Y-%m-%d')} 到 {last_date.strftime('%Y-%m-%d')}"
    )

    val_sliding_data, _ = preprocess_val_data(val_sliding_df, stockid2idx=stockid2idx)
    print(
        f"滑动验证预处理后: {len(val_sliding_data)} 行, {val_sliding_data['日期'].nunique()} 唯一日期"
    )
    val_sliding_data[features] = val_sliding_data[features].replace(
        [np.inf, -np.inf], np.nan
    )
    val_sliding_data = val_sliding_data.dropna(subset=features)
    val_sliding_data[features] = scaler.transform(val_sliding_data[features])

    # 提取 HS300 收益率用于计算超额收益
    hs300_code = "510300.XSHG"
    hs300_data = full_df[full_df["股票代码"] == hs300_code].sort_values("日期").copy()
    hs300_data["label"] = hs300_data["收盘"].pct_change()
    hs300_labels_map = {}
    for _, row in hs300_data.iterrows():
        hs300_labels_map[str(row["日期"])[:10]] = row["label"]

    train_sequences, train_targets, train_relevance, train_stock_indices, _, train_hs300_rets = (
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
    hs300_code = "510300.XSHG"
    hs300_data = full_df[full_df["股票代码"] == hs300_code].sort_values("日期").copy()
    hs300_data["label"] = hs300_data["收盘"].pct_change()
    hs300_labels_map = {}
    for _, row in hs300_data.iterrows():
        hs300_labels_map[str(row["日期"])[:10]] = row["label"]

    (
        val_sliding_sequences,
        val_sliding_targets,
        val_sliding_relevance,
        val_sliding_stock_indices,
        _,
        val_sliding_hs300_rets,
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
    }

    preprocessed_path = os.path.join(search_dir, "preprocessed_data.pkl")
    joblib.dump(preprocessed_data, preprocessed_path)
    joblib.dump(scaler, os.path.join(search_dir, "scaler.pkl"))

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
    exp_config.update(params)
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

    best_score = -float("inf")
    best_sliding_score = -float("inf")
    best_sliding_score_all = -float("inf")
    best_epoch = -1
    best_sliding_epoch = -1

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

        # 保存当前epoch的预测结果到单独文件 (使用masked后的结果，和训练时评估一致)
        model.eval()
        with torch.no_grad():
            # Weekly predictions
            batch_preds_weekly = []
            for batch in val_loader:
                sequences = batch["sequences"].to(device)
                outputs = model(sequences)
                masks = batch["masks"].to(device)
                # 应用mask：将无效股票设为很小的负数
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
                # 应用mask
                masked_outputs = outputs * masks + (1 - masks) * (-1e9)
                batch_preds_sliding.append(masked_outputs.cpu().numpy())
            epoch_preds_sliding = np.concatenate(batch_preds_sliding, axis=0)
            np.save(
                os.path.join(output_dir, f"preds_sliding_epoch{epoch + 1}.npy"),
                epoch_preds_sliding,
            )

        if current_score > best_score:
            best_score = current_score
            best_sliding_score = current_sliding_score
            best_epoch = epoch + 1
            torch.save(model.state_dict(), os.path.join(output_dir, "best_model.pth"))

        if current_sliding_score > best_sliding_score_all:
            best_sliding_score_all = current_sliding_score
            best_sliding_epoch = epoch + 1
            torch.save(
                model.state_dict(), os.path.join(output_dir, "best_model_sliding.pth")
            )

    if not os.path.exists(os.path.join(output_dir, "best_model_sliding.pth")):
        torch.save(
            model.state_dict(), os.path.join(output_dir, "best_model_sliding.pth")
        )

    scheduler.step()

    with open(os.path.join(output_dir, "final_score.txt"), "w") as f:
        f.write(
            f"Best epoch: {best_epoch}\nBest weekly_final_score: {best_score:.6f}\nBest sliding_epoch: {best_sliding_epoch}\nBest sliding_final_score: {best_sliding_score_all:.6f}\n"
        )

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
        "score": best_score,
        "sliding_score": best_sliding_score,
        "best_epoch": best_epoch,
        "params": params,
    }


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

    # Clear any previous GPU memory state
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        # Enable memory allocator settings to reduce fragmentation
        torch.cuda.set_per_process_memory_fraction(0.8)
    gc.collect()

    data_file = config.get("data_file", "data.csv")
    topk = config.get("top_k", 5)
    model_type = config.get("model_type", "tcn")
    file_prefix = data_file.split("_")[1].split(".")[0]
    search_dir = f"./model/search_{model_type}_{file_prefix}_{topk}"
    os.makedirs(search_dir, exist_ok=True)

    preprocessed_path = os.path.join(search_dir, "preprocessed_data.pkl")

    if args.resume and os.path.exists(preprocessed_path):
        print(f"Loading preprocessed data from {preprocessed_path}...")
        preprocessed_data = joblib.load(preprocessed_path)
        scaler = joblib.load(os.path.join(search_dir, "scaler.pkl"))
    else:
        preprocessed_data, scaler = preprocess_and_save(config, search_dir)

    PARAM_GRID = config_module.get_param_grid(config["model_type"])

    results = []
    results_path = os.path.join(search_dir, "search_results.json")
    if args.resume and os.path.exists(results_path):
        with open(results_path, "r") as f:
            results = json.load(f)
        print(f"Resuming from {len(results)} existing results...")

        # 找到最后一个完成的实验（通过检查final_score.txt或epoch_scores.txt是否完整）
        completed_indices = set()
        for r in results:
            completed_indices.add(r.get("exp_idx", -1))

        # 也检查目录中是否有完整的exp_X目录（final_score.txt 或 epoch_scores.txt完整）
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

        # 从最大的已完成索引+1继续
        if completed_indices:
            start_idx = max(completed_indices) + 1
            print(f"Continuing from experiment {start_idx + 1}")
        else:
            start_idx = 0
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
                print(
                    f"Experiment {i + 1}/{len(PARAM_GRID)} - SKIPPED (already completed)"
                )
                print(f"Params: {params}")
                print(f"{'=' * 50}")
                continue

        print(f"\n{'=' * 50}")
        print(f"Experiment {i + 1}/{len(PARAM_GRID)}")
        print(f"Params: {params}")
        print(f"{'=' * 50}")

        # Debug: Check GPU memory before running
        if torch.cuda.is_available():
            print(f"GPU before: allocated={torch.cuda.memory_allocated()/1e9:.2f}GB, reserved={torch.cuda.memory_reserved()/1e9:.2f}GB")

        start_time = time.time()
        result = run_experiment(
            params, config, preprocessed_data, scaler, search_dir, i, config_module
        )
        result["exp_idx"] = i  # 添加exp_idx便于resume
        elapsed = time.time() - start_time

        # Force garbage collection after experiment
        if torch.cuda.is_available():
            torch.cuda.synchronize()
            torch.cuda.empty_cache()
        gc.collect()
        elapsed = time.time() - start_time

        results.append(result)

        with open(results_path, "w") as f:
            json.dump(results, f, indent=2)

        # Progress summary
        completed = i + 1
        remaining = len(PARAM_GRID) - completed
        avg_time = (
            (results[-1].get("time", elapsed) + results[-1].get("time", elapsed)) / 2
            if results
            else elapsed
        )
        if "time" not in results[-1]:
            results[-1]["time"] = elapsed

        print(
            f"\n📊 Progress: {completed}/{len(PARAM_GRID)} ({completed / len(PARAM_GRID) * 100:.1f}%)"
        )
        print(f"⏱️  Last exp took: {elapsed:.1f}s")
        if remaining > 0:
            print(f"⏳ Est. remaining: {remaining * avg_time / 60:.1f} min")
        print(
            f"✅ Best score so far: {max(r['score'] for r in results if r['success']):.4f}"
        )

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

    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, default="config")
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--model-type", type=str, default=None)
    parser.add_argument("--feature-num", type=str, default=None)
    parser.add_argument("--data-file", type=str, default=None)
    parser.add_argument("--topk", type=int, default=None)
    parser.add_argument("--sequence-length", type=int, default=None)
    parser.add_argument("--N", type=int, default=None)
    args = parser.parse_args()
    main(args)
