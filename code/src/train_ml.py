import os, json, argparse, warnings
import numpy as np
import pandas as pd
import joblib
from datetime import datetime

from train import _preprocess_common
from config import config

warnings.filterwarnings("ignore")


def load_and_preprocess(data_path, cutoff_date="2099-01-01"):
    df = pd.read_csv(data_path)
    all_stock_ids = df["股票代码"].unique()
    stockid2idx = {sid: idx for idx, sid in enumerate(sorted(all_stock_ids))}
    config["feature_num"] = "39"
    processed, feature_columns = _preprocess_common(
        df, stockid2idx, desc="ML feature engineering", drop_small_open=True
    )
    processed["日期"] = pd.to_datetime(processed["日期"])
    cutoff = pd.Timestamp(cutoff_date)
    processed = processed[processed["日期"] <= cutoff].copy()
    return processed, feature_columns, stockid2idx


PRICE_LEVEL_COLS = {"开盘", "收盘", "最高", "最低", "成交额", "volume_ma_5", "volume_ma_20",
                    "boll_mid", "ema_12", "ema_26", "ema_60", "sma_5", "sma_20"}


def ml_feature_engineering(processed, feature_columns, momentum=True):
    ignore = {"股票代码", "日期", "label", "instrument"}
    feat_cols = [c for c in feature_columns if c not in ignore and c in processed.columns]
    added = []

    for col in feat_cols:
        rk = f"csr_{col}"
        processed[rk] = processed.groupby("日期")[col].rank(pct=True).values
        added.append(rk)
        zk = f"csz_{col}"
        gb = processed.groupby("日期")[col]
        processed[zk] = ((processed[col] - gb.transform("mean")) / (gb.transform("std") + 1e-8)).values
        added.append(zk)

    if momentum:
        rank_cols = [c for c in added if c.startswith("csr_")]
        for col in rank_cols:
            mm = f"mmt_{col}"
            processed[mm] = processed.groupby("股票代码")[col].diff(1).values
            added.append(mm)

    return added


def timeseries_cv_splits(dates_sorted, n_folds=4, val_days=20, gap_days=5, min_train_days=120):
    N = len(dates_sorted)
    step = val_days + gap_days
    for i in range(n_folds):
        fold_idx = n_folds - 1 - i
        val_end = N - i * step
        val_start = val_end - val_days
        train_end = val_start - gap_days
        if train_end < min_train_days:
            break
        yield fold_idx, list(dates_sorted[:train_end]), list(dates_sorted[val_start:val_end])


def _intraday_y_rank(df, max_label=31):
    """Convert continuous labels to integer ranks [0, max_label-1] per date group."""
    per_date_max = df.groupby("日期")["label"].transform("nunique") - 1
    ranks = df.groupby("日期", group_keys=False)["label"].rank(ascending=True, method="dense") - 1
    ranks = (ranks / per_date_max.replace(0, 1) * (max_label - 1)).astype(np.int32)
    return ranks.values


def prepare_data(processed, feature_columns, train_dates, val_dates=None):
    raw_cols = {"股票代码", "日期", "label"}
    feat_cols = [c for c in feature_columns if c in processed.columns and c not in raw_cols]

    train_df = processed[processed["日期"].isin(train_dates)].sort_values("日期").copy()

    def _split(df):
        X = df[feat_cols].values.astype(np.float32)
        X = np.nan_to_num(X, nan=0.0, posinf=0.0, neginf=0.0)
        y_cont = df["label"].values.astype(np.float32)
        y_rank = _intraday_y_rank(df)
        groups = df["日期"].value_counts().sort_index().values.tolist()
        return X, y_rank, y_cont, groups

    X_tr, y_rk, y_ct, g_tr = _split(train_df)
    if val_dates is not None:
        val_df = processed[processed["日期"].isin(val_dates)].sort_values("日期").copy()
        X_va, y_rv, y_cv, g_va = _split(val_df)
        return X_tr, y_rk, y_ct, g_tr, X_va, y_rv, y_cv, g_va
    return X_tr, y_rk, y_ct, g_tr


def train_xgb_ranker(X_train, y_train, groups_train, X_val, y_val, groups_val, params):
    import xgboost as xgb
    dtrain = xgb.DMatrix(X_train, label=y_train)
    dtrain.set_group(groups_train)
    dval = xgb.DMatrix(X_val, label=y_val)
    dval.set_group(groups_val)
    xgb_params = {
        "objective": "rank:ndcg",
        "ndcg_exp_gain": False,
        "eval_metric": "ndcg@3",
        "learning_rate": params["learning_rate"],
        "max_depth": params["max_depth"],
        "subsample": params["subsample"],
        "colsample_bytree": params["colsample_bytree"],
        "lambda": params["l2_reg"],
        "seed": params["seed"],
        "verbosity": 0,
        "tree_method": "hist",
    }
    es = params.get("early_stop", 0)
    model = xgb.train(
        xgb_params, dtrain,
        num_boost_round=params["num_round"],
        evals=[(dtrain, "train"), (dval, "val")] if es else None,
        early_stopping_rounds=es or None,
        verbose_eval=False,
    )
    return model


def train_lgb_ranker(X_train, y_train, groups_train, X_val, y_val, groups_val, params):
    import lightgbm as lgb
    train_data = lgb.Dataset(X_train, label=y_train, group=groups_train)
    val_data = lgb.Dataset(X_val, label=y_val, group=groups_val, reference=train_data)
    lgb_params = {
        "objective": params.get("lgb_objective", "lambdarank"),
        "metric": "ndcg",
        "ndcg_eval_at": [3],
        "learning_rate": params.get("learning_rate", 0.05),
        "max_depth": params.get("max_depth", 6),
        "num_leaves": params.get("num_leaves", 31),
        "subsample": params.get("subsample", 0.8),
        "feature_fraction": params.get("colsample_bytree", 0.8),
        "lambda_l2": params.get("l2_reg", 1.0),
        "seed": params.get("seed", 42),
        "verbosity": -1,
    }
    cb = [lgb.log_evaluation(0)]
    if params.get("early_stop", 0) > 0:
        cb.append(lgb.early_stopping(params["early_stop"]))
    model = lgb.train(
        lgb_params, train_data,
        num_boost_round=params["num_round"],
        valid_sets=[val_data] if params.get("early_stop", 0) > 0 else None,
        callbacks=cb,
    )
    return model


def train_cb_ranker(X_train, y_train, groups_train, X_val, y_val, groups_val, params):
    from catboost import CatBoost, Pool
    def _to_ids(groups):
        return np.repeat(np.arange(len(groups)), groups).astype(np.int32)
    train_pool = Pool(X_train, label=y_train, group_id=_to_ids(groups_train))
    val_pool = Pool(X_val, label=y_val, group_id=_to_ids(groups_val))
    model = CatBoost({
        "loss_function": "YetiRank",
        "learning_rate": params.get("learning_rate", 0.03),
        "depth": params.get("max_depth", 6),
        "subsample": params.get("subsample", 0.8),
        "colsample_bylevel": params.get("colsample_bytree", 0.8),
        "l2_leaf_reg": params.get("l2_reg", 1.0),
        "random_seed": params.get("seed", 42),
        "early_stopping_rounds": params.get("early_stop", 50),
        "verbose": False,
        "thread_count": -1,
    })
    model.fit(train_pool, eval_set=val_pool, plot=False)
    return model


def evaluate_ranker(model, X_val, y_val_cont, groups_val, top_k=3):
    import xgboost as xgb
    if isinstance(model, xgb.Booster):
        dval = xgb.DMatrix(X_val)
        y_pred = model.predict(dval)
    else:
        y_pred = model.predict(X_val)

    final_scores, ndcg_scores, hit_rates, mrr_scores = [], [], [], []
    start = 0
    for g in groups_val:
        end = start + g
        y_true = y_val_cont[start:end]
        y_pred_g = y_pred[start:end]
        if g < top_k:
            start = end
            continue
        sort_idx = np.argsort(-y_pred_g)
        true_top_idx = np.argsort(-y_true)[:top_k]
        pred_sum = y_true[sort_idx[:top_k]].sum()
        max_sum = y_true[true_top_idx].sum()
        rand_sum = top_k * y_true.mean()
        div = max_sum - rand_sum
        final_scores.append((pred_sum - rand_sum) / div if abs(div) > 1e-6 else 0.0)
        ideal = np.sort(y_true)[::-1][:top_k]
        ideal_dcg = np.sum(ideal / np.log2(np.arange(2, top_k + 2)))
        actual = y_true[sort_idx[:top_k]]
        dcg = np.sum(actual / np.log2(np.arange(2, top_k + 2)))
        ndcg_scores.append(dcg / ideal_dcg if ideal_dcg > 0 else 0.0)
        hit = len(set(sort_idx[:top_k]) & set(true_top_idx))
        hit_rates.append(hit / top_k)
        for rank, idx in enumerate(sort_idx):
            if idx in set(true_top_idx):
                mrr_scores.append(1.0 / (rank + 1))
                break
        else:
            mrr_scores.append(0.0)
        start = end
    return {
        "final_score": float(np.mean(final_scores)) if final_scores else 0.0,
        "ndcg": float(np.mean(ndcg_scores)) if ndcg_scores else 0.0,
        "hit_rate": float(np.mean(hit_rates)) if hit_rates else 0.0,
        "mrr": float(np.mean(mrr_scores)) if mrr_scores else 0.0,
    }


def main():
    parser = argparse.ArgumentParser(description="ML Ranker Training for ETF")
    parser.add_argument("--model-type", required=True, choices=["xgb", "lightgbm", "catboost"])
    parser.add_argument("-o", "--output-dir", required=True)
    parser.add_argument("--data-path", default="./etf_data/etf_74_train.csv")
    parser.add_argument("--train-cutoff", default="2026-01-01",
                        help="Training data cutoff (data before this used for training pool)")
    parser.add_argument("--val-start", default="2026-01-01",
                        help="Held-out validation start")
    parser.add_argument("--val-end", default="2026-03-31",
                        help="Held-out validation end")
    parser.add_argument("--n-folds", type=int, default=4)
    parser.add_argument("--val-days", type=int, default=20)
    parser.add_argument("--gap-days", type=int, default=5)
    parser.add_argument("--min-train-days", type=int, default=120)
    parser.add_argument("--num-round", type=int, default=1000)
    parser.add_argument("--early-stop", type=int, default=50)
    parser.add_argument("--learning-rate", type=float, default=0.05)
    parser.add_argument("--max-depth", type=int, default=8)
    parser.add_argument("--subsample", type=float, default=0.8)
    parser.add_argument("--colsample-bytree", type=float, default=0.8)
    parser.add_argument("--num-leaves", type=int, default=31)
    parser.add_argument("--l2-reg", type=float, default=1.0)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--objective", default="lambdarank",
                        choices=["lambdarank", "rank_xendcg"],
                        help="LightGBM ranking objective")
    parser.add_argument("--add-cs-features", action="store_true",
                        help="Add ML engineered features")
    parser.add_argument("--no-momentum", action="store_true",
                        help="Skip CS rank momentum features")
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)

    print("=" * 60)
    print(f"ML Ranker Training | model={args.model_type}")
    print(f"Output: {os.path.abspath(args.output_dir)}")
    print("=" * 60)

    # 1. Load full data
    print(f"\n[1] Loading data (up to {args.val_end})...")
    processed, feature_columns, stockid2idx = load_and_preprocess(args.data_path, args.val_end)
    print(f"  Rows={processed.shape[0]}, Features={len(feature_columns)}, Stocks={len(stockid2idx)}")
    print(f"  Date range: {processed['日期'].min().date()} ~ {processed['日期'].max().date()}")

    # Add ML features (cross-sectional + rank momentum)
    if args.add_cs_features:
        new_cols = ml_feature_engineering(processed, feature_columns, momentum=not args.no_momentum)
        feature_columns = feature_columns + new_cols
        print(f"  Added {len(new_cols)} ML features → {len(feature_columns)} total")

    processed = processed.sort_values("日期").reset_index(drop=True)
    all_dates = sorted(processed["日期"].unique())

    # Split pool and held-out
    cutoff = pd.Timestamp(args.train_cutoff)
    pool_dates = [d for d in all_dates if d <= cutoff]
    val_start = pd.Timestamp(args.val_start)
    val_end = pd.Timestamp(args.val_end)
    held_out_dates = [d for d in all_dates if val_start <= d <= val_end]
    print(f"  Pool dates: {len(pool_dates)} (up to {cutoff.date()})")
    print(f"  Held-out dates: {len(held_out_dates)} ({val_start.date()} ~ {val_end.date()})")

    if args.model_type == "xgb":
        train_fn = train_xgb_ranker
    elif args.model_type == "catboost":
        train_fn = train_cb_ranker
    else:
        train_fn = train_lgb_ranker
    params = {k: getattr(args, k) for k in
              ["num_round", "early_stop", "learning_rate", "max_depth",
               "subsample", "colsample_bytree", "l2_reg", "seed"]}
    if args.model_type == "lightgbm":
        params["num_leaves"] = args.num_leaves
        params["lgb_objective"] = args.objective

    # 2. CV on training pool
    print(f"\n[2] Time series CV on pool ({len(pool_dates)} days)...")
    cv_splits = list(timeseries_cv_splits(
        pool_dates, args.n_folds, args.val_days, args.gap_days, args.min_train_days
    ))
    print(f"  Folds: {len(cv_splits)}")
    cv_results = {}
    cv_best_iters = []
    for fid, tr_d, va_d in cv_splits:
        X_tr, y_rk, y_ct, g_tr, X_va, y_rv, y_cv, g_va = prepare_data(
            processed, feature_columns, tr_d, va_d
        )
        print(f"  Fold {fid}: train={len(tr_d)}d, val={va_d[0].date()}~{va_d[-1].date()}")
        m = train_fn(X_tr, y_rk, g_tr, X_va, y_rv, g_va, params)
        metrics = evaluate_ranker(m, X_va, y_cv, g_va, top_k=3)
        print(f"    final_score={metrics['final_score']:.4f}  ndcg={metrics['ndcg']:.4f}  hit={metrics['hit_rate']:.4f}")
        cv_results[str(fid)] = metrics["final_score"]
        bi = getattr(m, "best_iteration", getattr(m, "best_iteration_", None))
        if bi is not None:
            cv_best_iters.append(bi)
        fd = os.path.join(args.output_dir, f"fold_{fid}")
        os.makedirs(fd, exist_ok=True)
        joblib.dump(m, os.path.join(fd, "model.pkl"))
        with open(os.path.join(fd, "config.json"), "w") as f:
            json.dump({
                "num_round": bi or args.num_round,
                "train_dates": [str(d.date()) for d in [tr_d[0], tr_d[-1]]],
                "val_dates": [str(d.date()) for d in [va_d[0], va_d[-1]]],
                "metrics": metrics,
            }, f, indent=2)

    # 3. Retrain on full pool with fixed rounds (median of CV best iterations)
    fixed_rounds = int(np.median(cv_best_iters)) if cv_best_iters else args.num_round
    print(f"\n[3] Retrain on full pool ({fixed_rounds} fixed rounds, no early stopping)...")
    print(f"    CV best iterations: {cv_best_iters} → median = {fixed_rounds}")
    X_tr, y_rk, y_ct, g_tr = prepare_data(processed, feature_columns, pool_dates, None)[:4]
    retrain_params = dict(params)
    retrain_params["num_round"] = fixed_rounds
    retrain_params["early_stop"] = 0
    model = train_fn(X_tr, y_rk, g_tr, X_tr, y_rk, g_tr, retrain_params)

    # Evaluate on held-out (pure test, never seen by model)
    print(f"\n[4] Pure test eval on {args.val_start}~{args.val_end}...")
    _, _, _, _, X_va, _, y_cv, g_va = prepare_data(
        processed, feature_columns, pool_dates, held_out_dates
    )
    held_out_metrics = evaluate_ranker(model, X_va, y_cv, g_va, top_k=3)
    print(f"  final_score={held_out_metrics['final_score']:.4f}")
    print(f"  ndcg@3={held_out_metrics['ndcg']:.4f}")
    print(f"  hit_rate@3={held_out_metrics['hit_rate']:.4f}")
    print(f"  mrr={held_out_metrics['mrr']:.4f}")

    final_dir = os.path.join(args.output_dir, "final")
    os.makedirs(final_dir, exist_ok=True)
    joblib.dump(model, os.path.join(final_dir, "model.pkl"))
    with open(os.path.join(final_dir, "config.json"), "w") as f:
        json.dump({
            "train_dates": [str(pool_dates[0].date()), str(pool_dates[-1].date())],
            "val_dates": [str(held_out_dates[0].date()), str(held_out_dates[-1].date())],
            "metrics": held_out_metrics,
            "fixed_rounds": fixed_rounds,
            "add_cs_features": args.add_cs_features,
            "feature_columns": feature_columns,
        }, f, indent=2, default=str)

    # 4. Save root config
    fs_list = list(cv_results.values())
    cv_mean = float(np.mean(fs_list))
    cv_std = float(np.std(fs_list))
    with open(os.path.join(args.output_dir, "config.json"), "w") as f:
        json.dump({
            "model_type": args.model_type,
            "feature_num": "39",
            "num_features": len(feature_columns),
            "add_cs_features": args.add_cs_features,
            "cv_folds": len(cv_splits),
            "cv_best_iters": cv_best_iters,
            "fixed_rounds": fixed_rounds,
            "cv_metrics": {"final_score_mean": cv_mean, "final_score_std": cv_std},
            "held_out_metrics": held_out_metrics,
            "params": params,
            "data_path": args.data_path,
            "num_stocks": len(stockid2idx),
            "trained_at": datetime.now().isoformat(),
        }, f, indent=2, default=str)

    print(f"\n{'='*60}")
    print(f"CV final_score:      {cv_mean:.4f} +/- {cv_std:.4f}")
    print(f"CV best iterations:  {cv_best_iters} → fixed={fixed_rounds}")
    print(f"Test final_score:    {held_out_metrics['final_score']:.4f}")
    print(f"Output: {os.path.abspath(args.output_dir)}")
    print("Done!")


if __name__ == "__main__":
    main()
