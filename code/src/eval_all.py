"""
Re-evaluate all trained models (DL + ML) on validation sets.
Saves eval_results.json in each experiment directory.
"""
import os, sys, json, glob, warnings, argparse
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd
import joblib

warnings.filterwarnings("ignore")

BASE_DIR = os.path.join(os.path.dirname(__file__), "..", "..", "model")
ALL_DL_TYPES = ['dlinear', 'nlinear', 'itransformer', 'mamba', 'patchtst', 'timesnet']
ALL_ML_TYPES = ['xgb', 'lightgbm', 'catboost']


def discover_experiments():
    experiments = []
    # DL: model/{bayes,grid}_{type}_74_3_{date}/exp_*/
    for prefix in ['bayes', 'grid']:
        for mt in ALL_DL_TYPES:
            pattern = os.path.join(BASE_DIR, f"{prefix}_{mt}_74_3_*", "exp_*")
            for exp_dir in sorted(glob.glob(pattern)):
                cfg_path = os.path.join(exp_dir, "config.json")
                model_path = os.path.join(exp_dir, "best_model_sliding.pth")
                if os.path.exists(cfg_path) and os.path.exists(model_path):
                    experiments.append({
                        "exp_dir": exp_dir,
                        "model_type": mt,
                        "search_method": prefix,
                        "kind": "dl",
                    })
    # ML: final/model.pkl in search dir (retrain from train_ml_search.py)
    # or exp_*/final/model.pkl (standalone train_ml.py per config)
    for prefix in ['bayes', 'grid']:
        for mt in ALL_ML_TYPES:
            for search_dir in sorted(glob.glob(os.path.join(BASE_DIR, f"{prefix}_{mt}_*"))):
                # Case 1: retrain at search level (train_ml_search.py)
                model_path = os.path.join(search_dir, "final", "model.pkl")
                cfg_path = os.path.join(search_dir, "config.json")
                if os.path.exists(model_path) and os.path.exists(cfg_path):
                    experiments.append({
                        "exp_dir": search_dir,  # search dir itself, not exp_*
                        "model_type": mt,
                        "search_method": prefix,
                        "kind": "ml",
                    })
                # Case 2: per-experiment (train_ml.py exp_*/)
                for exp_dir in sorted(glob.glob(os.path.join(search_dir, "exp_*"))):
                    model_path = os.path.join(exp_dir, "final", "model.pkl")
                    cfg_path = os.path.join(exp_dir, "config.json")
                    if os.path.exists(model_path) and os.path.exists(cfg_path):
                        experiments.append({
                            "exp_dir": exp_dir,
                            "model_type": mt,
                            "search_method": prefix,
                            "kind": "ml",
                        })
                # Case 3: per-experiment CV results (train_ml_search.py search trials)
                for exp_dir in sorted(glob.glob(os.path.join(search_dir, "exp_*"))):
                    cfg_path = os.path.join(exp_dir, "config.json")
                    if os.path.exists(cfg_path):
                        with open(cfg_path) as f:
                            cfg = json.load(f)
                        if "cv_mean" in cfg and cfg.get("cv_mean", {}).get("final_score") is not None:
                            experiments.append({
                                "exp_dir": exp_dir,
                                "model_type": mt,
                                "search_method": prefix,
                                "kind": "ml_cv",
                            })
    n_dl = sum(1 for e in experiments if e['kind'] == 'dl')
    n_ml = sum(1 for e in experiments if e['kind'] == 'ml')
    n_cv = sum(1 for e in experiments if e['kind'] == 'ml_cv')
    print(f"  Discovered {len(experiments)} experiments: {n_dl} DL, {n_ml} ML, {n_cv} ML-CV")
    return experiments


def evaluate_ml(exp_dir, device="cpu"):
    """Load an ML model and evaluate on held-out dates."""
    sys.path.insert(0, os.path.join(os.path.dirname(__file__), "."))
    from train_ml import (
        load_and_preprocess, ml_feature_engineering,
        prepare_data, evaluate_ranker, compute_per_date_metrics,
        train_xgb_ranker, train_lgb_ranker, train_cb_ranker,
    )

    # Load root config for data_path and settings
    root_cfg_path = os.path.join(exp_dir, "config.json")
    with open(root_cfg_path) as f:
        cfg = json.load(f)
    data_path = cfg.get("data_path", os.path.join(os.path.dirname(__file__), "..", "etf_data", "etf_74_train.csv"))
    model_type = cfg.get("model_type", "xgb")
    add_cs = cfg.get("add_cs_features", False)
    no_momentum = cfg.get("no_momentum", False)

    # Load final config for feature_columns, held_out_dates
    final_cfg_path = os.path.join(exp_dir, "final", "config.json")
    if not os.path.exists(final_cfg_path):
        return None
    with open(final_cfg_path) as f:
        final_cfg = json.load(f)
    feature_columns = final_cfg.get("feature_columns", [])
    held_out_dates_str = final_cfg.get("val_dates", [])
    if not held_out_dates_str or len(held_out_dates_str) < 2 or not feature_columns:
        return None
    # val_dates might be [start, end] or full list; generate full list from processed data
    try:
        val_start = pd.Timestamp(held_out_dates_str[0])
        val_end = pd.Timestamp(held_out_dates_str[-1])
    except:
        return None

    try:
        processed, feats, _ = load_and_preprocess(data_path)
        if add_cs:
            new_cols = ml_feature_engineering(processed, feats, momentum=not no_momentum)
            feats = feats + new_cols
        processed = processed.sort_values("日期").reset_index(drop=True)
        all_dates = sorted(processed["日期"].unique())
        # Generate held_out_dates from val_start ~ val_end using all_dates
        held_out_dates = [d for d in all_dates if val_start <= d <= val_end]

        # Load model
        model_path = os.path.join(exp_dir, "final", "model.pkl")
        model = joblib.load(model_path)

        # Per-date prediction
        import xgboost as xgb
        import lightgbm as lgb
        feat_cols = [c for c in feature_columns if c in processed.columns]
        sliding_preds, sliding_targets = [], []
        for d in held_out_dates:
            day_data = processed[processed["日期"] == d].sort_values("股票代码")
            if day_data.empty:
                continue
            X = day_data[feat_cols].values.astype(np.float32)
            X = np.nan_to_num(X, nan=0.0, posinf=0.0, neginf=0.0)
            if isinstance(model, xgb.Booster):
                y_pred = model.predict(xgb.DMatrix(X))
            elif isinstance(model, lgb.Booster):
                y_pred = model.predict(X, predict_disable_shape_check=True)
            else:
                y_pred = model.predict(X)
            sliding_preds.append(y_pred)
            sliding_targets.append(day_data["label"].values.astype(np.float32))
        if len(sliding_preds) < 2:
            return None
        # Compute per-date metrics (handles variable N per date)
        keys = [
            "final_score", "ndcg", "hit_rate", "mrr",
            "pred_return_sum", "max_return_sum", "random_return_sum",
            "excess_return", "proximity_score", "rank_ic",
            "precision", "recall",
        ]
        per_day = {k: [] for k in keys}
        from scipy.stats import spearmanr
        for y_pred, y_true in zip(sliding_preds, sliding_targets):
            N = len(y_true)
            sorted_true = np.sort(y_true)[::-1]
            true_top = sorted_true[:3]
            pred_top_idx = np.argsort(-y_pred)[:3]
            pred_top = y_true[pred_top_idx]
            true_top_idx = np.argsort(-y_true)[:3]
            pred_sum = pred_top.sum()
            max_sum = true_top.sum()
            rand_sum = 3 * y_true.mean()
            per_day["pred_return_sum"].append(pred_sum)
            per_day["max_return_sum"].append(max_sum)
            per_day["random_return_sum"].append(rand_sum)
            denom = max_sum - rand_sum
            fs = (pred_sum - rand_sum) / denom if abs(denom) > 1e-6 else 0.0
            per_day["final_score"].append(fs)
            per_day["excess_return"].append(pred_sum - rand_sum)
            hit = len(set(pred_top_idx) & set(true_top_idx)) / 3
            per_day["hit_rate"].append(hit)
            dcg = sum(y_true[idx] / np.log2(r + 2) for r, idx in enumerate(pred_top_idx))
            idcg = sum(y_true[idx] / np.log2(r + 2) for r, idx in enumerate(true_top_idx))
            ndcg_val = dcg / idcg if idcg > 0 else 0.0
            per_day["ndcg"].append(ndcg_val)
            mrr = 0.0
            for rank, idx in enumerate(pred_top_idx, 1):
                if y_true[idx] > 0:
                    mrr = 1.0 / rank
                    break
            per_day["mrr"].append(mrr)
            percentiles = []
            for idx in pred_top_idx:
                worse = (y_true <= y_true[idx]).sum() / N
                percentiles.append(worse)
            avg_pct = np.mean(percentiles)
            rand_bench = 0.5
            if avg_pct >= rand_bench:
                norm = 0.5 + (avg_pct - rand_bench) / (1.0 - rand_bench) * 0.5
            else:
                norm = avg_pct / rand_bench * 0.5
            per_day["proximity_score"].append(max(0.0, min(1.0, norm)))
            ic, _ = spearmanr(y_pred, y_true)
            per_day["rank_ic"].append(0.0 if np.isnan(ic) else ic)
            pos_count = (pred_top > 0).sum()
            total_pos = (y_true > 0).sum()
            per_day["precision"].append(pos_count / 3)
            per_day["recall"].append(pos_count / total_pos if total_pos > 0 else 0.0)
        avg_metrics = {k: float(np.mean(v)) for k, v in per_day.items()}
        return {
            "weekly_score": avg_metrics["final_score"],
            "sliding_score": avg_metrics["final_score"],
            "weekly_ndcg": avg_metrics["ndcg"],
            "sliding_ndcg": avg_metrics["ndcg"],
            "weekly_hit_rate": avg_metrics["hit_rate"],
            "sliding_hit_rate": avg_metrics["hit_rate"],
            "weekly_rank_ic": avg_metrics["rank_ic"],
            "sliding_rank_ic": avg_metrics["rank_ic"],
            "weekly_pred_return_sum": avg_metrics["pred_return_sum"],
            "sliding_pred_return_sum": avg_metrics["pred_return_sum"],
            "weekly_mrr": avg_metrics["mrr"],
            "sliding_mrr": avg_metrics["mrr"],
            "weekly_proximity_score": avg_metrics["proximity_score"],
            "sliding_proximity_score": avg_metrics["proximity_score"],
            "num_val_dates": len(sliding_preds),
            "kind": "ml",
        }
    except Exception as e:
        print(f"    ML eval error: {e}")
        return None


def evaluate_ml_cv(exp_dir):
    """Read CV scores from a train_ml_search.py experiment config."""
    with open(os.path.join(exp_dir, "config.json")) as f:
        cfg = json.load(f)
    m = cfg["cv_mean"]
    r = {
        "weekly_score": m.get("final_score", 0.0),
        "sliding_score": m.get("final_score", 0.0),
        "weekly_ndcg": m.get("ndcg", 0.0),
        "sliding_ndcg": m.get("ndcg", 0.0),
        "weekly_hit_rate": m.get("hit_rate", 0.0),
        "sliding_hit_rate": m.get("hit_rate", 0.0),
        "weekly_mrr": m.get("mrr", 0.0),
        "sliding_mrr": m.get("mrr", 0.0),
        "num_weekly": None,
        "num_sliding": None,
    }
    r.update(cfg.get("params", {}))
    return r


def evaluate_dl(exp_dir, device="cpu"):
    """Load a DL model and evaluate on weekly + sliding validation sets."""
    try:
        sys.path.insert(0, os.path.join(os.path.dirname(__file__), "."))
        import torch
        from model import create_model
        from train import (
            RankingDataset, collate_fn, evaluate_ranking_model,
            WeightedRankingLoss, set_seed,
        )
        import config as cfg_module

        set_seed(42)
        if device == "cpu":
            device_obj = torch.device("cpu")
        else:
            device_obj = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        # Load experiment config
        cfg_path = os.path.join(exp_dir, "config.json")
        with open(cfg_path) as f:
            exp_config = json.load(f)

        # Find preprocessed_data.pkl in parent search dir
        search_dir = str(Path(exp_dir).parent)
        pp_path = os.path.join(search_dir, "preprocessed_data.pkl")
        if not os.path.exists(pp_path):
            return None

        pp = joblib.load(pp_path)
        features = pp["features"]
        num_stocks = pp["num_stocks"]
        feature_dim = len(features)

        # Build model
        model_type = exp_config["model_type"]
        full_config = {**exp_config}
        full_config.setdefault("sequence_length", exp_config.get("sequence_length", 60))
        model = create_model(model_type, feature_dim, full_config, num_stocks)

        # Load weights
        model_path = os.path.join(exp_dir, "best_model_sliding.pth")
        if not os.path.exists(model_path):
            return None
        state_dict = torch.load(model_path, map_location=device_obj)
        try:
            model.load_state_dict(state_dict)
        except RuntimeError as e:
            print(f"    跳过（权重不匹配: {str(e).split(chr(10))[0]}）")
            return None
        model.to(device_obj)
        model.eval()

        top_k = exp_config.get("top_k", 5)

        # Create datasets & loaders
        def _make_loader(seq_key, tgt_key, rel_key, idx_key, hs_key):
            dataset = RankingDataset(
                pp[seq_key], pp[tgt_key], pp[rel_key], pp[idx_key],
                pp.get(hs_key),
            )
            return torch.utils.data.DataLoader(
                dataset,
                batch_size=exp_config.get("batch_size", 4),
                shuffle=False,
                collate_fn=collate_fn,
                num_workers=0,
            )

        # Weekly validation
        weekly_loader = _make_loader(
            "val_sequences", "val_targets", "val_relevance",
            "val_stock_indices", "val_hs300_rets",
        )
        # Sliding validation
        sliding_loader = _make_loader(
            "val_sliding_sequences", "val_sliding_targets", "val_sliding_relevance",
            "val_sliding_stock_indices", "val_sliding_hs300_rets",
        )

        criterion = WeightedRankingLoss(
            temperature=1.0,
            k=top_k,
            weight_factor=exp_config.get("top5_weight", 2.0),
            pairwise_weight=exp_config.get("pairwise_weight", 1),
            base_weight=exp_config.get("base_weight", 1.0),
        )

        _, weekly_metrics = evaluate_ranking_model(
            model, weekly_loader, criterion, device_obj,
            writer=None, epoch=0, prefix="eval_weekly", top_k=top_k,
        )
        _, sliding_metrics = evaluate_ranking_model(
            model, sliding_loader, criterion, device_obj,
            writer=None, epoch=0, prefix="eval_sliding", top_k=top_k,
        )

        return {
            "weekly_score": weekly_metrics.get("final_score", 0.0),
            "sliding_score": sliding_metrics.get("final_score", 0.0),
            "weekly_ndcg": weekly_metrics.get("ndcg", 0.0),
            "sliding_ndcg": sliding_metrics.get("ndcg", 0.0),
            "weekly_hit_rate": weekly_metrics.get("hit_rate", 0.0),
            "sliding_hit_rate": sliding_metrics.get("hit_rate", 0.0),
            "weekly_rank_ic": weekly_metrics.get("rank_ic", 0.0),
            "sliding_rank_ic": sliding_metrics.get("rank_ic", 0.0),
            "weekly_pred_return_sum": weekly_metrics.get("pred_return_sum", 0.0),
            "sliding_pred_return_sum": sliding_metrics.get("pred_return_sum", 0.0),
            "weekly_mrr": weekly_metrics.get("mrr", 0.0),
            "sliding_mrr": sliding_metrics.get("mrr", 0.0),
            "weekly_proximity_score": weekly_metrics.get("proximity_score", 0.0),
            "sliding_proximity_score": sliding_metrics.get("proximity_score", 0.0),
            "num_weekly": len(weekly_loader.dataset),
            "num_sliding": len(sliding_loader.dataset),
            "kind": "dl",
        }
    except Exception as e:
        print(f"    DL eval error: {e}")
        return None


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--device", default="cpu", choices=["cpu", "auto"])
    parser.add_argument("--force", action="store_true", help="Re-evaluate even if eval_results.json exists")
    args = parser.parse_args()

    experiments = discover_experiments()
    results = []
    skip_counts = {}
    eval_counts = {}
    fail_counts = {}
    for exp in experiments:
        exp_dir = exp["exp_dir"]
        mt = exp["model_type"]
        kind = exp["kind"]
        eval_path = os.path.join(exp_dir, "eval_results.json") if kind != "ml_cv" else None

        if eval_path and os.path.exists(eval_path) and not args.force:
            with open(eval_path) as f:
                r = json.load(f)
            r["model_type"] = mt
            r["search_method"] = exp["search_method"]
            r["kind"] = kind
            r["exp_dir"] = exp_dir
            results.append(r)
            skip_counts[mt] = skip_counts.get(mt, 0) + 1
            continue

        if kind == "ml_cv":
            r = evaluate_ml_cv(exp_dir)
            if r:
                r["model_type"] = mt
                r["search_method"] = exp["search_method"]
                r["kind"] = kind
                r["exp_dir"] = exp_dir
                results.append(r)
                eval_counts[mt] = eval_counts.get(mt, 0) + 1
            continue

        print(f"  [eval] {kind}: {exp_dir}...", end=" ", flush=True)
        if kind == "ml":
            r = evaluate_ml(exp_dir, args.device)
        else:
            r = evaluate_dl(exp_dir, args.device)

        if r is None:
            print("FAILED")
            fail_counts[mt] = fail_counts.get(mt, 0) + 1
            continue
        r["model_type"] = mt
        r["search_method"] = exp["search_method"]
        r["kind"] = kind
        r["exp_dir"] = exp_dir
        with open(eval_path, "w") as f:
            json.dump(r, f, indent=2, default=str)
        print(f"weekly={r['weekly_score']:.4f} sliding={r['sliding_score']:.4f}")
        results.append(r)
        eval_counts[mt] = eval_counts.get(mt, 0) + 1

    if skip_counts:
        parts = [f"{k}={v}" for k, v in sorted(skip_counts.items())]
        print(f"  [skip] {'  '.join(parts)}")
    if fail_counts:
        parts = [f"{k}={v}" for k, v in sorted(fail_counts.items())]
        print(f"  [fail] {'  '.join(parts)}")

    # Summary
    if not results:
        print("\nNo results.")
        return
    df = pd.DataFrame(results)
    print("\n" + "=" * 60)
    print("Evaluation Summary")
    print("=" * 60)
    print(f"Total: {len(df)} experiments")
    for kind in ["dl", "ml", "ml_cv"]:
        sub = df[df["kind"] == kind]
        if sub.empty:
            continue
        label = "ML-CV" if kind == "ml_cv" else kind.upper()
        print(f"\n{label} ({len(sub)}):")
        print(f"  score:    mean={sub['weekly_score'].mean():.4f}  max={sub['weekly_score'].max():.4f}")
        print(f"  ndcg:     mean={sub['weekly_ndcg'].mean():.4f}  max={sub['weekly_ndcg'].max():.4f}")
        print(f"  hit_rate: mean={sub['weekly_hit_rate'].mean():.4f}  max={sub['weekly_hit_rate'].max():.4f}")
        print(f"  mrr:      mean={sub['weekly_mrr'].mean():.4f}  max={sub['weekly_mrr'].max():.4f}")
        if kind != "ml_cv":
            print(f"  sliding_score: mean={sub['sliding_score'].mean():.4f}  max={sub['sliding_score'].max():.4f}")

    # Save combined results
    out_path = os.path.join(BASE_DIR, "..", "output", "eval_all_results.json")
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    df.to_json(out_path, orient="records", indent=2)
    print(f"\nCombined results saved to {out_path}")


if __name__ == "__main__":
    main()
