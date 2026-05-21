"""
ML 模型超参数搜索
支持 Grid Search 和 Optuna Bayesian Search。

用法:
  # 搜索全部模型（默认: xgb + lightgbm + catboost）
  python code/src/train_ml_search.py
  
  # 只搜索指定模型
  python code/src/train_ml_search.py --model-type xgb
  
  # Bayesian Search
  python code/src/train_ml_search.py --model-type lightgbm \\
      --search-method bayesian --n-trials 50
  
  # 指定输出目录（覆盖自动生成路径）
  python code/src/train_ml_search.py --model-type catboost -o ./ml_search/cb_bayes \\
      --search-method bayesian --n-trials 100 --resume

自动路径:
  ./model/grid_{model_type}_{val_start}_{val_end}/
  ./model/bayes_{model_type}_{val_start}_{val_end}/
"""

import os, sys, json, argparse, warnings, itertools, copy
from datetime import datetime

import numpy as np
import pandas as pd
import joblib

from train import _preprocess_common
from train_ml import (
    load_and_preprocess,
    ml_feature_engineering,
    timeseries_cv_splits,
    prepare_data,
    train_xgb_ranker,
    train_lgb_ranker,
    train_cb_ranker,
    evaluate_ranker,
)

warnings.filterwarnings("ignore")

SEARCH_MODEL_TYPES = ["xgb", "lightgbm", "catboost"]

# ============================================================
# 参数搜索空间
# ============================================================

COMMON_GRID = {
    "learning_rate": [0.01, 0.05, 0.1],
    "max_depth": [6, 8, 10],
    "subsample": [0.8, 1.0],
    "colsample_bytree": [0.8, 1.0],
    "l2_reg": [0.1, 1.0, 10.0],
}

XGB_GRID = {**COMMON_GRID, "min_child_weight": [1, 5, 10]}

LGB_GRID = {
    **COMMON_GRID,
    "num_leaves": [31, 63, 127],
    "min_data_in_leaf": [10, 50, 100],
}

CB_GRID = {**COMMON_GRID, "min_data_in_leaf": [1, 10, 50]}


def get_param_grid(model_type):
    if model_type == "xgb":
        return XGB_GRID
    elif model_type == "lightgbm":
        return LGB_GRID
    else:
        return CB_GRID


def param_product(grid):
    keys = list(grid.keys())
    for vals in itertools.product(*[grid[k] for k in keys]):
        yield dict(zip(keys, vals))


def suggest_params(trial, model_type):
    p = {
        "learning_rate": trial.suggest_float("learning_rate", 0.005, 0.3, log=True),
        "max_depth": trial.suggest_int("max_depth", 4, 14),
        "subsample": trial.suggest_float("subsample", 0.5, 1.0),
        "colsample_bytree": trial.suggest_float("colsample_bytree", 0.5, 1.0),
        "l2_reg": trial.suggest_float("l2_reg", 0.001, 20.0, log=True),
    }
    if model_type == "xgb":
        p["min_child_weight"] = trial.suggest_int("min_child_weight", 1, 20)
    elif model_type == "lightgbm":
        p["num_leaves"] = trial.suggest_int("num_leaves", 15, 255)
        p["min_data_in_leaf"] = trial.suggest_int("min_data_in_leaf", 5, 200)
    else:
        p["min_data_in_leaf"] = trial.suggest_int("min_data_in_leaf", 1, 100)
    return p


# ============================================================
# 数据预处理（一次性）
# ============================================================

def preprocess_and_save(args):
    print(f"\n[preprocess] 加载数据（至 {args.val_end}）...")
    processed, feature_columns, stockid2idx = load_and_preprocess(args.data_path, args.val_end)
    print(f"  Rows={processed.shape[0]}, Features={len(feature_columns)}, Stocks={len(stockid2idx)}")

    if args.add_cs_features:
        new_cols = ml_feature_engineering(processed, feature_columns, momentum=not args.no_momentum)
        feature_columns = feature_columns + new_cols
        print(f"  新增 {len(new_cols)} ML 特征 → {len(feature_columns)} 总计")

    processed = processed.sort_values("日期").reset_index(drop=True)
    all_dates = sorted(processed["日期"].unique())

    cutoff = pd.Timestamp(args.train_cutoff)
    pool_dates = [d for d in all_dates if d <= cutoff]
    val_start = pd.Timestamp(args.val_start)
    val_end = pd.Timestamp(args.val_end)
    held_out_dates = [d for d in all_dates if val_start <= d <= val_end]
    print(f"  Pool: {len(pool_dates)} 天, Held-out: {len(held_out_dates)} 天")

    data = {
        "processed": processed,
        "feature_columns": feature_columns,
        "pool_dates": [str(d.date()) for d in pool_dates],
        "held_out_dates": [str(d.date()) for d in held_out_dates],
    }
    cache_path = os.path.join(args.output_dir, "preprocessed_data.pkl")
    joblib.dump(data, cache_path)
    print(f"  缓存已保存: {cache_path}")
    return data


def load_preprocessed(args):
    cache_path = os.path.join(args.output_dir, "preprocessed_data.pkl")
    if os.path.exists(cache_path):
        print(f"  加载缓存: {cache_path}")
        data = joblib.load(cache_path)
        data["pool_dates"] = [pd.Timestamp(d) for d in data["pool_dates"]]
        data["held_out_dates"] = [pd.Timestamp(d) for d in data["held_out_dates"]]
        return data
    return None


# ============================================================
# 单次实验
# ============================================================

def run_experiment(params, processed, feature_columns, pool_dates, args):
    cv_splits = list(timeseries_cv_splits(
        pool_dates, args.n_folds, args.val_days, args.gap_days, args.min_train_days
    ))
    if not cv_splits:
        return None

    if args.model_type == "xgb":
        train_fn = train_xgb_ranker
    elif args.model_type == "catboost":
        train_fn = train_cb_ranker
    else:
        train_fn = train_lgb_ranker

    exp_params = copy.deepcopy(params)
    exp_params["seed"] = args.seed
    exp_params["num_round"] = args.num_round
    exp_params["early_stop"] = args.early_stop
    if args.model_type == "lightgbm":
        exp_params["lgb_objective"] = args.objective

    all_metrics = {"final_score": [], "ndcg": [], "hit_rate": [], "mrr": []}
    best_iters = []
    for fid, tr_d, va_d in cv_splits:
        X_tr, y_rk, y_ct, g_tr, X_va, y_rv, y_cv, g_va = prepare_data(
            processed, feature_columns, tr_d, va_d
        )
        m = train_fn(X_tr, y_rk, g_tr, X_va, y_rv, g_va, exp_params)
        metrics = evaluate_ranker(m, X_va, y_cv, g_va, top_k=3)
        for k in all_metrics:
            all_metrics[k].append(metrics[k])
        bi = getattr(m, "best_iteration", getattr(m, "best_iteration_", None))
        if bi is not None:
            best_iters.append(bi)

    avg_metrics = {k: float(np.mean(v)) for k, v in all_metrics.items()}
    std_metrics = {k: float(np.std(v)) for k, v in all_metrics.items()}
    return {
        "params": params,
        "cv_mean": avg_metrics,
        "cv_std": std_metrics,
        "cv_best_iters": best_iters,
        "n_folds": len(cv_splits),
    }


# ============================================================
# 搜索结果排序与保存
# ============================================================

def _load_search_results(output_dir):
    path = os.path.join(output_dir, "search_results.json")
    if os.path.exists(path):
        with open(path, "r") as f:
            return json.load(f)
    return {}


def _save_search_results(output_dir, results):
    path = os.path.join(output_dir, "search_results.json")
    with open(path, "w") as f:
        json.dump(results, f, indent=2, default=str)


def _save_trial(output_dir, exp_idx, result):
    exp_dir = os.path.join(output_dir, f"exp_{exp_idx}")
    os.makedirs(exp_dir, exist_ok=True)
    with open(os.path.join(exp_dir, "config.json"), "w") as f:
        json.dump(result, f, indent=2, default=str)


# ============================================================
# Grid Search
# ============================================================

def run_grid_search(args, data):
    processed = data["processed"]
    feature_columns = data["feature_columns"]
    pool_dates = data["pool_dates"]

    grid = get_param_grid(args.model_type)
    all_params = list(param_product(grid))
    total = len(all_params)
    print(f"\n[Grid Search] {total} 个组合")

    results = _load_search_results(args.output_dir)
    completed = {int(k) for k in results.keys()}

    for idx, params in enumerate(all_params):
        if idx in completed:
            print(f"  [{idx + 1}/{total}] 已存在，跳过")
            continue

        print(f"  [{idx + 1}/{total}] params={params}")
        result = run_experiment(params, processed, feature_columns, pool_dates, args)
        if result is None:
            print("    → CV 为空，跳过")
            continue

        m = result["cv_mean"]
        print(f"    → final_score={m['final_score']:.4f} ndcg={m['ndcg']:.4f} "
              f"hit={m['hit_rate']:.4f} mrr={m['mrr']:.4f}")

        _save_trial(args.output_dir, idx, result)
        results[str(idx)] = {
            "exp_idx": idx,
            "params": params,
            "score": m[args.search_metric],
            "metrics": m,
            "n_folds": result["n_folds"],
        }
        _save_search_results(args.output_dir, results)

    return results


# ============================================================
# Bayesian Search (Optuna)
# ============================================================

def run_bayesian_search(args, data):
    import optuna

    processed = data["processed"]
    feature_columns = data["feature_columns"]
    pool_dates = data["pool_dates"]

    storage_url = f"sqlite:///{os.path.join(args.output_dir, 'optuna_study.db')}"
    study_name = f"ml_{args.model_type}_{args.search_metric}"

    results = _load_search_results(args.output_dir)
    completed = {int(k) for k in results.keys()}

    study = optuna.create_study(
        study_name=study_name,
        storage=storage_url,
        load_if_exists=args.resume,
        direction="maximize",
        sampler=optuna.samplers.TPESampler(seed=args.seed),
    )

    if completed:
        print(f"  已存在 {len(completed)} 个完成实验")

    existing_trials = {t.number for t in study.trials} if args.resume else set()

    for trial_idx in range(args.n_trials):
        if trial_idx in existing_trials:
            print(f"  [Trial {trial_idx + 1}/{args.n_trials}] Optuna 已存在，跳过")
            continue

        if trial_idx in completed:
            print(f"  [Trial {trial_idx + 1}/{args.n_trials}] 本地已存在，跳过")
            continue

        def objective(trial):
            params = suggest_params(trial, args.model_type)
            result = run_experiment(params, processed, feature_columns, pool_dates, args)
            if result is None:
                return 0.0
            m = result["cv_mean"]
            print(f"    → final_score={m['final_score']:.4f} ndcg={m['ndcg']:.4f} "
                  f"hit={m['hit_rate']:.4f} mrr={m['mrr']:.4f}")
            _save_trial(args.output_dir, trial_idx, result)
            results[str(trial_idx)] = {
                "exp_idx": trial_idx,
                "params": params,
                "score": m[args.search_metric],
                "metrics": m,
                "n_folds": result["n_folds"],
            }
            _save_search_results(args.output_dir, results)
            return m[args.search_metric]

        print(f"  [Trial {trial_idx + 1}/{args.n_trials}]")
        study.optimize(objective, n_trials=1)

    return results


# ============================================================
# 最佳参数重训
# ============================================================

def retrain_best(args, data, results):
    best_exp = max(results.values(), key=lambda x: x["score"])
    best_params = best_exp["params"]
    print(f"\n{'='*60}")
    print(f"最佳参数 (exp_{best_exp['exp_idx']}):")
    for k, v in best_params.items():
        print(f"  {k}: {v}")
    print(f"  CV {args.search_metric}: {best_exp['score']:.4f}")
    print(f"{'='*60}")

    processed = data["processed"]
    feature_columns = data["feature_columns"]
    pool_dates = data["pool_dates"]
    held_out_dates = data["held_out_dates"]

    if args.model_type == "xgb":
        train_fn = train_xgb_ranker
    elif args.model_type == "catboost":
        train_fn = train_cb_ranker
    else:
        train_fn = train_lgb_ranker

    # Determine fixed rounds from best experiment
    best_exp_dir = os.path.join(args.output_dir, f"exp_{best_exp['exp_idx']}")
    best_cfg_path = os.path.join(best_exp_dir, "config.json")
    best_iters = []
    if os.path.exists(best_cfg_path):
        with open(best_cfg_path) as f:
            best_cfg = json.load(f)
        best_iters = best_cfg.get("cv_best_iters", [])
    fixed_rounds = int(np.median(best_iters)) if best_iters else args.num_round

    retrain_params = copy.deepcopy(best_params)
    retrain_params["seed"] = args.seed
    retrain_params["num_round"] = fixed_rounds
    retrain_params["early_stop"] = 0
    if args.model_type == "lightgbm":
        retrain_params["lgb_objective"] = args.objective

    print(f"\n[Retrain] CV best iters: {best_iters} → fixed={fixed_rounds}")
    print(f"[Retrain] 全 Pool 重训 ({len(pool_dates)} 天)...")
    X_tr, y_rk, y_ct, g_tr = prepare_data(processed, feature_columns, pool_dates, None)[:4]
    model = train_fn(X_tr, y_rk, g_tr, X_tr, y_rk, g_tr, retrain_params)

    print(f"\n[Eval] 在 held-out 验证 ({held_out_dates[0].date()} ~ {held_out_dates[-1].date()})...")
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
            "train_dates": [str(d.date()) for d in [pool_dates[0], pool_dates[-1]]],
            "val_dates": [str(d.date()) for d in [held_out_dates[0], held_out_dates[-1]]],
            "metrics": held_out_metrics,
            "fixed_rounds": fixed_rounds,
            "add_cs_features": args.add_cs_features,
            "feature_columns": feature_columns,
        }, f, indent=2, default=str)

    cv_scores = [v["score"] for v in results.values()]
    with open(os.path.join(args.output_dir, "config.json"), "w") as f:
        json.dump({
            "model_type": args.model_type,
            "feature_num": "39",
            "num_features": len(feature_columns),
            "add_cs_features": args.add_cs_features,
            "search_method": args.search_method,
            "search_metric": args.search_metric,
            "n_trials": len(results),
            "best_exp_idx": best_exp["exp_idx"],
            "best_cv_score": best_exp["score"],
            "cv_score_mean": float(np.mean(cv_scores)),
            "cv_score_std": float(np.std(cv_scores)),
            "held_out_metrics": held_out_metrics,
            "best_params": best_params,
            "data_path": args.data_path,
            "trained_at": datetime.now().isoformat(),
        }, f, indent=2, default=str)

    print(f"\n  [完成] 最佳模型保存至 {final_dir}/")
    return held_out_metrics


# ============================================================
# 入口
# ============================================================

def main(args):
    os.makedirs(args.output_dir, exist_ok=True)

    print("=" * 60)
    print(f"ML 超参数搜索 | model={args.model_type}")
    print(f"  方法: {args.search_method}, 指标: {args.search_metric}")
    print(f"  CS特征: {'开' if args.add_cs_features else '关'}, "
          f"动量: {'开' if not args.no_momentum else '关'}")
    print(f"  输出: {os.path.abspath(args.output_dir)}")
    print("=" * 60)

    # 加载或预处理数据
    data = load_preprocessed(args)
    if data is None:
        data = preprocess_and_save(args)

    data["pool_dates"] = [pd.Timestamp(d) for d in data["pool_dates"]]
    data["held_out_dates"] = [pd.Timestamp(d) for d in data["held_out_dates"]]

    # 搜索
    if not args.retrain_only:
        if args.search_method == "grid":
            results = run_grid_search(args, data)
        else:
            results = run_bayesian_search(args, data)
    else:
        results = _load_search_results(args.output_dir)

    if not results:
        print("错误: 无搜索结果")
        return

    # 重训最佳
    retrain_best(args, data, results)

    print(f"\n{'='*60}")
    print(f"完成! 输出: {os.path.abspath(args.output_dir)}/final/")
    print(f"{'='*60}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="ML 超参数搜索")
    parser.add_argument("--model-type", default=None,
                        choices=SEARCH_MODEL_TYPES,
                        help=f"模型类型，不传则搜索全部: {', '.join(SEARCH_MODEL_TYPES)}")
    parser.add_argument("-o", "--output-dir", default=None,
                        help="输出目录（默认自动生成: ./model/{method}_{model_type}_{val_start}_{val_end}）")
    parser.add_argument("--data-path", default="./etf_data/etf_74_train.csv")
    parser.add_argument("--train-cutoff", default="2026-01-01")
    parser.add_argument("--val-start", default="2026-01-01")
    parser.add_argument("--val-end", default="2026-03-31")
    parser.add_argument("--n-folds", type=int, default=4)
    parser.add_argument("--val-days", type=int, default=20)
    parser.add_argument("--gap-days", type=int, default=5)
    parser.add_argument("--min-train-days", type=int, default=120)
    parser.add_argument("--num-round", type=int, default=1000)
    parser.add_argument("--early-stop", type=int, default=50)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--objective", default="lambdarank",
                        choices=["lambdarank", "rank_xendcg"])
    parser.add_argument("--add-cs-features", action="store_true")
    parser.add_argument("--no-momentum", action="store_true")
    parser.add_argument("--search-method", default="grid",
                        choices=["grid", "bayesian"])
    parser.add_argument("--search-metric", default="final_score",
                        choices=["final_score", "ndcg", "hit_rate", "mrr"])
    parser.add_argument("--n-trials", type=int, default=80)
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--retrain-only", action="store_true",
                        help="仅从已有搜索结果重训最佳参数，跳过搜索")
    args = parser.parse_args()

    model_types = [args.model_type] if args.model_type else SEARCH_MODEL_TYPES
    for mt in model_types:
        args.model_type = mt
        output_dir = args.output_dir
        if output_dir is None:
            method_prefix = "grid" if args.search_method == "grid" else "bayes"
            date_tag = f"_{args.val_start}_{args.val_end}"
            output_dir = f"./model/{method_prefix}_{mt}{date_tag}"
        args.output_dir = output_dir
        print(f"\n{'=' * 60}")
        print(f"  搜索模型: {mt}  ({model_types.index(mt) + 1}/{len(model_types)})")
        print(f"{'=' * 60}")
        main(args)
