import os, json, warnings
import numpy as np
import pandas as pd
import joblib

from config import config
from train import _preprocess_common
from train_ml import ml_feature_engineering

warnings.filterwarnings("ignore")

_NON_FEATURE_COLS = {
    "日期", "股票代码", "label", "停牌",
    "前收盘", "前收盘_原始", "复权因子",
    "开盘_原始", "收盘_原始", "最低_原始", "最高_原始",
    "涨停价", "跌停价",
}


class MLBacktester:
    def __init__(self, model_dir, cached_data, model_file="model.pkl", verbose=False):
        self.verbose = verbose

        final_cfg_path = os.path.join(model_dir, "final", "config.json")
        with open(final_cfg_path, "r") as f:
            self.model_config = json.load(f)

        self.processed = cached_data["processed"]
        self.stockid2idx = cached_data["stockid2idx"]

        # feature_columns: prefer final/config.json, fallback to derivation
        if "feature_columns" in self.model_config:
            self.feature_columns = self.model_config["feature_columns"]
        else:
            self.feature_columns = [
                c for c in self.processed.columns if c not in _NON_FEATURE_COLS
            ]

        # Read root config for expected feature count (older models)
        self._expected_features = None
        root_cfg_path = os.path.join(model_dir, "config.json")
        if os.path.exists(root_cfg_path):
            with open(root_cfg_path) as f:
                root_cfg = json.load(f)
            expected = root_cfg.get("num_features", None)
            if expected is not None and expected < len(self.feature_columns):
                orig_n = len(self.feature_columns)
                # Try stripping mmt_* (momentum) features first (added last)
                trimmed = [c for c in self.feature_columns if not c.startswith("mmt_")]
                if len(trimmed) >= expected:
                    self.feature_columns = trimmed[:expected]
                else:
                    self.feature_columns = self.feature_columns[:expected]
                if verbose:
                    print(f"  Features: {orig_n} → {len(self.feature_columns)} (expected {expected})")
            self._expected_features = expected

        model_path = os.path.join(model_dir, "final", model_file)
        if os.path.exists(model_path):
            self.models = [joblib.load(model_path)]
            self.use_average = False
        else:
            fold_dirs = sorted([
                d for d in os.listdir(model_dir)
                if d.startswith("fold_") and os.path.isdir(os.path.join(model_dir, d))
            ])
            self.models = []
            for fd in fold_dirs:
                fp = os.path.join(model_dir, fd, model_file)
                if os.path.exists(fp):
                    self.models.append(joblib.load(fp))
            if not self.models:
                raise FileNotFoundError(f"No model found in {model_dir}/final/")
            self.use_average = True

        if verbose:
            n = f"avg of {len(self.models)} folds" if self.use_average else "single"
            print(f"  MLBacktester: {n}, {len(self.feature_columns)} features")

    @classmethod
    def load_data_once(cls, data_path, add_cs_features):
        df = pd.read_csv(data_path)
        all_ids = df["股票代码"].unique()
        stockid2idx = {sid: i for i, sid in enumerate(sorted(all_ids))}

        config["feature_num"] = "39"
        processed, base_fcols = _preprocess_common(
            df, stockid2idx, desc="ML load", drop_small_open=False
        )
        processed["日期"] = pd.to_datetime(processed["日期"])

        if add_cs_features:
            ml_feature_engineering(processed, base_fcols)

        return {"processed": processed, "stockid2idx": stockid2idx}

    @classmethod
    def from_cached_data(cls, model_dir, cached_data, model_file="model.pkl",
                         verbose=False):
        return cls(model_dir, cached_data, model_file, verbose)

    def _get_predictions(self, target_date):
        if isinstance(target_date, str):
            target_date = pd.Timestamp(target_date)

        mask = self.processed["日期"] == target_date
        day_data = self.processed[mask]
        if day_data.empty:
            return None

        day_data = day_data.sort_values("股票代码")

        feat_cols = [c for c in self.feature_columns if c in day_data.columns]
        X = day_data[feat_cols].values.astype(np.float32)
        X = np.nan_to_num(X, nan=0.0, posinf=0.0, neginf=0.0)

        import xgboost as xgb
        import lightgbm as lgb
        scores = np.zeros(len(day_data), dtype=np.float32)
        for m in self.models:
            if isinstance(m, xgb.Booster):
                scores += m.predict(xgb.DMatrix(X))
            elif isinstance(m, lgb.Booster):
                scores += m.predict(X, predict_disable_shape_check=True)
            else:
                scores += m.predict(X)
        scores /= len(self.models)

        stock_ids = day_data["股票代码"].values
        result = []
        for rank, idx in enumerate(np.argsort(-scores)):
            result.append({
                "rank": rank + 1,
                "stock_id": str(stock_ids[idx]),
                "score": float(scores[idx]),
            })
        return result
