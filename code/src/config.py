import os


# ETF 配置参数
sequence_length = 60
feature_num = "39"
model_type = "tcn"
val_months = 3
val_start_date = "2026-01-01"
val_end_date = "2026-03-31"
topk = 3
N = 74
_date_tag = f"_{val_start_date}_{val_end_date}" if val_start_date else ""
config = {
    "sequence_length": sequence_length,
    "batch_size": 4,
    "num_epochs": 30,
    "learning_rate": 1e-5,
    "feature_num": feature_num,
    "model_type": model_type,
    "max_grad_norm": 5.0,
    "pairwise_weight": 1,
    "base_weight": 1.0,
    "top_k": topk,
    "top5_weight": 2.0,
    "val_months": val_months,
    "val_start_date": val_start_date,
    "val_end_date": val_end_date,
    "output_dir": f"./model/search_{model_type}_{N}_{topk}{_date_tag}",# 参考
    "output_base": "./model",
    "data_path": "./etf_data",
    "data_file": "etf_74_train.csv",
    "search_metric": "ndcg",
    "n_trials": 80,
    # 回测参数
    "commission": 0.0003,   # 手续费率 (万分之三)
    "slippage": 0.001,      # 滑点 (千分之一)
}

# TSCV 交叉验证配置
use_tscv = True                     # 是否启用 TSCV 多折评估
tscv_eval_dir = "model/TSCV"        # TSCV 评估输出路径
tscv_folds = [
    {"val_start": "2024-07-01", "val_end": "2024-12-31"},
    {"val_start": "2025-01-01", "val_end": "2025-06-30"},
    {"val_start": "2025-07-01", "val_end": "2025-12-31"},
    {"val_start": "2026-01-01", "val_end": "2026-05-29"},
]

MODEL_CONFIGS = {
    "transformer": {
        "d_model": 64,
        "nhead": 4,
        "num_layers": 1,
        "dim_feedforward": 256,
        "dropout": 0.1,
    },
    "lstm": {
        "d_model": 64,
        "num_layers": 1,
        "nhead": 4,
        "num_experts": None,
        "dropout": 0.1,
    },
    "tcn": {
        "d_model": 64,
        "num_layers": 2,
        "kernel_size": 3,
        "nhead": 4,
        "num_experts": None,
        "dropout": 0.1,
    },
    "gru": {
        "d_model": 64,
        "num_layers": 1,
        "nhead": 4,
        "num_experts": None,
        "dropout": 0.1,
    },
    "itransformer": {
        "d_model": 64,
        "num_layers": 1,
        "nhead": 4,
        "num_experts": None,
        "dropout": 0.1,
    },
    "timesnet": {
        "d_model": 32,
        "d_ff": 32,
        "num_layers": 1,
        "num_kernels": 3,
        "fft_top_k": 1,
        "dropout": 0.1,
    },
    "dlinear": {
        "d_model": 32,
        "kernel_size": 25,
        "dropout": 0.1,
    },
    "nlinear": {
        "d_model": 32,
        "dropout": 0.1,
    },
    "patchtst": {
        "d_model": 64,
        "num_layers": 1,
        "nhead": 4,
        "patch_len": 8,
        "patch_stride": 4,
        "dropout": 0.1,
    },
    "mamba": {
        "d_model": 64,
        "num_layers": 1,
        "d_state": 16,
        "dt_rank": 8,
        "dropout": 0.1,
    },
}


def generate_param_grid(
    model_type,
    learning_rates,
    d_models,
    num_layers_range,
    dropout_values,
    extra_params=None,
    num_experts_values=None,
):
    """Generate parameter grid"""
    grid = []
    for lr in learning_rates:
        for dm in d_models:
            for nl in num_layers_range:
                for dp in dropout_values:
                    if num_experts_values is not None:
                        for ne in num_experts_values:
                            params = {
                                "learning_rate": lr,
                                "d_model": dm,
                                "num_layers": nl,
                                "dropout": dp,
                                "num_experts": ne,
                            }
                            if extra_params:
                                params.update(extra_params)
                            grid.append(params)
                    else:
                        params = {
                            "learning_rate": lr,
                            "d_model": dm,
                            "num_layers": nl,
                            "dropout": dp,
                        }
                        if extra_params:
                            params.update(extra_params)
                        grid.append(params)
    return grid


PARAM_GRID = {
    "transformer": generate_param_grid(
        "transformer",
        learning_rates=[1e-5, 5e-5, 1e-4],
        d_models=[64, 128],
        num_layers_range=[1, 2, 3],
        dropout_values=[0.1, 0.2],
        extra_params={"nhead": 4, "dim_feedforward": 256},
    ),
    "lstm": generate_param_grid(
        "lstm",
        learning_rates=[5e-5, 1e-4, 3e-4],
        d_models=[64, 128],
        num_layers_range=[1, 2],
        dropout_values=[0.1, 0.2],
        num_experts_values=[None, 3],
    ),
    "tcn": generate_param_grid(
        "tcn",
        learning_rates=[5e-5, 1e-4, 3e-4],
        d_models=[64, 128],
        num_layers_range=[2, 3, 4],
        dropout_values=[0.1, 0.2],
        extra_params={"kernel_size": 3},
        num_experts_values=[None, 3],
    ),
    "gru": generate_param_grid(
        "gru",
        learning_rates=[5e-5, 1e-4, 3e-4],
        d_models=[64, 128],
        num_layers_range=[1, 2],
        dropout_values=[0.1, 0.2],
        num_experts_values=[None, 3],
    ),
    "itransformer": generate_param_grid(
        "itransformer",
        learning_rates=[1e-5, 5e-5, 1e-4],
        d_models=[64, 128],
        num_layers_range=[1, 2],
        dropout_values=[0.1, 0.2],
        extra_params={"nhead": 4},
        num_experts_values=[None, 3],
    ),
    "timesnet": generate_param_grid(
        "timesnet",
        learning_rates=[1e-4, 3e-4, 5e-4],
        d_models=[32, 64],
        num_layers_range=[1, 2],
        dropout_values=[0.1, 0.2],
        extra_params={"num_kernels": 3, "fft_top_k": 1},
    ),
    "dlinear": generate_param_grid(
        "dlinear",
        learning_rates=[1e-4, 3e-4, 1e-3],
        d_models=[32, 64, 128],
        num_layers_range=[1],
        dropout_values=[0.1, 0.2],
        extra_params={"kernel_size": 25},
    ),
    "nlinear": generate_param_grid(
        "nlinear",
        learning_rates=[1e-4, 3e-4, 1e-3],
        d_models=[32, 64],
        num_layers_range=[1],
        dropout_values=[0.1, 0.2],
    ),
    "patchtst": generate_param_grid(
        "patchtst",
        learning_rates=[1e-5, 5e-5, 1e-4],
        d_models=[64, 128],
        num_layers_range=[1, 2],
        dropout_values=[0.1, 0.2],
        extra_params={"nhead": 4},
    ),
    "mamba": generate_param_grid(
        "mamba",
        learning_rates=[5e-5, 1e-4, 3e-4],
        d_models=[64, 128],
        num_layers_range=[1, 2],
        dropout_values=[0.1, 0.2],
        extra_params={"d_state": 16, "dt_rank": 8},
    ),
}


def get_model_config(model_type):
    if model_type not in MODEL_CONFIGS:
        raise ValueError(f"Unknown model type: {model_type}")
    return MODEL_CONFIGS[model_type].copy()


def get_param_grid(model_type):
    if model_type not in PARAM_GRID:
        raise ValueError(f"Unknown model type: {model_type}")
    return PARAM_GRID[model_type]


def load_tscv_config():
    """从 config.py 模块读取 TSCV 配置。

    Returns:
        (tscv_eval_dir, folds_list, enabled)
    """
    return tscv_eval_dir, tscv_folds, use_tscv


def get_search_space(model_type):
    """返回贝叶斯搜索的搜索空间函数（Optuna trial 风格）

    范围与 PARAM_GRID 一致，避免浪费 trial 在无效区域。
    """
    if model_type not in MODEL_CONFIGS:
        raise ValueError(f"Unknown model type: {model_type}")

    # 各模型类型的参数范围（与 PARAM_GRID 对齐）
    ranges = {
        "transformer": dict(
            d_models=[64, 128],
            num_layers_range=(1, 3),
            dropout_values=(0.1, 0.2),
            learning_rate_range=(1e-5, 1e-4),
            nhead_values=[4],
            dim_feedforward_values=[128, 256],
        ),
        "itransformer": dict(
            d_models=[64, 128],
            num_layers_range=(1, 2),
            dropout_values=(0.1, 0.2),
            learning_rate_range=(1e-5, 1e-4),
            nhead_values=[4],
            num_experts_values=[None, 3],
        ),
        "lstm": dict(
            d_models=[64, 128],
            num_layers_range=(1, 2),
            dropout_values=(0.1, 0.2),
            learning_rate_range=(5e-5, 3e-4),
            num_experts_values=[None, 3],
        ),
        "gru": dict(
            d_models=[64, 128],
            num_layers_range=(1, 2),
            dropout_values=(0.1, 0.2),
            learning_rate_range=(5e-5, 3e-4),
            num_experts_values=[None, 3],
        ),
        "tcn": dict(
            d_models=[64, 128],
            num_layers_range=(2, 4),
            dropout_values=(0.1, 0.2),
            learning_rate_range=(5e-5, 3e-4),
            kernel_size_values=[3],
            num_experts_values=[None, 3],
        ),
        "timesnet": dict(
            d_models=[32, 64],
            num_layers_range=(1, 2),
            dropout_values=(0.1, 0.2),
            learning_rate_range=(1e-4, 5e-4),
            num_kernels_values=[3],
            fft_top_k_values=[1],
        ),
        "dlinear": dict(
            d_models=[32, 64, 128],
            num_layers_range=(1, 1),
            dropout_values=(0.1, 0.2),
            learning_rate_range=(1e-4, 1e-3),
            kernel_size_values=[25],
        ),
        "nlinear": dict(
            d_models=[32, 64],
            num_layers_range=(1, 1),
            dropout_values=(0.1, 0.2),
            learning_rate_range=(1e-4, 1e-3),
        ),
        "patchtst": dict(
            d_models=[64, 128],
            num_layers_range=(1, 2),
            dropout_values=(0.1, 0.2),
            learning_rate_range=(1e-5, 1e-4),
            nhead_values=[4],
        ),
        "mamba": dict(
            d_models=[64, 128],
            num_layers_range=(1, 2),
            dropout_values=(0.1, 0.2),
            learning_rate_range=(5e-5, 3e-4),
            d_state_values=[16],
            dt_rank_values=[8],
        ),
    }

    r = ranges.get(model_type, ranges["dlinear"])

    def search_space(trial):
        params = {}
        params["learning_rate"] = trial.suggest_float(
            "learning_rate", r["learning_rate_range"][0], r["learning_rate_range"][1], log=True,
        )
        params["d_model"] = trial.suggest_categorical("d_model", r["d_models"])
        params["num_layers"] = trial.suggest_int("num_layers", *r["num_layers_range"])
        params["dropout"] = trial.suggest_float(
            "dropout", r["dropout_values"][0], r["dropout_values"][1],
        )

        if r.get("nhead_values"):
            params["nhead"] = trial.suggest_categorical(
                "nhead", r["nhead_values"],
            )

        if r.get("dim_feedforward_values"):
            params["dim_feedforward"] = trial.suggest_categorical(
                "dim_feedforward", r["dim_feedforward_values"],
            )

        if r.get("num_experts_values"):
            params["num_experts"] = trial.suggest_categorical(
                "num_experts", r["num_experts_values"],
            )

        if r.get("kernel_size_values"):
            params["kernel_size"] = trial.suggest_categorical(
                "kernel_size", r["kernel_size_values"],
            )

        if r.get("num_kernels_values"):
            params["num_kernels"] = trial.suggest_categorical(
                "num_kernels", r["num_kernels_values"],
            )

        if r.get("fft_top_k_values"):
            params["fft_top_k"] = trial.suggest_categorical(
                "fft_top_k", r["fft_top_k_values"],
            )

        return params

    return search_space
