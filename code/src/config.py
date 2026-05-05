# ETF 配置参数
sequence_length = 60
feature_num = "39"
model_type = "tcn"
val_months = 3
topk = 3
N = 74

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
    "output_dir": f"./model/search_{model_type}_{N}_{topk}",
    "output_base": "./model",
    "data_path": "./etf_data",
    "data_file": "etf_74_train.csv",
}

MODEL_CONFIGS = {
    "transformer": {
        "d_model": 128,
        "nhead": 4,
        "num_layers": 2,
        "dim_feedforward": 256,
        "dropout": 0.1,
    },
    "lstm": {
        "d_model": 128,
        "num_layers": 2,
        "nhead": 4,
        "num_experts": 4,
        "dropout": 0.1,
    },
    "tcn": {
        "d_model": 256,
        "num_layers": 3,
        "kernel_size": 3,
        "nhead": 4,
        "num_experts": 3,
        "dropout": 0.1,
    },
    "gru": {
        "d_model": 128,
        "num_layers": 2,
        "nhead": 4,
        "num_experts": None,
        "dropout": 0.1,
    },
    "itransformer": {
        "d_model": 256,
        "num_layers": 2,
        "nhead": 8,
        "num_experts": None,
        "dropout": 0.2,
    },
    "timesnet": {
        "d_model": 128,
        "d_ff": 512,
        "num_layers": 2,
        "num_kernels": 6,
        "fft_top_k": 1,
        "dropout": 0.1,
    },
    "dlinear": {
        "d_model": 64,
        "kernel_size": 25,
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
        d_models=[128],
        num_layers_range=[2, 3],
        dropout_values=[0.1, 0.2],
        extra_params={"nhead": 4},
    ),
    "lstm": generate_param_grid(
        "lstm",
        learning_rates=[1e-5, 5e-5, 1e-4],
        d_models=[64, 128, 256],
        num_layers_range=[1, 2],
        dropout_values=[0.1, 0.2],
        num_experts_values=[None, 3, 4],
    ),
    "tcn": generate_param_grid(
        "tcn",
        learning_rates=[1e-5, 5e-5, 1e-4],
        d_models=[128, 256],
        num_layers_range=[3, 4],
        dropout_values=[0.1, 0.2],
        extra_params={"kernel_size": 3},
        num_experts_values=[None, 2, 3],
    ),
    "gru": generate_param_grid(
        "gru",
        learning_rates=[1e-5, 5e-5, 1e-4],
        d_models=[64, 128, 256],
        num_layers_range=[1, 2],
        dropout_values=[0.1, 0.2],
        num_experts_values=[None, 3, 4],
    ),
    "itransformer": generate_param_grid(
        "itransformer",
        learning_rates=[1e-5, 5e-5, 1e-4],
        d_models=[128, 256],
        num_layers_range=[2, 3],
        dropout_values=[0.1, 0.2],
        extra_params={"nhead": 8},
        num_experts_values=[None, 3, 4],
    ),
    "timesnet": generate_param_grid(
        "timesnet",
        learning_rates=[1e-5, 5e-5, 1e-4],
        d_models=[64, 128, 256],
        num_layers_range=[2, 3],
        dropout_values=[0.1, 0.2],
        extra_params={"num_kernels": 6, "fft_top_k": 1},
    ),
    "dlinear": generate_param_grid(
        "dlinear",
        learning_rates=[1e-5, 5e-5, 1e-4],
        d_models=[64, 128, 256],
        num_layers_range=[1],
        dropout_values=[0.1, 0.2],
        extra_params={"kernel_size": 25},
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
