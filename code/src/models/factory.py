"""模型工厂函数"""


def create_model(model_type, input_dim, config, num_stocks):
    """模型工厂函数"""
    if model_type == "transformer":
        from .transformer import StockTransformer

        config = {**config}
        config.setdefault("dim_feedforward", 512)
        return StockTransformer(input_dim, config, num_stocks)
    elif model_type == "lstm":
        from .lstm import RankingLSTM

        return RankingLSTM(input_dim, config, num_stocks)
    elif model_type == "tcn":
        from .tcn import RankingTCN

        return RankingTCN(input_dim, config, num_stocks)
    elif model_type == "gru":
        from .gru import RankingGRU

        return RankingGRU(input_dim, config, num_stocks)
    elif model_type == "itransformer":
        from .itransformer import RankingiTransformer

        return RankingiTransformer(input_dim, config, num_stocks)
    elif model_type == "dlinear":
        from .dlinear import RankingDLinear

        return RankingDLinear(input_dim, config, num_stocks)
    elif model_type == "timesnet":
        from .timesnet import RankingTimesNet

        return RankingTimesNet(input_dim, config, num_stocks)
    elif model_type == "nlinear":
        from .nlinear import RankingNLinear

        return RankingNLinear(input_dim, config, num_stocks)
    elif model_type == "patchtst":
        from .patchtst import RankingPatchTST

        return RankingPatchTST(input_dim, config, num_stocks)
    elif model_type == "mamba":
        from .mamba_simple import RankingMamba

        return RankingMamba(input_dim, config, num_stocks)
    else:
        raise ValueError(f"Unknown model type: {model_type}")
