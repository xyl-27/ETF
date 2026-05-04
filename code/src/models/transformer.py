import torch.nn as nn

from .positional_encoding import PositionalEncoding
from .attention import CrossStockAttention, FeatureAttention


class StockTransformer(nn.Module):
    def __init__(self, input_dim, config, num_stocks, emb_dim=16):
        super(StockTransformer, self).__init__()
        self.model_type = "RankingTransformer"
        self.config = config
        self.num_stocks = num_stocks

        self.input_proj = nn.Linear(input_dim, config["d_model"])
        self.pos_encoder = PositionalEncoding(
            config["d_model"], config["dropout"], config["sequence_length"]
        )

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=config["d_model"],
            nhead=config["nhead"],
            dim_feedforward=config["dim_feedforward"],
            dropout=config["dropout"],
            batch_first=True,
        )
        self.temporal_encoder = nn.TransformerEncoder(
            encoder_layer, num_layers=config["num_layers"]
        )

        self.feature_attention = FeatureAttention(config["d_model"], config["dropout"])
        self.cross_stock_attention = CrossStockAttention(
            config["d_model"], config["nhead"], config["dropout"]
        )

        self.ranking_layers = nn.Sequential(
            nn.Linear(config["d_model"], config["d_model"]),
            nn.LayerNorm(config["d_model"]),
            nn.ReLU(),
            nn.Dropout(config["dropout"]),
            nn.Linear(config["d_model"], config["d_model"] // 2),
            nn.LayerNorm(config["d_model"] // 2),
            nn.ReLU(),
            nn.Dropout(config["dropout"]),
        )

        self.score_head = nn.Sequential(
            nn.Linear(config["d_model"] // 2, config["d_model"] // 4),
            nn.ReLU(),
            nn.Dropout(config["dropout"] * 0.5),
            nn.Linear(config["d_model"] // 4, 1),
        )

        self._init_weights()

    def _init_weights(self):
        for module in self.modules():
            if isinstance(module, nn.Linear):
                nn.init.xavier_uniform_(module.weight)
                if module.bias is not None:
                    nn.init.zeros_(module.bias)

    def forward(self, src):
        batch_size, num_stocks, seq_len, feature_dim = src.size()
        src_reshaped = src.view(batch_size * num_stocks, seq_len, feature_dim)

        src_proj = self.input_proj(src_reshaped)
        src_proj = self.pos_encoder(src_proj)

        temporal_features = self.temporal_encoder(src_proj)
        aggregated_features = self.feature_attention(temporal_features)

        stock_features = aggregated_features.view(batch_size, num_stocks, -1)
        interactive_features = self.cross_stock_attention(stock_features)

        interactive_features = interactive_features.view(batch_size * num_stocks, -1)
        ranking_features = self.ranking_layers(interactive_features)
        scores = self.score_head(ranking_features)

        output = scores.view(batch_size, num_stocks)
        return output
