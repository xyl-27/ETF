import torch.nn as nn

from .attention import CrossStockAttention, FeatureAttention
from .mmoe import MMoE


class RankingLSTM(nn.Module):
    def __init__(self, input_dim, config, num_stocks, emb_dim=16):
        super(RankingLSTM, self).__init__()
        self.model_type = "RankingLSTM"
        self.config = config
        self.num_stocks = num_stocks

        hidden_dim = config.get("d_model", 256)
        num_layers = config.get("num_layers", 2)
        dropout = config.get("dropout", 0.1)
        nhead = config.get("nhead", 4)
        use_mmoe = config.get("use_mmoe", False)
        num_experts = config.get("num_experts", None)

        self.input_proj = nn.Linear(input_dim, hidden_dim)

        self.lstm = nn.LSTM(
            hidden_dim,
            hidden_dim,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0,
        )

        self.feature_attention = FeatureAttention(hidden_dim, dropout)

        self.use_mmoe = num_experts is not None and num_experts > 0
        if self.use_mmoe:
            self.mmoe = MMoE(
                hidden_dim, hidden_dim, num_experts=num_experts, dropout=dropout
            )

        self.cross_stock_attention = CrossStockAttention(hidden_dim, nhead, dropout)

        self.ranking_layers = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.LayerNorm(hidden_dim // 2),
            nn.ReLU(),
            nn.Dropout(dropout),
        )

        self.score_head = nn.Sequential(
            nn.Linear(hidden_dim // 2, hidden_dim // 4),
            nn.ReLU(),
            nn.Dropout(dropout * 0.5),
            nn.Linear(hidden_dim // 4, 1),
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

        lstm_out, (h_n, c_n) = self.lstm(src_proj)

        attended = self.feature_attention(lstm_out)

        if self.use_mmoe:
            attended = self.mmoe(attended)

        stock_features = attended.view(batch_size, num_stocks, -1)
        stock_features = self.cross_stock_attention(stock_features)

        ranking_features = self.ranking_layers(stock_features)
        scores = self.score_head(ranking_features)

        output = scores.squeeze(-1)
        return output
