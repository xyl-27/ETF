import torch.nn as nn

from .attention import CrossStockAttention, FeatureAttention
from .mmoe import MMoE


class RankingiTransformer(nn.Module):
    def __init__(self, input_dim, config, num_stocks, emb_dim=16):
        super(RankingiTransformer, self).__init__()
        self.model_type = "RankingiTransformer"
        self.config = config
        self.num_stocks = num_stocks

        self.seq_len = config.get("sequence_length", 60)
        hidden_dim = config.get("d_model", 256)
        num_layers = config.get("num_layers", 3)
        nhead = config.get("nhead", 8)
        # 校验 nhead 合法性：必须整除 d_model
        while hidden_dim % nhead != 0 and nhead > 1:
            nhead = nhead // 2
        dropout = config.get("dropout", 0.1)
        use_mmoe = config.get("use_mmoe", False)
        num_experts = config.get("num_experts", None)

        self.input_proj = nn.Linear(self.seq_len, hidden_dim)

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=hidden_dim,
            nhead=nhead,
            dim_feedforward=hidden_dim * 4,
            dropout=dropout,
            batch_first=True,
        )
        self.transformer_encoder = nn.TransformerEncoder(
            encoder_layer, num_layers=num_layers
        )

        self.use_mmoe = num_experts is not None and num_experts > 0
        if self.use_mmoe:
            self.mmoe = MMoE(
                hidden_dim, hidden_dim, num_experts=num_experts, dropout=dropout
            )

        self.feature_attention = FeatureAttention(hidden_dim, dropout)
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

        src_proj = src_reshaped.transpose(1, 2)

        src_proj = self.input_proj(src_proj)

        transformer_out = self.transformer_encoder(src_proj)

        pooled = transformer_out.mean(dim=1)

        if self.use_mmoe:
            pooled = self.mmoe(pooled)

        stock_features = pooled.view(batch_size, num_stocks, -1)
        stock_features = self.cross_stock_attention(stock_features)

        ranking_features = self.ranking_layers(stock_features)
        scores = self.score_head(ranking_features)

        output = scores.squeeze(-1)
        return output
