import torch.nn as nn


class RankingNLinear(nn.Module):
    def __init__(self, input_dim, config, num_stocks):
        super().__init__()
        self.num_stocks = num_stocks
        self.seq_length = config.get("sequence_length", 60)
        self.d_model = config.get("d_model", 64)
        self.dropout = config.get("dropout", 0.1)

        self.input_proj = nn.Linear(input_dim, self.d_model)

        self.linear = nn.Linear(self.seq_length, self.seq_length)
        nn.init.xavier_uniform_(self.linear.weight)

        self.ranking_layers = nn.Sequential(
            nn.Linear(self.d_model, self.d_model // 2),
            nn.LayerNorm(self.d_model // 2),
            nn.ReLU(),
            nn.Dropout(self.dropout),
            nn.Linear(self.d_model // 2, self.d_model // 4),
            nn.ReLU(),
            nn.Dropout(self.dropout * 0.5),
        )
        self.score_head = nn.Linear(self.d_model // 4, 1)
        self._init_weights()

    def _init_weights(self):
        for module in self.modules():
            if isinstance(module, nn.Linear):
                if module.weight.requires_grad:
                    nn.init.xavier_uniform_(module.weight)
                if module.bias is not None:
                    nn.init.zeros_(module.bias)

    def forward(self, src):
        batch_size, num_stocks, seq_len, feature_dim = src.size()
        src_reshaped = src.view(batch_size * num_stocks, seq_len, feature_dim)
        x = self.input_proj(src_reshaped)

        last = x[:, -1:, :]
        x_norm = x - last
        x_norm = x_norm.transpose(1, 2)
        out = self.linear(x_norm)
        out = out.transpose(1, 2) + last
        out = out.mean(dim=1)

        stock_features = out.view(batch_size, num_stocks, -1)
        ranking_features = self.ranking_layers(stock_features)
        scores = self.score_head(ranking_features)
        return scores.squeeze(-1)
