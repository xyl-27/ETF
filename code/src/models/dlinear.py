import torch.nn as nn
import torch


class moving_avg(nn.Module):
    """Moving average block to highlight the trend of time series"""

    def __init__(self, kernel_size, stride):
        super(moving_avg, self).__init__()
        self.kernel_size = kernel_size
        self.avg = nn.AvgPool1d(kernel_size=kernel_size, stride=stride, padding=0)

    def forward(self, x):
        front = x[:, 0:1, :].repeat(1, (self.kernel_size - 1) // 2, 1)
        end = x[:, -1:, :].repeat(1, (self.kernel_size - 1) // 2, 1)
        x = torch.cat([front, x, end], dim=1)
        x = self.avg(x.permute(0, 2, 1))
        x = x.permute(0, 2, 1)
        return x


class series_decomp(nn.Module):
    """Series decomposition block"""

    def __init__(self, kernel_size):
        super(series_decomp, self).__init__()
        self.moving_avg = moving_avg(kernel_size, stride=1)

    def forward(self, x):
        moving_mean = self.moving_avg(x)
        res = x - moving_mean
        return res, moving_mean


class RankingDLinear(nn.Module):
    """DLinear模型用于股票排名"""

    def __init__(self, input_dim, config, num_stocks):
        super(RankingDLinear, self).__init__()
        self.num_stocks = num_stocks
        self.seq_length = config.get("sequence_length", 60)
        self.d_model = config.get("d_model", 64)
        self.dropout = config.get("dropout", 0.1)

        kernel_size = config.get("kernel_size", 25)
        self.decomposition = series_decomp(kernel_size)

        self.Linear_Seasonal = nn.Linear(self.seq_length, self.seq_length)
        self.Linear_Trend = nn.Linear(self.seq_length, self.seq_length)
        self.Linear_Seasonal.weight = nn.Parameter(
            (1 / self.seq_length) * torch.ones([self.seq_length, self.seq_length])
        )
        self.Linear_Trend.weight = nn.Parameter(
            (1 / self.seq_length) * torch.ones([self.seq_length, self.seq_length])
        )

        self.input_proj = nn.Linear(input_dim, self.d_model)

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

        x = self.input_proj(src_reshaped)  # [B*S, L, D]

        seasonal_init, trend_init = self.decomposition(x)  # [B*S, L, D]

        seasonal_init = seasonal_init.transpose(1, 2)  # [B*S, D, L]
        trend_init = trend_init.transpose(1, 2)  # [B*S, D, L]

        seasonal_output = self.Linear_Seasonal(seasonal_init)  # [B*S, D, L]
        trend_output = self.Linear_Trend(trend_init)  # [B*S, D, L]

        output = seasonal_output + trend_output  # [B*S, D, L]
        output = output.transpose(1, 2)  # [B*S, L, D]
        output = output.mean(dim=1)  # [B*S, D]

        stock_features = output.view(batch_size, num_stocks, -1)

        ranking_features = self.ranking_layers(stock_features)
        scores = self.score_head(ranking_features)

        return scores.squeeze(-1)
