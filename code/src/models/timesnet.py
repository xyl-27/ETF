import torch
import torch.nn as nn
import torch.nn.functional as F

from .layers import Inception_Block_V1


def FFT_for_Period(x, k=2):
    xf = torch.fft.rfft(x, dim=1)
    frequency_list = abs(xf).mean(0).mean(-1)
    frequency_list[0] = 0
    _, top_indices = torch.topk(frequency_list, k)
    period = x.shape[1] // (top_indices + 1)
    period_weight = abs(xf).mean(-1)[:, top_indices]
    return period, period_weight


class TimesBlock(nn.Module):
    def __init__(self, seq_len, d_model, d_ff, num_kernels=6, k=2):
        super(TimesBlock, self).__init__()
        self.seq_len = seq_len
        self.k = k
        self.conv = nn.Sequential(
            Inception_Block_V1(d_model, d_ff, num_kernels=num_kernels),
            nn.GELU(),
            Inception_Block_V1(d_ff, d_model, num_kernels=num_kernels),
        )

    def forward(self, x):
        B, T, N = x.size()
        period_list, period_weight = FFT_for_Period(x, self.k)

        res = []
        for i in range(self.k):
            period = int(period_list[i].item())
            if self.seq_len % period != 0:
                length = ((self.seq_len // period) + 1) * period
                padding = torch.zeros(
                    [x.shape[0], (length - self.seq_len), x.shape[2]], device=x.device
                )
                out = torch.cat([x, padding], dim=1)
            else:
                length = self.seq_len
                out = x
            out = (
                out.reshape(B, length // period, period, N)
                .permute(0, 3, 1, 2)
                .contiguous()
            )
            out = self.conv(out)
            out = out.permute(0, 2, 3, 1).reshape(B, -1, N)
            res.append(out[:, : self.seq_len, :])
        res = torch.stack(res, dim=-1)
        period_weight = F.softmax(period_weight, dim=1)
        period_weight = period_weight.unsqueeze(1).unsqueeze(1).repeat(1, T, N, 1)
        res = torch.sum(res * period_weight, -1)
        res = res + x
        return res


class RankingTimesNet(nn.Module):
    def __init__(self, input_dim, config, num_stocks):
        super(RankingTimesNet, self).__init__()
        self.num_stocks = num_stocks
        self.seq_len = config.get("sequence_length", 60)
        self.d_model = config.get("d_model", 128)
        self.d_ff = config.get("d_ff", self.d_model * 4)
        self.num_layers = config.get("num_layers", 2)
        self.num_kernels = config.get("num_kernels", 6)
        self.fft_top_k = config.get("fft_top_k", 1)
        self.dropout = config.get("dropout", 0.1)

        self.input_proj = nn.Linear(input_dim, self.d_model)

        self.times_blocks = nn.ModuleList(
            [
                TimesBlock(
                    self.seq_len,
                    self.d_model,
                    self.d_ff,
                    self.num_kernels,
                    self.fft_top_k,
                )
                for _ in range(self.num_layers)
            ]
        )

        self.layer_norm = nn.LayerNorm(self.d_model)

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
                nn.init.xavier_uniform_(module.weight)
                if module.bias is not None:
                    nn.init.zeros_(module.bias)

    def forward(self, src):
        batch_size, num_stocks, seq_len, feature_dim = src.size()

        src_reshaped = src.view(batch_size * num_stocks, seq_len, feature_dim)

        x = self.input_proj(src_reshaped)

        for block in self.times_blocks:
            x = self.layer_norm(block(x))

        pooled = x.mean(dim=1)

        stock_features = pooled.view(batch_size, num_stocks, -1)

        ranking_features = self.ranking_layers(stock_features)
        scores = self.score_head(ranking_features)

        return scores.squeeze(-1)
