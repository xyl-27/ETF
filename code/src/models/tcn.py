import torch.nn as nn

from .attention import CrossStockAttention, FeatureAttention
from .mmoe import MMoE


class TemporalBlock(nn.Module):
    def __init__(
        self, n_inputs, n_outputs, kernel_size, stride, dilation, padding, dropout=0.2
    ):
        super(TemporalBlock, self).__init__()
        self.conv1 = nn.Conv1d(
            n_inputs,
            n_outputs,
            kernel_size,
            stride=stride,
            padding=padding,
            dilation=dilation,
        )
        self.bn1 = nn.BatchNorm1d(n_outputs)
        self.dropout1 = nn.Dropout(dropout)

        self.conv2 = nn.Conv1d(
            n_outputs,
            n_outputs,
            kernel_size,
            stride=stride,
            padding=padding,
            dilation=dilation,
        )
        self.bn2 = nn.BatchNorm1d(n_outputs)
        self.dropout2 = nn.Dropout(dropout)

        self.downsample = (
            nn.Conv1d(n_inputs, n_outputs, 1) if n_inputs != n_outputs else None
        )
        self.relu = nn.ReLU()

    def forward(self, x):
        out = self.conv1(x)
        out = out[:, :, : -self.conv1.padding[0]] if self.conv1.padding[0] > 0 else out
        out = self.relu(self.bn1(out))
        out = self.dropout1(out)

        out = self.conv2(out)
        out = out[:, :, : -self.conv2.padding[0]] if self.conv2.padding[0] > 0 else out
        out = self.relu(self.bn2(out))
        out = self.dropout2(out)

        res = x if self.downsample is None else self.downsample(x)
        res = res[:, :, : -out.size(2)] if res.size(2) > out.size(2) else res
        res = res[:, :, : out.size(2)] if res.size(2) < out.size(2) else res

        return self.relu(out + res)


class RankingTCN(nn.Module):
    def __init__(self, input_dim, config, num_stocks, emb_dim=16):
        super(RankingTCN, self).__init__()
        self.model_type = "RankingTCN"
        self.config = config
        self.num_stocks = num_stocks

        hidden_dim = config.get("d_model", 256)
        num_layers = config.get("num_layers", 4)
        kernel_size = config.get("kernel_size", 3)
        dropout = config.get("dropout", 0.1)
        nhead = config.get("nhead", 4)
        use_mmoe = config.get("use_mmoe", False)
        num_experts = config.get("num_experts", None)

        self.input_proj = nn.Linear(input_dim, hidden_dim)

        self.tcn_layers = nn.ModuleList()
        for i in range(num_layers):
            dilation = 2**i
            in_channels = hidden_dim if i > 0 else hidden_dim
            out_channels = hidden_dim
            padding = (kernel_size - 1) * dilation
            self.tcn_layers.append(
                TemporalBlock(
                    in_channels,
                    out_channels,
                    kernel_size,
                    1,
                    dilation,
                    padding,
                    dropout,
                )
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
        src_proj = self.input_proj(src_reshaped)

        x = src_proj.transpose(1, 2)

        for layer in self.tcn_layers:
            x = layer(x)

        tcn_out = x.transpose(1, 2)
        attended = self.feature_attention(tcn_out)

        if self.use_mmoe:
            attended = self.mmoe(attended)

        stock_features = attended.view(batch_size, num_stocks, -1)
        stock_features = self.cross_stock_attention(stock_features)

        ranking_features = self.ranking_layers(stock_features)
        scores = self.score_head(ranking_features)

        output = scores.squeeze(-1)
        return output
