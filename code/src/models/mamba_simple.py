import torch
import torch.nn as nn
import torch.nn.functional as F


class SSMBlock(nn.Module):
    def __init__(self, d_model, d_state=16, dt_rank=8):
        super().__init__()
        self.d_model = d_model
        self.d_state = d_state
        self.dt_rank = dt_rank

        self.d_inner = d_model

        self.x_proj = nn.Linear(d_model, dt_rank + d_state * 2, bias=False)
        self.dt_proj = nn.Linear(dt_rank, d_model, bias=True)

        A = torch.arange(1, d_state + 1, dtype=torch.float32).unsqueeze(0).repeat(d_model, 1)
        self.A_log = nn.Parameter(torch.log(A))
        self.D = nn.Parameter(torch.ones(d_model))

    def forward(self, x):
        B, L, D = x.shape
        d_state = self.d_state

        delta = self.dt_proj(self.x_proj(x)[:, :, :self.dt_rank])
        delta = F.softplus(delta)

        bc = self.x_proj(x)[:, :, self.dt_rank:]
        B_param = bc[:, :, :d_state]
        C_param = bc[:, :, d_state:]

        A = -torch.exp(self.A_log)

        h = torch.zeros(B, D, d_state, device=x.device, dtype=x.float())
        y = []
        for t in range(L):
            delta_t = delta[:, t, :].unsqueeze(-1)
            A_bar = torch.exp(delta_t * A.unsqueeze(0))
            B_bar = delta_t * B_param[:, t, :].unsqueeze(1)
            h = A_bar * h + B_bar * x[:, t, :].unsqueeze(-1)
            y_t = (h @ C_param[:, t, :].unsqueeze(-1)).squeeze(-1)
            y_t = y_t + self.D.unsqueeze(0) * x[:, t, :]
            y.append(y_t)

        return torch.stack(y, dim=1)


class RankingMamba(nn.Module):
    def __init__(self, input_dim, config, num_stocks):
        super().__init__()
        self.num_stocks = num_stocks
        self.seq_length = config.get("sequence_length", 60)
        self.d_model = config.get("d_model", 128)
        self.dropout = config.get("dropout", 0.1)
        self.num_layers = config.get("num_layers", 2)
        d_state = config.get("d_state", 16)
        dt_rank = config.get("dt_rank", 8)

        self.input_proj = nn.Linear(input_dim, self.d_model)

        self.conv1d = nn.Conv1d(
            self.d_model, self.d_model, kernel_size=4,
            padding=3, groups=self.d_model,
        )

        self.ssm_layers = nn.ModuleList([
            SSMBlock(self.d_model, d_state, dt_rank)
            for _ in range(self.num_layers)
        ])
        self.norm_layers = nn.ModuleList([
            nn.LayerNorm(self.d_model) for _ in range(self.num_layers)
        ])

        self.gate_proj = nn.Linear(self.d_model, self.d_model)
        self.act = nn.SiLU()

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
            elif isinstance(module, nn.Conv1d):
                nn.init.xavier_uniform_(module.weight)
                if module.bias is not None:
                    nn.init.zeros_(module.bias)

    def forward(self, src):
        batch_size, num_stocks, seq_len, feature_dim = src.size()
        src_reshaped = src.view(batch_size * num_stocks, seq_len, feature_dim)
        x = self.input_proj(src_reshaped)

        gate = self.act(self.gate_proj(x))

        conv_in = x.transpose(1, 2)
        conv_out = self.act(self.conv1d(conv_in))[:, :, :x.size(1)]
        x = conv_out.transpose(1, 2)

        for ssm, norm in zip(self.ssm_layers, self.norm_layers):
            residual = x
            x = ssm(norm(x))
            x = x + residual

        x = x * gate
        out = x.mean(dim=1)

        stock_features = out.view(batch_size, num_stocks, -1)
        ranking_features = self.ranking_layers(stock_features)
        scores = self.score_head(ranking_features)
        return scores.squeeze(-1)
