import torch
import torch.nn as nn
import math


class RankingPatchTST(nn.Module):
    def __init__(self, input_dim, config, num_stocks):
        super().__init__()
        self.num_stocks = num_stocks
        self.seq_length = config.get("sequence_length", 60)
        self.d_model = config.get("d_model", 128)
        self.dropout = config.get("dropout", 0.1)
        self.nhead = config.get("nhead", 4)
        self.num_layers = config.get("num_layers", 2)
        patch_len = config.get("patch_len", 8)
        stride = config.get("patch_stride", 4)

        num_patches = ((self.seq_length - patch_len) // stride) + 1

        self.patch_len = patch_len
        self.stride = stride
        self.num_patches = num_patches

        self.input_proj = nn.Linear(input_dim, self.d_model)

        self.patch_proj = nn.Linear(patch_len * self.d_model, self.d_model)

        self.pos_embed = nn.Parameter(
            torch.randn(1, num_patches, self.d_model) * 0.02
        )
        self.pos_drop = nn.Dropout(self.dropout)

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=self.d_model,
            nhead=self.nhead,
            dim_feedforward=self.d_model * 4,
            dropout=self.dropout,
            activation="gelu",
            batch_first=True,
        )
        self.transformer_encoder = nn.TransformerEncoder(
            encoder_layer, num_layers=self.num_layers
        )

        self.norm = nn.LayerNorm(self.d_model)

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

    def _create_patches(self, x):
        B, L, D = x.size()
        patches = []
        for t in range(0, L - self.patch_len + 1, self.stride):
            patch = x[:, t : t + self.patch_len, :]
            patches.append(patch.reshape(B, self.patch_len * D))
        return torch.stack(patches, dim=1)

    def forward(self, src):
        batch_size, num_stocks, seq_len, feature_dim = src.size()
        src_reshaped = src.view(batch_size * num_stocks, seq_len, feature_dim)
        x = self.input_proj(src_reshaped)

        patches = self._create_patches(x)
        patches = self.patch_proj(patches)
        patches = patches + self.pos_embed[:, : patches.size(1), :]
        patches = self.pos_drop(patches)

        patches = self.transformer_encoder(patches)
        patches = self.norm(patches)
        out = patches.mean(dim=1)

        stock_features = out.view(batch_size, num_stocks, -1)
        ranking_features = self.ranking_layers(stock_features)
        scores = self.score_head(ranking_features)
        return scores.squeeze(-1)
