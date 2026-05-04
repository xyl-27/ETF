import torch
import torch.nn as nn


class CrossStockAttention(nn.Module):
    """股票间交互注意力模块"""

    def __init__(self, d_model, nhead, dropout=0.1):
        super(CrossStockAttention, self).__init__()
        self.cross_attention = nn.MultiheadAttention(
            d_model, nhead, dropout=dropout, batch_first=True
        )
        self.norm = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(dropout)

    def forward(self, stock_features):
        attended, _ = self.cross_attention(
            stock_features, stock_features, stock_features
        )
        output = self.norm(stock_features + self.dropout(attended))
        return output


class FeatureAttention(nn.Module):
    """特征注意力模块"""

    def __init__(self, d_model, dropout=0.1):
        super(FeatureAttention, self).__init__()
        self.attention = nn.Sequential(
            nn.Linear(d_model, d_model // 2),
            nn.Tanh(),
            nn.Linear(d_model // 2, 1),
            nn.Softmax(dim=1),
        )
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        attention_weights = self.attention(x)
        attended = torch.sum(x * attention_weights, dim=1)
        return self.dropout(attended)
