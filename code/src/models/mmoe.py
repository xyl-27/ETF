import torch
import torch.nn as nn


class MMoE(nn.Module):
    """Multi-Gate Mixture-of-Experts 模块"""

    def __init__(self, input_dim, output_dim, num_experts=4, dropout=0.1):
        super(MMoE, self).__init__()
        self.num_experts = num_experts
        self.experts = nn.ModuleList(
            [
                nn.Sequential(
                    nn.Linear(input_dim, input_dim // 2),
                    nn.ReLU(),
                    nn.Dropout(dropout),
                    nn.Linear(input_dim // 2, output_dim),
                )
                for _ in range(num_experts)
            ]
        )
        self.gates = nn.ModuleList(
            [
                nn.Sequential(
                    nn.Linear(input_dim, num_experts),
                    nn.Softmax(dim=-1),
                )
                for _ in range(1)
            ]
        )

    def forward(self, x):
        experts_output = torch.stack([expert(x) for expert in self.experts], dim=1)
        gate_output = self.gates[0](x)
        output = torch.sum(experts_output * gate_output.unsqueeze(-1), dim=1)
        return output
