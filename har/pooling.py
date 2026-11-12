"""Independent pooling modifications; no torchvision dependency."""
import torch
from torch import nn
from torch.nn import functional as F


class GeM(nn.Module):
    """Positive p is learned; p=1 is mean, larger p emphasizes peaks."""

    def __init__(self):
        super().__init__()
        self.raw_p = nn.Parameter(torch.tensor(2.95))

    def forward(self, x):
        p = (F.softplus(self.raw_p) + 0.05).clamp(max=8.0)
        return x.clamp(min=1e-6).pow(p).mean((-2, -1), keepdim=True).pow(1 / p)


class AvgMaxPool(nn.Module):
    def forward(self, x):
        return torch.cat((F.adaptive_avg_pool2d(x, 1), F.adaptive_max_pool2d(x, 1)), dim=1)


class AttentionPool(nn.Module):
    def __init__(self, channels=512):
        super().__init__()
        self.score = nn.Conv2d(channels, 1, 1)
        nn.init.zeros_(self.score.weight)
        nn.init.zeros_(self.score.bias)

    def forward(self, x):
        weights = self.score(x).flatten(2).softmax(dim=-1)
        return (x.flatten(2) * weights).sum(-1).unsqueeze(-1).unsqueeze(-1)
