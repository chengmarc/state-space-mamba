"""MambaSSM regression model for log-price residual forecasting.

Given a sequence of slice snapshots of shape (batch, n_slices, n_features),
predicts the log-price residual at t+predict_window.  The model outputs one
scalar per prediction step; use MSE loss during training.
"""

import math
from typing import Union
from dataclasses import dataclass

import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange, repeat

from .scans import selective_scan


@dataclass
class ModelArgs:
    d_model:   int
    n_layer:   int
    d_state:   int = 16
    expand:    int = 2
    dt_rank:   Union[int, str] = 'auto'
    d_conv:    int = 4
    conv_bias: bool = True
    bias:      bool = False
    scan_mode: str  = 'logcumsumexp'
    dropout:   float = 0.1

    def __post_init__(self):
        self.d_inner = int(self.expand * self.d_model)
        if self.dt_rank == 'auto':
            self.dt_rank = math.ceil(self.d_model / 16)


class MambaSSM(nn.Module):

    def __init__(self, args: ModelArgs, input_dimension: int, n_horizons: int):
        super().__init__()
        self.args = args

        self.input_layer = nn.Linear(input_dimension, args.d_model)
        self.layers      = nn.ModuleList([ResidualBlock(args) for _ in range(args.n_layer)])
        self.norm_f      = RMSNorm(args.d_model)
        self.dropout     = nn.Dropout(args.dropout)
        # One logit per prediction horizon.
        self.head        = nn.Linear(args.d_model, n_horizons)

    def forward(self, x):
        x    = self.input_layer(x)
        for layer in self.layers:
            x = layer(x)
        x    = self.norm_f(x)
        last = self.dropout(x[:, -1, :])   # (batch, d_model) — richest context
        return self.head(last)              # (batch, n_horizons) — raw logits


class ResidualBlock(nn.Module):

    def __init__(self, args: ModelArgs):
        super().__init__()
        self.mixer   = MambaBlock(args)
        self.norm    = RMSNorm(args.d_model)
        self.dropout = nn.Dropout(args.dropout)

    def forward(self, x):
        return self.dropout(self.mixer(self.norm(x))) + x


class MambaBlock(nn.Module):

    def __init__(self, args: ModelArgs):
        super().__init__()
        self.args = args

        self.in_proj  = nn.Linear(args.d_model, args.d_inner * 2, bias=args.bias)
        self.conv1d   = nn.Conv1d(
            in_channels=args.d_inner, out_channels=args.d_inner,
            bias=args.conv_bias, kernel_size=args.d_conv,
            groups=args.d_inner, padding=args.d_conv - 1,
        )
        self.x_proj   = nn.Linear(args.d_inner, args.dt_rank + args.d_state * 2, bias=False)
        self.dt_proj  = nn.Linear(args.dt_rank, args.d_inner, bias=True)

        A = repeat(torch.arange(1, args.d_state + 1), 'n -> d n', d=args.d_inner)
        self.A_log = nn.Parameter(torch.log(A))
        self.D     = nn.Parameter(torch.ones(args.d_inner))
        self.out_proj = nn.Linear(args.d_inner, args.d_model, bias=args.bias)

    def forward(self, x):
        (b, l, d) = x.shape
        x_and_res = self.in_proj(x)
        (x, res)  = x_and_res.split([self.args.d_inner, self.args.d_inner], dim=-1)

        x = rearrange(x, 'b l d_in -> b d_in l')
        x = self.conv1d(x)[:, :, :l]
        x = rearrange(x, 'b d_in l -> b l d_in')

        x = F.silu(x)
        y = self.ssm(x)
        y = y * F.silu(res)
        return self.out_proj(y)

    def ssm(self, x):
        (d_in, n) = self.A_log.shape
        A = -torch.exp(self.A_log.float())
        D = self.D.float()

        x_dbl = self.x_proj(x)
        (delta, B, C) = x_dbl.split([self.args.dt_rank, n, n], dim=-1)
        delta = F.softplus(self.dt_proj(delta))

        return selective_scan(x, delta, A, B, C, D, mode=self.args.scan_mode)


class RMSNorm(nn.Module):

    def __init__(self, d_model: int, eps: float = 1e-5):
        super().__init__()
        self.eps    = eps
        self.weight = nn.Parameter(torch.ones(d_model))

    def forward(self, x):
        return x * torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + self.eps) * self.weight


def create_model(data, n_horizons: int, device, d_model: int = 32, n_layer: int = 3, dropout: float = 0.1):
    """Build a MambaSSM classifier for ``n_horizons`` binary direction outputs."""
    input_dimension = len(data.columns)
    args  = ModelArgs(d_model=d_model, n_layer=n_layer, dropout=dropout)
    model = MambaSSM(args, input_dimension, n_horizons).to(device)
    print(f'MambaSSM initialised — input_dim={input_dimension}  d_model={d_model}  '
          f'n_layer={n_layer}  dropout={dropout}  n_horizons={n_horizons}')
    return model
