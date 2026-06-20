"""MambaSSM forecasting model.

A selective state-space model adapted for multi-feature sequence-to-sequence
forecasting. The block structure (selective scan, RMSNorm, dual-branch
projection) follows the community reference implementation
`johnma2006/mamba-minimal`; the forecasting wrapper — input/output projection and
the trailing-window output head — is project-specific.

The model maps an input window of shape ``(batch, length, input_dimension)`` to a
forecast of shape ``(batch, output_length, 1)``, taking the last ``output_length``
positions of the final hidden sequence as the prediction.
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
    d_model: int
    n_layer: int
    d_state: int = 16
    expand: int = 2
    dt_rank: Union[int, str] = 'auto'
    d_conv: int = 4
    conv_bias: bool = True
    bias: bool = False
    scan_mode: str = 'logcumsumexp'

    def __post_init__(self):
        self.d_inner = int(self.expand * self.d_model)
        if self.dt_rank == 'auto':
            self.dt_rank = math.ceil(self.d_model / 16)


class MambaSSM(nn.Module):

    def __init__(self, args: ModelArgs, input_dimension: int, output_length: int):
        super().__init__()
        self.args = args
        self.output_length = output_length

        self.input_layer  = nn.Linear(input_dimension, args.d_model)
        self.layers       = nn.ModuleList([ResidualBlock(args) for _ in range(args.n_layer)])
        self.norm_f       = RMSNorm(args.d_model)
        self.output_layer = nn.Linear(args.d_model, 1)

    def forward(self, x):
        x = self.input_layer(x)
        for layer in self.layers:
            x = layer(x)
        x = self.norm_f(x)
        return self.output_layer(x[:, -self.output_length:, :])


class ResidualBlock(nn.Module):

    def __init__(self, args: ModelArgs):
        super().__init__()
        self.mixer = MambaBlock(args)
        self.norm  = RMSNorm(args.d_model)

    def forward(self, x):
        return self.mixer(self.norm(x)) + x


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


def create_model(data, forecast_horizon: int, device, d_model: int = 32, n_layer: int = 8):
    """Build a :class:`MambaSSM` sized for ``data`` and move it to ``device``.

    ``data`` is the feature DataFrame; its column count fixes the model's input
    dimension. ``forecast_horizon`` sets the output length. ``d_model`` and
    ``n_layer`` default to the values used in the reported experiments; callers
    typically pass the project defaults from :mod:`ssm.config`.
    """
    input_dimension = len(data.columns)
    args  = ModelArgs(d_model=d_model, n_layer=n_layer)
    model = MambaSSM(args, input_dimension, forecast_horizon).to(device)
    print(f'MambaSSM initialised — input_dim={input_dimension}  d_model={d_model}  n_layer={n_layer}')
    return model
