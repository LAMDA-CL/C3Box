import math
from collections import OrderedDict
from typing import Optional, Dict

import torch
import torch.nn as nn
import torch.nn.functional as F


def init_ssf_scale_shift(dim: int):
    scale = nn.Parameter(torch.ones(dim))
    shift = nn.Parameter(torch.zeros(dim))
    nn.init.normal_(scale, mean=1.0, std=0.02)
    nn.init.normal_(shift, std=0.02)
    return scale, shift


def ssf_ada(x, scale, shift):
    assert scale.shape == shift.shape
    # x: [B, N, C]
    if x.shape[-1] == scale.shape[0]:
        return x * scale + shift
    # x: [B, C, H, W]
    if x.dim() == 4 and x.shape[1] == scale.shape[0]:
        return x * scale.view(1, -1, 1, 1) + shift.view(1, -1, 1, 1)
    raise ValueError("SSF: tensor shape mismatch")


class LayerNorm(nn.LayerNorm):
    def forward(self, x: torch.Tensor):
        orig_type = x.dtype
        return super().forward(x.float()).to(orig_type)


class QuickGELU(nn.Module):
    def forward(self, x: torch.Tensor):
        return x * torch.sigmoid(1.702 * x)


class ResidualAttentionBlockSSF(nn.Module):
    def __init__(self, d_model: int, n_head: int, attn_mask: Optional[torch.Tensor] = None, tuning_mode: str = "ssf"):
        super().__init__()
        self.attn = nn.MultiheadAttention(d_model, n_head)
        self.ln_1 = LayerNorm(d_model)

        self.mlp = nn.Sequential(OrderedDict([
            ("c_fc", nn.Linear(d_model, d_model * 4)),
            ("gelu", QuickGELU()),
            ("c_proj", nn.Linear(d_model * 4, d_model)),
        ]))
        self.ln_2 = LayerNorm(d_model)
        self.attn_mask = attn_mask
        self.tuning_mode = tuning_mode

        if self.tuning_mode == "ssf":
            self.ssf_scale_1, self.ssf_shift_1 = init_ssf_scale_shift(d_model)
            self.ssf_scale_2, self.ssf_shift_2 = init_ssf_scale_shift(d_model)

            self.ssf_scale_attn, self.ssf_shift_attn = init_ssf_scale_shift(d_model)
            self.ssf_scale_mlp, self.ssf_shift_mlp = init_ssf_scale_shift(d_model)

    def attention(self, x: torch.Tensor):
        attn_mask = None
        if self.attn_mask is not None:
            attn_mask = self.attn_mask.to(dtype=x.dtype, device=x.device)
        return self.attn(x, x, x, need_weights=False, attn_mask=attn_mask)[0]

    def forward(self, x: torch.Tensor):
        # x: [L,B,C]
        if self.tuning_mode == "ssf":
            x_norm = self.ln_1(x)
            x_norm = ssf_ada(x_norm, self.ssf_scale_1, self.ssf_shift_1)
            attn_out = self.attention(x_norm)
            attn_out = ssf_ada(attn_out, self.ssf_scale_attn, self.ssf_shift_attn)
            x = x + attn_out

            x_norm2 = self.ln_2(x)
            x_norm2 = ssf_ada(x_norm2, self.ssf_scale_2, self.ssf_shift_2)
            mlp_out = self.mlp(x_norm2)
            mlp_out = ssf_ada(mlp_out, self.ssf_scale_mlp, self.ssf_shift_mlp)
            x = x + mlp_out
            return x

        x = x + self.attention(self.ln_1(x))
        x = x + self.mlp(self.ln_2(x))
        return x


class Transformer(nn.Module):
    def __init__(self, width: int, layers: int, heads: int, attn_mask: Optional[torch.Tensor] = None, tuning_mode: str = "ssf"):
        super().__init__()
        self.width = width
        self.layers = layers
        self.resblocks = nn.ModuleList([
            ResidualAttentionBlockSSF(width, heads, attn_mask=attn_mask, tuning_mode=tuning_mode)
            for _ in range(layers)
        ])

    def forward(self, x: torch.Tensor):
        for blk in self.resblocks:
            x = blk(x)
        return x


class VisionTransformer(nn.Module):
    def __init__(
        self,
        input_resolution: int,
        patch_size: int,
        width: int,
        layers: int,
        heads: int,
        output_dim: int,
        tuning_mode: str = "ssf",
    ):
        super().__init__()
        self.input_resolution = input_resolution
        self.output_dim = output_dim
        self.tuning_mode = tuning_mode

        # CLIP: conv1 patchify
        self.conv1 = nn.Conv2d(3, width, kernel_size=patch_size, stride=patch_size, bias=False)

        scale = width ** -0.5
        self.class_embedding = nn.Parameter(scale * torch.randn(width))
        self.positional_embedding = nn.Parameter(scale * torch.randn((input_resolution // patch_size) ** 2 + 1, width))
        self.ln_pre = LayerNorm(width)

        self.transformer = Transformer(width, layers, heads, attn_mask=None, tuning_mode=tuning_mode)

        self.ln_post = LayerNorm(width)
        self.proj = nn.Parameter(scale * torch.randn(width, output_dim))

        if self.tuning_mode == "ssf":
            self.ssf_scale_patch, self.ssf_shift_patch = init_ssf_scale_shift(width)
            self.ssf_scale_pre, self.ssf_shift_pre = init_ssf_scale_shift(width)
            self.ssf_scale_post, self.ssf_shift_post = init_ssf_scale_shift(width)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: [B,3,H,W]
        x = self.conv1(x)                         # [B,C,gh,gw]
        x = x.reshape(x.shape[0], x.shape[1], -1) # [B,C,N]
        x = x.permute(0, 2, 1)                    # [B,N,C]

        if self.tuning_mode == "ssf":
            x = ssf_ada(x, self.ssf_scale_patch, self.ssf_shift_patch)

        x = torch.cat([
            self.class_embedding.to(x.dtype)
            + torch.zeros(x.shape[0], 1, x.shape[-1], dtype=x.dtype, device=x.device),
            x
        ], dim=1)                                 # [B,N+1,C]

        x = x + self.positional_embedding.to(x.dtype)  # [B,N+1,C]
        x = self.ln_pre(x)

        if self.tuning_mode == "ssf":
            x = ssf_ada(x, self.ssf_scale_pre, self.ssf_shift_pre)

        x = x.permute(1, 0, 2)                    # [L,B,C]
        x = self.transformer(x)
        x = x.permute(1, 0, 2)                    # [B,L,C]

        x = self.ln_post(x[:, 0, :])              # [B,C]

        if self.tuning_mode == "ssf":
            x = ssf_ada(x, self.ssf_scale_post, self.ssf_shift_post)

        x = x @ self.proj                         # [B,output_dim]
        return x
