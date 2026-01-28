from __future__ import annotations

from collections import OrderedDict
from typing import Tuple, Union, Optional, Dict
import math

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


class LayerNorm(nn.LayerNorm):
    """torch LayerNorm but safe for fp16"""
    def forward(self, x: torch.Tensor):
        orig_type = x.dtype
        ret = super().forward(x.float())
        return ret.to(orig_type)


class QuickGELU(nn.Module):
    def forward(self, x: torch.Tensor):
        return x * torch.sigmoid(1.702 * x)


class Adapter(nn.Module):
    def __init__(
        self,
        config=None,
        d_model: Optional[int] = None,
        bottleneck: Optional[int] = None,
        dropout: float = 0.0,
        init_option: str = "bert",
        adapter_scalar: str = "1.0",
        adapter_layernorm_option: str = "in",
    ):
        super().__init__()
        self.n_embd = config.d_model if d_model is None else d_model
        self.down_size = config.attn_bn if bottleneck is None else bottleneck

        self.adapter_layernorm_option = adapter_layernorm_option
        self.adapter_layer_norm_before = None
        if adapter_layernorm_option in ("in", "out"):
            self.adapter_layer_norm_before = nn.LayerNorm(self.n_embd)

        if adapter_scalar == "learnable_scalar":
            self.scale = nn.Parameter(torch.ones(1))
        else:
            self.scale = float(adapter_scalar)

        self.down_proj = nn.Linear(self.n_embd, self.down_size)
        self.non_linear_func = nn.ReLU()
        self.up_proj = nn.Linear(self.down_size, self.n_embd)

        self.dropout = dropout
        if init_option == "bert":
            raise NotImplementedError
        elif init_option == "lora":
            with torch.no_grad():
                nn.init.kaiming_uniform_(self.down_proj.weight, a=math.sqrt(5))
                nn.init.zeros_(self.up_proj.weight)
                nn.init.zeros_(self.down_proj.bias)
                nn.init.zeros_(self.up_proj.bias)

    def forward(self, x, add_residual=True, residual=None):
        residual = x if residual is None else residual
        if self.adapter_layernorm_option == "in":
            x = self.adapter_layer_norm_before(x)

        down = self.down_proj(x)
        down = self.non_linear_func(down)
        down = F.dropout(down, p=self.dropout, training=self.training)
        up = self.up_proj(down)

        up = up * self.scale

        if self.adapter_layernorm_option == "out":
            up = self.adapter_layer_norm_before(up)

        if add_residual:
            output = up + residual
        else:
            output = up
        return output


class ResidualAttentionBlock(nn.Module):
    def __init__(self, d_model: int, n_head: int, attn_mask: torch.Tensor = None, tuning_config=None):
        super().__init__()
        self.config = tuning_config

        self.attn = nn.MultiheadAttention(d_model, n_head)
        self.ln_1 = LayerNorm(d_model)

        self.mlp = nn.Sequential(OrderedDict([
            ("c_fc", nn.Linear(d_model, d_model * 4)),
            ("gelu", QuickGELU()),
            ("c_proj", nn.Linear(d_model * 4, d_model)),
        ]))
        self.ln_2 = LayerNorm(d_model)
        self.attn_mask = attn_mask

        if self.config is not None and getattr(self.config, "ffn_adapt", False):
            self.adaptmlp = Adapter(
                self.config,
                dropout=0.1,
                bottleneck=self.config.ffn_num,
                init_option=self.config.ffn_adapter_init_option,
                adapter_scalar=self.config.ffn_adapter_scalar,
                adapter_layernorm_option=self.config.ffn_adapter_layernorm_option,
            )
        else:
            self.adaptmlp = None

    def attention(self, x: torch.Tensor):
        attn_mask = None
        if self.attn_mask is not None:
            attn_mask = self.attn_mask.to(dtype=x.dtype, device=x.device)
        return self.attn(x, x, x, need_weights=False, attn_mask=attn_mask)[0]

    def forward(self, x: torch.Tensor):
        x = x + self.attention(self.ln_1(x))

        if (self.config is not None) and getattr(self.config, "ffn_adapt", False) and self.config.ffn_option == "parallel":
            adapt_x = self.adaptmlp(x, add_residual=False)
        else:
            adapt_x = None

        residual = x

        x = self.mlp(self.ln_2(x))

        if (self.config is not None) and getattr(self.config, "ffn_adapt", False):
            if self.config.ffn_option == "sequential":
                x = self.adaptmlp(x)
            elif self.config.ffn_option == "parallel":
                x = x + adapt_x
            else:
                raise ValueError(self.config.ffn_adapt)

        x = residual + x
        return x


class Transformer(nn.Module):
    def __init__(self, width: int, layers: int, heads: int, attn_mask: torch.Tensor = None, tuning_config=None):
        super().__init__()
        self.width = width
        self.layers = layers
        self.resblocks = nn.ModuleList([
            ResidualAttentionBlock(width, heads, attn_mask, tuning_config=tuning_config)
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
        tuning_config=None,
    ):
        super().__init__()
        self.input_resolution = input_resolution
        self.output_dim = output_dim
        self.tuning_config = tuning_config

        self.conv1 = nn.Conv2d(3, width, kernel_size=patch_size, stride=patch_size, bias=False)

        scale = width ** -0.5
        self.class_embedding = nn.Parameter(scale * torch.randn(width))
        self.positional_embedding = nn.Parameter(
            scale * torch.randn((input_resolution // patch_size) ** 2 + 1, width)
        )
        self.ln_pre = LayerNorm(width)

        self.transformer = Transformer(width, layers, heads, tuning_config=tuning_config)

        self.ln_post = LayerNorm(width)
        self.proj = nn.Parameter(scale * torch.randn(width, output_dim))

        self.vpt_on = bool(getattr(tuning_config, "vpt_on", False)) if tuning_config is not None else False
        self.vpt_num = int(getattr(tuning_config, "vpt_num", 0)) if tuning_config is not None else 0
        if self.vpt_on:
            assert self.vpt_num > 0, "vpt_num must be > 0 when vpt_on=True"
            self.embeddings = nn.ParameterList([
                nn.Parameter(torch.empty(self.vpt_num, 1, width)) for _ in range(layers)
            ])
            for e in self.embeddings:
                nn.init.xavier_uniform_(e.data)

    def _forward_impl(self, x: torch.Tensor) -> torch.Tensor:
        x = self.conv1(x)                             # [B, C, gh, gw]
        x = x.reshape(x.shape[0], x.shape[1], -1)     # [B, C, N]
        x = x.permute(0, 2, 1)                        # [B, N, C]

        x = torch.cat([
            self.class_embedding.to(x.dtype)
            + torch.zeros(x.shape[0], 1, x.shape[-1], dtype=x.dtype, device=x.device),
            x
        ], dim=1)                                     # [B, N+1, C]

        x = x + self.positional_embedding.to(x.dtype)
        x = self.ln_pre(x)

        x = x.permute(1, 0, 2)                        # [N+1, B, C] (LND)

        if self.vpt_on:
            for i, blk in enumerate(self.transformer.resblocks):
                p = self.embeddings[i].to(dtype=x.dtype, device=x.device)  # [P,1,C]
                p = p.expand(-1, x.shape[1], -1)                           # [P,B,C]
                x = torch.cat([p, x], dim=0)                               # [P+N+1,B,C]
                x = blk(x)
                x = x[self.vpt_num:, :, :]                                 # remove prompt
        else:
            x = self.transformer(x)

        x = x.permute(1, 0, 2)                        # [B, N+1, C]

        x = self.ln_post(x[:, 0, :])                  # [B, C]
        x = x @ self.proj                             # [B, output_dim]
        return x

    def forward(self, x: torch.Tensor) -> Dict[str, torch.Tensor]:
        feat = self._forward_impl(x)
        return {"features": feat}


class CLIP(nn.Module):
    def __init__(
        self,
        embed_dim: int,
        # vision
        image_resolution: int,
        vision_layers: Union[Tuple[int, int, int, int], int],
        vision_width: int,
        vision_patch_size: int,
        # text
        context_length: int,
        vocab_size: int,
        transformer_width: int,
        transformer_heads: int,
        transformer_layers: int,
        # tuning
        tuning_config=None,
    ):
        super().__init__()
        self.context_length = context_length

        if isinstance(vision_layers, (tuple, list)):
            raise NotImplementedError("This reproduction targets CLIP ViT checkpoints only.")

        vision_heads = vision_width // 64
        self.visual = VisionTransformer(
            input_resolution=image_resolution,
            patch_size=vision_patch_size,
            width=vision_width,
            layers=vision_layers,
            heads=vision_heads,
            output_dim=embed_dim,
            tuning_config=tuning_config,
        )

        self.transformer = Transformer(
            width=transformer_width,
            layers=transformer_layers,
            heads=transformer_heads,
            attn_mask=self.build_attention_mask(),
            tuning_config=None,
        )

        self.vocab_size = vocab_size
        self.token_embedding = nn.Embedding(vocab_size, transformer_width)
        self.positional_embedding = nn.Parameter(torch.empty(self.context_length, transformer_width))
        self.ln_final = LayerNorm(transformer_width)

        self.text_projection = nn.Parameter(torch.empty(transformer_width, embed_dim))
        self.logit_scale = nn.Parameter(torch.ones([]) * np.log(1 / 0.07))

        self.initialize_parameters()

    def initialize_parameters(self):
        nn.init.normal_(self.token_embedding.weight, std=0.02)
        nn.init.normal_(self.positional_embedding, std=0.01)

        proj_std = (self.transformer.width ** -0.5) * ((2 * self.transformer.layers) ** -0.5)
        attn_std = self.transformer.width ** -0.5
        fc_std = (2 * self.transformer.width) ** -0.5

        for block in self.transformer.resblocks:
            nn.init.normal_(block.attn.in_proj_weight, std=attn_std)
            nn.init.normal_(block.attn.out_proj.weight, std=proj_std)
            nn.init.normal_(block.mlp.c_fc.weight, std=fc_std)
            nn.init.normal_(block.mlp.c_proj.weight, std=proj_std)

        nn.init.normal_(self.text_projection, std=self.transformer.width ** -0.5)

    def build_attention_mask(self):
        mask = torch.empty(self.context_length, self.context_length)
        mask.fill_(float("-inf"))
        mask.triu_(1)
        return mask

    @property
    def dtype(self):
        return self.visual.conv1.weight.dtype

    def encode_image(self, image: torch.Tensor) -> torch.Tensor:
        return self.visual(image.type(self.dtype))["features"]

    def encode_text(self, text: torch.Tensor) -> torch.Tensor:
        x = self.token_embedding(text).type(self.dtype)
        x = x + self.positional_embedding.type(self.dtype)
        x = x.permute(1, 0, 2)
        x = self.transformer(x)
        x = x.permute(1, 0, 2)
        x = self.ln_final(x).type(self.dtype)
        x = x[torch.arange(x.shape[0]), text.argmax(dim=-1)] @ self.text_projection
        return x

    def forward(self, image: torch.Tensor, text: torch.Tensor):
        image_features = self.encode_image(image)
        text_features = self.encode_text(text)

        image_features = image_features / image_features.norm(dim=1, keepdim=True)
        text_features = text_features / text_features.norm(dim=1, keepdim=True)

        logit_scale = self.logit_scale.exp()
        logits_per_image = logit_scale * image_features @ text_features.t()
        logits_per_text = logits_per_image.t()
        return logits_per_image, logits_per_text


def convert_weights(model: nn.Module):
    def _convert_weights_to_fp16(l):
        if isinstance(l, (nn.Conv1d, nn.Conv2d, nn.Linear)):
            l.weight.data = l.weight.data.half()
            if l.bias is not None:
                l.bias.data = l.bias.data.half()

        if isinstance(l, nn.MultiheadAttention):
            for attr in [*[f"{s}_proj_weight" for s in ["in", "q", "k", "v"]],
                         "in_proj_bias", "bias_k", "bias_v"]:
                tensor = getattr(l, attr, None)
                if tensor is not None:
                    tensor.data = tensor.data.half()

        for name in ["text_projection", "proj"]:
            if hasattr(l, name):
                attr = getattr(l, name)
                if attr is not None:
                    attr.data = attr.data.half()

    model.apply(_convert_weights_to_fp16)


def build_model(state_dict: dict, tuning_config=None) -> CLIP:
    vit = "visual.proj" in state_dict
    if not vit:
        raise NotImplementedError("This build_model targets CLIP ViT checkpoints only.")

    vision_width = state_dict["visual.conv1.weight"].shape[0]
    vision_layers = len([k for k in state_dict.keys()
                         if k.startswith("visual.") and k.endswith(".attn.in_proj_weight")])
    vision_patch_size = state_dict["visual.conv1.weight"].shape[-1]
    grid_size = round((state_dict["visual.positional_embedding"].shape[0] - 1) ** 0.5)
    image_resolution = vision_patch_size * grid_size

    embed_dim = state_dict["text_projection"].shape[1]
    context_length = state_dict["positional_embedding"].shape[0]
    vocab_size = state_dict["token_embedding.weight"].shape[0]
    transformer_width = state_dict["ln_final.weight"].shape[0]
    transformer_heads = transformer_width // 64
    transformer_layers = len(set(k.split(".")[2] for k in state_dict if k.startswith("transformer.resblocks")))

    model = CLIP(
        embed_dim,
        image_resolution, vision_layers, vision_width, vision_patch_size,
        context_length, vocab_size, transformer_width, transformer_heads, transformer_layers,
        tuning_config=tuning_config,
    )

    for key in ["input_resolution", "context_length", "vocab_size"]:
        if key in state_dict:
            del state_dict[key]
    msg = model.load_state_dict(state_dict, strict=False)
    print("missing:", len(msg.missing_keys), "unexpected:", len(msg.unexpected_keys))
    print("missing sample:", msg.missing_keys[:20])
    print("unexpected sample:", msg.unexpected_keys[:20])


    missing = set(msg.missing_keys)

    for _, p in model.named_parameters():
            p.requires_grad = False

    for name, p in model.named_parameters():
            if name in missing:
                p.requires_grad = True

    convert_weights(model)

    return model.eval()

