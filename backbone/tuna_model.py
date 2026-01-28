from __future__ import annotations

from collections import OrderedDict
from typing import Tuple, Union, Optional, Dict, Any
import math
import copy

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F



class TaskVector:
    def __init__(self, pretrained_state_dict=None, vector=None):
        if vector is not None:
            self.vector = vector
        else:
            with torch.no_grad():
                self.vector = {}
                for key in pretrained_state_dict:
                    if pretrained_state_dict[key].dtype in [torch.int64, torch.uint8]:
                        continue
                    self.vector[key] = pretrained_state_dict[key]

    def __add__(self, other):
        with torch.no_grad():
            new_vector = {}
            for key in self.vector:
                if key not in other.vector:
                    print(f"Warning, key {key} is not present in both task vectors.")
                    continue
                new_vector[key] = self.vector[key] + other.vector[key]
        return TaskVector(vector=new_vector)

    def __radd__(self, other):
        if other is None or isinstance(other, int):
            return self
        return self.__add__(other)

    def __neg__(self):
        with torch.no_grad():
            new_vector = {}
            for key in self.vector:
                new_vector[key] = -self.vector[key]
        return TaskVector(vector=new_vector)

    def weightmerging(self, taskvectors, coefficients):
        with torch.no_grad():
            new_vector = {}
            for key in taskvectors[0].vector:
                new_vector[key] = sum(coefficients[k] * taskvectors[k][key] for k in range(len(taskvectors)))
        return TaskVector(vector=new_vector)


def emr_merge(task_vectors):
    sum_param = {}
    n2p = []
    for m in range(len(task_vectors)):
        n2p_temp = task_vectors[m].vector
        n2p.append(n2p_temp)
        for n in n2p_temp:
            if n not in sum_param:
                sum_param[n] = []
            sum_param[n].append(n2p_temp[n])
    sum_param = {k: torch.stack(v, 0).mean(0) for k, v in sum_param.items()}

    vector_unified = {}
    scales = torch.zeros(len(task_vectors))
    masks = {}
    for n in sum_param:
        masks[n] = []
        flag = (sum_param[n] > 0) * 2 - 1
        param_max = torch.zeros_like(n2p[0][n])
        for m in range(len(task_vectors)):
            param = task_vectors[m].vector[n]
            mask = (param * flag) > 0
            masks[n].append(mask)
            param_abs = torch.abs(mask * param)
            param_max = torch.where(param_abs > param_max, param_abs, param_max)
            scales[m] += torch.mean(torch.abs(param))
        vector_unified[n] = param_max * flag

    new_scales = torch.zeros(len(task_vectors))
    for m in range(len(task_vectors)):
        for n in vector_unified:
            p = vector_unified[n] * masks[n][m]
            new_scales[m] += torch.mean(torch.abs(p))
    rescalers = scales / new_scales
    return vector_unified, masks, rescalers


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
    """
    Minimal FFN adapter. Supports x shape [..., C].
    """
    def __init__(
        self,
        d_model: int,
        bottleneck: int,
        dropout: float = 0.0,
        init_option: str = "lora",
        adapter_scalar: str = "1.0",
        adapter_layernorm_option: str = "in",
    ):
        super().__init__()
        self.n_embd = d_model
        self.down_size = bottleneck
        self.dropout = dropout

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

        if init_option == "lora":
            with torch.no_grad():
                nn.init.kaiming_uniform_(self.down_proj.weight, a=math.sqrt(5))
                nn.init.zeros_(self.up_proj.weight)
                nn.init.zeros_(self.down_proj.bias)
                nn.init.zeros_(self.up_proj.bias)
        elif init_option == "bert":
            raise NotImplementedError("bert init not implemented.")
        else:
            raise ValueError(f"Unknown init_option={init_option}")

    def forward(self, x: torch.Tensor, add_residual: bool = True, residual: Optional[torch.Tensor] = None):
        residual = x if residual is None else residual

        if self.adapter_layernorm_option == "in" and self.adapter_layer_norm_before is not None:
            x = self.adapter_layer_norm_before(x)

        down = self.down_proj(x)
        down = self.non_linear_func(down)
        down = F.dropout(down, p=self.dropout, training=self.training)
        up = self.up_proj(down) * self.scale

        if self.adapter_layernorm_option == "out" and self.adapter_layer_norm_before is not None:
            up = self.adapter_layer_norm_before(up)

        return (up + residual) if add_residual else up


class ResidualAttentionBlock(nn.Module):
    def __init__(self, d_model: int, n_head: int, attn_mask: torch.Tensor = None, tuning_config=None):
        super().__init__()
        self.tuning_config = tuning_config

        self.attn = nn.MultiheadAttention(d_model, n_head)
        self.ln_1 = LayerNorm(d_model)
        self.mlp = nn.Sequential(OrderedDict([
            ("c_fc", nn.Linear(d_model, d_model * 4)),
            ("gelu", QuickGELU()),
            ("c_proj", nn.Linear(d_model * 4, d_model)),
        ]))
        self.ln_2 = LayerNorm(d_model)
        self.attn_mask = attn_mask

    def attention(self, x: torch.Tensor):
        attn_mask = None
        if self.attn_mask is not None:
            attn_mask = self.attn_mask.to(dtype=x.dtype, device=x.device)
        return self.attn(x, x, x, need_weights=False, attn_mask=attn_mask)[0]

    def forward(self, x: torch.Tensor, adapt: Optional[Adapter] = None):
        # Attention
        x = x + self.attention(self.ln_1(x))

        # FFN + Adapter
        cfg = self.tuning_config
        use_adapt = (adapt is not None) and (cfg is not None) and getattr(cfg, "ffn_adapt", False)

        adapt_x = None
        if use_adapt and getattr(cfg, "ffn_option", "parallel") == "parallel":
            adapt_x = adapt(x, add_residual=False)

        residual = x
        ffn_out = self.mlp(self.ln_2(x))

        if use_adapt:
            if cfg.ffn_option == "sequential":
                ffn_out = adapt(ffn_out)
            elif cfg.ffn_option == "parallel":
                ffn_out = ffn_out + adapt_x
            else:
                raise ValueError(f"Unknown ffn_option={cfg.ffn_option}")

        x = residual + ffn_out
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

    def forward(self, x: torch.Tensor, adapters: Optional[nn.ModuleList] = None):
        for i, blk in enumerate(self.resblocks):
            adapt = None if adapters is None else adapters[i]
            x = blk(x, adapt=adapt)
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
        tuning_config=None
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

        self._device = getattr(tuning_config, "_device", None) if tuning_config is not None else None
        self.adapter_list = nn.ModuleList()
        self.cur_adapter = nn.ModuleList()
        self.merged_adapter = nn.ModuleList()

        if tuning_config is not None and getattr(tuning_config, "ffn_adapt", False):
            self.init_adapters()

    @property
    def base_out_dim(self) -> int:
        return self.output_dim

    def out_dim(self, adapter_id: int = -1) -> int:
        return self.output_dim

    def init_adapters(self):
        cfg = self.tuning_config
        bottleneck = getattr(cfg, "ffn_num", 16)
        dropout = getattr(cfg, "ffn_adapter_dropout", 0.1)
        init_opt = getattr(cfg, "ffn_adapter_init_option", "lora")
        scalar = getattr(cfg, "ffn_adapter_scalar", "0.1")
        lnopt = getattr(cfg, "ffn_adapter_layernorm_option", "none")

        self.cur_adapter = nn.ModuleList()
        for _ in range(self.transformer.layers):
            ad = Adapter(
                d_model=self.transformer.width,
                bottleneck=bottleneck,
                dropout=dropout,
                init_option=init_opt,
                adapter_scalar=scalar,
                adapter_layernorm_option=lnopt,
            )
            if self._device is not None:
                ad = ad.to(self._device)
            self.cur_adapter.append(ad)

        self.cur_adapter.requires_grad_(True)

    def freeze_visual_backbone(self, train_proj: bool = False):
        for p in self.parameters():
            p.requires_grad = False
        for p in self.cur_adapter.parameters():
            p.requires_grad = True
        if train_proj and self.proj is not None:
            self.proj.requires_grad_(True)

    def adapter_update(self, reset_new: bool = False):
        frozen = copy.deepcopy(self.cur_adapter)
        frozen.requires_grad_(False)
        self.adapter_list.append(frozen)
        if reset_new:
            self.init_adapters()

    def merge(self):
        if len(self.adapter_list) == 0:
            self.merged_adapter = copy.deepcopy(self.cur_adapter)
            if self._device is not None:
                self.merged_adapter = self.merged_adapter.to(self._device)
            return

        task_vectors = [
            TaskVector(pretrained_state_dict=self.adapter_list[i].cpu().state_dict())
            for i in range(len(self.adapter_list))
        ]
        vector_unified, masks, rescalers = emr_merge(task_vectors)

        self.merged_adapter = copy.deepcopy(self.cur_adapter).cpu()
        self.merged_adapter.load_state_dict(vector_unified, strict=False)
        if self._device is not None:
            self.merged_adapter = self.merged_adapter.to(self._device)

        if self._device is not None:
            for i in range(len(self.adapter_list)):
                self.adapter_list[i] = self.adapter_list[i].to(self._device)

    def _select_adapters(self, adapter_id: int, train: bool) -> Optional[nn.ModuleList]:
        cfg = self.tuning_config
        if cfg is None or not getattr(cfg, "ffn_adapt", False):
            return None

        if adapter_id == -1:
            return None

        if train:
            return self.cur_adapter

        if adapter_id < len(self.adapter_list):
            return self.adapter_list[adapter_id]
        if adapter_id == len(self.adapter_list):
            return self.cur_adapter
        # adapter_id > len(adapter_list)
        if len(self.merged_adapter) == 0:
            return self.cur_adapter
        return self.merged_adapter

    def _forward_with_adapters(self, x: torch.Tensor, adapters: Optional[nn.ModuleList]) -> torch.Tensor:
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
        x = self.transformer(x, adapters=adapters)
        x = x.permute(1, 0, 2)                        # [B, N+1, C]

        x = self.ln_post(x[:, 0, :])                  # [B, C]
        x = x @ self.proj                              # [B, output_dim]
        return x

    def forward(self, x: torch.Tensor, adapter_id: int = -1, train: bool = False) -> Dict[str, torch.Tensor]:
        adapters = self._select_adapters(adapter_id, train=train)
        feat = self._forward_with_adapters(x, adapters=adapters)
        return {"features": feat}

    @torch.no_grad()
    def forward_proto(self, x: torch.Tensor, adapter_id: int) -> torch.Tensor:
        adapters = self._select_adapters(adapter_id, train=False)
        return self._forward_with_adapters(x, adapters=adapters)

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
            raise NotImplementedError("This implementation focuses on CLIP ViT visual backbone only.")
        else:
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


    def encode_image(self, image: torch.Tensor, adapter_id: int = -1, train: bool = False) -> torch.Tensor:
        res = self.visual(image.type(self.dtype), adapter_id=adapter_id, train=train)
        return res["features"]

    def encode_text(self, text: torch.Tensor) -> torch.Tensor:
        x = self.token_embedding(text).type(self.dtype)     # [B, n_ctx, d_model]
        x = x + self.positional_embedding.type(self.dtype)
        x = x.permute(1, 0, 2)                              # NLD -> LND
        x = self.transformer(x)                             # no adapters
        x = x.permute(1, 0, 2)                              # LND -> NLD
        x = self.ln_final(x).type(self.dtype)
        x = x[torch.arange(x.shape[0]), text.argmax(dim=-1)] @ self.text_projection
        return x

    def forward(self, image: torch.Tensor, text: torch.Tensor, adapter_id: int = -1, train: bool = False):
        image_features = self.encode_image(image, adapter_id=adapter_id, train=train)
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

    # fp16
    convert_weights(model)

    msg = model.load_state_dict(state_dict, strict=False)


    return model.eval()
