from collections import OrderedDict
from typing import Tuple, Union, Optional
import math

import numpy as np
import torch
import torch.nn.functional as F
from torch import nn


class Adapter(nn.Module):
    """
    A minimal EASE-style FFN adapter.
    Supports input of shape [..., C] where C = d_model (works for [N,B,C] and [B,N,C]).
    """
    def __init__(self,
                 d_model: int,
                 bottleneck: int,
                 dropout: float = 0.0,
                 init_option: str = "lora",
                 adapter_scalar: str = "1.0",
                 adapter_layernorm_option: str = "in"):
        super().__init__()
        self.n_embd = d_model
        self.down_size = bottleneck

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

        # EASE LoRA-style init
        if init_option == "lora":
            with torch.no_grad():
                nn.init.kaiming_uniform_(self.down_proj.weight, a=math.sqrt(5))
                nn.init.zeros_(self.up_proj.weight)
                nn.init.zeros_(self.down_proj.bias)
                nn.init.zeros_(self.up_proj.bias)
        elif init_option == "bert":
            raise NotImplementedError("bert init not implemented (matches your original).")

    def forward(self, x: torch.Tensor, add_residual: bool = True, residual: Optional[torch.Tensor] = None):
        residual = x if residual is None else residual

        if self.adapter_layernorm_option == "in":
            x = self.adapter_layer_norm_before(x)

        down = self.down_proj(x)
        down = self.non_linear_func(down)
        down = F.dropout(down, p=self.dropout, training=self.training)
        up = self.up_proj(down) * self.scale

        if self.adapter_layernorm_option == "out":
            up = self.adapter_layer_norm_before(up)

        return (up + residual) if add_residual else up


class Bottleneck(nn.Module):
    expansion = 4

    def __init__(self, inplanes, planes, stride=1):
        super().__init__()
        self.conv1 = nn.Conv2d(inplanes, planes, 1, bias=False)
        self.bn1 = nn.BatchNorm2d(planes)
        self.relu1 = nn.ReLU(inplace=True)

        self.conv2 = nn.Conv2d(planes, planes, 3, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(planes)
        self.relu2 = nn.ReLU(inplace=True)

        self.avgpool = nn.AvgPool2d(stride) if stride > 1 else nn.Identity()

        self.conv3 = nn.Conv2d(planes, planes * self.expansion, 1, bias=False)
        self.bn3 = nn.BatchNorm2d(planes * self.expansion)
        self.relu3 = nn.ReLU(inplace=True)

        self.downsample = None
        self.stride = stride

        if stride > 1 or inplanes != planes * Bottleneck.expansion:
            self.downsample = nn.Sequential(OrderedDict([
                ("-1", nn.AvgPool2d(stride)),
                ("0", nn.Conv2d(inplanes, planes * self.expansion, 1, stride=1, bias=False)),
                ("1", nn.BatchNorm2d(planes * self.expansion))
            ]))

    def forward(self, x: torch.Tensor):
        identity = x
        out = self.relu1(self.bn1(self.conv1(x)))
        out = self.relu2(self.bn2(self.conv2(out)))
        out = self.avgpool(out)
        out = self.bn3(self.conv3(out))
        if self.downsample is not None:
            identity = self.downsample(x)
        out += identity
        out = self.relu3(out)
        return out


class AttentionPool2d(nn.Module):
    def __init__(self, spacial_dim: int, embed_dim: int, num_heads: int, output_dim: int = None):
        super().__init__()
        self.positional_embedding = nn.Parameter(torch.randn(spacial_dim ** 2 + 1, embed_dim) / embed_dim ** 0.5)
        self.k_proj = nn.Linear(embed_dim, embed_dim)
        self.q_proj = nn.Linear(embed_dim, embed_dim)
        self.v_proj = nn.Linear(embed_dim, embed_dim)
        self.c_proj = nn.Linear(embed_dim, output_dim or embed_dim)
        self.num_heads = num_heads

    def forward(self, x):
        x = x.flatten(start_dim=2).permute(2, 0, 1)  # NCHW -> (HW)NC
        x = torch.cat([x.mean(dim=0, keepdim=True), x], dim=0)  # (HW+1)NC
        x = x + self.positional_embedding[:, None, :].to(x.dtype)  # (HW+1)NC
        x, _ = F.multi_head_attention_forward(
            query=x[:1], key=x, value=x,
            embed_dim_to_check=x.shape[-1],
            num_heads=self.num_heads,
            q_proj_weight=self.q_proj.weight,
            k_proj_weight=self.k_proj.weight,
            v_proj_weight=self.v_proj.weight,
            in_proj_weight=None,
            in_proj_bias=torch.cat([self.q_proj.bias, self.k_proj.bias, self.v_proj.bias]),
            bias_k=None,
            bias_v=None,
            add_zero_attn=False,
            dropout_p=0,
            out_proj_weight=self.c_proj.weight,
            out_proj_bias=self.c_proj.bias,
            use_separate_proj_weight=True,
            training=self.training,
            need_weights=False
        )
        return x.squeeze(0)


class ModifiedResNet(nn.Module):
    def __init__(self, layers, output_dim, heads, input_resolution=224, width=64):
        super().__init__()
        self.output_dim = output_dim
        self.input_resolution = input_resolution

        self.conv1 = nn.Conv2d(3, width // 2, kernel_size=3, stride=2, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(width // 2)
        self.relu1 = nn.ReLU(inplace=True)
        self.conv2 = nn.Conv2d(width // 2, width // 2, kernel_size=3, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(width // 2)
        self.relu2 = nn.ReLU(inplace=True)
        self.conv3 = nn.Conv2d(width // 2, width, kernel_size=3, padding=1, bias=False)
        self.bn3 = nn.BatchNorm2d(width)
        self.relu3 = nn.ReLU(inplace=True)
        self.avgpool = nn.AvgPool2d(2)

        self._inplanes = width
        self.layer1 = self._make_layer(width, layers[0])
        self.layer2 = self._make_layer(width * 2, layers[1], stride=2)
        self.layer3 = self._make_layer(width * 4, layers[2], stride=2)
        self.layer4 = self._make_layer(width * 8, layers[3], stride=2)

        embed_dim = width * 32
        self.attnpool = AttentionPool2d(input_resolution // 32, embed_dim, heads, output_dim)

    def _make_layer(self, planes, blocks, stride=1):
        layers = [Bottleneck(self._inplanes, planes, stride)]
        self._inplanes = planes * Bottleneck.expansion
        for _ in range(1, blocks):
            layers.append(Bottleneck(self._inplanes, planes))
        return nn.Sequential(*layers)

    def forward(self, x):
        def stem(z):
            z = self.relu1(self.bn1(self.conv1(z)))
            z = self.relu2(self.bn2(self.conv2(z)))
            z = self.relu3(self.bn3(self.conv3(z)))
            z = self.avgpool(z)
            return z

        x = x.type(self.conv1.weight.dtype)
        x = stem(x)
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        x = self.attnpool(x)
        return x


class LayerNorm(nn.LayerNorm):
    """Subclass torch's LayerNorm to handle fp16."""
    def forward(self, x: torch.Tensor):
        orig_type = x.dtype
        ret = super().forward(x.type(torch.float32))
        return ret.type(orig_type)


class QuickGELU(nn.Module):
    def forward(self, x: torch.Tensor):
        return x * torch.sigmoid(1.702 * x)


class ResidualAttentionBlock(nn.Module):
    """
    Original CLIP block, with EASE adapter optionally injected on FFN.
    Note: CLIP uses LND (N, B, C) inside transformer.
    """
    def __init__(self, d_model: int, n_head: int, attn_mask: torch.Tensor = None, tuning_config=None):
        super().__init__()
        self.tuning_config = tuning_config

        self.attn = nn.MultiheadAttention(d_model, n_head)
        self.ln_1 = LayerNorm(d_model)
        self.mlp = nn.Sequential(OrderedDict([
            ("c_fc", nn.Linear(d_model, d_model * 4)),
            ("gelu", QuickGELU()),
            ("c_proj", nn.Linear(d_model * 4, d_model))
        ]))
        self.ln_2 = LayerNorm(d_model)
        self.attn_mask = attn_mask

    def attention(self, x: torch.Tensor):
        self.attn_mask = self.attn_mask.to(dtype=x.dtype, device=x.device) if self.attn_mask is not None else None
        return self.attn(x, x, x, need_weights=False, attn_mask=self.attn_mask)[0]

    def forward(self, x: torch.Tensor, adapt: Optional[Adapter] = None):
        # attn
        x = x + self.attention(self.ln_1(x))

        # ffn + adapter (EASE)
        cfg = self.tuning_config
        use_adapt = (adapt is not None) and (cfg is not None) and getattr(cfg, "ffn_adapt", False)

        # parallel branch precompute
        adapt_x = None
        if use_adapt and getattr(cfg, "ffn_option", "parallel") == "parallel":
            adapt_x = adapt(x, add_residual=False)

        residual = x
        ffn_out = self.mlp(self.ln_2(x))

        if use_adapt:
            if cfg.ffn_option == "sequential":
                ffn_out = adapt(ffn_out)  # adapter adds residual internally
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
        self.resblocks = nn.ModuleList(
            [ResidualAttentionBlock(width, heads, attn_mask, tuning_config=tuning_config) for _ in range(layers)]
        )

    def forward(self, x: torch.Tensor, adapters: Optional[nn.ModuleList] = None):
        for i, blk in enumerate(self.resblocks):
            adapt = None if adapters is None else adapters[i]
            x = blk(x, adapt=adapt)
        return x


class VisionTransformer(nn.Module):
    def __init__(self,
                 input_resolution: int,
                 patch_size: int,
                 width: int,
                 layers: int,
                 heads: int,
                 output_dim: int,
                 tuning_config=None):
        super().__init__()
        self.input_resolution = input_resolution
        self.output_dim = output_dim
        self.tuning_config = tuning_config

        self.conv1 = nn.Conv2d(in_channels=3, out_channels=width,
                               kernel_size=patch_size, stride=patch_size, bias=False)

        scale = width ** -0.5
        self.class_embedding = nn.Parameter(scale * torch.randn(width))
        self.positional_embedding = nn.Parameter(
            scale * torch.randn((input_resolution // patch_size) ** 2 + 1, width)
        )
        self.ln_pre = LayerNorm(width)

        self.transformer = Transformer(width, layers, heads, tuning_config=tuning_config)

        self.ln_post = LayerNorm(width)
        self.proj = nn.Parameter(scale * torch.randn(width, output_dim))

        self.adapter_list = []
        self.cur_adapter = nn.ModuleList()

        if tuning_config is not None and getattr(tuning_config, "ffn_adapt", False):
            self._device = getattr(tuning_config, "_device", None)
            self.get_new_adapter()
        else:
            self._device = None

    def get_new_adapter(self):
        cfg = self.tuning_config
        self.cur_adapter = nn.ModuleList()
        bottleneck = getattr(cfg, "ffn_num", None)
        assert bottleneck is not None, "tuning_config.ffn_num is required for adapters"

        for _ in range(self.transformer.layers):
            ad = Adapter(
                d_model=self.transformer.width,
                bottleneck=bottleneck,
                dropout=0.1,
                init_option=getattr(cfg, "ffn_adapter_init_option", "lora"),
                adapter_scalar=getattr(cfg, "ffn_adapter_scalar", "1.0"),
                adapter_layernorm_option=getattr(cfg, "ffn_adapter_layernorm_option", "in"),
            )
            if self._device is not None:
                ad = ad.to(self._device)
            self.cur_adapter.append(ad)

        self.cur_adapter.requires_grad_(True)

    def add_adapter_to_list(self):
        import copy
        self.adapter_list.append(copy.deepcopy(self.cur_adapter.requires_grad_(False)))
        self.get_new_adapter()

    def freeze_visual_backbone(self, train_proj: bool = False):
        # freeze all
        for p in self.parameters():
            p.requires_grad = False
        # unfreeze adapters
        for p in self.cur_adapter.parameters():
            p.requires_grad = True
        # optional: train projection
        if train_proj and self.proj is not None:
            self.proj.requires_grad_(True)

    def _forward_with_adapters(self, x: torch.Tensor, adapters: Optional[nn.ModuleList]):
        # original CLIP ViT forward, only difference is transformer(adapters=...)
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

        x = x.permute(1, 0, 2)                        # [N+1, B, C]  (LND)
        x = self.transformer(x, adapters=adapters)
        x = x.permute(1, 0, 2)                        # [B, N+1, C]

        x = self.ln_post(x[:, 0, :])                  # [B, C]
        if self.proj is not None:
            x = x @ self.proj                         # [B, output_dim]
        return x

    def forward_train(self, x: torch.Tensor):
        cfg = self.tuning_config
        if cfg is not None and getattr(cfg, "ffn_adapt", False):
            return self._forward_with_adapters(x, adapters=self.cur_adapter)
        else:
            return self._forward_with_adapters(x, adapters=None)

    def forward_test(self, x: torch.Tensor, use_init_ptm: bool = False):
        outs = []
        if use_init_ptm:
            outs.append(self._forward_with_adapters(x, adapters=None))
        for old in self.adapter_list:
            outs.append(self._forward_with_adapters(x, adapters=old))
        outs.append(self._forward_with_adapters(x, adapters=self.cur_adapter))
        return torch.cat(outs, dim=1)  # [B, output_dim * num_versions]

    def forward(self, x: torch.Tensor, test: bool = False, use_init_ptm: bool = False):
        return self.forward_test(x, use_init_ptm=use_init_ptm) if test else self.forward_train(x)

    @torch.no_grad()
    def forward_proto(self, x: torch.Tensor, adapt_index: int):
        if adapt_index == -1:
            return self._forward_with_adapters(x, adapters=None)

        if adapt_index < len(self.adapter_list):
            return self._forward_with_adapters(x, adapters=self.adapter_list[adapt_index])

        return self._forward_with_adapters(x, adapters=self.cur_adapter)



class CLIP(nn.Module):
    def __init__(self,
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
                 tuning_config=None):
        super().__init__()
        self.context_length = context_length

        # vision
        if isinstance(vision_layers, (tuple, list)):
            vision_heads = vision_width * 32 // 64
            self.visual = ModifiedResNet(
                layers=vision_layers,
                output_dim=embed_dim,
                heads=vision_heads,
                input_resolution=image_resolution,
                width=vision_width
            )
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

        if isinstance(self.visual, ModifiedResNet):
            if self.visual.attnpool is not None:
                std = self.visual.attnpool.c_proj.in_features ** -0.5
                nn.init.normal_(self.visual.attnpool.q_proj.weight, std=std)
                nn.init.normal_(self.visual.attnpool.k_proj.weight, std=std)
                nn.init.normal_(self.visual.attnpool.v_proj.weight, std=std)
                nn.init.normal_(self.visual.attnpool.c_proj.weight, std=std)

            for resnet_block in [self.visual.layer1, self.visual.layer2, self.visual.layer3, self.visual.layer4]:
                for name, param in resnet_block.named_parameters():
                    if name.endswith("bn3.weight"):
                        nn.init.zeros_(param)

        proj_std = (self.transformer.width ** -0.5) * ((2 * self.transformer.layers) ** -0.5)
        attn_std = self.transformer.width ** -0.5
        fc_std = (2 * self.transformer.width) ** -0.5

        # text transformer init stays original
        for block in self.transformer.resblocks:
            nn.init.normal_(block.attn.in_proj_weight, std=attn_std)
            nn.init.normal_(block.attn.out_proj.weight, std=proj_std)
            nn.init.normal_(block.mlp.c_fc.weight, std=fc_std)
            nn.init.normal_(block.mlp.c_proj.weight, std=proj_std)

        if self.text_projection is not None:
            nn.init.normal_(self.text_projection, std=self.transformer.width ** -0.5)

    def build_attention_mask(self):
        mask = torch.empty(self.context_length, self.context_length)
        mask.fill_(float("-inf"))
        mask.triu_(1)
        return mask

    @property
    def dtype(self):
        # visual could be ModifiedResNet or VisionTransformer
        if hasattr(self.visual, "conv1"):
            return self.visual.conv1.weight.dtype
        # fallback
        return next(self.parameters()).dtype

    def encode_image(self, image: torch.Tensor, test: bool = False, use_init_ptm: bool = False):
        if isinstance(self.visual, VisionTransformer):
            return self.visual(image.type(self.dtype), test=test, use_init_ptm=use_init_ptm)
        else:
            return self.visual(image.type(self.dtype))
    def encode_text(self, text: torch.Tensor):
        x = self.token_embedding(text).type(self.dtype)  # [B, n_ctx, d_model]
        x = x + self.positional_embedding.type(self.dtype)
        x = x.permute(1, 0, 2)                           # NLD -> LND
        x = self.transformer(x)                          # no adapters
        x = x.permute(1, 0, 2)                           # LND -> NLD
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
                tensor = getattr(l, attr)
                if tensor is not None:
                    tensor.data = tensor.data.half()

        for name in ["text_projection", "proj"]:
            if hasattr(l, name):
                attr = getattr(l, name)
                if attr is not None:
                    attr.data = attr.data.half()

    model.apply(_convert_weights_to_fp16)


def build_model(state_dict: dict, tuning_config=None):
    vit = "visual.proj" in state_dict

    if vit:
        vision_width = state_dict["visual.conv1.weight"].shape[0]
        vision_layers = len([k for k in state_dict.keys()
                             if k.startswith("visual.") and k.endswith(".attn.in_proj_weight")])
        vision_patch_size = state_dict["visual.conv1.weight"].shape[-1]
        grid_size = round((state_dict["visual.positional_embedding"].shape[0] - 1) ** 0.5)
        image_resolution = vision_patch_size * grid_size
    else:
        counts: list = [len(set(k.split(".")[2] for k in state_dict
                                if k.startswith(f"visual.layer{b}")))
                        for b in [1, 2, 3, 4]]
        vision_layers = tuple(counts)
        vision_width = state_dict["visual.layer1.0.conv1.weight"].shape[0]
        output_width = round((state_dict["visual.attnpool.positional_embedding"].shape[0] - 1) ** 0.5)
        vision_patch_size = None
        assert output_width ** 2 + 1 == state_dict["visual.attnpool.positional_embedding"].shape[0]
        image_resolution = output_width * 32

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

    convert_weights(model)

    msg = model.load_state_dict(state_dict, strict=False)

    return model.eval()
