# Source: https://github.com/SivanDoveh/TSVLC/blob/main/src/open_clip/loralib/layers.py
#
#  ------------------------------------------------------------------------------------------
#  Copyright (c) Microsoft Corporation. All rights reserved.
#  Licensed under the MIT License (MIT). See LICENSE in the repo root for license information.
#  ------------------------------------------------------------------------------------------
#
import pdb
import torch
import torch.nn as nn
import torch.nn.functional as F

import math
from typing import Optional, List, Tuple

from torch.nn.init import xavier_uniform_
from torch.nn.init import constant_
from torch.nn.init import xavier_normal_
from torch.nn.init import zeros_
import warnings

class LoRALayer():
    def __init__(
        self, 
        r: int, 
        lora_alpha: int, 
        lora_dropout: float,
        merge_weights: bool,
    ):
        self.r = r
        self.lora_alpha = lora_alpha
        # Optional dropout
        if lora_dropout > 0.:
            self.lora_dropout = nn.Dropout(p=lora_dropout)
        else:
            self.lora_dropout = lambda x: x
        # Mark the weight as unmerged
        self.merged = False
        self.merge_weights = merge_weights


class Embedding(nn.Embedding, LoRALayer):
    # LoRA implemented in a dense layer
    def __init__(
        self,
        num_embeddings: int,
        embedding_dim: int,
        r: int = 0,
        lora_alpha: int = 1,
        merge_weights: bool = True,
        **kwargs
    ):
        nn.Embedding.__init__(self, num_embeddings, embedding_dim, **kwargs)
        LoRALayer.__init__(self, r=r, lora_alpha=lora_alpha, lora_dropout=0,
                           merge_weights=merge_weights)
        # Actual trainable parameters
        if r > 0:
            self.lora_A = nn.Parameter(self.weight.new_zeros((r, num_embeddings)))
            self.lora_B = nn.Parameter(self.weight.new_zeros((embedding_dim, r)))
            self.scaling = self.lora_alpha / self.r
            # Freezing the pre-trained weight matrix
            self.weight.requires_grad = False
        self.reset_parameters()

    def reset_parameters(self):
        nn.Embedding.reset_parameters(self)
        if hasattr(self, 'lora_A'):
            # initialize A the same way as the default for nn.Linear and B to zero
            nn.init.zeros_(self.lora_A)
            nn.init.normal_(self.lora_B)

    def train(self, mode: bool = True):
        nn.Embedding.train(self, mode)
        if self.merge_weights and self.merged:
            # Make sure that the weights are not merged
            if self.r > 0:
                self.weight.data -= (self.lora_B @ self.lora_A).T * self.scaling
            self.merged = False
    
    def eval(self):
        nn.Linear.eval(self)
        if self.merge_weights and not self.merged:
            # Merge the weights and mark it
            if self.r > 0:
                self.weight.data += (self.lora_B @ self.lora_A) * self.scaling
            self.merged = True

    def forward(self, x: torch.Tensor):
        if self.r > 0 and not self.merged:
            result = nn.Embedding.forward(self, x)
            if self.r > 0:
                after_A = F.embedding(
                    x, self.lora_A.T, self.padding_idx, self.max_norm,
                    self.norm_type, self.scale_grad_by_freq, self.sparse
                )
                result += (after_A @ self.lora_B.T) * self.scaling
            return result
        else:
            return nn.Embedding.forward(self, x)
            

class Linear(nn.Linear, LoRALayer):
    # LoRA implemented in a dense layer
    def __init__(
        self, 
        in_features: int, 
        out_features: int, 
        r: int = 0, 
        lora_alpha: int = 1, 
        lora_dropout: float = 0.,
        fan_in_fan_out: bool = False, # Set this to True if the layer to replace stores weight like (fan_in, fan_out)
        merge_weights: bool = True,
        **kwargs
    ):
        nn.Linear.__init__(self, in_features, out_features, **kwargs)
        LoRALayer.__init__(self, r=r, lora_alpha=lora_alpha, lora_dropout=lora_dropout,
                           merge_weights=merge_weights)

        self.fan_in_fan_out = fan_in_fan_out
        # Actual trainable parameters
        if r > 0:
            self.lora_A = nn.Parameter(self.weight.new_zeros((r, in_features)))
            self.lora_B = nn.Parameter(self.weight.new_zeros((out_features, r)))
            self.scaling = self.lora_alpha / self.r
            # Freezing the pre-trained weight matrix
            self.weight.requires_grad = False
        self.reset_parameters()
        if fan_in_fan_out:
            self.weight.data = self.weight.data.T

    def reset_parameters(self):
        nn.Linear.reset_parameters(self)
        if hasattr(self, 'lora_A'):
            # initialize A the same way as the default for nn.Linear and B to zero
            nn.init.kaiming_uniform_(self.lora_A, a=math.sqrt(5))
            nn.init.zeros_(self.lora_B)

    def train(self, mode: bool = True):
        def T(w):
            return w.T if self.fan_in_fan_out else w
        nn.Linear.train(self, mode)
        if self.merge_weights and self.merged:
            # Make sure that the weights are not merged
            if self.r > 0:
                self.weight.data -= T(self.lora_B @ self.lora_A) * self.scaling
            self.merged = False
    
    def eval(self):
        def T(w):
            return w.T if self.fan_in_fan_out else w
        nn.Linear.eval(self)
        if self.merge_weights and not self.merged:
            # Merge the weights and mark it
            if self.r > 0:
                self.weight.data += T(self.lora_B @ self.lora_A) * self.scaling
            self.merged = True

    def forward(self, x: torch.Tensor):
        def T(w):
            return w.T if self.fan_in_fan_out else w
        if self.r > 0 and not self.merged:
            result = F.linear(x, T(self.weight), bias=self.bias)
            if self.r > 0:
                result += (self.lora_dropout(x) @ self.lora_A.T @ self.lora_B.T) * self.scaling
            return result
        else:
            return F.linear(x, T(self.weight), bias=self.bias)


class MergedLinear(nn.Linear, LoRALayer):
    # LoRA implemented in a dense layer
    def __init__(
        self, 
        in_features: int, 
        out_features: int, 
        r: int = 0, 
        lora_alpha: int = 1, 
        lora_dropout: float = 0.,
        enable_lora: List[bool] = [False],
        fan_in_fan_out: bool = False,
        merge_weights: bool = True,
        **kwargs
    ):
        nn.Linear.__init__(self, in_features, out_features, **kwargs)
        LoRALayer.__init__(self, r=r, lora_alpha=lora_alpha, lora_dropout=lora_dropout,
                           merge_weights=merge_weights)
        assert out_features % len(enable_lora) == 0, \
            'The length of enable_lora must divide out_features'
        self.enable_lora = enable_lora
        self.fan_in_fan_out = fan_in_fan_out
        # Actual trainable parameters
        if r > 0 and any(enable_lora):
            self.lora_A = nn.Parameter(
                self.weight.new_zeros((r * sum(enable_lora), in_features)))
            self.lora_B = nn.Parameter(
                self.weight.new_zeros((out_features // len(enable_lora) * sum(enable_lora), r))
            ) # weights for Conv1D with groups=sum(enable_lora)
            self.scaling = self.lora_alpha / self.r
            # Freezing the pre-trained weight matrix
            self.weight.requires_grad = False
            # Compute the indices
            self.lora_ind = self.weight.new_zeros(
                (out_features, ), dtype=torch.bool
            ).view(len(enable_lora), -1)
            self.lora_ind[enable_lora, :] = True
            self.lora_ind = self.lora_ind.view(-1)
        self.reset_parameters()
        if fan_in_fan_out:
            self.weight.data = self.weight.data.T

    def reset_parameters(self):
        nn.Linear.reset_parameters(self)
        if hasattr(self, 'lora_A'):
            # initialize A the same way as the default for nn.Linear and B to zero
            nn.init.kaiming_uniform_(self.lora_A, a=math.sqrt(5))
            nn.init.zeros_(self.lora_B)

    def zero_pad(self, x):
        result = x.new_zeros((*x.shape[:-1], self.out_features))
        result = result.view(-1, self.out_features)
        result[:, self.lora_ind] = x.reshape(
            -1, self.out_features // len(self.enable_lora) * sum(self.enable_lora)
        )
        return result.view((*x.shape[:-1], self.out_features))

    def train(self, mode: bool = True):
        def T(w):
            return w.T if self.fan_in_fan_out else w
        nn.Linear.train(self, mode)
        if self.merge_weights and self.merged:
            # Make sure that the weights are not merged
            if self.r > 0 and any(self.enable_lora):
                delta_w = F.conv1d(
                    self.lora_A.data.unsqueeze(0), 
                    self.lora_B.data.unsqueeze(-1), 
                    groups=sum(self.enable_lora)
                ).squeeze(0)
                self.weight.data -= self.zero_pad(T(delta_w * self.scaling))
            self.merged = False
    
    def eval(self):
        def T(w):
            return w.T if self.fan_in_fan_out else w
        nn.Linear.eval(self)
        if self.merge_weights and not self.merged:
            # Merge the weights and mark it
            if self.r > 0 and any(self.enable_lora):
                delta_w = F.conv1d(
                    self.lora_A.data.unsqueeze(0), 
                    self.lora_B.data.unsqueeze(-1), 
                    groups=sum(self.enable_lora)
                ).squeeze(0)
                self.weight.data += self.zero_pad(T(delta_w * self.scaling))
            self.merged = True

    def forward(self, x: torch.Tensor):
        def T(w):
            return w.T if self.fan_in_fan_out else w
        if self.merged:
            return F.linear(x, T(self.weight), bias=self.bias)
        else:
            result = F.linear(x, T(self.weight), bias=self.bias)
            if self.r > 0:
                after_A = F.linear(self.lora_dropout(x), self.lora_A)
                after_B = F.conv1d(
                    after_A.transpose(-2, -1), 
                    self.lora_B.unsqueeze(-1), 
                    groups=sum(self.enable_lora)
                ).transpose(-2, -1)
                result += self.zero_pad(after_B) * self.scaling
            return result
            

class Conv2d(nn.Conv2d, LoRALayer):
    # LoRA implemented in a dense layer
    def __init__(
        self, 
        in_channels: int, 
        out_channels: int,
        kernel_size: int,
        r: int = 0, 
        lora_alpha: int = 1, 
        lora_dropout: float = 0.,
        merge_weights: bool = True,
        **kwargs
    ):
        nn.Conv2d.__init__(self, in_channels, out_channels, kernel_size, **kwargs)
        LoRALayer.__init__(self, r=r, lora_alpha=lora_alpha, lora_dropout=lora_dropout,
                           merge_weights=merge_weights)
        assert type(kernel_size) is int
        # Actual trainable parameters
        if r > 0:
            self.lora_A = nn.Parameter(
                self.weight.new_zeros((r*kernel_size, in_channels*kernel_size))
            )
            self.lora_B = nn.Parameter(
                self.weight.new_zeros((out_channels*kernel_size, r*kernel_size))
            )
            self.scaling = self.lora_alpha / self.r
            # Freezing the pre-trained weight matrix
            self.weight.requires_grad = False
        self.reset_parameters()

    def reset_parameters(self):
        nn.Conv2d.reset_parameters(self)
        if hasattr(self, 'lora_A'):
            # initialize A the same way as the default for nn.Linear and B to zero
            nn.init.kaiming_uniform_(self.lora_A, a=math.sqrt(5))
            nn.init.zeros_(self.lora_B)

    def train(self, mode: bool = True):
        nn.Conv2d.train(self, mode)
        if self.merge_weights and self.merged:
            # Make sure that the weights are not merged
            self.weight.data -= (self.lora_B @ self.lora_A).view(self.weight.shape) * self.scaling
            self.merged = False

    def eval(self):
        nn.Conv2d.eval(self)
        if self.merge_weights and not self.merged:
            # Merge the weights and mark it
            self.weight.data += (self.lora_B @ self.lora_A).view(self.weight.shape) * self.scaling
            self.merged = True

    def forward(self, x: torch.Tensor):
        if self.r > 0 and not self.merged:
            return F.conv2d(
                x, 
                self.weight + (self.lora_B @ self.lora_A).view(self.weight.shape) * self.scaling,
                self.bias, self.stride, self.padding, self.dilation, self.groups
            )
        return nn.Conv2d.forward(self, x)

##########################
##########################
##########################

class MultiheadAttention(nn.Module):
    r"""Allows the model to jointly attend to information
    from different representation subspaces.
    See reference: Attention Is All You Need
    .. math::
        \text{MultiHead}(Q, K, V) = \text{Concat}(head_1,\dots,head_h)W^O
        \text{where} head_i = \text{Attention}(QW_i^Q, KW_i^K, VW_i^V)
    Args:
        embed_dim: total dimension of the model.
        num_heads: parallel attention heads.
        dropout: a Dropout layer on attn_output_weights. Default: 0.0.
        bias: add bias as module parameter. Default: True.
        add_bias_kv: add bias to the key and value sequences at dim=0.
        add_zero_attn: add a new batch of zeros to the key and
                       value sequences at dim=1.
        kdim: total number of features in key. Default: None.
        vdim: total number of features in value. Default: None.
        Note: if kdim and vdim are None, they will be set to embed_dim such that
        query, key, and value have the same number of features.
    Examples::
        >>> multihead_attn = nn.MultiheadAttention(embed_dim, num_heads)
        >>> attn_output, attn_output_weights = multihead_attn(query, key, value)
    """
    bias_k: Optional[torch.Tensor]
    bias_v: Optional[torch.Tensor]

    def __init__(self, embed_dim, num_heads, dropout=0., bias=True, add_bias_kv=False, add_zero_attn=False, kdim=None, vdim=None, lora_alpha: int = 1, r=0, only_kv=False,mlp=False):
        super(MultiheadAttention, self).__init__()
        self.embed_dim = embed_dim
        self.kdim = kdim if kdim is not None else embed_dim
        self.vdim = vdim if vdim is not None else embed_dim
        self._qkv_same_embed_dim = self.kdim == embed_dim and self.vdim == embed_dim
        self.only_kv = only_kv
        self.mlp = mlp
        self.num_heads = num_heads
        self.dropout = dropout
        self.head_dim = embed_dim // num_heads
        assert self.head_dim * num_heads == self.embed_dim, "embed_dim must be divisible by num_heads"

        assert r > 0

        self.lora_alpha = lora_alpha
        self.r = r
        self.scaling = self.lora_alpha / self.r

        if self._qkv_same_embed_dim is False:
            self.q_proj_weight = nn.Parameter(torch.Tensor(embed_dim, embed_dim))
            self.k_proj_weight = nn.Parameter(torch.Tensor(embed_dim, self.kdim))
            self.v_proj_weight = nn.Parameter(torch.Tensor(embed_dim, self.vdim))
            self.q_proj_weight_lora_A = nn.Parameter(torch.Tensor(r, embed_dim))
            self.k_proj_weight_lora_A = nn.Parameter(torch.Tensor(r, self.kdim))
            self.v_proj_weight_lora_A = nn.Parameter(torch.Tensor(r, self.vdim))
            self.q_proj_weight_lora_B = nn.Parameter(torch.Tensor(embed_dim, r))
            self.k_proj_weight_lora_B = nn.Parameter(torch.Tensor(embed_dim, r))
            self.v_proj_weight_lora_B = nn.Parameter(torch.Tensor(embed_dim, r))
            self.register_parameter('in_proj_weight', None)
            self.register_parameter('in_proj_weight_lora_A', None)
            self.register_parameter('in_proj_weight_lora_B', None)
        else:
            if only_kv and not mlp:
                self.q_proj_weight = nn.Parameter(torch.Tensor(embed_dim, embed_dim))
                self.k_proj_weight = nn.Parameter(torch.Tensor(embed_dim, embed_dim))
                self.v_proj_weight = nn.Parameter(torch.Tensor(embed_dim, embed_dim))
                # self.q_proj_weight_lora_A = nn.Parameter(torch.Tensor(r, embed_dim))
                self.register_parameter('q_proj_weight_lora_A', None)
                self.k_proj_weight_lora_A = nn.Parameter(torch.Tensor(r, self.kdim))
                self.v_proj_weight_lora_A = nn.Parameter(torch.Tensor(r, self.vdim))
                # self.q_proj_weight_lora_B = nn.Parameter(torch.Tensor(embed_dim, r))
                self.register_parameter('q_proj_weight_lora_B', None)
                self.k_proj_weight_lora_B = nn.Parameter(torch.Tensor(embed_dim, r))
                self.v_proj_weight_lora_B = nn.Parameter(torch.Tensor(embed_dim, r))
                self.register_parameter('in_proj_weight', None)
                self.register_parameter('in_proj_weight_lora_A', None)
                self.register_parameter('in_proj_weight_lora_B', None)
                pass
            elif mlp:
                self.q_proj_weight = nn.Parameter(torch.Tensor(embed_dim, embed_dim))
                self.k_proj_weight = nn.Parameter(torch.Tensor(embed_dim, embed_dim))
                self.v_proj_weight = nn.Parameter(torch.Tensor(embed_dim, embed_dim))
                # self.q_proj_weight_lora_A = nn.Parameter(torch.Tensor(r, embed_dim))
                self.register_parameter('q_proj_weight_lora_A', None)
                # self.k_proj_weight_lora_A = nn.Parameter(torch.Tensor(r, self.kdim))
                # self.v_proj_weight_lora_A = nn.Parameter(torch.Tensor(r, self.vdim))
                self.register_parameter('k_proj_weight_lora_A', None)
                self.register_parameter('v_proj_weight_lora_A', None)
                # self.q_proj_weight_lora_B = nn.Parameter(torch.Tensor(embed_dim, r))
                self.register_parameter('q_proj_weight_lora_B', None)
                # self.k_proj_weight_lora_B = nn.Parameter(torch.Tensor(embed_dim, r))
                # self.v_proj_weight_lora_B = nn.Parameter(torch.Tensor(embed_dim, r))
                self.register_parameter('k_proj_weight_lora_B', None)
                self.register_parameter('v_proj_weight_lora_B', None)
                self.register_parameter('in_proj_weight', None)
                self.register_parameter('in_proj_weight_lora_A', None)
                self.register_parameter('in_proj_weight_lora_B', None)
                pass
            else:
                self.in_proj_weight = nn.Parameter(torch.empty(3 * embed_dim, embed_dim))
                self.in_proj_weight_lora_A = nn.Parameter(torch.empty(r, embed_dim))
                self.in_proj_weight_lora_B = nn.Parameter(torch.empty(3 * embed_dim, r))
                self.register_parameter('q_proj_weight', None)
                self.register_parameter('k_proj_weight', None)
                self.register_parameter('v_proj_weight', None)
                self.register_parameter('q_proj_weight_lora_A', None)
                self.register_parameter('k_proj_weight_lora_A', None)
                self.register_parameter('v_proj_weight_lora_A', None)
                self.register_parameter('q_proj_weight_lora_B', None)
                self.register_parameter('k_proj_weight_lora_B', None)
                self.register_parameter('v_proj_weight_lora_B', None)

        if bias:
            self.in_proj_bias = nn.Parameter(torch.empty(3 * embed_dim))
        else:
            self.register_parameter('in_proj_bias', None)
        if only_kv:
            self.out_proj = Linear(embed_dim, embed_dim, bias=True, merge_weights=False, lora_alpha=lora_alpha, r=0)
        else:
            self.out_proj = Linear(embed_dim, embed_dim, bias=True, merge_weights=False, lora_alpha=lora_alpha, r=r)

        if add_bias_kv:
            self.bias_k = nn.Parameter(torch.empty(1, 1, embed_dim))
            self.bias_v = nn.Parameter(torch.empty(1, 1, embed_dim))
        else:
            self.bias_k = self.bias_v = None

        self.add_zero_attn = add_zero_attn

        self._reset_parameters()

    def _reset_parameters(self):
        if self._qkv_same_embed_dim:
            if self.only_kv and not self.mlp:
                xavier_uniform_(self.q_proj_weight)
                xavier_uniform_(self.k_proj_weight)
                xavier_uniform_(self.v_proj_weight)
                # xavier_uniform_(self.q_proj_weight_lora_A)
                xavier_uniform_(self.k_proj_weight_lora_A)
                xavier_uniform_(self.v_proj_weight_lora_A)
                # zeros_(self.q_proj_weight_lora_B)
                zeros_(self.k_proj_weight_lora_B)
                zeros_(self.v_proj_weight_lora_B)
                pass
            elif self.mlp:
                xavier_uniform_(self.q_proj_weight)
                xavier_uniform_(self.k_proj_weight)
                xavier_uniform_(self.v_proj_weight)
                # xavier_uniform_(self.q_proj_weight_lora_A)
                # xavier_uniform_(self.k_proj_weight_lora_A)
                # xavier_uniform_(self.v_proj_weight_lora_A)
                # zeros_(self.q_proj_weight_lora_B)
                # zeros_(self.k_proj_weight_lora_B)
                # zeros_(self.v_proj_weight_lora_B)
                pass
            else:
                xavier_uniform_(self.in_proj_weight)
                xavier_uniform_(self.in_proj_weight_lora_A)
                # B should be zeros
                zeros_(self.in_proj_weight_lora_B)
                # xavier_uniform_(self.in_proj_weight_lora_B)
        else:
            xavier_uniform_(self.q_proj_weight)
            xavier_uniform_(self.k_proj_weight)
            xavier_uniform_(self.v_proj_weight)
            xavier_uniform_(self.q_proj_weight_lora_A)
            xavier_uniform_(self.k_proj_weight_lora_A)
            xavier_uniform_(self.v_proj_weight_lora_A)
            zeros_(self.q_proj_weight_lora_B)
            zeros_(self.k_proj_weight_lora_B)
            zeros_(self.v_proj_weight_lora_B)
            # xavier_uniform_(self.q_proj_weight_lora_B)
            # xavier_uniform_(self.k_proj_weight_lora_B)
            # xavier_uniform_(self.v_proj_weight_lora_B)

        if self.in_proj_bias is not None:
            constant_(self.in_proj_bias, 0.)
            constant_(self.out_proj.bias, 0.)
        if self.bias_k is not None:
            xavier_normal_(self.bias_k)
        if self.bias_v is not None:
            xavier_normal_(self.bias_v)

    def __setstate__(self, state):
        # Support loading old MultiheadAttention checkpoints generated by v1.1.0
        if '_qkv_same_embed_dim' not in state:
            state['_qkv_same_embed_dim'] = True

        super(MultiheadAttention, self).__setstate__(state)

    def forward(self, query, key, value, key_padding_mask=None,
                need_weights=True, attn_mask=None):
        # type: (Tensor, Tensor, Tensor, Optional[Tensor], bool, Optional[Tensor]) -> Tuple[Tensor, Optional[Tensor]]
        r"""
    Args:
        query, key, value: map a query and a set of key-value pairs to an output.
            See "Attention Is All You Need" for more details.
        key_padding_mask: if provided, specified padding elements in the key will
            be ignored by the attention. When given a binary mask and a value is True,
            the corresponding value on the attention layer will be ignored. When given
            a byte mask and a value is non-zero, the corresponding value on the attention
            layer will be ignored
        need_weights: output attn_output_weights.
        attn_mask: 2D or 3D mask that prevents attention to certain positions. A 2D mask will be broadcasted for all
            the batches while a 3D mask allows to specify a different mask for the entries of each batch.
    Shape:
        - Inputs:
        - query: :math:`(L, N, E)` where L is the target sequence length, N is the batch size, E is
          the embedding dimension.
        - key: :math:`(S, N, E)`, where S is the source sequence length, N is the batch size, E is
          the embedding dimension.
        - value: :math:`(S, N, E)` where S is the source sequence length, N is the batch size, E is
          the embedding dimension.
        - key_padding_mask: :math:`(N, S)` where N is the batch size, S is the source sequence length.
          If a ByteTensor is provided, the non-zero positions will be ignored while the position
          with the zero positions will be unchanged. If a BoolTensor is provided, the positions with the
          value of ``True`` will be ignored while the position with the value of ``False`` will be unchanged.
        - attn_mask: 2D mask :math:`(L, S)` where L is the target sequence length, S is the source sequence length.
          3D mask :math:`(N*num_heads, L, S)` where N is the batch size, L is the target sequence length,
          S is the source sequence length. attn_mask ensure that position i is allowed to attend the unmasked
          positions. If a ByteTensor is provided, the non-zero positions are not allowed to attend
          while the zero positions will be unchanged. If a BoolTensor is provided, positions with ``True``
          is not allowed to attend while ``False`` values will be unchanged. If a FloatTensor
          is provided, it will be added to the attention weight.
        - Outputs:
        - attn_output: :math:`(L, N, E)` where L is the target sequence length, N is the batch size,
          E is the embedding dimension.
        - attn_output_weights: :math:`(N, L, S)` where N is the batch size,
          L is the target sequence length, S is the source sequence length.
        """
        if not self._qkv_same_embed_dim:
            return multi_head_attention_forward(
                query, key, value, self.embed_dim, self.num_heads,
                self.in_proj_weight, self.in_proj_bias, self.in_proj_weight_lora_A, self.in_proj_weight_lora_B, self.scaling,
                self.bias_k, self.bias_v, self.add_zero_attn,
                self.dropout, self.out_proj.weight, self.out_proj.bias, self.out_proj.lora_A, self.out_proj.lora_B, self.out_proj.scaling,
                training=self.training,
                key_padding_mask=key_padding_mask, need_weights=need_weights,
                attn_mask=attn_mask, use_separate_proj_weight=True,
                q_proj_weight=self.q_proj_weight, k_proj_weight=self.k_proj_weight, v_proj_weight=self.v_proj_weight,
                q_proj_weight_A=self.q_proj_weight_lora_A, k_proj_weight_A=self.k_proj_weight_lora_A, v_proj_weight_A=self.v_proj_weight_lora_A,
                q_proj_weight_B=self.q_proj_weight_lora_B, k_proj_weight_B=self.k_proj_weight_lora_B, v_proj_weight_B=self.v_proj_weight_lora_B,
                q_proj_weight_scaling=self.scaling, k_proj_weight_scaling=self.scaling, v_proj_weight_scaling=self.scaling
            )
        else:
            if self.only_kv and not self.mlp:
                # return multi_head_attention_forward(
                #     query, key, value, self.embed_dim, self.num_heads,
                #     self.in_proj_weight, self.in_proj_bias, self.in_proj_weight_lora_A, self.in_proj_weight_lora_B, self.scaling,
                #     self.bias_k, self.bias_v, self.add_zero_attn,
                #     self.dropout, self.out_proj.weight, self.out_proj.bias,
                #     training=self.training,
                #     key_padding_mask=key_padding_mask, need_weights=need_weights,
                #     attn_mask=attn_mask)
                return multi_head_attention_forward(
                    query, key, value, self.embed_dim, self.num_heads,
                    self.in_proj_weight, self.in_proj_bias, self.in_proj_weight_lora_A, self.in_proj_weight_lora_B, self.scaling,
                    self.bias_k, self.bias_v, self.add_zero_attn,
                    self.dropout, self.out_proj.weight, self.out_proj.bias, 
                    training=self.training,
                    key_padding_mask=key_padding_mask, need_weights=need_weights,
                    attn_mask=attn_mask, use_separate_proj_weight=True,
                    q_proj_weight=self.q_proj_weight, k_proj_weight=self.k_proj_weight, v_proj_weight=self.v_proj_weight,
                    q_proj_weight_A=self.q_proj_weight_lora_A, k_proj_weight_A=self.k_proj_weight_lora_A, v_proj_weight_A=self.v_proj_weight_lora_A,
                    q_proj_weight_B=self.q_proj_weight_lora_B, k_proj_weight_B=self.k_proj_weight_lora_B, v_proj_weight_B=self.v_proj_weight_lora_B,
                    q_proj_weight_scaling=self.scaling, k_proj_weight_scaling=self.scaling, v_proj_weight_scaling=self.scaling,
                    only_kv=True
                )
            elif self.mlp:
                return multi_head_attention_forward(
                    query, key, value, self.embed_dim, self.num_heads,
                    self.in_proj_weight, self.in_proj_bias, self.in_proj_weight_lora_A, self.in_proj_weight_lora_B, self.scaling,
                    self.bias_k, self.bias_v, self.add_zero_attn,
                    self.dropout, self.out_proj.weight, self.out_proj.bias, 
                    training=self.training,
                    key_padding_mask=key_padding_mask, need_weights=need_weights,
                    attn_mask=attn_mask, use_separate_proj_weight=True,
                    q_proj_weight=self.q_proj_weight, k_proj_weight=self.k_proj_weight, v_proj_weight=self.v_proj_weight,
                    only_kv=True,
                    mlp=True,
                )

            else:
                return multi_head_attention_forward(
                    query, key, value, self.embed_dim, self.num_heads,
                    self.in_proj_weight, self.in_proj_bias, self.in_proj_weight_lora_A, self.in_proj_weight_lora_B, self.scaling,
                    self.bias_k, self.bias_v, self.add_zero_attn,
                    self.dropout, self.out_proj.weight, self.out_proj.bias, self.out_proj.lora_A, self.out_proj.lora_B, self.out_proj.scaling,
                    training=self.training,
                    key_padding_mask=key_padding_mask, need_weights=need_weights,
                    attn_mask=attn_mask)

from torch import Tensor
from torch.nn.functional import linear, pad, softmax, dropout
def multi_head_attention_forward(query: Tensor,
                                 key: Tensor,
                                 value: Tensor,
                                 embed_dim_to_check: int,
                                 num_heads: int,
                                 in_proj_weight: Tensor,
                                 in_proj_bias: Tensor,
                                 in_proj_weight_A: Tensor,
                                 in_proj_weight_B: Tensor,
                                 in_proj_weight_scaling: Tensor,
                                 bias_k: Optional[Tensor],
                                 bias_v: Optional[Tensor],
                                 add_zero_attn: bool,
                                 dropout_p: float,
                                 out_proj_weight: Tensor,
                                 out_proj_bias: Tensor,
                                 out_proj_weight_A: Tensor = None,
                                 out_proj_weight_B: Tensor = None,
                                 out_proj_weight_scaling: Tensor = None,
                                 training: bool = True,
                                 key_padding_mask: Optional[Tensor] = None,
                                 need_weights: bool = True,
                                 attn_mask: Optional[Tensor] = None,
                                 use_separate_proj_weight: bool = False,
                                 q_proj_weight: Optional[Tensor] = None,
                                 k_proj_weight: Optional[Tensor] = None,
                                 v_proj_weight: Optional[Tensor] = None,
                                 q_proj_weight_A: Optional[Tensor] = None,
                                 k_proj_weight_A: Optional[Tensor] = None,
                                 v_proj_weight_A: Optional[Tensor] = None,
                                 q_proj_weight_B: Optional[Tensor] = None,
                                 k_proj_weight_B: Optional[Tensor] = None,
                                 v_proj_weight_B: Optional[Tensor] = None,
                                 q_proj_weight_scaling = None,
                                 k_proj_weight_scaling = None,
                                 v_proj_weight_scaling = None,
                                 static_k: Optional[Tensor] = None,
                                 static_v: Optional[Tensor] = None,
                                 only_kv: bool = False,
                                 mlp: bool = False
                                 ) -> Tuple[Tensor, Optional[Tensor]]:
    r"""
    Args:
        query, key, value: map a query and a set of key-value pairs to an output.
            See "Attention Is All You Need" for more details.
        embed_dim_to_check: total dimension of the model.
        num_heads: parallel attention heads.
        in_proj_weight, in_proj_bias: input projection weight and bias.
        bias_k, bias_v: bias of the key and value sequences to be added at dim=0.
        add_zero_attn: add a new batch of zeros to the key and
                       value sequences at dim=1.
        dropout_p: probability of an element to be zeroed.
        out_proj_weight, out_proj_bias: the output projection weight and bias.
        training: apply dropout if is ``True``.
        key_padding_mask: if provided, specified padding elements in the key will
            be ignored by the attention. This is an binary mask. When the value is True,
            the corresponding value on the attention layer will be filled with -inf.
        need_weights: output attn_output_weights.
        attn_mask: 2D or 3D mask that prevents attention to certain positions. A 2D mask will be broadcasted for all
            the batches while a 3D mask allows to specify a different mask for the entries of each batch.
        use_separate_proj_weight: the function accept the proj. weights for query, key,
            and value in different forms. If false, in_proj_weight will be used, which is
            a combination of q_proj_weight, k_proj_weight, v_proj_weight.
        q_proj_weight, k_proj_weight, v_proj_weight, in_proj_bias: input projection weight and bias.
        static_k, static_v: static key and value used for attention operators.
    Shape:
        Inputs:
        - query: :math:`(L, N, E)` where L is the target sequence length, N is the batch size, E is
          the embedding dimension.
        - key: :math:`(S, N, E)`, where S is the source sequence length, N is the batch size, E is
          the embedding dimension.
        - value: :math:`(S, N, E)` where S is the source sequence length, N is the batch size, E is
          the embedding dimension.
        - key_padding_mask: :math:`(N, S)` where N is the batch size, S is the source sequence length.
          If a ByteTensor is provided, the non-zero positions will be ignored while the zero positions
          will be unchanged. If a BoolTensor is provided, the positions with the
          value of ``True`` will be ignored while the position with the value of ``False`` will be unchanged.
        - attn_mask: 2D mask :math:`(L, S)` where L is the target sequence length, S is the source sequence length.
          3D mask :math:`(N*num_heads, L, S)` where N is the batch size, L is the target sequence length,
          S is the source sequence length. attn_mask ensures that position i is allowed to attend the unmasked
          positions. If a ByteTensor is provided, the non-zero positions are not allowed to attend
          while the zero positions will be unchanged. If a BoolTensor is provided, positions with ``True``
          are not allowed to attend while ``False`` values will be unchanged. If a FloatTensor
          is provided, it will be added to the attention weight.
        - static_k: :math:`(N*num_heads, S, E/num_heads)`, where S is the source sequence length,
          N is the batch size, E is the embedding dimension. E/num_heads is the head dimension.
        - static_v: :math:`(N*num_heads, S, E/num_heads)`, where S is the source sequence length,
          N is the batch size, E is the embedding dimension. E/num_heads is the head dimension.
        Outputs:
        - attn_output: :math:`(L, N, E)` where L is the target sequence length, N is the batch size,
          E is the embedding dimension.
        - attn_output_weights: :math:`(N, L, S)` where N is the batch size,
          L is the target sequence length, S is the source sequence length.
    """

    tgt_len, bsz, embed_dim = query.size()
    assert embed_dim == embed_dim_to_check
    # allow MHA to have different sizes for the feature dimension
    assert key.size(0) == value.size(0) and key.size(1) == value.size(1)

    head_dim = embed_dim // num_heads
    assert head_dim * num_heads == embed_dim, "embed_dim must be divisible by num_heads"
    scaling = float(head_dim) ** -0.5

    if not use_separate_proj_weight:
        if torch.equal(query, key) and torch.equal(key, value):
            # self-attention
            qkv = linear(query, in_proj_weight, in_proj_bias)
            qkv += linear(linear(query, in_proj_weight_A), in_proj_weight_B) * in_proj_weight_scaling
            q, k, v = qkv.chunk(3, dim=-1)

        elif torch.equal(key, value):
            # encoder-decoder attention
            # This is inline in_proj function with in_proj_weight and in_proj_bias
            _b = in_proj_bias
            _start = 0
            _end = embed_dim
            _w = in_proj_weight[_start:_end, :]
            if _b is not None:
                _b = _b[_start:_end]
            q = linear(query, _w, _b)
            q += linear(linear(query, in_proj_weight_A), in_proj_weight_B[_start:_end, :]) * in_proj_weight_scaling

            if key is None:
                assert value is None
                k = None
                v = None
            else:

                # This is inline in_proj function with in_proj_weight and in_proj_bias
                _b = in_proj_bias
                _start = embed_dim
                _end = None
                _w = in_proj_weight[_start:, :]
                if _b is not None:
                    _b = _b[_start:]
                kv = linear(key, _w, _b)
                kv += linear(linear(key, in_proj_weight_A), in_proj_weight_B[_start:, :]) * in_proj_weight_scaling
                k, v = kv.chunk(2, dim=-1)

        else:
            # This is inline in_proj function with in_proj_weight and in_proj_bias
            _b = in_proj_bias
            _start = 0
            _end = embed_dim
            _w = in_proj_weight[_start:_end, :]
            if _b is not None:
                _b = _b[_start:_end]
            q = linear(query, _w, _b)
            q += linear(linear(query, in_proj_weight_A), in_proj_weight_B[_start:_end, :]) * in_proj_weight_scaling

            # This is inline in_proj function with in_proj_weight and in_proj_bias
            _b = in_proj_bias
            _start = embed_dim
            _end = embed_dim * 2
            _w = in_proj_weight[_start:_end, :]
            if _b is not None:
                _b = _b[_start:_end]
            k = linear(key, _w, _b)
            k += linear(linear(key, in_proj_weight_A), in_proj_weight_B[_start:_end, :]) * in_proj_weight_scaling

            # This is inline in_proj function with in_proj_weight and in_proj_bias
            _b = in_proj_bias
            _start = embed_dim * 2
            _end = None
            _w = in_proj_weight[_start:, :]
            if _b is not None:
                _b = _b[_start:]
            v = linear(value, _w, _b)
            v += linear(linear(value, in_proj_weight_A), in_proj_weight_B[_start:, :]) * in_proj_weight_scaling
    else:
        if only_kv and not mlp:
            q_proj_weight_non_opt = torch.jit._unwrap_optional(q_proj_weight)
            len1, len2 = q_proj_weight_non_opt.size()
            assert len1 == embed_dim and len2 == query.size(-1)

            k_proj_weight_non_opt = torch.jit._unwrap_optional(k_proj_weight)
            len1, len2 = k_proj_weight_non_opt.size()
            assert len1 == embed_dim and len2 == key.size(-1)

            v_proj_weight_non_opt = torch.jit._unwrap_optional(v_proj_weight)
            len1, len2 = v_proj_weight_non_opt.size()
            assert len1 == embed_dim and len2 == value.size(-1)

            # q_proj_weight_non_opt_A = torch.jit._unwrap_optional(q_proj_weight_A)
            k_proj_weight_non_opt_A = torch.jit._unwrap_optional(k_proj_weight_A)
            v_proj_weight_non_opt_A = torch.jit._unwrap_optional(v_proj_weight_A)
            # q_proj_weight_non_opt_B = torch.jit._unwrap_optional(q_proj_weight_B)
            k_proj_weight_non_opt_B = torch.jit._unwrap_optional(k_proj_weight_B)
            v_proj_weight_non_opt_B = torch.jit._unwrap_optional(v_proj_weight_B)

            if in_proj_bias is not None:
                # pdb.set_trace()
                q = linear(query, q_proj_weight_non_opt, in_proj_bias[0:embed_dim])
                k = linear(key, k_proj_weight_non_opt, in_proj_bias[embed_dim:(embed_dim * 2)])
                v = linear(value, v_proj_weight_non_opt, in_proj_bias[(embed_dim * 2):])
            else:
                q = linear(query, q_proj_weight_non_opt, in_proj_bias)
                k = linear(key, k_proj_weight_non_opt, in_proj_bias)
                v = linear(value, v_proj_weight_non_opt, in_proj_bias)

            # q += linear(linear(query, q_proj_weight_non_opt_A), q_proj_weight_non_opt_B) * q_proj_weight_scaling
            k += linear(linear(key, k_proj_weight_non_opt_A), k_proj_weight_non_opt_B) * k_proj_weight_scaling
            v += linear(linear(value, v_proj_weight_non_opt_A), v_proj_weight_non_opt_B) * v_proj_weight_scaling

            pass
        elif mlp:
            q_proj_weight_non_opt = torch.jit._unwrap_optional(q_proj_weight)
            len1, len2 = q_proj_weight_non_opt.size()
            assert len1 == embed_dim and len2 == query.size(-1)

            k_proj_weight_non_opt = torch.jit._unwrap_optional(k_proj_weight)
            len1, len2 = k_proj_weight_non_opt.size()
            assert len1 == embed_dim and len2 == key.size(-1)

            v_proj_weight_non_opt = torch.jit._unwrap_optional(v_proj_weight)
            len1, len2 = v_proj_weight_non_opt.size()
            assert len1 == embed_dim and len2 == value.size(-1)

            # q_proj_weight_non_opt_A = torch.jit._unwrap_optional(q_proj_weight_A)
            # k_proj_weight_non_opt_A = torch.jit._unwrap_optional(k_proj_weight_A)
            # v_proj_weight_non_opt_A = torch.jit._unwrap_optional(v_proj_weight_A)
            # q_proj_weight_non_opt_B = torch.jit._unwrap_optional(q_proj_weight_B)
            # k_proj_weight_non_opt_B = torch.jit._unwrap_optional(k_proj_weight_B)
            # v_proj_weight_non_opt_B = torch.jit._unwrap_optional(v_proj_weight_B)

            if in_proj_bias is not None:
                # pdb.set_trace()
                q = linear(query, q_proj_weight_non_opt, in_proj_bias[0:embed_dim])
                k = linear(key, k_proj_weight_non_opt, in_proj_bias[embed_dim:(embed_dim * 2)])
                v = linear(value, v_proj_weight_non_opt, in_proj_bias[(embed_dim * 2):])
            else:
                q = linear(query, q_proj_weight_non_opt, in_proj_bias)
                k = linear(key, k_proj_weight_non_opt, in_proj_bias)
                v = linear(value, v_proj_weight_non_opt, in_proj_bias)

            # q += linear(linear(query, q_proj_weight_non_opt_A), q_proj_weight_non_opt_B) * q_proj_weight_scaling
            # k += linear(linear(key, k_proj_weight_non_opt_A), k_proj_weight_non_opt_B) * k_proj_weight_scaling
            # v += linear(linear(value, v_proj_weight_non_opt_A), v_proj_weight_non_opt_B) * v_proj_weight_scaling

        else:
            q_proj_weight_non_opt = torch.jit._unwrap_optional(q_proj_weight)
            len1, len2 = q_proj_weight_non_opt.size()
            assert len1 == embed_dim and len2 == query.size(-1)

            k_proj_weight_non_opt = torch.jit._unwrap_optional(k_proj_weight)
            len1, len2 = k_proj_weight_non_opt.size()
            assert len1 == embed_dim and len2 == key.size(-1)

            v_proj_weight_non_opt = torch.jit._unwrap_optional(v_proj_weight)
            len1, len2 = v_proj_weight_non_opt.size()
            assert len1 == embed_dim and len2 == value.size(-1)

            q_proj_weight_non_opt_A = torch.jit._unwrap_optional(q_proj_weight_A)
            k_proj_weight_non_opt_A = torch.jit._unwrap_optional(k_proj_weight_A)
            v_proj_weight_non_opt_A = torch.jit._unwrap_optional(v_proj_weight_A)
            q_proj_weight_non_opt_B = torch.jit._unwrap_optional(q_proj_weight_B)
            k_proj_weight_non_opt_B = torch.jit._unwrap_optional(k_proj_weight_B)
            v_proj_weight_non_opt_B = torch.jit._unwrap_optional(v_proj_weight_B)

            if in_proj_bias is not None:
                q = linear(query, q_proj_weight_non_opt, in_proj_bias[0:embed_dim])
                k = linear(key, k_proj_weight_non_opt, in_proj_bias[embed_dim:(embed_dim * 2)])
                v = linear(value, v_proj_weight_non_opt, in_proj_bias[(embed_dim * 2):])
            else:
                q = linear(query, q_proj_weight_non_opt, in_proj_bias)
                k = linear(key, k_proj_weight_non_opt, in_proj_bias)
                v = linear(value, v_proj_weight_non_opt, in_proj_bias)

            q += linear(linear(query, q_proj_weight_non_opt_A), q_proj_weight_non_opt_B) * q_proj_weight_scaling
            k += linear(linear(key, k_proj_weight_non_opt_A), k_proj_weight_non_opt_B) * k_proj_weight_scaling
            v += linear(linear(value, v_proj_weight_non_opt_A), v_proj_weight_non_opt_B) * v_proj_weight_scaling

    q = q * scaling

    if attn_mask is not None:
        assert attn_mask.dtype == torch.float32 or attn_mask.dtype == torch.float64 or \
            attn_mask.dtype == torch.float16 or attn_mask.dtype == torch.uint8 or attn_mask.dtype == torch.bool, \
            'Only float, byte, and bool types are supported for attn_mask, not {}'.format(attn_mask.dtype)
        if attn_mask.dtype == torch.uint8:
            warnings.warn("Byte tensor for attn_mask in nn.MultiheadAttention is deprecated. Use bool tensor instead.")
            attn_mask = attn_mask.to(torch.bool)

        if attn_mask.dim() == 2:
            attn_mask = attn_mask.unsqueeze(0)
            if list(attn_mask.size()) != [1, query.size(0), key.size(0)]:
                raise RuntimeError('The size of the 2D attn_mask is not correct.')
        elif attn_mask.dim() == 3:
            if list(attn_mask.size()) != [bsz * num_heads, query.size(0), key.size(0)]:
                raise RuntimeError('The size of the 3D attn_mask is not correct.')
        else:
            raise RuntimeError("attn_mask's dimension {} is not supported".format(attn_mask.dim()))
        # attn_mask's dim is 3 now.

    # convert ByteTensor key_padding_mask to bool
    if key_padding_mask is not None and key_padding_mask.dtype == torch.uint8:
        warnings.warn("Byte tensor for key_padding_mask in nn.MultiheadAttention is deprecated. Use bool tensor instead.")
        key_padding_mask = key_padding_mask.to(torch.bool)

    if bias_k is not None and bias_v is not None:
        if static_k is None and static_v is None:
            k = torch.cat([k, bias_k.repeat(1, bsz, 1)])
            v = torch.cat([v, bias_v.repeat(1, bsz, 1)])
            if attn_mask is not None:
                attn_mask = pad(attn_mask, (0, 1))
            if key_padding_mask is not None:
                key_padding_mask = pad(key_padding_mask, (0, 1))
        else:
            assert static_k is None, "bias cannot be added to static key."
            assert static_v is None, "bias cannot be added to static value."
    else:
        assert bias_k is None
        assert bias_v is None

    q = q.contiguous().view(tgt_len, bsz * num_heads, head_dim).transpose(0, 1)
    if k is not None:
        k = k.contiguous().view(-1, bsz * num_heads, head_dim).transpose(0, 1)
    if v is not None:
        v = v.contiguous().view(-1, bsz * num_heads, head_dim).transpose(0, 1)

    if static_k is not None:
        assert static_k.size(0) == bsz * num_heads
        assert static_k.size(2) == head_dim
        k = static_k

    if static_v is not None:
        assert static_v.size(0) == bsz * num_heads
        assert static_v.size(2) == head_dim
        v = static_v

    src_len = k.size(1)

    if key_padding_mask is not None:
        assert key_padding_mask.size(0) == bsz
        assert key_padding_mask.size(1) == src_len

    if add_zero_attn:
        src_len += 1
        k = torch.cat([k, torch.zeros((k.size(0), 1) + k.size()[2:], dtype=k.dtype, device=k.device)], dim=1)
        v = torch.cat([v, torch.zeros((v.size(0), 1) + v.size()[2:], dtype=v.dtype, device=v.device)], dim=1)
        if attn_mask is not None:
            attn_mask = pad(attn_mask, (0, 1))
        if key_padding_mask is not None:
            key_padding_mask = pad(key_padding_mask, (0, 1))

    attn_output_weights = torch.bmm(q, k.transpose(1, 2))
    assert list(attn_output_weights.size()) == [bsz * num_heads, tgt_len, src_len]

    if attn_mask is not None:
        if attn_mask.dtype == torch.bool:
            attn_output_weights.masked_fill_(attn_mask, float('-inf'))
        else:
            attn_output_weights += attn_mask


    if key_padding_mask is not None:
        attn_output_weights = attn_output_weights.view(bsz, num_heads, tgt_len, src_len)
        attn_output_weights = attn_output_weights.masked_fill(
            key_padding_mask.unsqueeze(1).unsqueeze(2),
            float('-inf'),
        )
        attn_output_weights = attn_output_weights.view(bsz * num_heads, tgt_len, src_len)

    attn_output_weights = softmax(
        attn_output_weights, dim=-1)
    attn_output_weights = dropout(attn_output_weights, p=dropout_p, training=training)

    attn_output = torch.bmm(attn_output_weights, v)
    assert list(attn_output.size()) == [bsz * num_heads, tgt_len, head_dim]
    attn_output_ = attn_output.transpose(0, 1).contiguous().view(tgt_len, bsz, embed_dim)
    attn_output = linear(attn_output_, out_proj_weight, out_proj_bias)
    if out_proj_weight_scaling is not None:
        attn_output += linear(linear(attn_output_, out_proj_weight_A), out_proj_weight_B) * out_proj_weight_scaling

    if need_weights:
        # average attention weights over heads
        attn_output_weights = attn_output_weights.view(bsz, num_heads, tgt_len, src_len)
        return attn_output, attn_output_weights.sum(dim=1) / num_heads
    else:
        return attn_output, None
