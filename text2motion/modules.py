# Copyright 2023 The HuggingFace Team. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,

# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple, Union

import torch
import math
from torch import nn
from torch.nn import functional as F

from diffusers.configuration_utils import ConfigMixin, register_to_config
from diffusers.utils import BaseOutput, logging
from diffusers.models.embeddings import TimestepEmbedding, Timesteps
from diffusers.models.modeling_utils import ModelMixin
from diffusers.models.resnet import Downsample2D, ResnetBlock2D
from diffusers.models.unets.unet_2d_blocks import UNetMidBlock2D
from diffusers.models.upsampling import Upsample2D
from diffusers.utils import deprecate, is_torch_version
from einops import rearrange
from diffusers.models.attention import Attention,FeedForward
from diffusers.models.embeddings import TimestepEmbedding, Timesteps, get_3d_sincos_pos_embed
from timm.models.layers import Mlp
from transformers import CLIPTokenizer, CLIPTextModel, CLIPImageProcessor

try:
    import xformers
    import xformers.ops
except:
    XFORMERS_IS_AVAILBLE = False


logger = logging.get_logger(__name__)  # pylint: disable=invalid-name

    
class BasicTransformerBlock(nn.Module):
    r"""
    Parameters:
        dim (`int`): The number of channels in the input and output.
        num_attention_heads (`int`): The number of heads to use for multi-head attention.
        attention_head_dim (`int`): The number of channels in each head.
        dropout (`float`, *optional*, defaults to 0.0): The dropout probability to use.
        activation_fn (`str`, *optional*, defaults to `"geglu"`): Activation function to be used in feed-forward.
        attention_bias (:
            obj: `bool`, *optional*, defaults to `False`): Configure if the attentions should contain a bias parameter.
        qk_norm (`bool`, defaults to `True`):
            Whether or not to use normalization after query and key projections in Attention.
        norm_elementwise_affine (`bool`, defaults to `True`):
            Whether to use learnable elementwise affine parameters for normalization.
        norm_eps (`float`, defaults to `1e-5`):
            Epsilon value for normalization layers.
        final_dropout (`bool` defaults to `False`):
            Whether to apply a final dropout after the last feed-forward layer.
        ff_inner_dim (`int`, *optional*, defaults to `None`):
            Custom hidden dimension of Feed-forward layer. If not provided, `4 * dim` is used.
        ff_bias (`bool`, defaults to `True`):
            Whether or not to use bias in Feed-forward layer.
        attention_out_bias (`bool`, defaults to `True`):
            Whether or not to use bias in Attention output projection layer.
    """

    def __init__(
        self,
        dim: int,
        num_attention_heads: int,
        attention_head_dim: int,
        time_embed_dim: Optional[int] = None,
        dropout: float = 0.0,
        activation_fn: str = "gelu-approximate",
        attention_bias: bool = False,
        qk_norm: bool = True,
        norm_elementwise_affine: bool = True,
        norm_eps: float = 1e-5,
        final_dropout: bool = True,
        ff_inner_dim: Optional[int] = None,
        ff_bias: bool = True,
        attention_out_bias: bool = True,
    ):
        super().__init__()

        # 1. Self Attention
        self.norm1 = nn.LayerNorm(dim, eps=norm_eps, elementwise_affine=norm_elementwise_affine)
        
        self.attn1 = Attention(
            query_dim=dim,
            dim_head=attention_head_dim,
            heads=num_attention_heads,
            qk_norm="layer_norm" if qk_norm else None,
            eps=1e-6,
            bias=attention_bias,
            out_bias=attention_out_bias,
        )

        # 2. Feed Forward
        self.norm2 = nn.LayerNorm(dim, eps=norm_eps, elementwise_affine=norm_elementwise_affine)

        self.ff = FeedForward(
            dim,
            dropout=dropout,
            activation_fn=activation_fn,
            final_dropout=final_dropout,
            inner_dim=ff_inner_dim,
            bias=ff_bias,
        )

    def forward(
        self,
        hidden_states: torch.Tensor,
    ) -> torch.Tensor:
        
        # norm1
        norm_hidden_states = self.norm1(hidden_states)

        # attention
        attn_output = self.attn1(
            hidden_states=norm_hidden_states,
            encoder_hidden_states=None,
        )
        hidden_states = hidden_states + attn_output

        # norm & modulate
        norm_hidden_states = self.norm2(hidden_states)

        # feed-forward
        ff_output = self.ff(norm_hidden_states)
        hidden_states = hidden_states + ff_output
        
        return hidden_states
    


# patch embed without positional encoding
class PatchEmbed(nn.Module):
    def __init__(
        self,
        patch_size: int = 2,
        in_channels: int = 16,
        embed_dim: int = 1920,
        bias: bool = True,
    ) -> None:
        super().__init__()
        self.patch_size = patch_size

        self.proj = nn.Conv2d(
            in_channels, embed_dim, kernel_size=(patch_size, patch_size), stride=patch_size, bias=bias
        )

    def forward(self, image_embeds: torch.Tensor):
        r"""
        Args:
            image_embeds (`torch.Tensor`):
                Input image embeddings. Expected shape: (batch_size, num_frames, channels, height, width) or (batch_size, channels, height, width)
        Returns:
            embeds (`torch.Tensor`):
                (batch_size,num_frames x height x width,embed_dim) or (batch_size,1 x height x width,embed_dim)
        """
        if image_embeds.dim() == 5:
            batch, num_frames, channels, height, width = image_embeds.shape
            image_embeds = image_embeds.reshape(-1, channels, height, width)
        else:
            batch, channels, height, width = image_embeds.shape
            num_frames = 1
            
        image_embeds = self.proj(image_embeds)
        image_embeds = image_embeds.view(batch, num_frames, *image_embeds.shape[1:])
        image_embeds = image_embeds.flatten(3).transpose(2, 3)  # [batch, num_frames, height x width, channels]
        image_embeds = image_embeds.flatten(1, 2)  # [batch, num_frames x height x width, channels]

        return image_embeds # [batch, num_frames x height x width, channels]
def modulate(x, shift, scale):
    return x * (1 + scale.unsqueeze(1)) + shift.unsqueeze(1) 
# class Attention(nn.Module):
#     def __init__(self, dim, num_heads=8, qkv_bias=False, attn_drop=0., proj_drop=0., use_lora=False, attention_mode='math'):
#         super().__init__()
#         assert dim % num_heads == 0, 'dim should be divisible by num_heads'
#         self.num_heads = num_heads
#         head_dim = dim // num_heads
#         self.scale = head_dim ** -0.5
#         self.attention_mode = attention_mode
#         self.qkv = nn.Linear(dim, dim * 3, bias=qkv_bias)
#         self.attn_drop = nn.Dropout(attn_drop)
#         self.proj = nn.Linear(dim, dim)
#         self.proj_drop = nn.Dropout(proj_drop)

#     def forward(self, x):
#         B, N, C = x.shape
#         qkv = self.qkv(x).reshape(B, N, 3, self.num_heads, C // self.num_heads).permute(2, 0, 3, 1, 4).contiguous()
#         q, k, v = qkv.unbind(0)   # make torchscript happy (cannot use tensor as tuple)
        
#         if self.attention_mode == 'xformers': # cause loss nan while using with amp
#             # https://github.com/facebookresearch/xformers/blob/e8bd8f932c2f48e3a3171d06749eecbbf1de420c/xformers/ops/fmha/__init__.py#L135
#             q_xf = q.transpose(1,2).contiguous()
#             k_xf = k.transpose(1,2).contiguous()
#             v_xf = v.transpose(1,2).contiguous()
#             x = xformers.ops.memory_efficient_attention(q_xf, k_xf, v_xf).reshape(B, N, C)

#         elif self.attention_mode == 'flash':
#             # cause loss nan while using with amp
#             # Optionally use the context manager to ensure one of the fused kerenels is run
#             with torch.backends.cuda.sdp_kernel(enable_math=False):
#                 x = torch.nn.functional.scaled_dot_product_attention(q, k, v).reshape(B, N, C) # require pytorch 2.0

#         elif self.attention_mode == 'math':
#             attn = (q @ k.transpose(-2, -1)) * self.scale
#             attn = attn.softmax(dim=-1)
#             attn = self.attn_drop(attn)
#             x = (attn @ v).transpose(1, 2).reshape(B, N, C)

#         else:
#             raise NotImplemented

#         x = self.proj(x)
#         x = self.proj_drop(x)
#         return x
class TransformerBlock(nn.Module):
    """
    A Latte block with adaptive layer norm zero (adaLN-Zero) conditioning.
    """
    def __init__(self, 
                head_dim:int,
                num_heads:int,
                mlp_ratio=4.0,
                **block_kwargs):
        super().__init__()
        hidden_size =  head_dim * num_heads
        self.norm1 = nn.LayerNorm(hidden_size, elementwise_affine=False, eps=1e-6)

        self.attn = Attention(
            query_dim=num_heads * head_dim,
            dim_head=head_dim,
            heads=num_heads,
            eps=1e-6,
            **block_kwargs
        )
        self.norm2 = nn.LayerNorm(hidden_size, elementwise_affine=False, eps=1e-6)
        mlp_hidden_dim = int(hidden_size * mlp_ratio)
        approx_gelu = lambda: nn.GELU(approximate="tanh")
        self.mlp = Mlp(in_features=hidden_size, hidden_features=mlp_hidden_dim, act_layer=approx_gelu, drop=0)
        self.adaLN_modulation = nn.Sequential(
            nn.SiLU(),
            nn.Linear(hidden_size, 6 * hidden_size, bias=True)
        )

    def forward(self, x, c):
        shift_msa, scale_msa, gate_msa, shift_mlp, scale_mlp, gate_mlp = self.adaLN_modulation(c).chunk(6, dim=1)
        x = x + gate_msa.unsqueeze(1) * self.attn(modulate(self.norm1(x), shift_msa, scale_msa))
        x = x + gate_mlp.unsqueeze(1) * self.mlp(modulate(self.norm2(x), shift_mlp, scale_mlp))
        return x  
class LabelEncoder(nn.Module):
    def __init__(self,num_labels:int,out_dim:int,emb_dim:int,drop_ratio:float=0.0,act_fn=nn.GELU):
        super().__init__()
        self.embed = nn.Embedding(num_labels,emb_dim)
        self.act1 = act_fn()
        self.dropout = nn.Dropout(drop_ratio) if drop_ratio > 0 else nn.Identity()
        self.proj = nn.Linear(emb_dim,out_dim)
        self.act2 = act_fn()
    def forward(self,label_id:int):
        x = self.dropout(self.act1(self.embed(label_id)))
        x = self.act2(self.proj(x))
        return x
class TextEncoder(nn.Module):
    """Uses the CLIP transformer encoder for text (from Hugging Face)"""
    def __init__(self, path, device="cuda", max_length=77):
        super().__init__()
        self.tokenizer = CLIPTokenizer.from_pretrained(path)
        self.transformer = CLIPTextModel.from_pretrained(path)
        self.device = device
        self.max_length = max_length
        self.freeze()

    def freeze(self):
        self.transformer = self.transformer.eval()
        for param in self.parameters():
            param.requires_grad = False

    def forward(self, text):
        batch_encoding = self.tokenizer(text, truncation=True, max_length=self.max_length, return_length=True,
                                        return_overflowing_tokens=False, padding="max_length", return_tensors="pt")
        tokens = batch_encoding["input_ids"].to(self.device)
        outputs = self.transformer(input_ids=tokens)

        z = outputs.last_hidden_state
        pooled_z = outputs.pooler_output
        return z, pooled_z

    def encode(self, text):
        return self(text)