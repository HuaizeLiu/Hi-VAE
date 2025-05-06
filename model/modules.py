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
import einops
from timm.models.layers import Mlp

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
from diffusers.models.embeddings import get_2d_sincos_pos_embed,get_1d_sincos_pos_embed_from_grid,get_3d_sincos_pos_embed,TimestepEmbedding, Timesteps

VIS_ATTEN_FLAG = True
attention_maps = []
def get_attention_maps():
    global attention_maps
    return attention_maps
def clear_attention_maps():
    global attention_maps
    attention_maps.clear()
def set_vis_atten_flag(flag):
    global VIS_ATTEN_FLAG
    VIS_ATTEN_FLAG = flag


logger = logging.get_logger(__name__)  # pylint: disable=invalid-name

# ----------------------- A2M Type1  Predict Pose ------------------------
class DownEncoderBlock2D(nn.Module):
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        dropout: float = 0.0,
        num_layers: int = 1,
        resnet_eps: float = 1e-6,
        resnet_time_scale_shift: str = "default",
        resnet_act_fn: str = "swish",
        resnet_groups: int = 32,
        resnet_pre_norm: bool = True,
        output_scale_factor: float = 1.0,
        add_downsample: bool = True,
        downsample_padding: int = 1,
    ):
        super().__init__()
        resnets = []

        for i in range(num_layers):
            in_channels = in_channels if i == 0 else out_channels
            resnets.append(
                ResnetBlock2D(
                    in_channels=in_channels,
                    out_channels=out_channels,
                    temb_channels=None,
                    eps=resnet_eps,
                    groups=resnet_groups,
                    dropout=dropout,
                    time_embedding_norm=resnet_time_scale_shift,
                    non_linearity=resnet_act_fn,
                    output_scale_factor=output_scale_factor,
                    pre_norm=resnet_pre_norm,
                )
            )

        self.resnets = nn.ModuleList(resnets)

        if add_downsample:
            self.downsamplers = nn.ModuleList(
                [
                    Downsample2D(
                        out_channels, use_conv=True, out_channels=out_channels, padding=downsample_padding, name="op"
                    )
                ]
            )
        else:
            self.downsamplers = None

    def forward(self, hidden_states: torch.Tensor, *args, **kwargs) -> torch.Tensor:
        if len(args) > 0 or kwargs.get("scale", None) is not None:
            deprecation_message = "The `scale` argument is deprecated and will be ignored. Please remove it, as passing it will raise an error in the future. `scale` should directly be passed while calling the underlying pipeline component i.e., via `cross_attention_kwargs`."
            deprecate("scale", "1.0.0", deprecation_message)

        for resnet in self.resnets:
            hidden_states = resnet(hidden_states, temb=None)

        if self.downsamplers is not None:
            for downsampler in self.downsamplers:
                hidden_states = downsampler(hidden_states)

        return hidden_states

class MidBlock2D(nn.Module):
    def __init__(
        self,
        in_channel: int = 64,
        out_channel: int = 1280,
    ):
        super().__init__()
        
        self.mid_convs = nn.ModuleList()
        self.mid_convs.append(nn.Sequential(
            nn.Conv2d(
                in_channels=in_channel,
                out_channels=in_channel,
                kernel_size=3,
                stride=1,
                padding=1
            ),
            nn.ReLU(),
            nn.Conv2d(
                in_channels=in_channel,
                out_channels=in_channel,
                kernel_size=3,
                stride=1,
                padding=1
            ),
        ))
        self.mid_convs.append(nn.Conv2d(
            in_channels=in_channel,
            out_channels=out_channel,
            kernel_size=1,
            stride=1,
        ))
    
    def forward(self, x):
        for mid_conv in self.mid_convs:
            sample = mid_conv(x)
        
        return sample
        
class UpDecoderBlock2D(nn.Module):
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        resolution_idx: Optional[int] = None,
        dropout: float = 0.0,
        num_layers: int = 1,
        resnet_eps: float = 1e-6,
        resnet_time_scale_shift: str = "default",  # default, spatial
        resnet_act_fn: str = "swish",
        resnet_groups: int = 32,
        resnet_pre_norm: bool = True,
        output_scale_factor: float = 1.0,
        add_upsample: bool = True,
        temb_channels: Optional[int] = None,
    ):
        super().__init__()
        resnets = []

        for i in range(num_layers):
            input_channels = in_channels if i == 0 else out_channels
            resnets.append(
                ResnetBlock2D(
                    in_channels=input_channels,
                    out_channels=out_channels,
                    temb_channels=temb_channels,
                    eps=resnet_eps,
                    groups=resnet_groups,
                    dropout=dropout,
                    time_embedding_norm=resnet_time_scale_shift,
                    non_linearity=resnet_act_fn,
                    output_scale_factor=output_scale_factor,
                    pre_norm=resnet_pre_norm,
                )
            )

        self.resnets = nn.ModuleList(resnets)

        if add_upsample:
            self.upsamplers = nn.ModuleList([Upsample2D(out_channels, use_conv=True, out_channels=out_channels)])
        else:
            self.upsamplers = None

        self.resolution_idx = resolution_idx

    def forward(self, hidden_states: torch.Tensor, temb: Optional[torch.Tensor] = None) -> torch.Tensor:
        for resnet in self.resnets:
            hidden_states = resnet(hidden_states, temb=temb)

        if self.upsamplers is not None:
            for upsampler in self.upsamplers:
                hidden_states = upsampler(hidden_states)

        return hidden_states

class DuoFrameDownEncoder(ModelMixin, ConfigMixin):

    _supports_gradient_checkpointing = True

    @register_to_config
    def __init__(
        self,
        in_channel: int = 4,
        block_out_channels : Tuple[int] = (64, 128, 256, 256),
        norm_groups : int = 4,
        resnet_layers_per_block: int = 2,
        add_attention : bool = True,
    ):
        super().__init__()

        # conv_in 
        self.conv_in = nn.Conv2d(   
                            in_channel,
                            block_out_channels[0],
                            kernel_size=3,
                            stride=1,
                            padding=1,
                        )
        
        # downblock
        self.downblock = nn.ModuleList()
        
        output_channel = block_out_channels[0]
        for i,channels in enumerate(block_out_channels):
            
            input_channel = output_channel
            output_channel = block_out_channels[i]
            is_final_block = i == len(block_out_channels) - 1
            
            self.downblock.append(
                DownEncoderBlock2D(
                    in_channels=input_channel,
                    out_channels=output_channel,
                    num_layers= resnet_layers_per_block,
                    resnet_groups = norm_groups,
                    add_downsample=not is_final_block,
                )
            )

        # mid_block
        self.mid_block = UNetMidBlock2D(
            in_channels=block_out_channels[-1],
            resnet_eps=1e-6,
            output_scale_factor=1,
            resnet_time_scale_shift="default",
            attention_head_dim=block_out_channels[-1],
            resnet_groups=norm_groups,
            temb_channels=None,
            add_attention=add_attention,
        )

        # conv_out
        self.conv_norm_out = nn.GroupNorm(num_channels=block_out_channels[-1], num_groups=norm_groups, eps=1e-6)
        self.conv_act = nn.SiLU()
        self.conv_out = nn.Conv2d(block_out_channels[-1], block_out_channels[-1], 3, padding=1)


    def _set_gradient_checkpointing(self, module, value=False):
        if hasattr(module, "gradient_checkpointing"):
            module.gradient_checkpointing = value

    def forward(self, x: torch.FloatTensor)  -> torch.Tensor:
        """
        Args:
            * x : (b,c,h,w)
        Output:
            * x : (b,c',h/8,w/8)
        """
        
        # conv_in
        x = self.conv_in(x)
        
        # downblock
        for downblock in self.downblock:
            x = downblock(x)
        
        # mid
        x = self.mid_block(x)

        # out
        x = self.conv_norm_out(x)
        x = self.conv_act(x)
        x = self.conv_out(x)
        
        return x

class MotionDownEncoder(ModelMixin, ConfigMixin):

    _supports_gradient_checkpointing = True

    @register_to_config
    def __init__(
        self,
        in_channel: int = 4,
        block_out_channels : Tuple[int] = (64, 128, 256, 256),
        norm_groups : int = 32,
        resnet_layers_per_block: int = 2,
        add_attention : bool = True,
    ):
        super().__init__()

        # conv_in 
        self.conv_in = nn.Conv2d(   
                            in_channel,
                            block_out_channels[0],
                            kernel_size=1,
                            stride=1,
                        )
        
        # downblock
        self.downblock = nn.ModuleList()
        
        output_channel = block_out_channels[0]
        for i,channels in enumerate(block_out_channels):
            
            input_channel = output_channel
            output_channel = block_out_channels[i]
            is_final_block = i == len(block_out_channels) - 1
            
            self.downblock.append(
                DownEncoderBlock2D(
                    in_channels=input_channel,
                    out_channels=output_channel,
                    num_layers= resnet_layers_per_block,
                    resnet_groups = norm_groups,
                    add_downsample=not is_final_block,
                )
            )

        # mid_block
        self.mid_block = UNetMidBlock2D(
            in_channels=block_out_channels[-1],
            resnet_eps=1e-6,
            output_scale_factor=1,
            resnet_time_scale_shift="default",
            attention_head_dim=block_out_channels[-1],
            resnet_groups=norm_groups,
            temb_channels=None,
            add_attention=add_attention,
        )

        # conv_out
        self.conv_norm_out = nn.GroupNorm(num_channels=block_out_channels[-1], num_groups=norm_groups, eps=1e-6)
        self.conv_act = nn.SiLU()
        self.conv_out = nn.Conv2d(block_out_channels[-1], block_out_channels[-1], 3, padding=1)


    def _set_gradient_checkpointing(self, module, value=False):
        if hasattr(module, "gradient_checkpointing"):
            module.gradient_checkpointing = value

    def forward(self, x: torch.FloatTensor)  -> torch.Tensor:
        """
        Args:
            * x : (b,c,h,w)
        Output:
            * x : (b,c',h/8,w/8)
        """
        
        # conv_in
        x = self.conv_in(x)
        
        # downblock
        for downblock in self.downblock:
            x = downblock(x)
        
        # mid
        x = self.mid_block(x)

        # out
        x = self.conv_norm_out(x)
        x = self.conv_act(x)
        x = self.conv_out(x)
        
        return x

class DownEncoder(ModelMixin, ConfigMixin):

    _supports_gradient_checkpointing = True

    @register_to_config
    def __init__(
        self,
        in_channel: int = 4,
        block_out_channels : Tuple[int] = (64, 128, 256, 256),
        norm_groups : int = 8,
        resnet_layers_per_block: int = 2,
        add_attention : bool = True,
    ):
        super().__init__()

        # conv_in 
        self.conv_in = nn.Conv2d(   
                            in_channel,
                            block_out_channels[0],
                            kernel_size=1,
                            stride=1,
                        )
        
        # downblock
        self.downblock = nn.ModuleList()
        
        output_channel = block_out_channels[0]
        for i,channels in enumerate(block_out_channels):
            
            input_channel = output_channel
            output_channel = block_out_channels[i]
            is_final_block = i == len(block_out_channels) - 1
            
            self.downblock.append(
                DownEncoderBlock2D(
                    in_channels=input_channel,
                    out_channels=output_channel,
                    num_layers= resnet_layers_per_block,
                    resnet_groups = norm_groups,
                    add_downsample=not is_final_block,
                )
            )

        # mid_block
        self.mid_block = UNetMidBlock2D(
            in_channels=block_out_channels[-1],
            resnet_eps=1e-6,
            output_scale_factor=1,
            resnet_time_scale_shift="default",
            attention_head_dim=block_out_channels[-1],
            resnet_groups=norm_groups,
            temb_channels=None,
            add_attention=add_attention,
        )

        # conv_out
        self.conv_norm_out = nn.GroupNorm(num_channels=block_out_channels[-1], num_groups=norm_groups, eps=1e-6)
        self.conv_act = nn.SiLU()
        self.conv_out = nn.Conv2d(block_out_channels[-1], block_out_channels[-1], 3, padding=1)


    def _set_gradient_checkpointing(self, module, value=False):
        if hasattr(module, "gradient_checkpointing"):
            module.gradient_checkpointing = value

    def forward(self, x: torch.FloatTensor)  -> torch.Tensor:
        """
        Args:
            * x : (b,c,h,w)
        Output:
            * x : (b,c',h/8,w/8)
        """
        
        # conv_in
        x = self.conv_in(x)
        
        # downblock
        for downblock in self.downblock:
            x = downblock(x)
        
        # mid
        x = self.mid_block(x)

        # out
        x = self.conv_norm_out(x)
        x = self.conv_act(x)
        x = self.conv_out(x)
        
        return x

class Upsampler(ModelMixin, ConfigMixin):

    _supports_gradient_checkpointing = True

    @register_to_config
    def __init__(
        self,
        in_channel: int = 256,
        out_channel: Optional[int] = None,
        block_out_channels : Tuple[int] = (256, 256, 128, 64),
        norm_groups : int = 8,
        resnet_layers_per_block: int = 2,
        add_attention : bool = True,
    ):
        super().__init__()

        self.out_channel = out_channel

        # conv_in 
        self.conv_in = nn.Conv2d(   
                            in_channel,
                            block_out_channels[0],
                            kernel_size=3,
                            stride=1,
                            padding=1,
                        )

        # mid_block
        self.mid_block = UNetMidBlock2D(
            in_channels=block_out_channels[0],
            resnet_eps=1e-6,
            output_scale_factor=1,
            resnet_time_scale_shift="default",
            attention_head_dim=block_out_channels[0],
            resnet_groups=norm_groups,
            temb_channels=None,
            add_attention=add_attention,
        )
        
        # upblock
        self.upblock = nn.ModuleList()
        output_channel = block_out_channels[0]
        for i,channels in enumerate(block_out_channels):
            
            input_channel = output_channel
            output_channel = block_out_channels[i]
            is_final_block = i == len(block_out_channels) - 1
            
            self.upblock.append(
                UpDecoderBlock2D(
                    in_channels=input_channel,
                    out_channels=output_channel,
                    num_layers= resnet_layers_per_block,
                    resnet_groups = norm_groups,
                    add_upsample=not is_final_block,
                )
            )

        # conv_out
        self.conv_norm_out = nn.GroupNorm(num_channels=block_out_channels[-1], num_groups=norm_groups, eps=1e-6)
        self.conv_act = nn.SiLU()
        self.conv_out = nn.Conv2d(block_out_channels[-1], block_out_channels[-1], 3, padding=1)

        # channel 
        if self.out_channel:
            self.conv_final =  nn.Conv2d(   
                                    block_out_channels[-1],
                                    out_channel,
                                    kernel_size=3,
                                    stride=1,
                                    padding=1,
                                )


    def _set_gradient_checkpointing(self, module, value=False):
        if hasattr(module, "gradient_checkpointing"):
            module.gradient_checkpointing = value

    def forward(self, x: torch.FloatTensor) -> torch.tensor:
        """
        Args:
            * x : (b,c,h,w)
        Output:
            * x : (b,c',h*8,w*8)
        """
        
        # conv_in
        x = self.conv_in(x)

        # mid
        x = self.mid_block(x)
        
        # upblock
        for upblock in self.upblock:
            x = upblock(x)

        # out
        x = self.conv_norm_out(x)
        x = self.conv_act(x)
        x = self.conv_out(x)

        # final
        if self.out_channel:
            x = self.conv_final(x)
        
        return x

# mapping for same shape & different channels
class MapConv(nn.Module):
    def __init__(self,
                 in_channel: int = 8,
                 hidden : int = 640,
                 out_channel: int = 4,
                 block_layer : int = 8,
                 goups : int = 2,):
        super().__init__()

        
        # conv_in
        self.conv_in =  nn.Conv2d(   
                            in_channel,
                            hidden,
                            kernel_size=3,
                            stride=1,
                            padding=1,
                        )

        # attn
        self.mid_block = UNetMidBlock2D(
            in_channels=hidden,
            resnet_eps=1e-6,
            output_scale_factor=1,
            resnet_time_scale_shift="default",
            attention_head_dim=64,
            resnet_groups=goups,
            temb_channels=None,
            add_attention=True,
        )

        # map
        self.map = nn.ModuleList()
        for i in range(block_layer):
            resnet = ResnetBlock2D(
                in_channels=hidden,
                out_channels=hidden,
                temb_channels=None,
                groups=goups,
            )
            self.map.append(resnet)
        
        # conv_out
        self.conv_out =  nn.Conv2d(   
                            hidden,
                            out_channel,
                            kernel_size=3,
                            stride=1,
                            padding=1,
                        )
    
    def forward(self, x: torch.tensor , temb: Optional[torch.tensor] = None) -> torch.tensor:
        
        x = self.conv_in(x)
        x = self.mid_block(x)
        for l in self.map:
            x = l(x,None)
        x = self.conv_out(x)

        return x
    
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
    
class BasicCrossTransformerBlock(nn.Module):
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
        cross_dim:int,
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
            cross_attention_dim=cross_dim,
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
        encoder_hidden_state: torch.Tensor,
    ) -> torch.Tensor:
        
        # norm1
        norm_hidden_states = self.norm1(hidden_states)

        # attention
        attn_output = self.attn1(
            hidden_states=norm_hidden_states,
            encoder_hidden_states=encoder_hidden_state,
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
    
class AMDLayerNormZero(nn.Module):
    def __init__(
        self,
        conditioning_dim: int,
        embedding_dim: int,
        elementwise_affine: bool = True,
        eps: float = 1e-5,
        bias: bool = True,
    ) -> None:
        super().__init__()

        self.embed_dim = embedding_dim
        self.silu = nn.SiLU()
        self.linear = nn.Linear(conditioning_dim, 6 * embedding_dim, bias=bias)
        self.norm = nn.LayerNorm(embedding_dim, eps=eps, elementwise_affine=elementwise_affine)

    def forward(
        self, hidden_states: torch.Tensor, encoder_hidden_states: torch.Tensor, temb: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        
        shift, scale, gate, enc_shift, enc_scale, enc_gate = self.linear(self.silu(temb)).chunk(6, dim=1)
        hidden_states = self.norm(hidden_states) * (1 + scale)[:, None, :] + shift[:, None, :]
        encoder_hidden_states = self.norm(encoder_hidden_states) * (1 + enc_scale)[:, None, :] + enc_shift[:, None, :]
        return hidden_states, encoder_hidden_states, gate[:, None, :], enc_gate[:, None, :]

class AMDLayerNormZero_OneVariable(nn.Module):
    def __init__(
        self,
        conditioning_dim: int,
        embedding_dim: int,
        elementwise_affine: bool = True,
        eps: float = 1e-5,
        bias: bool = True,
    ) -> None:
        super().__init__()

        self.embed_dim = embedding_dim
        self.silu = nn.SiLU()
        self.linear = nn.Linear(conditioning_dim, 3 * embedding_dim, bias=bias)
        self.norm = nn.LayerNorm(embedding_dim, eps=eps, elementwise_affine=elementwise_affine)

    def forward(
        self, hidden_states: torch.Tensor, temb: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        
        shift, scale, gate = self.linear(self.silu(temb)).chunk(3, dim=1)
        hidden_states = self.norm(hidden_states) * (1 + scale)[:, None, :] + shift[:, None, :]
        return hidden_states, gate[:, None, :]

class AMDLayerNormZero2Condition(nn.Module):
    def __init__(
        self,
        conditioning_dim: int,
        embedding_dim: int,
        elementwise_affine: bool = True,
        eps: float = 1e-5,
        bias: bool = True,
    ) -> None:
        super().__init__()

        self.embed_dim = embedding_dim
        self.silu = nn.SiLU()
        self.linear = nn.Linear(conditioning_dim, 9 * embedding_dim, bias=bias)
        self.norm = nn.LayerNorm(embedding_dim, eps=eps, elementwise_affine=elementwise_affine)

    def forward(
        self, hidden_states: torch.Tensor, condition_states1: torch.Tensor,condition_states2:torch.Tensor, temb: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        
        shift, scale, gate, c1_shift, c1_scale, c1_gate,c2_shift, c2_scale, c2_gate = self.linear(self.silu(temb)).chunk(9, dim=1)

        hidden_states = self.norm(hidden_states) * (1 + scale)[:, None, :] + shift[:, None, :]
        condition_states1 = self.norm(condition_states1) * (1 + c1_scale)[:, None, :] + c1_shift[:, None, :]
        condition_states2 = self.norm(condition_states2) * (1 + c2_scale)[:, None, :] + c2_shift[:, None, :]
        
        return hidden_states, condition_states1,condition_states2, gate[:, None, :], c1_gate[:, None, :],c2_gate[:, None, :]
    
class AdaLayerNorm(nn.Module):
    r"""
    Norm layer modified to incorporate timestep embeddings.

    Parameters:
        embedding_dim (`int`): The size of each embedding vector.
        num_embeddings (`int`, *optional*): The size of the embeddings dictionary.
        output_dim (`int`, *optional*):
        norm_elementwise_affine (`bool`, defaults to `False):
        norm_eps (`bool`, defaults to `False`):
        chunk_dim (`int`, defaults to `0`):
    """

    def __init__(
        self,
        embedding_dim: int,
        num_embeddings: Optional[int] = None,
        output_dim: Optional[int] = None,
        norm_elementwise_affine: bool = False,
        norm_eps: float = 1e-5,
        chunk_dim: int = 0,
    ):
        super().__init__()

        self.chunk_dim = chunk_dim
        output_dim = output_dim or embedding_dim * 2

        if num_embeddings is not None:
            self.emb = nn.Embedding(num_embeddings, embedding_dim)
        else:
            self.emb = None

        self.silu = nn.SiLU()
        self.linear = nn.Linear(embedding_dim, output_dim)
        self.norm = nn.LayerNorm(output_dim // 2, norm_eps, norm_elementwise_affine)

    def forward(
        self, x: torch.Tensor, timestep: Optional[torch.Tensor] = None, temb: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        # temb (8*15,512) x (8*15,16,256)
        if self.emb is not None:
            temb = self.emb(timestep)

        temb = self.linear(self.silu(temb)) 

        if self.chunk_dim == 1:
            # This is a bit weird why we have the order of "shift, scale" here and "scale, shift" in the
            # other if-branch. This branch is specific to CogVideoX for now.
            shift, scale = temb.chunk(2, dim=1)
            shift = shift[:, None, :]
            scale = scale[:, None, :]
        else:
            scale, shift = temb.chunk(2, dim=0)

        x = self.norm(x) * (1 + scale) + shift
        return x

    
class AMDTransformerBlock(nn.Module):
    r"""
        AMDTransformerBlock
    """

    def __init__(
        self,
        dim: int,
        num_attention_heads: int,
        attention_head_dim: int,
        time_embed_dim: int,
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
        self.norm1 = AMDLayerNormZero(time_embed_dim, dim, norm_elementwise_affine, norm_eps, bias=True)

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
        self.norm2 = AMDLayerNormZero(time_embed_dim, dim, norm_elementwise_affine, norm_eps, bias=True)

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
        temb: torch.Tensor,
        encoder_hidden_states: torch.Tensor = None,
    ) -> torch.Tensor:
        """
        *************************  ******************
        * encoder_hidden_states *   * hidden_states *
        *************************  ******************
        """
        if encoder_hidden_states == None:
            encoder_hidden_states = hidden_states

        norm_hidden_states, norm_encoder_hidden_states, gate_msa, enc_gate_msa = self.norm1(
            hidden_states, encoder_hidden_states, temb
        )

        # attention
        image_length = norm_encoder_hidden_states.shape[1]

        # AMD uses concatenated image + motion embeddings with self-attention instead of using
        # them in cross-attention individually
        norm_hidden_states = torch.cat([norm_encoder_hidden_states, norm_hidden_states], dim=1)
        attn_output = self.attn1(
            hidden_states=norm_hidden_states,
            encoder_hidden_states=None,
        )

        hidden_states = hidden_states + gate_msa * attn_output[:, image_length:]
        encoder_hidden_states = encoder_hidden_states + enc_gate_msa * attn_output[:, :image_length]

        # norm & modulate
        norm_hidden_states, norm_encoder_hidden_states, gate_ff, enc_gate_ff = self.norm2(
            hidden_states, encoder_hidden_states, temb
        )

        # feed-forward
        norm_hidden_states = torch.cat([norm_encoder_hidden_states, norm_hidden_states], dim=1)
        ff_output = self.ff(norm_hidden_states)

        hidden_states = hidden_states + gate_ff * ff_output[:, image_length:]
        encoder_hidden_states = encoder_hidden_states + enc_gate_ff * ff_output[:, :image_length]
        
        return hidden_states, encoder_hidden_states

class BasicDiTBlock(nn.Module):
    r"""
        AMDTransformerBlock
    """

    def __init__(
        self,
        dim: int,
        num_attention_heads: int,
        attention_head_dim: int,
        time_embed_dim: int,
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
        self.norm1 = AMDLayerNormZero_OneVariable(time_embed_dim, dim, norm_elementwise_affine, norm_eps, bias=True)

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
        self.norm2 = AMDLayerNormZero_OneVariable(time_embed_dim, dim, norm_elementwise_affine, norm_eps, bias=True)

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
        temb: torch.Tensor,
    ) -> torch.Tensor:


        norm_hidden_states, gate_msa = self.norm1(
            hidden_states, temb
        ) # N,F,D

        attn_output = self.attn1(
            hidden_states=norm_hidden_states,
            encoder_hidden_states=None,
        )

        hidden_states = hidden_states + gate_msa * attn_output

        # norm & modulate
        norm_hidden_states, gate_ff = self.norm2(
            hidden_states, temb
        )

        # feed-forward
        ff_output = self.ff(norm_hidden_states)

        hidden_states = hidden_states + gate_ff * ff_output
        
        return hidden_states


class AMDTransformerMotionBlock(nn.Module):
    r"""
        AMDTransformerBlock
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
        if time_embed_dim:
            self.norm1 = AMDLayerNormZero_OneVariable(time_embed_dim, dim, norm_elementwise_affine, norm_eps, bias=True)
        else:
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
        if time_embed_dim:
            self.norm2 = AMDLayerNormZero_OneVariable(time_embed_dim, dim, norm_elementwise_affine, norm_eps, bias=True)
        else:
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
        temb: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:

        if temb:
            norm_hidden_states, gate_msa = self.norm1(
                hidden_states, temb
            ) # N,F,D
        else:
            norm_hidden_states = self.norm1(hidden_states)

        attn_output = self.attn1(
            hidden_states=norm_hidden_states,
            encoder_hidden_states=None,
        )

        if temb:
            hidden_states = hidden_states + gate_msa * attn_output
        else:
            hidden_states = hidden_states + attn_output

        # norm & modulate
        if temb:
            norm_hidden_states, gate_ff = self.norm2(
                hidden_states, temb
            ) # N,F,D
        else:
            norm_hidden_states = self.norm2(hidden_states)


        # feed-forward
        ff_output = self.ff(norm_hidden_states)
        if temb:
            hidden_states = hidden_states + gate_ff * ff_output
        else:
            hidden_states = hidden_states + ff_output

        return hidden_states


class TransformerBlock2Condition(nn.Module):
    r"""
        AMDTransformerBlock
    """

    def __init__(
        self,
        dim: int,
        num_attention_heads: int,
        attention_head_dim: int,
        time_embed_dim: int,
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
        self.norm1 = AMDLayerNormZero2Condition(time_embed_dim, dim, norm_elementwise_affine, norm_eps, bias=True)

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
        self.norm2 = AMDLayerNormZero2Condition(time_embed_dim, dim, norm_elementwise_affine, norm_eps, bias=True)

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
        condition_states1: torch.Tensor,
        condition_states2: torch.Tensor,
        temb: torch.Tensor,
    ) -> torch.Tensor:
        """
        *************************  ******************   ******************** 
        *  hidden_states        *  *condition_states1*   * condition_states2* 
        *************************  ******************   ********************
        """
        hidden_length = hidden_states.shape[1]
        condition1_length = condition_states1.shape[1]
        condition2_length = condition_states2.shape[1]

        norm_hidden_states, norm_condition_states1,norm_condition_states2, gate_msa, c_gate_msa1,c_gate_msa2 = self.norm1(
            hidden_states, condition_states1,condition_states2, temb
        )
        
        # AMD uses concatenated image + motion embeddings with self-attention instead of using
        # them in cross-attention individually
        norm_hidden_states = torch.cat([norm_hidden_states, norm_condition_states1,norm_condition_states2], dim=1)
        attn_output = self.attn1(
            hidden_states=norm_hidden_states,
            encoder_hidden_states=None,
        )

        hidden_states = hidden_states + gate_msa * attn_output[:,:hidden_length]
        condition_states1 = condition_states1 + c_gate_msa1 * attn_output[:, hidden_length:hidden_length+condition1_length]
        condition_states2 = condition_states2 + c_gate_msa2 * attn_output[:, hidden_length+condition1_length:]

        # norm & modulate
        norm_hidden_states, norm_condition_states1,norm_condition_states2, gate_ff, c_gate_ff1,c_gate_ff2 = self.norm2(
            hidden_states, condition_states1,condition_states2, temb
        )

        # feed-forward
        norm_hidden_states = torch.cat([norm_hidden_states, norm_condition_states1,norm_condition_states2], dim=1)
        ff_output = self.ff(norm_hidden_states)

        hidden_states = hidden_states + gate_ff * ff_output[:, :hidden_length]
        condition_states1 = condition_states1 + c_gate_ff1 * ff_output[:, hidden_length:hidden_length+condition1_length]
        condition_states2 = condition_states2 + c_gate_ff2 * ff_output[:, hidden_length+condition1_length:]
        
        return hidden_states, condition_states1,condition_states2
    
class TransformerBlock2Condition_SimpleAdaLN(nn.Module):
    r"""
        TransformerBlock2Condition_SimpleAdaLN
    """

    def __init__(
        self,
        dim: int,
        num_attention_heads: int,
        attention_head_dim: int,
        time_embed_dim: int,
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
        self.norm1 = AMDLayerNormZero_OneVariable(time_embed_dim, dim, norm_elementwise_affine, norm_eps, bias=True)
        self.norm1_condition1 = nn.LayerNorm(dim, eps=norm_eps, elementwise_affine=norm_elementwise_affine)
        self.norm1_condition2 = nn.LayerNorm(dim, eps=norm_eps, elementwise_affine=norm_elementwise_affine)

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
        self.norm2 = AMDLayerNormZero_OneVariable(time_embed_dim, dim, norm_elementwise_affine, norm_eps, bias=True)
        self.norm2_condition1 = nn.LayerNorm(dim, eps=norm_eps, elementwise_affine=norm_elementwise_affine)
        self.norm2_condition2 = nn.LayerNorm(dim, eps=norm_eps, elementwise_affine=norm_elementwise_affine)
        
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
        condition_states1: torch.Tensor,
        condition_states2: torch.Tensor,
        temb: torch.Tensor,
    ) -> torch.Tensor:
        """
        *************************  ******************   ******************** 
        *  hidden_states        *  *condition_states1*   * condition_states2* 
        *************************  ******************   ********************
        """
        hidden_length = hidden_states.shape[1]
        condition1_length = condition_states1.shape[1]
        condition2_length = condition_states2.shape[1]

        # norm
        norm_hidden_states,gate = self.norm1(hidden_states, temb=temb)
        norm_condition_states1 = self.norm1_condition1(condition_states1)
        norm_condition_states2 = self.norm1_condition2(condition_states2)

        # AMD uses concatenated image + motion embeddings with self-attention instead of using
        # them in cross-attention individually
        norm_hidden_states = torch.cat([norm_hidden_states, norm_condition_states1,norm_condition_states2], dim=1)
        attn_output = self.attn1(
            hidden_states=norm_hidden_states,
            encoder_hidden_states=None,
        )

        hidden_states = hidden_states + gate * attn_output[:,:hidden_length]
        condition_states1 = condition_states1 + attn_output[:, hidden_length:hidden_length+condition1_length]
        condition_states2 = condition_states2 + attn_output[:, hidden_length+condition1_length:]

        # norm & modulate
        norm_hidden_states,gate = self.norm2(hidden_states, temb=temb)
        norm_condition_states1 = self.norm2_condition1(condition_states1)
        norm_condition_states2 = self.norm2_condition2(condition_states2)


        # feed-forward
        norm_hidden_states = torch.cat([norm_hidden_states, norm_condition_states1,norm_condition_states2], dim=1)
        ff_output = self.ff(norm_hidden_states)

        hidden_states = hidden_states + gate * ff_output[:, :hidden_length]
        condition_states1 = condition_states1 + ff_output[:, hidden_length:hidden_length+condition1_length]
        condition_states2 = condition_states2 + ff_output[:, hidden_length+condition1_length:]
        
        return hidden_states, condition_states1,condition_states2
    

class Any2MotionTransformerBlock(nn.Module):

    def __init__(
        self,
        dim: int,
        num_attention_heads: int,
        attention_head_dim: int,
        time_embed_dim: int,
        motion_frames : int,
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

        self.motion_frames = motion_frames

        # 1.1 norm_in
        self.norm1 = AdaLayerNorm(
            embedding_dim=time_embed_dim,
            output_dim=dim*2,
            norm_elementwise_affine=norm_elementwise_affine,
            norm_eps=norm_eps,
            chunk_dim=1
        )

        # 1.2 self-attention
        self.attn1 = Attention(
            query_dim=dim,
            dim_head=attention_head_dim,
            heads=num_attention_heads,
            qk_norm="layer_norm" if qk_norm else None,
            eps=1e-6,
            bias=attention_bias,
            out_bias=attention_out_bias,
        )

        # 2.1 norm
        self.norm2 = AdaLayerNorm(
            embedding_dim=time_embed_dim,
            output_dim=dim*2,
            norm_elementwise_affine=norm_elementwise_affine,
            norm_eps=norm_eps,
            chunk_dim=1
        )

        # 2.2 cross-attention for refimg
        self.attn2 = Attention(
                    query_dim=dim,
                    cross_attention_dim=dim,
                    heads=num_attention_heads,
                    dim_head=attention_head_dim,
                    dropout=dropout,
                    bias=attention_bias,
                    out_bias=attention_out_bias,
        )

        # 3.1 norm
        self.norm3 = AdaLayerNorm(
            embedding_dim=time_embed_dim,
            output_dim=dim*2,
            norm_elementwise_affine=norm_elementwise_affine,
            norm_eps=norm_eps,
            chunk_dim=1
        )

        # 3.2 cross-attention for extra-condition
        self.attn3 = Attention(
                    query_dim=dim,
                    cross_attention_dim=dim,
                    heads=num_attention_heads,
                    dim_head=attention_head_dim,
                    dropout=dropout,
                    bias=attention_bias,
                    out_bias=attention_out_bias,
        )

        # 4. Feed Forward
        self.norm4 = AdaLayerNorm(
            embedding_dim=time_embed_dim,
            output_dim=dim*2,
            norm_elementwise_affine=norm_elementwise_affine,
            norm_eps=norm_eps,
            chunk_dim=1
        )

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
        refimg_states: torch.Tensor,
        extra_states: torch.Tensor,
        temb: torch.Tensor,
    ) -> torch.Tensor:

        assert hidden_states.dim() == refimg_states.dim() and hidden_states.dim() == extra_states.dim() , f"hidden_states.dim():{hidden_states.dim()},refimg_states.dim():{refimg_states.dim()},extra_states.dim():{extra_states.dim()}"

        # 1.1 norm
        hidden_states = self.norm1(hidden_states, temb=temb)

        # 1.2 3D self-attention
        hidden_states = einops.rearrange(hidden_states, '(b f) l d -> b (f l) d',f=self.motion_frames)
        attn_output = self.attn1(hidden_states, None) 
        hidden_states = hidden_states + attn_output
        hidden_states = einops.rearrange(hidden_states, 'b (f l) d -> (b f) l d',f=self.motion_frames)

        # 2.1 norm
        hidden_states = self.norm2(hidden_states, temb=temb)

        # 2.2 cross-attention for refimg
        attn_output = self.attn2(hidden_states, refimg_states)

        # 3.1 norm
        hidden_states = hidden_states + attn_output
        hidden_states = self.norm3(hidden_states, temb=temb)

        # 3.2 cross-attention for extra-condition
        attn_output = self.attn3(hidden_states, extra_states)

        # 4.1 norm
        hidden_states = hidden_states + attn_output
        hidden_states = self.norm4(hidden_states, temb=temb)

        # 4.2 ff
        hidden_states = self.ff(hidden_states) + hidden_states

        return hidden_states


class A2MCrossAttnBlock(nn.Module):

    def __init__(
        self,
        dim: int,
        num_attention_heads: int,
        attention_head_dim: int,
        time_embed_dim: int,
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

        # 1.1 norm_in
        self.norm1 = AMDLayerNormZero(conditioning_dim=time_embed_dim,
                                      embedding_dim=dim)

        # 2.2 cross-attention for refimg
        self.attn = Attention(
                    query_dim=dim,
                    cross_attention_dim=dim,
                    heads=num_attention_heads,
                    dim_head=attention_head_dim,
                    dropout=dropout,
                    bias=attention_bias,
                    out_bias=attention_out_bias,
        )

        # Feed Forward
        self.norm2 = AMDLayerNormZero(conditioning_dim=time_embed_dim,
                                      embedding_dim=dim)

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
        motion_hidden_states: torch.Tensor, # N,FL,D
        ref_motion_hidden_states: torch.Tensor, # N,L,D
        conditon_hidden_states: torch.Tensor, # N,F+1,W,D
        temb: torch.Tensor,
    ) -> torch.Tensor:

        N,FL,D = motion_hidden_states.shape
        N,L,D = ref_motion_hidden_states.shape
        F = FL // L


        if conditon_hidden_states.dim()==4 :
            assert F + 1 == conditon_hidden_states.shape[1] ,f'conditon_hidden_states {conditon_hidden_states.shape}'
            conditon_hidden_states = einops.rearrange(conditon_hidden_states,'n f w d -> (n f) w d') # N(F+1),W,D


        # norm1
        norm_motion_hidden_states, norm_ref_motion_hidden_states, gate_msa, enc_gate_msa = self.norm1(
            motion_hidden_states, ref_motion_hidden_states, temb
        )

        # transform for cross attn
        hidden_states = torch.cat([norm_ref_motion_hidden_states,norm_motion_hidden_states],dim=1) # N,L+FL,D
        hidden_states = einops.rearrange(hidden_states,'n (f l) d -> (n f) l d',l=L) #  N(F+1),L,D
        assert hidden_states.shape[0] == conditon_hidden_states.shape[0] ,f'hidden_states.shape {hidden_states.shape} ,audio_hidden_states.shape {audio_hidden_states.shape}'

        # cross-attention for audio
        attn_output = self.attn(hidden_states, conditon_hidden_states)  #  N(F+1),L,D
        attn_output = einops.rearrange(attn_output,'(n f) l d -> n f l d',n=N).flatten(1,2) # N,L+FL,D
        motion_hidden_states = motion_hidden_states + gate_msa * attn_output[:,L:] # N,FL,D
        ref_motion_hidden_states = ref_motion_hidden_states + enc_gate_msa * attn_output[:,:L] # N,L,D

        # norm2
        norm_motion_hidden_states, norm_ref_motion_hidden_states, gate_msa, enc_gate_msa = self.norm2(
            motion_hidden_states, ref_motion_hidden_states, temb
        )  
        hidden_states = torch.cat([norm_ref_motion_hidden_states,norm_motion_hidden_states],dim=1) # N,L+FL,D

        # ff
        hidden_states = self.ff(hidden_states)  #  N,L+FL,D
        motion_hidden_states = motion_hidden_states + gate_msa * hidden_states[:,L:] # N,FL,D
        ref_motion_hidden_states = ref_motion_hidden_states + enc_gate_msa * hidden_states[:,:L] # N,L,D 

        return motion_hidden_states,ref_motion_hidden_states


class A2MMotionSelfAttnBlock(nn.Module):

    def __init__(
        self,
        dim: int,
        num_attention_heads: int,
        attention_head_dim: int,
        time_embed_dim: int,
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


        # 1.1 norm_in
        self.norm1 = AMDLayerNormZero(conditioning_dim=time_embed_dim,
                                      embedding_dim=dim)

        # 1.2 self-attention
        self.attn = Attention(
            query_dim=dim,
            dim_head=attention_head_dim,
            heads=num_attention_heads,
            qk_norm="layer_norm" if qk_norm else None,
            eps=1e-6,
            bias=attention_bias,
            out_bias=attention_out_bias,
        )

        # 2.1 norm
        self.norm2 = AMDLayerNormZero(conditioning_dim=time_embed_dim,
                                      embedding_dim=dim)

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
        motion_hidden_states: torch.Tensor, # N,FL,D
        ref_motion_hidden_states: torch.Tensor, # N,L,D
        temb: torch.Tensor,
    ) -> torch.Tensor:

        N,FL,D = motion_hidden_states.shape
        N,L,D = ref_motion_hidden_states.shape
        F = FL // L

        # norm1
        norm_motion_hidden_states, norm_ref_motion_hidden_states, gate_msa, enc_gate_msa = self.norm1(
            motion_hidden_states, ref_motion_hidden_states, temb
        )
        hidden_states = torch.cat([norm_ref_motion_hidden_states,norm_motion_hidden_states],dim=1) # N,L+FL,D

        # self-attention
        attn_output = self.attn(hidden_states, None) 
        motion_hidden_states = motion_hidden_states + gate_msa * attn_output[:,L:] # N,FL,D
        ref_motion_hidden_states = ref_motion_hidden_states + enc_gate_msa * attn_output[:,:L] # N,L,D

        # norm2
        norm_motion_hidden_states, norm_ref_motion_hidden_states, gate_msa, enc_gate_msa = self.norm2(
            motion_hidden_states, ref_motion_hidden_states, temb
        )  
        hidden_states = torch.cat([norm_ref_motion_hidden_states,norm_motion_hidden_states],dim=1) # N,L+FL,D

        # ff
        hidden_states = self.ff(hidden_states)  #  N,L+FL,D
        motion_hidden_states = motion_hidden_states + gate_msa * hidden_states[:,L:] # N,FL,D
        ref_motion_hidden_states = ref_motion_hidden_states + enc_gate_msa * hidden_states[:,:L] # N,L,D    

        return motion_hidden_states,ref_motion_hidden_states


# ----------------------- A2M audio ------------------------
class AudioToImageShapeMlp(nn.Module):
    def __init__(self,
                audio_dim:int = 384,
                audio_block:int = 50,
                outchannel:int = 256,
    
                out_height:int = 4,
                out_width:int = 4,
                **kwargs
                ):
        super().__init__()
        self.outchannel = outchannel
        self.out_height = out_height
        self.out_width = out_width


        outdim = outchannel * out_height * out_width
        self.mlp = Mlp(in_features=audio_dim*audio_block,hidden_features=outdim,out_features=outdim)
                          
    def forward(self,audio_feature:torch.Tensor):
        """
        Args:
            audio_feature (torch.Tensor):  (N,F,M,C)

        Returns:
            audio_feature (torch.Tensor):  (N,F,D)
        """        
        n,f,m,d = audio_feature.shape

        audio_feature = einops.rearrange(audio_feature,'n f m d -> n f (m d)')
        audio_feature = self.mlp(audio_feature)
        audio_feature = einops.rearrange(audio_feature,'n f (c h w) -> n f c h w',c=self.outchannel,h=self.out_height,w=self.out_width)

        return audio_feature

class AudioFeatureMlp(nn.Module):
    def __init__(self,
                audio_dim:int = 384,
                audio_block:int = 50,
                hidden_dim:int = 128,
                outdim:int = 1024,
                **kwargs
                ):
        super().__init__()

        # self.mlp1 = Mlp(in_features=audio_dim,
        #                     hidden_features=hidden_dim,
        #                     out_features=hidden_dim)

        # self.mlp2 = Mlp(in_features=audio_block * hidden_dim,
        #                     hidden_features=outdim,
        #                     out_features=outdim)

        self.mlp = Mlp(in_features=audio_dim*audio_block,hidden_features=outdim,out_features=outdim)

                            
    def forward(self,audio_feature:torch.Tensor):
        """
        Args:
            audio_feature (torch.Tensor):  (N,F,M,C)

        Returns:
            audio_feature (torch.Tensor):  (N,F,D)
        """        
        # n,f,m,d = audio_feature.shape
        # audio_feature = self.mlp1(audio_feature)
        # audio_feature = audio_feature.reshape(n,f,-1)
        # audio_feature = self.mlp2(audio_feature)

        audio_feature = einops.rearrange(audio_feature,'n f m d -> n f (m d)')
        audio_feature = self.mlp(audio_feature)

        return audio_feature

class AudioFeatureWindowMlp(nn.Module):
    def __init__(self,
                audio_dim:int = 384,
                audio_block:int = 50,
                intermediate_dim :int = 1024,
                window_size:int = 12,
                outdim:int = 768,
                **kwargs
                ):
        super().__init__()

        self.window_size = window_size
        self.ff1 =  nn.Linear(audio_dim*audio_block, intermediate_dim)
        self.ff2 =  nn.Linear(intermediate_dim,intermediate_dim)
        self.ff3 =  nn.Linear(intermediate_dim,window_size * outdim)
        self.norm = nn.LayerNorm(outdim)
                            
    def forward(self,audio_feature:torch.Tensor):
        """
        Args:
            audio_feature (torch.Tensor):  (N,F,M,C)

        Returns:
            audio_feature (torch.Tensor):  (N,F,W,D)
        """        
        n,f,m,d = audio_feature.shape

        audio_feature = einops.rearrange(audio_feature,'n f m d -> n f (m d)')
        audio_feature = torch.relu(self.ff1(audio_feature)) # n f inter
        audio_feature = torch.relu(self.ff2(audio_feature)) # n f inter
        audio_feature = torch.relu(self.ff3(audio_feature)) # n f w*d
        audio_feature = einops.rearrange(audio_feature,"n f (w d) -> n f w d",w= self.window_size)

        audio_feature = self.norm(audio_feature)

        return audio_feature


class RefMotionRefImgeBlock(nn.Module):

    def __init__(
        self,
        dim: int,
        num_attention_heads: int,
        attention_head_dim: int,
        time_embed_dim: int,
        dropout: float = 0.0,
        activation_fn: str = "gelu-approximate",
        attention_bias: bool = True,
        qk_norm: bool = True,
        norm_elementwise_affine: bool = True,
        norm_eps: float = 1e-5,
        final_dropout: bool = True,
        ff_inner_dim: Optional[int] = None,
        ff_bias: bool = True,
        attention_out_bias: bool = True,
    ):
        super().__init__()


        # 1.1 norm_in
        self.norm1 = AdaLayerNorm(
            embedding_dim=time_embed_dim,
            output_dim=dim*2,
            norm_elementwise_affine=norm_elementwise_affine,
            norm_eps=norm_eps,
            chunk_dim=1
        )

        # 1.2 self-attention
        self.attn1 = Attention(
            query_dim=dim,
            dim_head=attention_head_dim,
            heads=num_attention_heads,
            qk_norm="layer_norm" if qk_norm else None,
            eps=1e-6,
            bias=attention_bias,
            out_bias=attention_out_bias,
        )

        # 2.1 norm
        self.norm2 = AdaLayerNorm(
            embedding_dim=time_embed_dim,
            output_dim=dim*2,
            norm_elementwise_affine=norm_elementwise_affine,
            norm_eps=norm_eps,
            chunk_dim=1
        )

        # 2.2 cross-attention for refmotion
        self.attn2 = Attention(
                    query_dim=dim,
                    cross_attention_dim=dim,
                    heads=num_attention_heads,
                    dim_head=attention_head_dim,
                    dropout=dropout,
                    bias=attention_bias,
                    out_bias=attention_out_bias,
        )

        # 3.1 norm
        self.norm3 = AdaLayerNorm(
            embedding_dim=time_embed_dim,
            output_dim=dim*2,
            norm_elementwise_affine=norm_elementwise_affine,
            norm_eps=norm_eps,
            chunk_dim=1
        )

        # 3.2 cross-attention for refimg
        self.attn3 = Attention(
                    query_dim=dim,
                    cross_attention_dim=dim,
                    heads=num_attention_heads,
                    dim_head=attention_head_dim,
                    dropout=dropout,
                    bias=attention_bias,
                    out_bias=attention_out_bias,
        )

        # 4. Feed Forward
        self.norm4 = AdaLayerNorm(
            embedding_dim=time_embed_dim,
            output_dim=dim*2,
            norm_elementwise_affine=norm_elementwise_affine,
            norm_eps=norm_eps,
            chunk_dim=1
        )

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
        hidden_states: torch.Tensor, # N,L1,D
        refmotion_states: torch.Tensor, # N,L2,D
        refimg_states: torch.Tensor, # N,L3,D
        temb: torch.Tensor,
    ) -> torch.Tensor:

        assert hidden_states.dim() == refimg_states.dim() and hidden_states.dim() == refmotion_states.dim() , f"hidden_states.dim():{hidden_states.dim()},refimg_states.dim():{refimg_states.dim()}"

        # 1.1 norm
        hidden_states = self.norm1(hidden_states, temb=temb)

        # 1.2 3D self-attention
        attn_output = self.attn1(hidden_states, None) 
        hidden_states = hidden_states + attn_output

        # 2.1 norm
        hidden_states = self.norm2(hidden_states, temb=temb)

        # 2.2 cross-attention for refmotion
        attn_output = self.attn2(hidden_states, refmotion_states)

        # 3.1 norm
        hidden_states = hidden_states + attn_output
        hidden_states = self.norm3(hidden_states, temb=temb)

        # 3.2 cross-attention for refimg
        attn_output = self.attn3(hidden_states, refimg_states)

        # 4.1 norm
        hidden_states = hidden_states + attn_output
        hidden_states = self.norm4(hidden_states, temb=temb)

        # 4.2 ff
        hidden_states = self.ff(hidden_states) + hidden_states

        return hidden_states

class Audio2Pose(nn.Module):
    def __init__(self,
                audio_dim:int = 384,
                audio_block:int = 50,

                motion_height:int = 4,
                motion_width:int = 4,
                motion_dim:int = 256,

                pose_width:int = 32,
                pose_height:int = 32,
                pose_dim:int = 4,
                num_frames:int = 15,

                **kwargs
                ):
        super().__init__()
        self.num_frames = num_frames
        self.pw = pose_width
        self.ph = pose_height
        self.pc = pose_dim
        self.audio_encoder = AudioToImageShapeMlp(
            audio_dim=audio_dim,
            audio_block = audio_block,
            outchannel=motion_dim,
            out_height=motion_height,
            out_width=motion_width,
        ) # (NF,256,4,4)

        self.pose_predictor = Upsampler(
            in_channel=motion_dim,
            out_channel=pose_dim,
            block_out_channels=(motion_dim,128,64,32),
        )

        self.pose_downsample = DownEncoder(in_channel=pose_dim,block_out_channels=(32,64,128,motion_dim))

    def forward(self,audio_feature:torch.Tensor,pose_gt:torch.Tensor):
        """
        Args:
            audio_feature (torch.Tensor): (N,F,M,D)
            pose_gt (torch.Tensor): (N,F,C,H,W)

        Returns:
            pose_pred (torch.Tensor): (N,F,C,H,W), used for loss calculation
            pose_transform (torch.Tensor): (N,F,256,4,4), used for diffusion
            audio_hidden_state (torch.Tensor): (N,F,256,4,4), used for audio condition injection
        """        

        b,f,m,d = audio_feature.shape
        audio_hidden_state = self.audio_encoder(audio_feature) # (N,F,256,4,4)

        audio_hidden_state = einops.rearrange(audio_hidden_state,'n f c h w -> (n f) c h w')
        pose_pre = self.pose_predictor(audio_hidden_state)

        pose_gt = einops.rearrange(pose_gt,'n f c h w -> (n f) c h w')
        pose_gt_transform = self.pose_downsample(pose_gt)

        pose_pre = einops.rearrange(pose_pre,'(n f) c h w -> n f c h w',n=b) # (8,15,256,4,4)
        pose_gt_transform = einops.rearrange(pose_gt_transform,'(n f) c h w -> n f c h w',n=b) # (4,15,4,32,32)
        audio_hidden_state = einops.rearrange(audio_hidden_state,'(n f) c h w -> n f c h w',n=b) # (4,15,256,4,4)

        return pose_pre, pose_gt_transform, audio_hidden_state 
    def prepare_extra(self,audio:torch.Tensor,pose:torch.Tensor):
        b = audio.shape[0]
        audio_hidden_state = self.audio_encoder(audio)
        pose_pred = self.pose_predictor(audio_hidden_state)
        pose_pred = self.pose_downsample(pose_pred)
        pose_pred = einops.rearrange(pose_pred,'(n f) c h w -> n f c h w',n=b) # (4,15,4,32,32)
        audio_hidden_state = einops.rearrange(audio_hidden_state,'(n f) c h w -> n f c h w',n=b) # (4,15,256,4,4)
        return audio_hidden_state, pose_pred

class MotionTrensferBlock(nn.Module):
    r"""
        MotionTrensferBlock
    """

    def __init__(
        self,
        dim: int,
        num_attention_heads: int,
        attention_head_dim: int,
        time_embed_dim: int,
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
        self.norm1 = AMDLayerNormZero(time_embed_dim, dim, norm_elementwise_affine, norm_eps, bias=True)

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
        self.norm2 = AMDLayerNormZero(time_embed_dim, dim, norm_elementwise_affine, norm_eps, bias=True)

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
        hidden_states: torch.Tensor, # NF,L1,D
        encoder_hidden_states: torch.Tensor, # NF,L2,D
        temb: torch.Tensor,
    ) -> torch.Tensor:
        """
        *************************  ******************
        *  hidden_states*   * encoder_hidden_states *
        *************************  ******************
        """
        norm_hidden_states, norm_encoder_hidden_states, gate_msa, enc_gate_msa = self.norm1(
            hidden_states, encoder_hidden_states, temb
        )

        # attention
        motion_length = norm_hidden_states.shape[1]

        # AMD uses concatenated image + motion embeddings with self-attention instead of using
        # them in cross-attention individually
        norm_hidden_states = torch.cat([norm_hidden_states, norm_encoder_hidden_states ], dim=1)

        attn_output = self.attn1(
            hidden_states=norm_hidden_states,
            encoder_hidden_states=None,
        )

        hidden_states = hidden_states + gate_msa * attn_output[:, :motion_length]
        encoder_hidden_states = encoder_hidden_states + enc_gate_msa * attn_output[:, motion_length:]

        # norm & modulate
        norm_hidden_states, norm_encoder_hidden_states, gate_ff, enc_gate_ff = self.norm2(
            hidden_states, encoder_hidden_states, temb
        )

        # feed-forward
        norm_hidden_states = torch.cat([norm_encoder_hidden_states, norm_hidden_states], dim=1)
        ff_output = self.ff(norm_hidden_states)

        hidden_states = hidden_states + gate_ff * ff_output[:, :motion_length]
        encoder_hidden_states = encoder_hidden_states + enc_gate_ff * ff_output[:, motion_length:]
        
        return hidden_states, encoder_hidden_states

# ----------------------- A2P ------------------------

class A2PTemporalSpatialBlock(nn.Module):

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

        # Temporal Attention
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

        # Spatial Attention
        self.norm2 = nn.LayerNorm(dim, eps=norm_eps, elementwise_affine=norm_elementwise_affine)
        
        self.attn2 = Attention(
            query_dim=dim,
            dim_head=attention_head_dim,
            heads=num_attention_heads,
            qk_norm="layer_norm" if qk_norm else None,
            eps=1e-6,
            bias=attention_bias,
            out_bias=attention_out_bias,
        )

        # Feed Forward
        self.norm3 = nn.LayerNorm(dim, eps=norm_eps, elementwise_affine=norm_elementwise_affine)

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
        
        N,F,L,D = hidden_states.shape
        
        # norm1
        hidden_states = einops.rearrange(hidden_states,'n f l d -> (n l) f d') # NL,F,D
        norm_hidden_states = self.norm1(hidden_states) # NL,F,D

        # temporal attention
        attn_output = self.attn1(
            hidden_states=norm_hidden_states,
            encoder_hidden_states=None,
        )
        hidden_states = hidden_states + attn_output

        # norm2
        hidden_states = einops.rearrange(hidden_states,'(n l) f d -> (n f) l d',n=N,l=L)
        norm_hidden_states = self.norm2(hidden_states) # NF,L,D
        
        # spatial attention
        attn_output = self.attn2(
            hidden_states=norm_hidden_states,
            encoder_hidden_states=None,
        )
        hidden_states = hidden_states + attn_output

        # norm & modulate
        norm_hidden_states = self.norm3(hidden_states) # NF,L,D

        # feed-forward
        ff_output = self.ff(norm_hidden_states)
        hidden_states = hidden_states + ff_output

        # transform
        hidden_states = einops.rearrange(hidden_states,'(n f) l d -> n f l d',n=N)
        
        return hidden_states # N,F,L,D


class A2PCrossAudioBlock(nn.Module):

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

        # Temporal Attention
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

        # Feed Forward
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
        audio_hidden_states: torch.Tensor,
    ) -> torch.Tensor:
        
        N,F,L,D = hidden_states.shape
        N,F,W,D = audio_hidden_states.shape
        
        # norm1
        hidden_states = einops.rearrange(hidden_states,'n f l d -> (n f) l d') # NF,L,D
        norm_hidden_states = self.norm1(hidden_states) # NF,L,D
        audio_hidden_states = einops.rearrange(audio_hidden_states,'n f w d -> (n f) w d') # NF,W,D

        # temporal attention
        attn_output = self.attn1(
            hidden_states=norm_hidden_states,
            encoder_hidden_states=audio_hidden_states,
        )
        hidden_states = hidden_states + attn_output # NF,L,D
 

        # norm & modulate
        norm_hidden_states = self.norm2(hidden_states) # NF,L,D

        # feed-forward
        ff_output = self.ff(norm_hidden_states)
        hidden_states = hidden_states + ff_output

        # transform
        hidden_states = einops.rearrange(hidden_states,'(n f) l d -> n f l d',n=N,f=F)
        
        return hidden_states # N,F,L,D
   