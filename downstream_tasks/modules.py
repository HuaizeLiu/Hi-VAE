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
from torch import nn
from torch.nn import functional as F
from diffusers.utils import BaseOutput, logging
from einops import rearrange
from diffusers.utils import is_torch_version
from diffusers.models.attention import Attention,FeedForward
from timm.models.layers import Mlp
from transformers import CLIPTokenizer, CLIPTextModel
from diffusers.models.modeling_utils import ModelMixin
from diffusers.models.embeddings import get_2d_sincos_pos_embed,get_1d_sincos_pos_embed_from_grid,get_3d_sincos_pos_embed,TimestepEmbedding, Timesteps
import einops

def modulate(x, shift, scale):
    return x * (1 + scale.unsqueeze(1)) + shift.unsqueeze(1) 

logger = logging.get_logger(__name__)  # pylint: disable=invalid-name


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
        conditon_states1: torch.Tensor,
        condition_states2: torch.Tensor,
        temb: torch.Tensor,
    ) -> torch.Tensor:
        """
        *************************  ******************   ******************** 
        *  hidden_states        *  *conditon_states1*   * condition_states2* 
        *************************  ******************   ********************
        """
        hidden_length = hidden_states.shape[1]
        condition1_length = conditon_states1.shape[1]
        condition2_length = condition_states2.shape[1]

        norm_hidden_states, norm_conditon_states1,norm_conditon_states2, gate_msa, c_gate_msa1,c_gate_msa2 = self.norm1(
            hidden_states, conditon_states1,condition_states2, temb
        )
        
        # AMD uses concatenated image + motion embeddings with self-attention instead of using
        # them in cross-attention individually
        norm_hidden_states = torch.cat([norm_hidden_states, norm_conditon_states1,norm_conditon_states2], dim=1)
        attn_output = self.attn1(
            hidden_states=norm_hidden_states,
            encoder_hidden_states=None,
        )

        hidden_states = hidden_states + gate_msa * attn_output[:,:hidden_length]
        conditon_states1 = conditon_states1 + c_gate_msa1 * attn_output[:, hidden_length:hidden_length+condition1_length]
        condition_states2 = condition_states2 + c_gate_msa2 * attn_output[:, hidden_length+condition1_length:]

        # norm & modulate
        norm_hidden_states, norm_conditon_states1,norm_conditon_states2, gate_ff, c_gate_ff1,c_gate_ff2 = self.norm2(
            hidden_states, conditon_states1,condition_states2, temb
        )

        # feed-forward
        norm_hidden_states = torch.cat([norm_hidden_states, norm_conditon_states1,norm_conditon_states2], dim=1)
        ff_output = self.ff(norm_hidden_states)

        hidden_states = hidden_states + gate_ff * ff_output[:, :hidden_length]
        conditon_states1 = conditon_states1 + c_gate_ff1 * ff_output[:, hidden_length:hidden_length+condition1_length]
        condition_states2 = condition_states2 + c_gate_ff2 * ff_output[:, hidden_length+condition1_length:]
        
        return hidden_states, conditon_states1,condition_states2

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
# This block only do wav2vec model forwarding and reversed for future development.

class AudioFeatureEncoder(nn.Module):
    def __init__(self, feature_dim:int=384,latent_dim:int = 512,out_dim:int = 768):
        super().__init__()
        self.feature_dim = feature_dim
        self.mlp = nn.Sequential(
            nn.Linear(feature_dim, latent_dim),
            nn.GELU(),
            nn.Linear(latent_dim, out_dim),
        )
    def forward(self,audio_feature):
        feature = self.mlp(audio_feature)
        return feature
class AudioProjModel(ModelMixin):
    """Audio Projection Model

    This class defines an audio projection model that takes audio embeddings as input
    and produces context tokens as output. The model is based on the ModelMixin class
    and consists of multiple linear layers and activation functions. It can be used
    for various audio processing tasks.

    Attributes:
        seq_len (int): The length of the audio sequence.
        blocks (int): The number of blocks in the audio projection model.
        channels (int): The number of channels in the audio projection model.
        intermediate_dim (int): The intermediate dimension of the model.
        context_tokens (int): The number of context tokens in the output.
        output_dim (int): The output dimension of the context tokens.

    Methods:
        __init__(self, seq_len=5, blocks=12, channels=768, intermediate_dim=512, context_tokens=32, output_dim=768):
            Initializes the AudioProjModel with the given parameters.
        forward(self, audio_embeds):
            Defines the forward pass for the AudioProjModel.
            Parameters:
            audio_embeds (torch.Tensor): The input audio embeddings with shape (batch_size, video_length, blocks, channels).
            Returns:
            context_tokens (torch.Tensor): The output context tokens with shape (batch_size, video_length, context_tokens, output_dim).

    """

    def __init__(
        self,
        seq_len=5,
        blocks=12,  # add a new parameter blocks
        channels=768,  # add a new parameter channels
        intermediate_dim=512,
        output_dim=768,
        context_tokens=32,
        **kwargs
    ):
        super().__init__()

        self.seq_len = seq_len
        self.blocks = blocks
        self.channels = channels
        self.input_dim = (
            seq_len * blocks * channels
        )  # update input_dim to be the product of blocks and channels.
        self.intermediate_dim = intermediate_dim
        self.context_tokens = context_tokens
        self.output_dim = output_dim

        # define multiple linear layers
        self.proj1 = nn.Linear(self.input_dim, intermediate_dim)
        self.proj2 = nn.Linear(intermediate_dim, intermediate_dim)
        self.proj3 = nn.Linear(intermediate_dim, context_tokens * output_dim)

        self.norm = nn.LayerNorm(output_dim)

    def forward(self, audio_embeds):
        """
        Defines the forward pass for the AudioProjModel.

        Parameters:
            audio_embeds (torch.Tensor): The input audio embeddings with shape (batch_size, video_length, blocks, channels).

        Returns:
            context_tokens (torch.Tensor): The output context tokens with shape (batch_size, video_length, context_tokens, output_dim).
        """
        # merge
        video_length = audio_embeds.shape[1]
        audio_embeds = rearrange(audio_embeds, "bz f w b c -> (bz f) w b c")
        batch_size, window_size, blocks, channels = audio_embeds.shape
        audio_embeds = audio_embeds.view(batch_size, window_size * blocks * channels)

        audio_embeds = torch.relu(self.proj1(audio_embeds))
        audio_embeds = torch.relu(self.proj2(audio_embeds))

        context_tokens = self.proj3(audio_embeds).reshape(
            batch_size, self.context_tokens, self.output_dim
        )

        context_tokens = self.norm(context_tokens)
        context_tokens = rearrange(
            context_tokens, "(bz f) m c -> bz f m c", f=video_length
        )

        return context_tokens #  bz f m c
class PosePredictor(nn.Module):
    def __init__(
        self,
        pose_dim:int,
        pose_width:int,
        pose_height:int,
        num_frames : int,
        num_attention_heads: int,
        attention_head_dim: int,
        num_layers:int = 4,
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
        patch_size:int=2,
    ):
        super().__init__()
        assert num_layers % 2 == 0, f'num_layes should be odd, but receive {num_layers}'
        dim = num_attention_heads * attention_head_dim
        self.num_frames = num_frames
        self.pose_dim = pose_dim
        self.patch_embed = PatchEmbed(patch_size=patch_size,in_channels=pose_dim,embed_dim=dim, bias=True)
        self.patch_size = patch_size
        # 3. 2D&3D positional embeddings
        ph,pw = pose_height // patch_size, pose_width // patch_size
        pos_embedding_ = get_2d_sincos_pos_embed(dim, (ph,pw))
        pos_embedding_ = torch.from_numpy(pos_embedding_) 
        pos_embedding = torch.zeros(1,ph*pw, dim, requires_grad=False)
        pos_embedding.data.copy_(pos_embedding_)
        self.register_buffer("pos_embedding", pos_embedding, persistent=False)
        self.blocks = nn.ModuleList(
            [
                BasicTransformerBlock(
                    dim,
                    num_attention_heads,
                    attention_head_dim,
                    dropout,
                    activation_fn,
                    attention_bias,
                    qk_norm,
                    norm_elementwise_affine,
                    norm_eps,
                    final_dropout,
                    ff_inner_dim,
                    ff_bias,
                    attention_out_bias
                )
            ] * num_layers
        )
        self.norm_final = nn.LayerNorm(dim, norm_eps, norm_elementwise_affine)
        self.proj_out = nn.Linear(dim, patch_size * patch_size * pose_dim)
    def forward(self,pose:torch.Tensor,pose_cond:torch.Tensor):
        '''
        pose (torch.Tensor): (N,F,C,H,W)
        pose_cond (torch.Tensor): (N,C,H,W)
        '''
        n,f,c,h,w = pose.shape
        one_pose_len = h * w // (self.patch_size ** 2)
        hidden_states = torch.concat([pose_cond[:,None],pose],dim=1)
        hidden_states = einops.rearrange(hidden_states,'n f c h w -> (n f) c h w')
        hidden_states = self.patch_embed(hidden_states) + self.pos_embedding

        for i, block in enumerate(self.blocks):
            hidden_states = block(
                hidden_states 
            )
            hidden_states = einops.rearrange(hidden_states,'(n f) l d -> (n l) f d',n=n)
        hidden_states = self.norm_final(hidden_states)
        motion_hidden_states = self.proj_out(hidden_states)
        motion_hidden_states = einops.rearrange(motion_hidden_states,'(n f) l d -> n f l d',n=n)
        motion_hidden_states = motion_hidden_states[:,1:]
        p = self.patch_size
        print(motion_hidden_states.shape)
        output = motion_hidden_states.reshape(n, self.num_frames, h // p, w // p, self.pose_dim, p, p)
        output = output.permute(0, 1, 4, 2, 5, 3, 6).flatten(5, 6).flatten(3, 4) 
        return output
        
class Audio2Pose(nn.Module):
    def __init__(self,
                audio_dim:int = 768,
                pose_width:int = 32,
                pose_height:int = 32,
                pose_dim:int = 4,
                num_frames:int = 15,
                outdim:int = 1024,
                audio_latent_dim:int=1024,

                num_attention_heads : int = 8,
                attention_dim : int = 64,
                audio_seq_len:int = 1,
                **kwargs
                ):
        super().__init__()
        self.num_frames = num_frames
        self.pw = pose_width
        self.ph = pose_height
        self.pc = pose_dim
        self.audio_encoder = AudioFeatureEncoder(
            audio_dim * audio_seq_len,
            audio_latent_dim,
            pose_width * pose_height * pose_dim
        )
        self.pose_predictor = PosePredictor(
            pose_dim,
            pose_width,
            pose_height,
            num_frames,
            num_attention_heads,
            attention_dim,
            **kwargs
        )
        self.mlp_out = Mlp(in_features=pose_width*pose_height*pose_dim,
                            hidden_features=outdim,
                            out_features=outdim)
    def forward(self,audio_feature:torch.Tensor,pose_cond:torch.Tensor):
        """
        Args:
            audio_feature (torch.Tensor): (N,F,D) or (N,F,M,D)
            pose_cond (torch.Tensor): (N,C,H,W)

        Returns:
            pose_pred (torch.Tensor): (N,F,C,H,W), used for loss calculation
            pose_hidden_state (torch.Tensor): (N,F,C,H,W), used for extra condition injection
        """        
        if len(audio_feature.shape) == 4:
            audio_feature = audio_feature.flatten(-2,-1)
        b,f,d = audio_feature.shape
        pose = self.audio_encoder(audio_feature).reshape(b,f,self.pc,self.ph,self.pw)
        pose_pred = self.pose_predictor(pose,pose_cond) 

        extra = pose_pred + pose
        extra = einops.rearrange(extra,'n f c h w -> n f (c h w)')
        extra = self.mlp_out(extra)
        return pose_pred, extra
    
class Any2MotionDiffusionTransformer(nn.Module):
    """
    Diffusion Transformer
    """

    _supports_gradient_checkpointing = True
    def __init__(
        self,
        num_attention_heads: int = 8,
        attention_head_dim: int = 64,
        motion_in_channels: Optional[int] = 256,
        refimg_in_channels: Optional[int] = 4,
        extra_in_channels: Optional[int] = 768,
        out_channels: Optional[int] = 256,
        num_layers: int = 16,
        
        image_width: int = 32,
        image_height: int = 32,
        image_patch_size: int = 2,

        motion_width: int = 4,
        motion_height: int = 4,
        motion_patch_size: int = 1,
        motion_frames: int = 15,

        time_embed_dim: int = 512,

        flip_sin_to_cos: bool = True,
        freq_shift: int = 0,


        dropout: float = 0.0,
        attention_bias: bool = True,
        temporal_compression_ratio: int = 4,

        activation_fn: str = "gelu-approximate",
        timestep_activation_fn: str = "silu",
        norm_elementwise_affine: bool = True,
        norm_eps: float = 1e-5,
        spatial_interpolation_scale: float = 1.0,
        temporal_interpolation_scale: float = 1.0,
        **kwargs
    ):
        super().__init__()
        
        # 1. Setting
        hidden_dim = num_attention_heads * attention_head_dim

        iph = image_height // image_patch_size
        ipw = image_width // image_patch_size
        itl = iph * ipw # image token length

        mph = motion_height // motion_patch_size
        mpw = motion_width // motion_patch_size
        mtl = mph * mpw * motion_frames # motion token num
        
        self.image_patch_size = image_patch_size
        self.motion_patch_size = motion_patch_size
        self.out_channels = out_channels
        self.motion_frames = motion_frames

        # 2. Patch embedding (N,F,C,H,W) -> (B,S,D)
        self.refimg_patch_embed = PatchEmbed(patch_size=image_patch_size,in_channels=refimg_in_channels,embed_dim=hidden_dim, bias=True)
        self.motion_patch_embed = PatchEmbed(patch_size=motion_patch_size,in_channels=motion_in_channels,embed_dim=hidden_dim, bias=True)
        self.extra_embed = nn.Linear(extra_in_channels,hidden_dim)
        self.embedding_dropout = nn.Dropout(dropout)

        # 3. 2D&3D positional embeddings
        image_pos_embedding = get_2d_sincos_pos_embed(hidden_dim, (iph, ipw)) # (iph*ipw,D)
        image_pos_embedding = torch.from_numpy(image_pos_embedding) # (iph*ipw,D)
        pos_embedding = torch.zeros(1, itl, hidden_dim, requires_grad=False)
        pos_embedding.data[:, :itl].copy_(image_pos_embedding)
        self.register_buffer("refimg_pos_embedding", pos_embedding, persistent=False)

        spatial_pos_embedding = get_3d_sincos_pos_embed(  
            hidden_dim,
            (mph, mph),
            motion_frames,
            spatial_interpolation_scale,
            temporal_interpolation_scale,
        ) 
        spatial_pos_embedding = torch.from_numpy(spatial_pos_embedding).flatten(0, 1) # [T*H*W, D]
        pos_embedding = torch.zeros(1,*spatial_pos_embedding.shape, requires_grad=False)
        pos_embedding.data.copy_(spatial_pos_embedding)
        self.register_buffer("motion_pos_embedding", pos_embedding, persistent=False)
        

        # 5. Time embeddings
        self.time_proj = Timesteps(hidden_dim, flip_sin_to_cos, freq_shift)
        self.time_embedding = TimestepEmbedding(hidden_dim, time_embed_dim, timestep_activation_fn)

        # 6. Define spatio-temporal transformers blocks
        self.transformer_blocks = nn.ModuleList(
            [
                Any2MotionTransformerBlock(
                    dim=hidden_dim,
                    num_attention_heads=num_attention_heads,
                    attention_head_dim=attention_head_dim,
                    time_embed_dim=time_embed_dim,
                    motion_frames=motion_frames,
                    dropout=dropout,
                    activation_fn=activation_fn,
                    attention_bias=attention_bias,
                    norm_elementwise_affine=norm_elementwise_affine,
                    norm_eps=norm_eps,
                )
                for _ in range(num_layers)
            ]
        )

        self.norm_final = nn.LayerNorm(hidden_dim, norm_eps, norm_elementwise_affine)

        # 5. Output blocks
        self.norm_out = AdaLayerNorm(
            embedding_dim=time_embed_dim,
            output_dim=hidden_dim*2,
            norm_elementwise_affine=norm_elementwise_affine,
            norm_eps=norm_eps,
            chunk_dim=1,
        )
        self.proj_out = nn.Linear(hidden_dim, motion_patch_size * motion_patch_size * out_channels)

        self.gradient_checkpointing = False

    def _set_gradient_checkpointing(self, module, value=False):
        self.gradient_checkpointing = value

    def forward(
        self,
        motion_hidden_states: torch.Tensor,
        refimg_hidden_states: torch.Tensor,
        extra_hidden_states: torch.Tensor,
        timestep: Union[int, float, torch.LongTensor], # Timesteps should be a 1d-array
        timestep_cond: Optional[torch.Tensor] = None,
        return_dict: bool = True,
    ):
        """
        motion_hidden_states : (N*F,D,h,w)   
        refimg_hidden_states : (N,C,H,W)  or (N*F,C,H,W)  
        extra_hidden_states  : (N*F,M,D')    这个需要提前做 position encoding if needed ,前面两个固定在这里做position encoding
        """
        # motion (N,F,D,h,w) -> (NF,D,h,w)
        if motion_hidden_states.dim() == 5:
            motion_hidden_states = motion_hidden_states.flatten(0,1) 
        if extra_hidden_states.dim() == 4:
            extra_hidden_states = extra_hidden_states.flatten(0,1)
        if refimg_hidden_states.shape[0] < motion_hidden_states.shape[0] and refimg_hidden_states.dim()==4:
            refimg_hidden_states = refimg_hidden_states.unsqueeze(1).repeat(1,self.motion_frames,1,1,1)

            refimg_hidden_states = refimg_hidden_states.flatten(0,1)  # (N*F,C,H,W)


        assert motion_hidden_states.shape[0] == refimg_hidden_states.shape[0] and motion_hidden_states.shape[0] == extra_hidden_states.shape[0] ,f"motion_hidden_states.shape:{motion_hidden_states.shape},refimg_hidden_states.shape:{refimg_hidden_states.shape},extra_hidden_states.shape:{extra_hidden_states.shape}"

        NF,Ci,Hi,Wi = refimg_hidden_states.shape
        NF,Cm,Hm,Wm = motion_hidden_states.shape
        refimg_seq_length = Hi * Wi // (self.image_patch_size**2)
        motion_seq_length = Hm * Wm // (self.motion_patch_size**2)
        
        # 1. Time embedding   # Timesteps should be a 1d-array
        t_emb = self.time_proj(timestep)
        t_emb = t_emb.to(dtype=motion_hidden_states.dtype)
        emb = self.time_embedding(t_emb, timestep_cond) # (batch,time_embed_dim) (8,512)
        emb = emb.unsqueeze(1).repeat(1,self.motion_frames,1) # (8,15,512)
        emb = emb.flatten(0,1) # (120,512)

        # 2. Patch embedding
        motion_hidden_states = self.motion_patch_embed(motion_hidden_states) # [N,L1,D]
        refimg_hidden_states = self.refimg_patch_embed(refimg_hidden_states) # [N,L2,D]
        extra_hidden_states  = self.extra_embed(extra_hidden_states) # [N,L3,D]

        # 3. Position embedding
        refimg_hidden_states = refimg_hidden_states + self.refimg_pos_embedding[:, :refimg_seq_length]
        motion_hidden_states = motion_hidden_states + self.motion_pos_embedding[:, :motion_seq_length]
        self.embedding_dropout(motion_hidden_states)

        # 4. Transformer blocks
        for i, block in enumerate(self.transformer_blocks):
            if self.training and self.gradient_checkpointing:

                def create_custom_forward(module):
                    def custom_forward(*inputs):
                        return module(*inputs)

                    return custom_forward

                ckpt_kwargs: Dict[str, Any] = {"use_reentrant": False} if is_torch_version(">=", "1.11.0") else {}
                motion_hidden_states = torch.utils.checkpoint.checkpoint(
                    create_custom_forward(block),
                    motion_hidden_states,
                    refimg_hidden_states,
                    extra_hidden_states,
                    emb,
                    **ckpt_kwargs,
                )
            else:
                motion_hidden_states = block(
                    motion_hidden_states,
                    refimg_hidden_states,
                    extra_hidden_states,
                    temb=emb,
                )

        motion_hidden_states = self.norm_final(motion_hidden_states)

        # 6. Final block
        motion_hidden_states = self.norm_out(motion_hidden_states, temb=emb)
        motion_hidden_states = self.proj_out(motion_hidden_states) # [N,L1,D]

        # 7. Unpatchify
        p = self.motion_patch_size
        output = motion_hidden_states.reshape(NF // self.motion_frames, self.motion_frames, Hm // p, Wm // p, self.out_channels, p, p)
        output = output.permute(0, 1, 4, 2, 5, 3, 6).flatten(5, 6).flatten(3, 4) # [N,C,H,W]

        return output # [NF,C,H,W]


class AudioFeatureMlp(nn.Module):
    def __init__(self,
                audio_dim:int = 768,
                num_frames:int = 15,
                block = 12,
                outdim:int = 1024,
                **kwargs
                ):
        super().__init__()
        self.num_frames = num_frames

        self.mlp = Mlp(in_features=block*audio_dim,
                            hidden_features=outdim,
                            out_features=outdim)
    def forward(self,audio_feature:torch.Tensor,pose_cond:torch.Tensor):
        """
        Args:
            audio_feature (torch.Tensor):  (N,F,M,D)

        Returns:
            audio_feature (torch.Tensor):  (N,F,D)
        """        
        if len(audio_feature.shape) == 4:
            audio_feature = audio_feature.flatten(-2,-1)
        b,f,d = audio_feature.shape
        audio_feature = self.mlp(audio_feature)

        return audio_feature

# Audio + cat(pose + refimg) + motion_t
class Audio2MotionAllSequence(nn.Module):
    _supports_gradient_checkpointing = True
    def __init__(
        self,
        num_attention_heads: int = 8,
        attention_head_dim: int = 64,
        motion_in_channels: Optional[int] = 256,
        refimg_in_channels: Optional[int] = 4,
        extra_in_channels: Optional[int] = 768,
        out_channels: Optional[int] = 256,
        num_layers: int = 16,
        
        image_width: int = 32,
        image_height: int = 32,
        image_patch_size: int = 2,

        motion_width: int = 4,
        motion_height: int = 4,
        motion_patch_size: int = 1,
        motion_frames: int = 15,

        flip_sin_to_cos: bool = True,
        freq_shift: int = 0,
        time_embed_dim: int = 512,

        dropout: float = 0.0,
        attention_bias: bool = True,
        temporal_compression_ratio: int = 4,

        activation_fn: str = "gelu-approximate",
        timestep_activation_fn: str = "silu",
        norm_elementwise_affine: bool = True,
        norm_eps: float = 1e-5,
        spatial_interpolation_scale: float = 1.0,
        temporal_interpolation_scale: float = 1.0,
    ):
        super().__init__()
        
        # 1. Setting
        hidden_dim = num_attention_heads * attention_head_dim

        iph = image_height // image_patch_size
        ipw = image_width // image_patch_size
        itl = iph * ipw # image token length

        mph = motion_height // motion_patch_size
        mpw = motion_width // motion_patch_size
        mtl = mph * mpw * motion_frames # motion token num
        
        self.image_patch_size = image_patch_size
        self.motion_patch_size = motion_patch_size
        self.out_channels = out_channels
        self.motion_frames = motion_frames

        # 2. Patch embedding (N,F,C,H,W) -> (B,S,D)
        self.refimg_pose_patch_embed = PatchEmbed(patch_size=image_patch_size,in_channels=refimg_in_channels*2,embed_dim=hidden_dim, bias=True)
        self.motion_patch_embed = PatchEmbed(patch_size=motion_patch_size,in_channels=motion_in_channels,embed_dim=hidden_dim, bias=True)


        self.extra_embed = nn.Linear(extra_in_channels,hidden_dim)
        self.embedding_dropout = nn.Dropout(dropout)

        # 3. 2D&3D positional embeddings
        image_pos_embedding = get_2d_sincos_pos_embed(hidden_dim, (iph, ipw)) # (iph*ipw,D)
        image_pos_embedding = torch.from_numpy(image_pos_embedding) # (iph*ipw,D)
        pos_embedding = torch.zeros(1, itl, hidden_dim, requires_grad=False)
        pos_embedding.data[:, :itl].copy_(image_pos_embedding)
        self.register_buffer("refimg_pos_embedding", pos_embedding, persistent=False)

        spatial_pos_embedding = get_3d_sincos_pos_embed(  
            hidden_dim,
            (mph, mph),
            motion_frames,
            spatial_interpolation_scale,
            temporal_interpolation_scale,
        ) 
        spatial_pos_embedding = torch.from_numpy(spatial_pos_embedding).flatten(0, 1) # [T*H*W, D]
        pos_embedding = torch.zeros(1,*spatial_pos_embedding.shape, requires_grad=False)
        pos_embedding.data.copy_(spatial_pos_embedding)
        self.register_buffer("motion_pos_embedding", pos_embedding, persistent=False)
            
        

        # 5. Time embeddings
        self.time_proj = Timesteps(hidden_dim, flip_sin_to_cos, freq_shift)
        self.time_embedding = TimestepEmbedding(hidden_dim, time_embed_dim, timestep_activation_fn)

        # 6. Define spatio-temporal transformers blocks
        self.transformer_blocks = nn.ModuleList(
            [
                TransformerBlock2Condition(
                    dim=hidden_dim,
                    num_attention_heads=num_attention_heads,
                    attention_head_dim=attention_head_dim,
                    time_embed_dim=time_embed_dim,
                    dropout=dropout,
                    activation_fn=activation_fn,
                    attention_bias=attention_bias,
                    norm_elementwise_affine=norm_elementwise_affine,
                    norm_eps=norm_eps,
                )
                for _ in range(num_layers)
            ]
        )

        self.norm_final = nn.LayerNorm(hidden_dim, norm_eps, norm_elementwise_affine)

        # 5. Output blocks
        self.norm_out = AdaLayerNorm(
            embedding_dim=time_embed_dim,
            output_dim=hidden_dim*2,
            norm_elementwise_affine=norm_elementwise_affine,
            norm_eps=norm_eps,
            chunk_dim=1,
        )
        self.proj_out = nn.Linear(hidden_dim, motion_patch_size * motion_patch_size * out_channels)

        self.gradient_checkpointing = False

    def _set_gradient_checkpointing(self, module, value=False):
        self.gradient_checkpointing = value

    def forward(
        self,
        motion_hidden_states: torch.Tensor,
        refimg_hidden_states: torch.Tensor,
        pose_hidden_states: torch.Tensor,
        extra_hidden_states: torch.Tensor,
        timestep: Union[int, float, torch.LongTensor], # Timesteps should be a 1d-array
        timestep_cond: Optional[torch.Tensor] = None,
        return_dict: bool = True,
    ):
        """
        motion_hidden_states : (N,F,D,h,w)   
        refimg_hidden_states : (N,C,H,W)  
        pose_hidden_states: (N,C,H,W) 
        extra_hidden_states  : (N,F,D)   这个需要提前做 position encoding if needed ,前面两个固定在这里做position encoding
        """
        assert motion_hidden_states.shape[1] == extra_hidden_states.shape[1]
        assert motion_hidden_states.shape[0] == refimg_hidden_states.shape[0]

        N,Ci,Hi,Wi = refimg_hidden_states.shape
        N,F,Cm,Hm,Wm = motion_hidden_states.shape
        refimg_seq_length = Hi * Wi // (self.image_patch_size**2)
        motion_seq_length = F *Hm * Wm // (self.motion_patch_size**2)
        
        # 1. Time embedding   # Timesteps should be a 1d-array
        t_emb = self.time_proj(timestep)
        t_emb = t_emb.to(dtype=motion_hidden_states.dtype)
        emb = self.time_embedding(t_emb, timestep_cond) # (batch,time_embed_dim) (8,512)

        # 2. Patch embedding
        motion_hidden_states = self.motion_patch_embed(motion_hidden_states) # [N,L1,D]
        ref_pose_hidden_states =torch.cat((refimg_hidden_states,pose_hidden_states),dim=1)
        ref_pose_hidden_states = self.refimg_pose_patch_embed(ref_pose_hidden_states) # [N,L2,D]
        extra_hidden_states  = self.extra_embed(extra_hidden_states) # [N,L3,D]

        # 3. Position embedding
        ref_pose_hidden_states = ref_pose_hidden_states + self.refimg_pos_embedding[:, :refimg_seq_length]
        motion_hidden_states = motion_hidden_states + self.motion_pos_embedding[:, :motion_seq_length]
        self.embedding_dropout(motion_hidden_states)

        # 4. Transformer blocks
        for i, block in enumerate(self.transformer_blocks):
            if self.training and self.gradient_checkpointing:

                def create_custom_forward(module):
                    def custom_forward(*inputs):
                        return module(*inputs)

                    return custom_forward

                ckpt_kwargs: Dict[str, Any] = {"use_reentrant": False} if is_torch_version(">=", "1.11.0") else {}
                motion_hidden_states,ref_pose_hidden_states,extra_hidden_states = torch.utils.checkpoint.checkpoint(
                    create_custom_forward(block),
                    motion_hidden_states,
                    ref_pose_hidden_states,
                    extra_hidden_states,
                    emb,
                    **ckpt_kwargs,
                )
            else:
                motion_hidden_states,ref_pose_hidden_states,extra_hidden_states = block(
                    motion_hidden_states,
                    ref_pose_hidden_states,
                    extra_hidden_states,
                    temb=emb,
                )

        motion_hidden_states = self.norm_final(motion_hidden_states)

        # 6. Final block
        motion_hidden_states = self.norm_out(motion_hidden_states, temb=emb)
        motion_hidden_states = self.proj_out(motion_hidden_states) # [N,L1,D]

        # 7. Unpatchify
        p = self.motion_patch_size
        output = motion_hidden_states.reshape(N*F, 1, Hm // p, Wm // p, self.out_channels, p, p)
        output = output.permute(0, 1, 4, 2, 5, 3, 6).flatten(5, 6).flatten(3, 4).squeeze(1) # [N,C,H,W]
        output = einops.rearrange(output,'(n f) c h w -> n f c h w',f=F)
        

        return output # N,F,C,H,W
