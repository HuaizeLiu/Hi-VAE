import torch
import torch.nn as nn
from typing import Optional, Union, Dict, Any
import einops
import pdb

from diffusers.utils import is_torch_version

from transformers import AutoModel
from diffusers import CogVideoXPipeline

from diffusers import CogVideoXPipeline, CogVideoXDDIMScheduler

from diffusers.models.attention import Attention,FeedForward

from .modules import (BasicTransformerBlock,
                      PatchEmbed,
                      AMDTransformerBlock,
                      AdaLayerNorm,
                      TransformerBlock2Condition,
                      TransformerBlock2Condition_SimpleAdaLN,
                      A2MMotionSelfAttnBlock,
                      A2MCrossAttnBlock,
                      A2PTemporalSpatialBlock,
                      A2PCrossAudioBlock,
                      AMDTransformerMotionBlock,
                      BasicDiTBlock,
                      BasicCrossTransformerBlock)
from diffusers.models.embeddings import TimestepEmbedding, Timesteps, get_3d_sincos_pos_embed,get_2d_sincos_pos_embed,get_1d_sincos_pos_embed_from_grid
from diffusers.configuration_utils import ConfigMixin, register_to_config
from diffusers.utils import BaseOutput, logging
from diffusers.models.embeddings import TimestepEmbedding, Timesteps
from diffusers.models.modeling_utils import ModelMixin

# ------------ motion encoder ---------------
class MotionEncoderLearnTokenTransformer(nn.Module):
    r"""
        Motion Encoder With Learnable Token
    """

    def __init__(
        self,
        # ----- img
        img_height: int = 32,
        img_width: int = 32,
        img_inchannel: int = 4,
        img_patch_size: int = 2,
        # ----- motion
        motion_token_num:int = 12,
        motion_channel:int = 128,
        need_norm_out :bool = True,
        # ----- attention
        num_attention_heads: int = 12,
        attention_head_dim: int = 64,
        num_layers: int = 8,
        freq_shift: int = 0,       
        dropout: float = 0.0,
        attention_bias: bool = True,
        activation_fn: str = "gelu-approximate",
        norm_elementwise_affine: bool = True,
        norm_eps: float = 1e-5,
        spatial_interpolation_scale: float = 1.0,
        temporal_interpolation_scale: float = 1.0,
    ):
        super().__init__()

        # setting
        hidden_dim = num_attention_heads * attention_head_dim
        iph = img_height // img_patch_size
        ipw = img_width // img_patch_size
        itl = iph * ipw
        self.img_token_len = itl

        # motion token
        INIT_CONST = 0.02
        self.motion_token = nn.Parameter(torch.randn(1, motion_token_num, motion_channel) * INIT_CONST)
        self.motion_embed = nn.Linear(motion_channel,hidden_dim)
        
        # img embedding        
        self.patch_embed = PatchEmbed(img_patch_size,img_inchannel,hidden_dim)
        self.embedding_dropout = nn.Dropout(dropout)

        # 2D positional embeddings
        image_pos_embedding = get_2d_sincos_pos_embed(hidden_dim, (iph, ipw)) # (iph*ipw,D)
        image_pos_embedding = torch.from_numpy(image_pos_embedding) # (iph*ipw,D)
        pos_embedding = torch.zeros(1, itl, hidden_dim, requires_grad=False)
        pos_embedding.data[:, :itl].copy_(image_pos_embedding)
        self.register_buffer("pos_embedding", pos_embedding, persistent=False) # (1,itl,hidden_dim)

        # transformers blocks
        self.transformer_blocks = nn.ModuleList(
            [
                BasicTransformerBlock(
                    dim=hidden_dim,
                    num_attention_heads=num_attention_heads,
                    attention_head_dim=attention_head_dim,
                    dropout=dropout,
                    activation_fn=activation_fn,
                    attention_bias=attention_bias,
                    norm_elementwise_affine=norm_elementwise_affine,
                    norm_eps=norm_eps,
                )
                for _ in range(num_layers)
            ]
        )

        # Output blocks
        self.norm_final = nn.LayerNorm(hidden_dim, eps=norm_eps, elementwise_affine=norm_elementwise_affine)
        self.proj_out = nn.Linear(hidden_dim, motion_channel)
        self.need_norm_out = need_norm_out
        if self.need_norm_out:
            self.norm_out = nn.LayerNorm(motion_channel, eps=norm_eps, elementwise_affine=False)

        self.gradient_checkpointing = False

    def forward(
        self,
        img_hidden_states: torch.Tensor, # N,T,C,H,W | ref_frame + frames
        mask_ratio = None,
    ):
        N,T,C,H,W = img_hidden_states.shape

        # motion token
        motion_token = self.motion_embed(self.motion_token) # (1,motion_token_num,D)
        motion_token = motion_token.repeat(N*T,1,1) # (NT,motion_token_num,D)

        # img token
        img_hidden_states = einops.rearrange(img_hidden_states, 'n t c h w -> (n t) c h w')
        img_hidden_states = self.patch_embed(img_hidden_states) # [NT,hw,D]
        assert self.img_token_len == img_hidden_states.shape[1] , 'img_token_len should be equal!'

        # img position encoding
        pos_embeds = self.pos_embedding[:, :self.img_token_len] # [1,hw, D]
        img_hidden_states = img_hidden_states + pos_embeds
        img_hidden_states = self.embedding_dropout(img_hidden_states)

        # random mask
        if mask_ratio is not None:
            img_hidden_states,_,_ = self.random_masking(img_hidden_states,mask_ratio)



        # cat
        hidden_states = torch.cat([motion_token,img_hidden_states],dim=1) # [NT,token_m + token_i ,D]

        # Transformer blocks
        for i, block in enumerate(self.transformer_blocks):
            hidden_states = block(
                hidden_states=hidden_states,
            )

        # Final block
        motion_token = hidden_states[:, :motion_token.shape[1],:] # [NT,motion_token_num,D]
        motion_token = self.norm_final(motion_token) # [NT,motion_token_num ,D]
        motion_token = self.proj_out(motion_token) # [NT,motion_token_num ,motion_channel]
        if self.need_norm_out:
            motion_token = self.norm_out(motion_token)

        # Unpatchify
        motion_token = einops.rearrange(motion_token, '(n t) l d -> n t l d',n=N)  # [N,T,motion_token_num,motion_channel]

        return motion_token # [N,T,motion_token_num,motion_channel]

    def random_masking(self, x, mask_ratio):
        """
        Perform per-sample random masking by per-sample shuffling.
        Per-sample shuffling is done by argsort random noise.
        x: [N, L, D], sequence
        """
        N, L, D = x.shape  # batch, length, dim
        len_keep = int(L * (1 - mask_ratio)) # L*0.25
        
        noise = torch.rand(N, L, device=x.device)  # noise in [0, 1]
        
        # sort noise for each sample
        # [0.3,0.5,0.9,0.1,0.6] -> [3,0,1,4,2] from small to large
        ids_shuffle = torch.argsort(noise, dim=1)  # ascend: small is keep, large is remove , Smallest row front
        # [3,0,1,4,2] -> [1,2,4,0,3] restore the original order
        ids_restore = torch.argsort(ids_shuffle, dim=1) # (N,L)

        # keep the first subset
        ids_keep = ids_shuffle[:, :len_keep] # (N, len_keep) e.g. [3,0,1,4,2] -> [3,0,1]
        # index (N,len_keep,D)  x_masked (N,len_keep,D)
        x_masked = torch.gather(x, dim=1, index=ids_keep.unsqueeze(-1).repeat(1, 1, D))

        # generate the binary mask: 0 is keep, 1 is remove
        mask = torch.ones([N, L], device=x.device)
        mask[:, :len_keep] = 0
        # unshuffle to get the binary mask
        mask = torch.gather(mask, dim=1, index=ids_restore) # (N,L) 0 is keep, 1 is remove

        return x_masked, mask, ids_restore # (N,len_keep,D), (N,L), (N,L)


class MotionEncoderLearnTokenTemporalTransformer(nn.Module):
    r"""
        Motion Encoder With Learnable Token
    """

    def __init__(
        self,
        # ----- img
        img_height: int = 32,
        img_width: int = 32,
        img_inchannel: int = 4,
        img_patch_size: int = 2,
        # ----- motion
        motion_token_num:int = 12,
        motion_channel:int = 128,
        need_norm_out :bool = True,
        video_frames:int = 16,
        # ----- attention
        num_attention_heads: int = 12,
        attention_head_dim: int = 64,
        num_layers: int = 8,
        freq_shift: int = 0,       
        dropout: float = 0.0,
        attention_bias: bool = True,
        activation_fn: str = "gelu-approximate",
        norm_elementwise_affine: bool = True,
        norm_eps: float = 1e-5,
        spatial_interpolation_scale: float = 1.0,
        temporal_interpolation_scale: float = 1.0,
    ):
        super().__init__()

        # setting
        hidden_dim = num_attention_heads * attention_head_dim
        iph = img_height // img_patch_size
        ipw = img_width // img_patch_size
        itl = iph * ipw
        self.img_token_len = itl
        self.video_frames = video_frames

        # motion token
        INIT_CONST = 0.02
        self.motion_token = nn.Parameter(torch.randn(1, motion_token_num, motion_channel) * INIT_CONST)
        self.motion_embed = nn.Linear(motion_channel,hidden_dim)
        
        # img embedding        
        self.patch_embed = PatchEmbed(img_patch_size,img_inchannel,hidden_dim)
        self.embedding_dropout = nn.Dropout(dropout)

        # 2D positional embeddings
        image_pos_embedding = get_2d_sincos_pos_embed(hidden_dim, (iph, ipw)) # (iph*ipw,D)
        image_pos_embedding = torch.from_numpy(image_pos_embedding) # (iph*ipw,D)
        pos_embedding = torch.zeros(1, itl, hidden_dim, requires_grad=False)
        pos_embedding.data[:, :itl].copy_(image_pos_embedding)
        self.register_buffer("pos_embedding", pos_embedding, persistent=False) # (1,itl,hidden_dim)

        temporal_embedding = get_1d_sincos_pos_embed_from_grid(hidden_dim,torch.arange(self.video_frames * motion_token_num)) # ref img and 
        motion_temporal_embedding = torch.zeros(1,*temporal_embedding.shape,requires_grad=False)
        motion_temporal_embedding.data.copy_(torch.from_numpy(temporal_embedding))
        self.register_buffer("motion_temporal_embedding",motion_temporal_embedding,persistent=False)

        # transformers blocks
        self.transformer_blocks = nn.ModuleList(
            [
                BasicTransformerBlock(
                    dim=hidden_dim,
                    num_attention_heads=num_attention_heads,
                    attention_head_dim=attention_head_dim,
                    dropout=dropout,
                    activation_fn=activation_fn,
                    attention_bias=attention_bias,
                    norm_elementwise_affine=norm_elementwise_affine,
                    norm_eps=norm_eps,
                )
                for _ in range(num_layers)
            ]
        )

        self.motion_blocks = nn.ModuleList(
            [ 
                AMDTransformerMotionBlock(
                    dim=hidden_dim,
                    num_attention_heads=num_attention_heads,
                    attention_head_dim=attention_head_dim,
                    dropout=dropout,
                    activation_fn=activation_fn,
                    attention_bias=attention_bias,
                    norm_elementwise_affine=norm_elementwise_affine,
                    norm_eps=norm_eps,
                )
                for _ in range(num_layers)
            ]
        )

        # Output blocks
        self.norm_final = nn.LayerNorm(hidden_dim, eps=norm_eps, elementwise_affine=norm_elementwise_affine)
        self.proj_out = nn.Linear(hidden_dim, motion_channel)
        self.need_norm_out = need_norm_out
        if self.need_norm_out:
            self.norm_out = nn.LayerNorm(motion_channel, eps=norm_eps, elementwise_affine=False)

        self.gradient_checkpointing = False

    def forward(
        self,
        img_hidden_states: torch.Tensor, # N,T,C,H,W | ref_frame + frames
        mask_ratio = None,
    ):
        N,T,C,H,W = img_hidden_states.shape

        # motion token
        motion_token = self.motion_embed(self.motion_token) # (1,motion_token_num,D)
        motion_token = motion_token.repeat(N,T,1,1) # (NT,motion_token_num,D)
        source_token, target_token = motion_token.chunk(2,dim=1)
        n, t, l, d = target_token.shape
        target_token = einops.rearrange(target_token,'n t l d -> n (t l) d') # n,(t*l),d
        target_token = target_token[:,:] + self.motion_temporal_embedding[:,:(T//2)*l] # n,(t*l),d
        target_token = einops.rearrange(target_token,'n (t l) d -> n t l d',t=t)
        motion_token = torch.cat([source_token,target_token],dim=1)
        motion_token = einops.rearrange(motion_token,'n t l d -> (n t) l d')

        # img token
        img_hidden_states = einops.rearrange(img_hidden_states, 'n t c h w -> (n t) c h w')
        img_hidden_states = self.patch_embed(img_hidden_states) # [NT,hw,D]
        assert self.img_token_len == img_hidden_states.shape[1] , 'img_token_len should be equal!'

        # img position encoding
        pos_embeds = self.pos_embedding[:, :self.img_token_len] # [1,hw, D]
        img_hidden_states = img_hidden_states + pos_embeds
        img_hidden_states = self.embedding_dropout(img_hidden_states)


        # random mask
        if mask_ratio is not None:
            img_hidden_states,_,_ = self.random_masking(img_hidden_states,mask_ratio)

        # cat
        hidden_states = torch.cat([motion_token,img_hidden_states],dim=1) # [NT,token_m + token_i ,D]

        # Transformer blocks
        for i, (block, m_block) in enumerate(zip(self.transformer_blocks,self.motion_blocks)):
            
            hidden_states = block(
                hidden_states=hidden_states,
            )
            
            motion_token = hidden_states[:, :motion_token.shape[1],:]
            source_token, target_token = motion_token.chunk(2,dim=0)

            img_hidden_states = hidden_states[:, motion_token.shape[1]:, :]

            target_token = einops.rearrange(target_token,'(n t) l d -> (n l) t d',n=n) # n,(t*l),d
            
            target_token = m_block(
                hidden_states=target_token,
            )

            target_token = einops.rearrange(target_token,'(n l) t d -> (n t) l d',n=n,t=t)
            motion_token = torch.cat([source_token,target_token],dim=0)
            hidden_states = torch.cat([motion_token,img_hidden_states],dim=1) # [NT,token_m + token_i ,D]


        # Final block
        motion_token = hidden_states[:, :motion_token.shape[1],:] # [NT,motion_token_num,D]
        motion_token = self.norm_final(motion_token) # [NT,motion_token_num ,D]
        motion_token = self.proj_out(motion_token) # [NT,motion_token_num ,motion_channel]
        if self.need_norm_out:
            motion_token = self.norm_out(motion_token)

        # Unpatchify
        motion_token = einops.rearrange(motion_token, '(n t) l d -> n t l d',n=N)  # [N,T,motion_token_num,motion_channel]

        return motion_token # [N,T,motion_token_num,motion_channel]

    def random_masking(self, x, mask_ratio):
        """
        Perform per-sample random masking by per-sample shuffling.
        Per-sample shuffling is done by argsort random noise.
        x: [N, L, D], sequence
        """
        N, L, D = x.shape  # batch, length, dim
        len_keep = int(L * (1 - mask_ratio)) # L*0.25
        
        noise = torch.rand(N, L, device=x.device)  # noise in [0, 1]
        
        # sort noise for each sample
        # [0.3,0.5,0.9,0.1,0.6] -> [3,0,1,4,2] from small to large
        ids_shuffle = torch.argsort(noise, dim=1)  # ascend: small is keep, large is remove , Smallest row front
        # [3,0,1,4,2] -> [1,2,4,0,3] restore the original order
        ids_restore = torch.argsort(ids_shuffle, dim=1) # (N,L)

        # keep the first subset
        ids_keep = ids_shuffle[:, :len_keep] # (N, len_keep) e.g. [3,0,1,4,2] -> [3,0,1]
        # index (N,len_keep,D)  x_masked (N,len_keep,D)
        x_masked = torch.gather(x, dim=1, index=ids_keep.unsqueeze(-1).repeat(1, 1, D))

        # generate the binary mask: 0 is keep, 1 is remove
        mask = torch.ones([N, L], device=x.device)
        mask[:, :len_keep] = 0
        # unshuffle to get the binary mask
        mask = torch.gather(mask, dim=1, index=ids_restore) # (N,L) 0 is keep, 1 is remove

        return x_masked, mask, ids_restore # (N,len_keep,D), (N,L), (N,L)


'''
    只处理时序信息,进行时序编码,适用于camera运动的提取
    只需要一个learnable token,学习时序运动,模糊外观信息
    也不要和object相加,因为motion信息不一样,最好就concat或者分层注入
'''

class MotionEncoderLearnTokenOnlyTemporalTransformer(nn.Module):
    r"""
        Motion Encoder With Learnable Token
    """

    def __init__(
        self,
        # ----- img
        img_height: int = 32,
        img_width: int = 32,
        img_inchannel: int = 4,
        img_patch_size: int = 2,
        # ----- motion
        motion_token_num:int = 12,
        motion_channel:int = 128,
        need_norm_out :bool = True,
        video_frames:int = 16,
        # ----- attention
        num_attention_heads: int = 12,
        attention_head_dim: int = 64,
        num_layers: int = 8,
        freq_shift: int = 0,       
        dropout: float = 0.0,
        attention_bias: bool = True,
        activation_fn: str = "gelu-approximate",
        norm_elementwise_affine: bool = True,
        norm_eps: float = 1e-5,
        spatial_interpolation_scale: float = 1.0,
        temporal_interpolation_scale: float = 1.0,
    ):
        super().__init__()

        # setting
        hidden_dim = num_attention_heads * attention_head_dim
        iph = img_height // img_patch_size
        ipw = img_width // img_patch_size
        itl = iph * ipw
        self.img_token_len = itl
        self.video_frames = video_frames

        # motion token
        INIT_CONST = 0.02
        self.motion_token = nn.Parameter(torch.randn(1, motion_token_num, motion_channel) * INIT_CONST)
        self.motion_embed = nn.Linear(motion_channel,hidden_dim)
        
        # img embedding        
        self.patch_embed = PatchEmbed(img_patch_size,img_inchannel,hidden_dim)
        self.embedding_dropout = nn.Dropout(dropout)

        # 2D positional embeddings
        image_pos_embedding = get_2d_sincos_pos_embed(hidden_dim, (iph, ipw)) # (iph*ipw,D)
        image_pos_embedding = torch.from_numpy(image_pos_embedding) # (iph*ipw,D)
        pos_embedding = torch.zeros(1, itl, hidden_dim, requires_grad=False)
        pos_embedding.data[:, :itl].copy_(image_pos_embedding)
        self.register_buffer("pos_embedding", pos_embedding, persistent=False) # (1,itl,hidden_dim)

        temporal_embedding = get_1d_sincos_pos_embed_from_grid(hidden_dim,torch.arange(self.video_frames)) # ref img and 
        motion_temporal_embedding = torch.zeros(1,*temporal_embedding.shape,requires_grad=False)
        motion_temporal_embedding.data.copy_(torch.from_numpy(temporal_embedding))
        self.register_buffer("motion_temporal_embedding",motion_temporal_embedding,persistent=False)

        # transformers blocks
        self.transformer_blocks = nn.ModuleList(
            [
                BasicCrossTransformerBlock(
                    dim=hidden_dim, # Q motion token channel
                    cross_dim=hidden_dim, # KV img_hidden_state channel
                    num_attention_heads=num_attention_heads,
                    attention_head_dim=attention_head_dim,
                    dropout=dropout,
                    activation_fn=activation_fn,
                    attention_bias=attention_bias,
                    norm_elementwise_affine=norm_elementwise_affine,
                    norm_eps=norm_eps,
                )
                for _ in range(num_layers)
            ]
        )


        # Output blocks
        self.norm_final = nn.LayerNorm(hidden_dim, eps=norm_eps, elementwise_affine=norm_elementwise_affine)
        self.proj_out = nn.Linear(hidden_dim, motion_channel)
        self.need_norm_out = need_norm_out
        if self.need_norm_out:
            self.norm_out = nn.LayerNorm(motion_channel, eps=norm_eps, elementwise_affine=False)

        self.gradient_checkpointing = False

    def forward(
        self,
        img_hidden_states: torch.Tensor, # N,T,C,H,W | ref_frame + frames
        mask_ratio = None,
    ):
        N,T,C,H,W = img_hidden_states.shape

        # img token
        img_hidden_states = einops.rearrange(img_hidden_states, 'n t c h w -> (n t) c h w')
        img_hidden_states = self.patch_embed(img_hidden_states) # [NT,hw // 4,D]
        assert self.img_token_len == img_hidden_states.shape[1] , 'img_token_len should be equal!'

        # img spaical position embedding
        pos_embeds = self.pos_embedding[:, :self.img_token_len] # [1,hw//4, D]
        img_hidden_states = img_hidden_states + pos_embeds
        # img_hidden_states = self.embedding_dropout(img_hidden_states)

        # img temporal position embedding
        img_hidden_states = einops.rearrange(img_hidden_states, '(n t) (h w) c -> (n h w) t c', n=N, h=H//2)
        img_hidden_states = img_hidden_states[:,:] + self.motion_temporal_embedding[:,:T]
        img_hidden_states = einops.rearrange(img_hidden_states, '(n h w) t c -> n t (h w) c', n=N, h=H//2)

        # random mask,
        if mask_ratio is not None:
            img_hidden_states = self.random_masking(img_hidden_states,mask_ratio)

        # motion token
        _,Tm,Sm,_ = img_hidden_states.shape
        n,t,c = self.motion_token.shape
        # pdb.set_trace()
        # print("self.motion_token.shape",self.motion_token.shape)
        # print("img_hidden_states.shape",img_hidden_states.shape)
        motion_token = self.motion_embed(self.motion_token).unsqueeze(0) # (1,1,T,D)
        # motion_token = motion_token.repeat(N, Sm, Tm//t, 1) # (N,H*W,T,D)
        motion_token = motion_token.repeat(N, Sm, 1, 1) # (N,H*W,t,D)
        if t != Tm:
            motion_token = motion_token.repeat_interleave(Tm//t, dim=2) # (N,H*W,T,D)
        # print("T Sm Tm t",T,Sm, Tm, t)
        motion_token = einops.rearrange(motion_token, 'n s t c -> (n s) t c')
        motion_token = motion_token[:,:] + self.motion_temporal_embedding[:,:T] # n*h*w, t, d

        img_hidden_states = einops.rearrange(img_hidden_states, 'n t s c -> (n s) t c')
        # Transformer blocks
        for i, block in enumerate(self.transformer_blocks):
            
            motion_token = block(
                hidden_states=motion_token,
                encoder_hidden_state=img_hidden_states
            )


        # Final block
        motion_token = self.norm_final(motion_token) # [NS, T, D]
        motion_token = self.proj_out(motion_token) # [NS, T, motion_channel]
        if self.need_norm_out:
            motion_token = self.norm_out(motion_token)

        # s=h*w
        # Unpatchify
        motion_token = einops.rearrange(motion_token, '(n s) t c -> n t s c',n=N)  # [N,T,motion_token_num,motion_channel]

        return motion_token # [N,T,motion_token_num,motion_channel]

    def random_masking(self, x, mask_ratio):
        """
        Perform per-sample random masking by per-sample shuffling.
        Per-sample shuffling is done by argsort random noise.
        x: [N, L, D], sequence
        """
        N, T, L, D = x.shape  # batch, length, dim
        len_keep = int(L * (1 - mask_ratio)) # L*0.25
        
        noise = torch.rand(N, L, device=x.device)  # noise in [0, 1]
        
        # sort noise for each sample
        # [0.3,0.5,0.9,0.1,0.6] -> [3,0,1,4,2] from small to large
        ids_shuffle = torch.argsort(noise, dim=1)  # ascend: small is keep, large is remove , Smallest row front
        # [3,0,1,4,2] -> [1,2,4,0,3] restore the original order
        ids_restore = torch.argsort(ids_shuffle, dim=1) # (N,L)

        # keep the first subset
        ids_keep = ids_shuffle[:, :len_keep] # (N, len_keep) e.g. [3,0,1,4,2] -> [3,0,1]
        # index (N,len_keep,D)  x_masked (N,len_keep,D)
        x_masked = torch.gather(x, dim=2, index=ids_keep.unsqueeze(-1).unsqueeze(1).repeat(1, T, 1, D))

        # generate the binary mask: 0 is keep, 1 is remove
        # mask = torch.ones([N, L], device=x.device)
        # mask[:, :len_keep] = 0
        # # unshuffle to get the binary mask
        # mask = torch.gather(mask, dim=1, index=ids_restore) # (N,L) 0 is keep, 1 is remove

        return x_masked # (N,T,len_keep,D)


# ------------ motion transformer ---------------
class MotionTransformer(nn.Module):
    """
    Motion 
    """

    _supports_gradient_checkpointing = True

    def __init__(
        self,
        motion_token_num : int = 4,
        motion_token_channel : int = 128,
        motion_frames : int = 128,
        attention_head_dim : int = 64,
        num_attention_heads: int = 16,
        num_layers: int = 8,
        
        freq_shift: int = 0,       
        dropout: float = 0.0,
        attention_bias: bool = True,
        activation_fn: str = "gelu-approximate",
        norm_elementwise_affine: bool = True,
        norm_eps: float = 1e-5,
        spatial_interpolation_scale: float = 1.0,
        temporal_interpolation_scale: float = 1.0,
    ):
        super().__init__()
        
        # 1. Setting
        hidden_dim = num_attention_heads * attention_head_dim
        self.out_channels = motion_token_channel
        self.motion_token_length = motion_token_num * motion_frames
        
        # 1. Patch embedding
        self.embed = nn.Linear(motion_token_channel,hidden_dim)
        self.embedding_dropout = nn.Dropout(dropout)

        # 2. 1D positional embeddings
        temporal_embedding = get_1d_sincos_pos_embed_from_grid(hidden_dim,torch.arange(self.motion_token_length)) 
        motion_pos_embedding = torch.zeros(1,*temporal_embedding.shape,requires_grad=False)
        motion_pos_embedding.data.copy_(torch.from_numpy(temporal_embedding))
        self.register_buffer("motion_pos_embedding",motion_pos_embedding,persistent=False)

        # 3. Define spatio-temporal transformers blocks
        self.transformer_blocks = nn.ModuleList(
            [
                BasicTransformerBlock(
                    dim=hidden_dim,
                    num_attention_heads=num_attention_heads,
                    attention_head_dim=attention_head_dim,
                    dropout=dropout,
                    activation_fn=activation_fn,
                    attention_bias=attention_bias,
                    norm_elementwise_affine=norm_elementwise_affine,
                    norm_eps=norm_eps,
                )
                for _ in range(num_layers)
            ]
        )
        self.norm_final = nn.LayerNorm(hidden_dim, eps=norm_eps, elementwise_affine=norm_elementwise_affine)

        # 4. Output blocks
        self.proj_out = nn.Linear(hidden_dim,motion_token_channel)

        self.gradient_checkpointing = False

    def _set_gradient_checkpointing(self, module, value=False):
        self.gradient_checkpointing = value

    def forward(
        self,
        hidden_states: torch.Tensor, # N,F,L,D
    ):
        N,F,L,D = hidden_states.shape

        # 2. Patch embedding
        hidden_states = self.embed(hidden_states) # N,F,L,D

        # 3. Position embedding
        hidden_states = hidden_states.flatten(1,2) + self.motion_pos_embedding[:,:F*L,:] # N,FL,D

        # 5. Transformer blocks
        for i, block in enumerate(self.transformer_blocks):
            if self.training and self.gradient_checkpointing:

                def create_custom_forward(module):
                    def custom_forward(*inputs):
                        return module(*inputs)

                    return custom_forward

                ckpt_kwargs: Dict[str, Any] = {"use_reentrant": False} if is_torch_version(">=", "1.11.0") else {}
                hidden_states = torch.utils.checkpoint.checkpoint(
                    create_custom_forward(block),
                    hidden_states,
                    **ckpt_kwargs,
                )
            else:
                hidden_states = block(
                    hidden_states=hidden_states,
                )

        hidden_states = self.norm_final(hidden_states) # N,FL,D

        # 6. Final block
        hidden_states = self.proj_out(hidden_states) # N,FL,D

        # 7. Unpatchify
        hidden_states = einops.rearrange(hidden_states,'n (f l) d -> n f l d',f=F)

        return hidden_states # N,F,L,D


# ------------ diffusion ---------------
class AMDReconstructTransformerModel(nn.Module):
    """
    Diffusion Transformer
    """

    _supports_gradient_checkpointing = True
    def __init__(
        self,
        num_attention_heads: int = 20,
        attention_head_dim: int = 64,
        out_channels: Optional[int] = 4,
        num_layers: int = 12,
        # ----- img
        image_width: int = 32,
        image_height: int = 32,
        image_patch_size: int = 2,
        image_in_channels: Optional[int] = 4, 
        # ----- motion
        motion_token_num:int = 12,
        motion_in_channels: Optional[int] = 128,

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
        """
        
        Traning:
        Z(N,1,C,H,W)
        Motion(N,k,d,h,w)
        
        Inference:
        Z(N,1,C,H,W)
        Motion(N,k,d,h,w)
        """
        super().__init__()
        
        # 1. Setting
        hidden_dim = num_attention_heads * attention_head_dim
        iph = image_height // image_patch_size
        ipw = image_width // image_patch_size
        itl = iph * ipw # image token length
        self.image_patch_size = image_patch_size
        self.out_channels = out_channels

        # Patch embedding (N,F,C,H,W) -> (B,S,D)
        self.image_patch_embed = PatchEmbed(patch_size=image_patch_size,in_channels=image_in_channels,embed_dim=hidden_dim, bias=True)
        self.motion_patch_embed = nn.Linear(motion_in_channels,hidden_dim)
        self.embedding_dropout = nn.Dropout(dropout)

        # 2D positional embeddings
        image_pos_embedding = get_2d_sincos_pos_embed(hidden_dim, (iph, ipw)) # (iph*ipw,D)
        image_pos_embedding = torch.from_numpy(image_pos_embedding) # (iph*ipw,D)
        pos_embedding = torch.zeros(1, itl, hidden_dim, requires_grad=False)
        pos_embedding.data[:, :itl].copy_(image_pos_embedding)
        self.register_buffer("pos_embedding", pos_embedding, persistent=False)

        # 1D position encoding
        temporal_embedding = get_1d_sincos_pos_embed_from_grid(hidden_dim,torch.arange(2+2*motion_token_num)) 
        motion_pos_embedding = torch.zeros(1,*temporal_embedding.shape,requires_grad=False)
        motion_pos_embedding.data.copy_(torch.from_numpy(temporal_embedding))
        self.register_buffer("motion_pos_embedding",motion_pos_embedding,persistent=False)
        
        # Split Token
        self.source_token = nn.Parameter(torch.zeros(1, 1, hidden_dim),requires_grad=True)
        self.target_token = nn.Parameter(torch.zeros(1, 1, hidden_dim),requires_grad=True)

        # Define spatio-temporal transformers blocks
        self.transformer_blocks = nn.ModuleList(
            [ 
                BasicTransformerBlock(
                    dim=hidden_dim,
                    num_attention_heads=num_attention_heads,
                    attention_head_dim=attention_head_dim,
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

        # Output blocks
        self.proj_out = nn.Linear(hidden_dim, image_patch_size * image_patch_size * out_channels)

        self.gradient_checkpointing = False

    def _set_gradient_checkpointing(self, module, value=False):
        self.gradient_checkpointing = value

    def forward(
        self,
        motion_source_hidden_states: torch.Tensor,
        motion_target_hidden_states: torch.Tensor,
        image_hidden_states: torch.Tensor,
    ):
        """
        motion_hidden_states : (b,d,h,w)
        image_hidden_states : (b,2c,H,W)
        time_step : (b,)

        """
        N,Ci,Hi,Wi = image_hidden_states.shape
        N,L,Cm = motion_source_hidden_states.shape
        image_seq_length = Hi * Wi // (self.image_patch_size**2)
        motion_seq_length = 2*L + 2

        # Patch embedding
        motion_source_hidden_states = self.motion_patch_embed(motion_source_hidden_states) # [N,S,D]
        motion_target_hidden_states = self.motion_patch_embed(motion_target_hidden_states) # [N,S,D]
        image_hidden_states = self.image_patch_embed(image_hidden_states)

        # cat
        source_token = self.source_token.repeat(N, 1, 1)
        target_token = self.target_token.repeat(N, 1, 1)
        motion_hidden_states = torch.cat([source_token,motion_source_hidden_states,target_token,motion_target_hidden_states],dim=1) # [N,2S+2,D]

        # Position embedding
        motion_hidden_states = motion_hidden_states + self.motion_pos_embedding[:, :motion_seq_length]
        image_hidden_states = image_hidden_states + self.pos_embedding[:, :image_seq_length]
        self.embedding_dropout(motion_hidden_states)

        # Transformer blocks
        hidden_states = torch.cat([image_hidden_states,motion_hidden_states],dim=1)
        for i, block in enumerate(self.transformer_blocks):
            hidden_states = block(
                hidden_states=hidden_states,
            )

        image_hidden_states = self.norm_final(hidden_states[:,:image_hidden_states.shape[1],:])

        # 6. Final block
        image_hidden_states = self.proj_out(image_hidden_states)

        # 7. Unpatchify
        p = self.image_patch_size
        output = image_hidden_states.reshape(N, 1, Hi // p, Wi // p, self.out_channels, p, p)
        output = output.permute(0, 1, 4, 2, 5, 3, 6).flatten(5, 6).flatten(3, 4).squeeze(1) # [N,C,H,W]

        return output

class AMDReconstructSplitTransformerModel(nn.Module):
    """
    Diffusion Transformer
    """

    _supports_gradient_checkpointing = True
    def __init__(
        self,
        num_attention_heads: int = 20,
        attention_head_dim: int = 64,
        out_channels: Optional[int] = 4,
        num_layers: int = 12,
        # ----- img
        image_width: int = 32,
        image_height: int = 32,
        image_patch_size: int = 2,
        image_in_channels: Optional[int] = 4, 
        # ----- motion
        motion_token_num:int = 12,
        motion_in_channels: Optional[int] = 128,

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
        """
        
        Traning:
        Z(N,1,C,H,W)
        Motion(N,k,d,h,w)
        
        Inference:
        Z(N,1,C,H,W)
        Motion(N,k,d,h,w)
        """
        super().__init__()
        
        # 1. Setting
        hidden_dim = num_attention_heads * attention_head_dim
        iph = image_height // image_patch_size
        ipw = image_width // image_patch_size
        itl = iph * ipw # image token length
        self.image_patch_size = image_patch_size
        self.out_channels = out_channels

        # Patch embedding (N,F,C,H,W) -> (B,S,D)
        self.zi_image_patch_embed = PatchEmbed(patch_size=image_patch_size,in_channels=image_in_channels//2,embed_dim=hidden_dim, bias=True)
        self.zt_image_patch_embed = PatchEmbed(patch_size=image_patch_size,in_channels=image_in_channels//2,embed_dim=hidden_dim, bias=True)
        self.motion_patch_embed = nn.Linear(motion_in_channels,hidden_dim)
        self.embedding_dropout = nn.Dropout(dropout)

        # 2D positional embeddings
        image_pos_embedding = get_2d_sincos_pos_embed(hidden_dim, (iph, ipw)) # (iph*ipw,D)
        image_pos_embedding = torch.from_numpy(image_pos_embedding) # (iph*ipw,D)
        pos_embedding = torch.zeros(1, itl, hidden_dim, requires_grad=False)
        pos_embedding.data[:, :itl].copy_(image_pos_embedding)
        self.register_buffer("pos_embedding", pos_embedding, persistent=False)

        # 1D position encoding
        temporal_embedding = get_1d_sincos_pos_embed_from_grid(hidden_dim,torch.arange(2+2*motion_token_num)) 
        motion_pos_embedding = torch.zeros(1,*temporal_embedding.shape,requires_grad=False)
        motion_pos_embedding.data.copy_(torch.from_numpy(temporal_embedding))
        self.register_buffer("motion_pos_embedding",motion_pos_embedding,persistent=False)
        
        # Split Token
        self.source_token = nn.Parameter(torch.zeros(1, 1, hidden_dim),requires_grad=True)
        self.target_token = nn.Parameter(torch.zeros(1, 1, hidden_dim),requires_grad=True)

        # Define spatio-temporal transformers blocks
        self.transformer_blocks = nn.ModuleList(
            [ 
                BasicTransformerBlock(
                    dim=hidden_dim,
                    num_attention_heads=num_attention_heads,
                    attention_head_dim=attention_head_dim,
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

        # Output blocks
        self.proj_out = nn.Linear(hidden_dim, image_patch_size * image_patch_size * out_channels)

        self.gradient_checkpointing = False

    def _set_gradient_checkpointing(self, module, value=False):
        self.gradient_checkpointing = value

    def forward(
        self,
        motion_source_hidden_states: torch.Tensor,
        motion_target_hidden_states: torch.Tensor,
        image_hidden_states: torch.Tensor, 
    ):
        """
        motion_hidden_states : (b,d,h,w)
        image_hidden_states : (b,2c,H,W)
        time_step : (b,)
        """
        N,Ci,Hi,Wi = image_hidden_states.shape
        N,L,Cm = motion_source_hidden_states.shape
        image_seq_length = Hi * Wi // (self.image_patch_size**2)
        motion_seq_length = 2*L + 2

        # Patch embedding
        motion_source_hidden_states = self.motion_patch_embed(motion_source_hidden_states) # [N,S,D]
        motion_target_hidden_states = self.motion_patch_embed(motion_target_hidden_states) # [N,S,D]

        zi_image_hidden_states = self.zi_image_patch_embed(image_hidden_states[:,:Ci//2,:,:])
        zt_image_hidden_states = self.zt_image_patch_embed(image_hidden_states[:,Ci//2:,:,:])

        # cat
        source_token = self.source_token.repeat(N, 1, 1)
        target_token = self.target_token.repeat(N, 1, 1)
        motion_hidden_states = torch.cat([source_token,motion_source_hidden_states,target_token,motion_target_hidden_states],dim=1) # [N,2S+2,D]

        # Position embedding
        motion_hidden_states = motion_hidden_states + self.motion_pos_embedding[:, :motion_seq_length]
        zi_image_hidden_states = zi_image_hidden_states + self.pos_embedding[:, :image_seq_length]
        zt_image_hidden_states = zt_image_hidden_states + self.pos_embedding[:, :image_seq_length]
        self.embedding_dropout(motion_hidden_states)

        # Transformer blocks
        hidden_states = torch.cat([zt_image_hidden_states,zi_image_hidden_states,motion_hidden_states],dim=1)
        for i, block in enumerate(self.transformer_blocks):
            hidden_states = block(
                hidden_states=hidden_states,
            )

        image_hidden_states = self.norm_final(hidden_states[:,:zt_image_hidden_states.shape[1],:])

        # 6. Final block
        image_hidden_states = self.proj_out(image_hidden_states)

        # 7. Unpatchify
        p = self.image_patch_size
        output = image_hidden_states.reshape(N, 1, Hi // p, Wi // p, self.out_channels, p, p)
        output = output.permute(0, 1, 4, 2, 5, 3, 6).flatten(5, 6).flatten(3, 4).squeeze(1) # [N,C,H,W]

        return output



class AMDDiffusionTransformerModel(nn.Module):
    """
    Diffusion Transformer
    """

    _supports_gradient_checkpointing = True
    def __init__(
        self,
        num_attention_heads: int = 20,
        attention_head_dim: int = 64,
        out_channels: Optional[int] = 4,
        num_layers: int = 12,
        motion_type: str = 'decouple',
        # ----- img
        image_width: int = 32,
        image_height: int = 32,
        image_patch_size: int = 2,
        image_in_channels: Optional[int] = 4, 
        # ----- motion
        motion_token_num:int = 12,
        motion_in_channels: Optional[int] = 128,

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
        """
        
        Traning:
        Z(N,1,C,H,W)
        Motion(N,k,d,h,w)
        
        Inference:
        Z(N,1,C,H,W)
        Motion(N,k,d,h,w)
        """
        super().__init__()
        
        # 1. Setting
        hidden_dim = num_attention_heads * attention_head_dim
        iph = image_height // image_patch_size
        ipw = image_width // image_patch_size
        itl = iph * ipw # image token length
        self.image_patch_size = image_patch_size
        self.out_channels = out_channels
        self.motion_type = motion_type

        # Patch embedding (N,F,C,H,W) -> (B,S,D)
        self.image_patch_embed = PatchEmbed(patch_size=image_patch_size,in_channels=image_in_channels,embed_dim=hidden_dim, bias=True)
        self.motion_patch_embed = nn.Linear(motion_in_channels,hidden_dim)
        self.embedding_dropout = nn.Dropout(dropout)

        # 2D positional embeddings
        image_pos_embedding = get_2d_sincos_pos_embed(hidden_dim, (iph, ipw)) # (iph*ipw,D)
        image_pos_embedding = torch.from_numpy(image_pos_embedding) # (iph*ipw,D)
        pos_embedding = torch.zeros(1, itl, hidden_dim, requires_grad=False)
        pos_embedding.data[:, :itl].copy_(image_pos_embedding)
        self.register_buffer("pos_embedding", pos_embedding, persistent=False)

        # 1D position encoding
        temporal_embedding = get_1d_sincos_pos_embed_from_grid(hidden_dim,torch.arange(2+2*motion_token_num)) 
        motion_pos_embedding = torch.zeros(1,*temporal_embedding.shape,requires_grad=False)
        motion_pos_embedding.data.copy_(torch.from_numpy(temporal_embedding))
        self.register_buffer("motion_pos_embedding",motion_pos_embedding,persistent=False)
        
        # Split Token
        self.source_token = nn.Parameter(torch.zeros(1, 1, hidden_dim),requires_grad=True)
        self.target_token = nn.Parameter(torch.zeros(1, 1, hidden_dim),requires_grad=True)

        # Time embeddings
        self.time_proj = Timesteps(hidden_dim, flip_sin_to_cos, freq_shift)
        self.time_embedding = TimestepEmbedding(hidden_dim, time_embed_dim, timestep_activation_fn)

        # Define spatio-temporal transformers blocks
        self.transformer_blocks = nn.ModuleList(
            [ 
                AMDTransformerBlock(
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

        # Output blocks
        self.norm_out = AdaLayerNorm(
            embedding_dim=time_embed_dim,
            output_dim=2 * hidden_dim,
            norm_elementwise_affine=norm_elementwise_affine,
            norm_eps=norm_eps,
            chunk_dim=1,
        )
        self.proj_out = nn.Linear(hidden_dim, image_patch_size * image_patch_size * out_channels)

        self.gradient_checkpointing = False

    def _set_gradient_checkpointing(self, module, value=False):
        self.gradient_checkpointing = value

    def forward(
        self,
        camera_motion_target_hidden_states: torch.Tensor,
        image_hidden_states: torch.Tensor,
        timestep: Union[int, float, torch.LongTensor], # Timesteps should be a 1d-array
        timestep_cond: Optional[torch.Tensor] = None,
        camera_motion_source_hidden_states: torch.Tensor = None,
        object_motion_source_hidden_states: torch.Tensor = None,
        object_motion_target_hidden_states: torch.Tensor = None,
        return_dict: bool = True,
    ):
        """
        motion_hidden_states : (b,d,h,w)
        image_hidden_states : (b,2c,H,W)
        time_step : (b,)

        """
        N,Ci,Hi,Wi = image_hidden_states.shape
        N,L,Cm = camera_motion_target_hidden_states.shape
        image_seq_length = Hi * Wi // (self.image_patch_size**2)
        
        camera_motion_seq_length = motion_seq_length = 2*L+2
        if camera_motion_source_hidden_states == None:
            camera_motion_seq_length = L + 1
        

        
        # Time embedding   # Timesteps should be a 1d-array
        t_emb = self.time_proj(timestep)
        t_emb = t_emb.to(dtype=camera_motion_target_hidden_states.dtype)
        emb = self.time_embedding(t_emb, timestep_cond)

        # Patch embedding
        # motion_source_hidden_states = self.motion_patch_embed(motion_source_hidden_states) # [N,S,D]
        # motion_target_hidden_states = self.motion_patch_embed(motion_target_hidden_states) # [N,S,D]
        image_hidden_states = self.image_patch_embed(image_hidden_states)

        # cat
        source_token = self.source_token.repeat(N, 1, 1)
        target_token = self.target_token.repeat(N, 1, 1)

        if self.motion_type == 'plus':
            
            motion_source_hidden_states = camera_motion_source_hidden_states + object_motion_source_hidden_states
            motion_target_hidden_states = camera_motion_target_hidden_states + object_motion_target_hidden_states

            motion_source_hidden_states = self.motion_patch_embed(motion_source_hidden_states) # [N,S,D]
            motion_target_hidden_states = self.motion_patch_embed(motion_target_hidden_states) # [N,S,D]
            motion_hidden_states = torch.cat([source_token,motion_source_hidden_states,target_token,motion_target_hidden_states],dim=1) # [N,2S+2,D]
            
            # Position embedding
            motion_hidden_states = motion_hidden_states + self.motion_pos_embedding[:, :motion_seq_length]
            self.embedding_dropout(motion_hidden_states)
        elif self.motion_type == 'decouple':
            camera_motion_target_hidden_states = self.motion_patch_embed(camera_motion_target_hidden_states) # [N,S,D]
            if camera_motion_source_hidden_states != None:
                camera_motion_source_hidden_states = self.motion_patch_embed(camera_motion_source_hidden_states) # [N,S,D]
                camera_motion_hidden_states = torch.cat([source_token,camera_motion_source_hidden_states,target_token,camera_motion_target_hidden_states],dim=1) # [N,2S+2,D]
            else:
                camera_motion_hidden_states = torch.cat([target_token,camera_motion_target_hidden_states],dim=1)
            # Position embedding
            camera_motion_hidden_states = camera_motion_hidden_states + self.motion_pos_embedding[:, :camera_motion_seq_length]
            self.embedding_dropout(camera_motion_hidden_states)

            if object_motion_source_hidden_states != None:
                object_motion_source_hidden_states = self.motion_patch_embed(object_motion_source_hidden_states) # [N,S,D]
                object_motion_target_hidden_states = self.motion_patch_embed(object_motion_target_hidden_states) # [N,S,D]
                object_motion_source_hidden_states = object_motion_source_hidden_states + self.motion_pos_embedding[:, 1:L+1]
                object_motion_target_hidden_states = object_motion_source_hidden_states + self.motion_pos_embedding[:, L+2:motion_seq_length]
                # object_motion_hidden_states = object_motion_hidden_states + self.motion_pos_embedding[:, :motion_seq_length]
                # self.embedding_dropout(object_motion_hidden_states)

        # Position embedding

        image_hidden_states = image_hidden_states + self.pos_embedding[:, :image_seq_length]
        # Transformer blocks  总共有12层
        if self.motion_type == 'decouple' and object_motion_source_hidden_states != None: 
            # motion_hidden_states赋值相机运动
            motion_hidden_states = camera_motion_hidden_states # [b,2l+2.1024]
            for i, block in enumerate(self.transformer_blocks):
                
                if i < 8: # 前8层注入相机运动
                    motion_hidden_states, image_hidden_states = block(
                        hidden_states=motion_hidden_states,
                        encoder_hidden_states=image_hidden_states,
                        temb=emb,
                    )

            if camera_motion_source_hidden_states != None:
                # 将motion latents中,camera_motion部分替换成高频object_motion
                source_token = motion_hidden_states[:,0,:].unsqueeze(1)
                camera_motion_source_hidden_states = motion_hidden_states[:,1:1+L,:]
                target_token = motion_hidden_states[:,1+L,:].unsqueeze(1)
                camera_motion_target_hidden_states = motion_hidden_states[:,2+L:,:]
            else:
                target_token = motion_hidden_states[:,0,:].unsqueeze(1)
                camera_motion_target_hidden_states = motion_hidden_states[:,1:,:]
            
            motion_hidden_states = torch.cat([source_token,object_motion_source_hidden_states,target_token,object_motion_target_hidden_states],dim=1) # [N,2S+2,D]

            for i, block in enumerate(self.transformer_blocks):
                if i>=6:
                    motion_hidden_states, image_hidden_states = block(
                        hidden_states=motion_hidden_states,
                        encoder_hidden_states=image_hidden_states,
                        temb=emb,
                    )
        else:
            # Transformer blocks
            if self.motion_type == 'decouple':
                motion_hidden_states = camera_motion_hidden_states
            for i, block in enumerate(self.transformer_blocks):
                motion_hidden_states, image_hidden_states = block(
                    hidden_states=motion_hidden_states,
                    encoder_hidden_states=image_hidden_states,
                    temb=emb,
                )

        image_hidden_states = self.norm_final(image_hidden_states)

        # 6. Final block
        image_hidden_states = self.norm_out(image_hidden_states, temb=emb)
        image_hidden_states = self.proj_out(image_hidden_states)

        # 7. Unpatchify
        p = self.image_patch_size
        output = image_hidden_states.reshape(N, 1, Hi // p, Wi // p, self.out_channels, p, p)
        output = output.permute(0, 1, 4, 2, 5, 3, 6).flatten(5, 6).flatten(3, 4).squeeze(1) # [N,C,H,W]

        return output

class AMDDiffusionTransformerModelDualStream(nn.Module):

    _supports_gradient_checkpointing = True
    def __init__(
        self,
        num_attention_heads: int = 20,
        attention_head_dim: int = 64,
        out_channels: Optional[int] = 4,
        num_layers: int = 12,
        # ----- img
        image_width: int = 32,
        image_height: int = 32,
        image_patch_size: int = 2,
        image_in_channels: Optional[int] = 4, 
        # ----- motion
        motion_token_num:int = 12,
        motion_in_channels: Optional[int] = 128,
        motion_target_num_frame : int = 16,

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
        """
        
        Traning:
        Z(N,1,C,H,W)
        Motion(N,k,d,h,w)
        
        Inference:
        Z(N,1,C,H,W)
        Motion(N,k,d,h,w)
        """
        super().__init__()
        
        # 1. Setting
        hidden_dim = num_attention_heads * attention_head_dim
        iph = image_height // image_patch_size
        ipw = image_width // image_patch_size
        itl = iph * ipw # image token length
        self.image_patch_size = image_patch_size
        self.out_channels = out_channels
        self.target_frame = motion_target_num_frame

        # Patch embedding (N,F,C,H,W) -> (B,S,D)
        self.image_patch_embed = PatchEmbed(patch_size=image_patch_size,in_channels=image_in_channels,embed_dim=hidden_dim, bias=True)
        self.motion_patch_embed = nn.Linear(motion_in_channels,hidden_dim)
        self.embedding_dropout = nn.Dropout(dropout)

        # learnable token
        INIT_CONST = 0.02
        self.source_token = nn.Parameter(torch.randn(1, 1, hidden_dim) * INIT_CONST)
        self.target_token = nn.Parameter(torch.randn(1, 1, hidden_dim) * INIT_CONST)

        # 2D positional embeddings
        image_pos_embedding = get_2d_sincos_pos_embed(hidden_dim, (iph, ipw)) # (iph*ipw,D)
        image_pos_embedding = torch.from_numpy(image_pos_embedding) # (iph*ipw,D)
        pos_embedding = torch.zeros(1, itl, hidden_dim, requires_grad=False)
        pos_embedding.data[:, :itl].copy_(image_pos_embedding)
        self.register_buffer("pos_embedding", pos_embedding, persistent=False)

        # 1D position encoding
        temporal_embedding = get_1d_sincos_pos_embed_from_grid(hidden_dim,torch.arange(2*motion_token_num+2)) 
        motion_pos_embedding = torch.zeros(1,*temporal_embedding.shape,requires_grad=False)
        motion_pos_embedding.data.copy_(torch.from_numpy(temporal_embedding))
        self.register_buffer("motion_pos_embedding",motion_pos_embedding,persistent=False)

        temporal_embedding = get_1d_sincos_pos_embed_from_grid(hidden_dim,torch.arange(2*self.target_frame * (motion_token_num+1))) 
        motion_temporal_embedding = torch.zeros(1,*temporal_embedding.shape,requires_grad=False)
        motion_temporal_embedding.data.copy_(torch.from_numpy(temporal_embedding))
        self.register_buffer("motion_temporal_embedding",motion_temporal_embedding,persistent=False)

        # Time embeddings
        self.time_proj = Timesteps(hidden_dim, flip_sin_to_cos, freq_shift)
        self.time_embedding = TimestepEmbedding(hidden_dim, time_embed_dim, timestep_activation_fn)

        # Define spatio-temporal transformers blocks
        self.transformer_blocks = nn.ModuleList(
            [ 
                AMDTransformerBlock(
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

        self.motion_blocks = nn.ModuleList(
            [ 
                AMDTransformerMotionBlock(
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

        # Output blocks
        self.norm_out = AdaLayerNorm(
            embedding_dim=time_embed_dim,
            output_dim=2 * hidden_dim,
            norm_elementwise_affine=norm_elementwise_affine,
            norm_eps=norm_eps,
            chunk_dim=1,
        )
        self.proj_out = nn.Linear(hidden_dim, image_patch_size * image_patch_size * out_channels)

        self.gradient_checkpointing = False

    def _set_gradient_checkpointing(self, module, value=False):
        self.gradient_checkpointing = value

    def forward(
        self,
        motion_source_hidden_states: torch.Tensor,
        motion_target_hidden_states: torch.Tensor,
        image_hidden_states: torch.Tensor,
        timestep: Union[int, float, torch.LongTensor], # Timesteps should be a 1d-array
        timestep_cond: Optional[torch.Tensor] = None,
        return_dict: bool = True,
    ):
        """
        motion_hidden_states : (nt,l,d)
        image_hidden_states : (b,2c,H,W)
        time_step : (b,)
        """
        N,Ci,Hi,Wi = image_hidden_states.shape
        N,L,Cm = motion_target_hidden_states.shape
        image_seq_length = Hi * Wi // (self.image_patch_size**2)
        motion_seq_length = 2*L 

        n = N // self.target_frame
        t = self.target_frame
        l = L
        d = Cm

        # Time embedding   # Timesteps should be a 1d-array
        t_emb = self.time_proj(timestep)
        t_emb = t_emb.to(dtype=motion_source_hidden_states.dtype)
        emb = self.time_embedding(t_emb, timestep_cond)

        emb_m = einops.rearrange(emb,'(n t) d -> n t d',n=n,t=t)# n,t,d
        emb_m = emb_m[:,0,:] # n,d

        # Patch embedding
        image_hidden_states = self.image_patch_embed(image_hidden_states) # nt,s,d

        motion_source_hidden_states = self.motion_patch_embed(motion_source_hidden_states) # nt,l,d
        motion_target_hidden_states = self.motion_patch_embed(motion_target_hidden_states) # nt,l,d
        
        source_token = self.source_token.repeat(n*t,1,1) # nt,l,d
        target_token = self.target_token.repeat(n*t,1,1) # nt,l,d
        motion_hidden_states = torch.cat([source_token,motion_source_hidden_states,target_token,motion_target_hidden_states],dim=1) # nt,2+2l,d
        # motion position encoding1
        motion_hidden_states = motion_hidden_states + self.motion_pos_embedding[:, :2*l+2] # nt,2+2l,d
        # motion position encoding2
        motion_hidden_states = einops.rearrange(motion_hidden_states,'(n t) l d -> n (t l) d',n=n) # n,t(2l+2),d
        motion_hidden_states = motion_hidden_states + self.motion_temporal_embedding[:,:t*(2*l+2)] # n,t(2l+2),d

        # img Position embedding
        image_hidden_states = image_hidden_states + self.pos_embedding[:, :image_seq_length]

        # Transformer blocks
        for block,m_block in zip(self.transformer_blocks,self.motion_blocks):

            # motion temporal block
            motion_hidden_states = m_block(
                hidden_states = motion_hidden_states,
                temb = emb_m,
            )   # n,t(2l+2),d

            # transform for block
            motion_hidden_states = einops.rearrange(motion_hidden_states,'n (t l) d -> (n t) l d',t=t)

            # img block
            motion_hidden_states, image_hidden_states = block(
                hidden_states=motion_hidden_states,
                encoder_hidden_states=image_hidden_states,
                temb=emb,
            )

            motion_hidden_states = einops.rearrange(motion_hidden_states,'(n t) l d -> n (t l) d',t=t)


        image_hidden_states = self.norm_final(image_hidden_states)

        # 6. Final block
        image_hidden_states = self.norm_out(image_hidden_states, temb=emb)
        image_hidden_states = self.proj_out(image_hidden_states)

        # 7. Unpatchify
        p = self.image_patch_size
        output = image_hidden_states.reshape(N, 1, Hi // p, Wi // p, self.out_channels, p, p)
        output = output.permute(0, 1, 4, 2, 5, 3, 6).flatten(5, 6).flatten(3, 4).squeeze(1) # [N,C,H,W]

        return output

class AMDDiffusionTransformerModelTempMotion(nn.Module):
    """
    Diffusion Transformer
    """

    _supports_gradient_checkpointing = True
    def __init__(
        self,
        num_attention_heads: int = 20,
        attention_head_dim: int = 64,
        out_channels: Optional[int] = 4,
        num_layers: int = 12,
        use_camera: bool = False,
        use_object: bool = False,
        # ----- img
        image_width: int = 32,
        image_height: int = 32,
        image_patch_size: int = 2,
        image_in_channels: Optional[int] = 4, 
        # ----- motion
        motion_token_num:int = 12,
        camera_motion_in_channels: Optional[int] = 16,
        object_motion_in_channels: Optional[int] = 64,
        motion_target_num_frame : int = 16,

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
        """
        
        Traning:
        Z(N,1,C,H,W)
        Motion(N,k,d,h,w)
        
        Inference:
        Z(N,1,C,H,W)
        Motion(N,k,d,h,w)
        """
        super().__init__()
        
        # 1. Setting
        hidden_dim = num_attention_heads * attention_head_dim
        iph = image_height // image_patch_size
        ipw = image_width // image_patch_size
        itl = iph * ipw # image token length
        self.image_patch_size = image_patch_size
        self.out_channels = out_channels
        self.target_frame = motion_target_num_frame
        self.use_camera = use_camera
        self.use_object = use_object

        # Patch embedding (N,F,C,H,W) -> (B,S,D)
        self.image_patch_embed = PatchEmbed(patch_size=image_patch_size,in_channels=image_in_channels,embed_dim=hidden_dim, bias=True)
        self.embedding_dropout = nn.Dropout(dropout)

        # 2D positional embeddings
        image_pos_embedding = get_2d_sincos_pos_embed(hidden_dim, (iph, ipw)) # (iph*ipw,D)
        image_pos_embedding = torch.from_numpy(image_pos_embedding) # (iph*ipw,D)
        pos_embedding = torch.zeros(1, itl, hidden_dim, requires_grad=False)
        pos_embedding.data[:, :itl].copy_(image_pos_embedding)
        self.register_buffer("pos_embedding", pos_embedding, persistent=False)

        # 1D position encoding
        temporal_embedding = get_1d_sincos_pos_embed_from_grid(hidden_dim,torch.arange(2+2*motion_token_num)) 
        motion_pos_embedding = torch.zeros(1,*temporal_embedding.shape,requires_grad=False)
        motion_pos_embedding.data.copy_(torch.from_numpy(temporal_embedding))
        self.register_buffer("motion_pos_embedding",motion_pos_embedding,persistent=False)

        # 1D img temporal position encoding
        temporal_embedding = get_1d_sincos_pos_embed_from_grid(hidden_dim,torch.arange(self.target_frame)) 
        img_pos_embedding = torch.zeros(1,*temporal_embedding.shape,requires_grad=False)
        img_pos_embedding.data.copy_(torch.from_numpy(temporal_embedding))
        self.register_buffer("img_temporal_embedding",img_pos_embedding,persistent=False)
        
        # Time embeddings
        self.time_proj = Timesteps(hidden_dim, flip_sin_to_cos, freq_shift)
        self.time_embedding = TimestepEmbedding(hidden_dim, time_embed_dim, timestep_activation_fn)

        # Define spatio-temporal transformers blocks
        # if self.use_camera:
        #     self.camera_motion_patch_embed = nn.Linear(camera_motion_in_channels,hidden_dim)
        #     self.camera_transformer_blocks = nn.ModuleList(
        #         [ 
        #             AMDTransformerBlock(
        #                 dim=hidden_dim,
        #                 num_attention_heads=num_attention_heads,
        #                 attention_head_dim=attention_head_dim,
        #                 time_embed_dim=time_embed_dim,
        #                 dropout=dropout,
        #                 activation_fn=activation_fn,
        #                 attention_bias=attention_bias,
        #                 norm_elementwise_affine=norm_elementwise_affine,
        #                 norm_eps=norm_eps,
        #             )
        #             for _ in range(num_layers)
        #         ]
        #     )
        
        if self.use_object:
            # Split Token
            self.source_token = nn.Parameter(torch.zeros(1, 1, hidden_dim),requires_grad=True)
            self.target_token = nn.Parameter(torch.zeros(1, 1, hidden_dim),requires_grad=True)

            self.object_motion_patch_embed = nn.Linear(object_motion_in_channels,hidden_dim)
            self.object_transformer_blocks = nn.ModuleList(
                [ 
                    AMDTransformerBlock(
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

        # Output blocks
        self.norm_out = AdaLayerNorm(
            embedding_dim=time_embed_dim,
            output_dim=2 * hidden_dim,
            norm_elementwise_affine=norm_elementwise_affine,
            norm_eps=norm_eps,
            chunk_dim=1,
        )
        self.proj_out = nn.Linear(hidden_dim, image_patch_size * image_patch_size * out_channels)

        self.gradient_checkpointing = False

    def _set_gradient_checkpointing(self, module, value=False):
        self.gradient_checkpointing = value

    def forward(
        self,
        image_hidden_states: torch.Tensor,
        timestep: Union[int, float, torch.LongTensor], # Timesteps should be a 1d-array
        timestep_cond: Optional[torch.Tensor] = None,
        object_motion_source_hidden_states: torch.Tensor = None,
        object_motion_target_hidden_states: torch.Tensor = None,
        return_dict: bool = True,
    ):
        """
        motion_hidden_states : (b,d,h,w)
        image_hidden_states : (b,2c,H,W)
        time_step : (b,)

        """
        N,Ci,Hi,Wi = image_hidden_states.shape
        image_seq_length = Hi * Wi // (self.image_patch_size**2)
        n = N // self.target_frame
        t = self.target_frame


        # Patch embedding
        image_hidden_states = self.image_patch_embed(image_hidden_states)

        # Time embedding   # Timesteps should be a 1d-array
        t_emb = self.time_proj(timestep)
        t_emb = t_emb.to(dtype=object_motion_source_hidden_states.dtype)
        emb = self.time_embedding(t_emb, timestep_cond)

        emb_s = einops.rearrange(emb,'(n t) d -> n t d',n=n,t=t)# n,t,d
        emb_s = emb_s[:,:1,:].repeat(1,image_hidden_states.shape[1],1) # n,s,d
        emb_s = emb_s.flatten(0,1) # ns,d

        # if camera_motion_target_hidden_states != None:
        #     Nc,Lc,_ = camera_motion_target_hidden_states.shape
        #     camera_motion_seq_length = Lc
        #     Ni = n*image_seq_length
        #     if Nc != Ni:
        #         camera_motion_target_hidden_states = camera_motion_target_hidden_states.repeat(Ni//Nc)
        #     camera_motion_hidden_states = self.motion_patch_embed(camera_motion_target_hidden_states) # [N,S,D]

        #     # Position embedding
        #     camera_motion_hidden_states = camera_motion_hidden_states + self.motion_pos_embedding[:, :camera_motion_seq_length]
        #     self.embedding_dropout(camera_motion_hidden_states)

        if object_motion_source_hidden_states != None:
            No,Lo,_ = object_motion_target_hidden_states.shape
            object_motion_seq_length = 2*Lo + 2

            source_token = self.source_token.repeat(N, 1, 1)
            target_token = self.target_token.repeat(N, 1, 1)

            object_motion_source_hidden_states = self.object_motion_patch_embed(object_motion_source_hidden_states) # [N,S,D]
            object_motion_target_hidden_states = self.object_motion_patch_embed(object_motion_target_hidden_states) # [N,S,D]
            # object_motion_source_hidden_states = object_motion_source_hidden_states + self.motion_pos_embedding[:, 1:L+1]
            # object_motion_target_hidden_states = object_motion_source_hidden_states + self.motion_pos_embedding[:, L+2:motion_seq_length]
            object_motion_hidden_states = torch.cat([source_token,object_motion_source_hidden_states,target_token,object_motion_target_hidden_states],dim=1) # [N,2S+2,D]
            object_motion_hidden_states = object_motion_hidden_states + self.motion_pos_embedding[:, :object_motion_seq_length]
            self.embedding_dropout(object_motion_hidden_states)


        # Position embedding
        # motion_hidden_states = motion_hidden_states + self.motion_pos_embedding[:, :motion_seq_length] # nt,2l+2,d
        image_hidden_states = image_hidden_states + self.pos_embedding[:, :image_seq_length]  # nt,s,d
        image_hidden_states = einops.rearrange(image_hidden_states,'(n t) s d -> (n s) t d',n=n)
        image_hidden_states = image_hidden_states + self.img_temporal_embedding[:,:t] # ns,t,d
        image_hidden_states = einops.rearrange(image_hidden_states,'(n s) t d -> (n t) s d',n=n) # nt,s,d
        # self.embedding_dropout(motion_hidden_states)

        # Transformer blocks        
        # motion_hidden_states赋值相机运动
        for i, object_block in enumerate(self.object_transformer_blocks):
            
            if object_motion_source_hidden_states != None:
                object_motion_hidden_states, image_hidden_states = object_block(
                    hidden_states=object_motion_hidden_states,
                    encoder_hidden_states=image_hidden_states,
                    temb=emb,
                )
                # Q img(1 ref + video noise)(n*h*w,1+t,c); KV camera_motion (n*h*w,t,c)
                # (n*h*w, 2t, c)
                # if camera_motion_target_hidden_states != None:
                #     image_hidden_states = einops.rearrange(image_hidden_states,'(n t) s d -> (n s) t d',n=n)
                #     camera_motion_hidden_states, image_hidden_states = camera_block(
                #         hidden_states=camera_motion_hidden_states,
                #         encoder_hidden_states=image_hidden_states,
                #         temb=emb,
                #     )
                #     image_hidden_states = einops.rearrange(image_hidden_states,'(n s) t d -> (n t) s d',n=n)
                
                # # Q img(t ref img + video noise)(n*h*w,2t,c);
                # image_hidden_states = einops.rearrange(image_hidden_states,'(n t) s d -> (n s) t d',n=n)
                # image_hidden_states = s_block(
                #     hidden_states = image_hidden_states,
                #     temb = emb_s,
                # )

                # image_hidden_states = einops.rearrange(image_hidden_states,'(n s) t d -> (n t) s d',n=n)

         

        image_hidden_states = self.norm_final(image_hidden_states)

        # 6. Final block
        image_hidden_states = self.norm_out(image_hidden_states, temb=emb)
        image_hidden_states = self.proj_out(image_hidden_states)

        # 7. Unpatchify
        p = self.image_patch_size
        output = image_hidden_states.reshape(N, 1, Hi // p, Wi // p, self.out_channels, p, p)
        output = output.permute(0, 1, 4, 2, 5, 3, 6).flatten(5, 6).flatten(3, 4).squeeze(1) # [N,C,H,W]

        return output


class AMDDiffusionTransformerModelImgSpatialTempMotion(nn.Module):
    """
    Diffusion Transformer
    """

    _supports_gradient_checkpointing = True
    def __init__(
        self,
        num_attention_heads: int = 20,
        attention_head_dim: int = 64,
        out_channels: Optional[int] = 4,
        num_layers: int = 12,
        use_camera: bool = False,
        use_object: bool = False,
        # ----- img
        image_width: int = 32,
        image_height: int = 32,
        image_patch_size: int = 2,
        image_in_channels: Optional[int] = 4, 
        # ----- motion
        motion_token_num:int = 12,
        camera_motion_in_channels: Optional[int] = 16,
        object_motion_in_channels: Optional[int] = 64,
        motion_target_num_frame : int = 16,

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
        """
        
        Traning:
        Z(N,1,C,H,W)
        Motion(N,k,d,h,w)
        
        Inference:
        Z(N,1,C,H,W)
        Motion(N,k,d,h,w)
        """
        super().__init__()
        
        # 1. Setting
        hidden_dim = num_attention_heads * attention_head_dim
        iph = image_height // image_patch_size
        ipw = image_width // image_patch_size
        itl = iph * ipw # image token length
        self.image_patch_size = image_patch_size
        self.out_channels = out_channels
        self.target_frame = motion_target_num_frame
        self.use_camera = use_camera
        self.use_object = use_object

        # Patch embedding (N,F,C,H,W) -> (B,S,D)
        self.image_patch_embed = PatchEmbed(patch_size=image_patch_size,in_channels=image_in_channels,embed_dim=hidden_dim, bias=True)
        self.embedding_dropout = nn.Dropout(dropout)

        # 2D positional embeddings
        image_pos_embedding = get_2d_sincos_pos_embed(hidden_dim, (iph, ipw)) # (iph*ipw,D)
        image_pos_embedding = torch.from_numpy(image_pos_embedding) # (iph*ipw,D)
        pos_embedding = torch.zeros(1, itl, hidden_dim, requires_grad=False)
        pos_embedding.data[:, :itl].copy_(image_pos_embedding)
        self.register_buffer("pos_embedding", pos_embedding, persistent=False)

        # 1D position encoding
        temporal_embedding = get_1d_sincos_pos_embed_from_grid(hidden_dim,torch.arange(2+2*motion_token_num)) 
        motion_pos_embedding = torch.zeros(1,*temporal_embedding.shape,requires_grad=False)
        motion_pos_embedding.data.copy_(torch.from_numpy(temporal_embedding))
        self.register_buffer("motion_pos_embedding",motion_pos_embedding,persistent=False)

        # 1D img temporal position encoding
        temporal_embedding = get_1d_sincos_pos_embed_from_grid(hidden_dim,torch.arange(self.target_frame)) 
        img_pos_embedding = torch.zeros(1,*temporal_embedding.shape,requires_grad=False)
        img_pos_embedding.data.copy_(torch.from_numpy(temporal_embedding))
        self.register_buffer("img_temporal_embedding",img_pos_embedding,persistent=False)
        

        # Time embeddings
        self.time_proj = Timesteps(hidden_dim, flip_sin_to_cos, freq_shift)
        self.time_embedding = TimestepEmbedding(hidden_dim, time_embed_dim, timestep_activation_fn)

        # Define spatio-temporal transformers blocks
        if self.use_camera:
            self.camera_motion_patch_embed = nn.Linear(camera_motion_in_channels,hidden_dim)
            self.camera_transformer_blocks = nn.ModuleList(
                [ 
                    AMDTransformerBlock(
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

        if self.use_object:
            # Split Token
            self.source_token = nn.Parameter(torch.zeros(1, 1, hidden_dim),requires_grad=True)
            self.target_token = nn.Parameter(torch.zeros(1, 1, hidden_dim),requires_grad=True)

            self.object_motion_patch_embed = nn.Linear(object_motion_in_channels,hidden_dim)
            self.object_transformer_blocks = nn.ModuleList(
                [ 
                    AMDTransformerBlock(
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

        self.spatial_blocks = nn.ModuleList(
            [ 
                BasicDiTBlock(
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

        # Output blocks
        self.norm_out = AdaLayerNorm(
            embedding_dim=time_embed_dim,
            output_dim=2 * hidden_dim,
            norm_elementwise_affine=norm_elementwise_affine,
            norm_eps=norm_eps,
            chunk_dim=1,
        )
        self.proj_out = nn.Linear(hidden_dim, image_patch_size * image_patch_size * out_channels)

        self.gradient_checkpointing = False

    def _set_gradient_checkpointing(self, module, value=False):
        self.gradient_checkpointing = value

    def forward(
        self,
        image_hidden_states: torch.Tensor,
        timestep: Union[int, float, torch.LongTensor], # Timesteps should be a 1d-array
        timestep_cond: Optional[torch.Tensor] = None,
        camera_motion_target_hidden_states: torch.Tensor = None,
        object_motion_source_hidden_states: torch.Tensor = None,
        object_motion_target_hidden_states: torch.Tensor = None,
        return_dict: bool = True,
    ):
        """
        motion_hidden_states : (b,d,h,w)
        image_hidden_states : (b,2c,H,W)
        time_step : (b,)

        """
        N,Ci,Hi,Wi = image_hidden_states.shape
        image_seq_length = Hi * Wi // (self.image_patch_size**2)

        n = N // self.target_frame

        t = self.target_frame

        # Patch embedding
        image_hidden_states = self.image_patch_embed(image_hidden_states)

        # Time embedding   # Timesteps should be a 1d-array
        # print("timestep.shape",timestep.shape)
        t_emb = self.time_proj(timestep)
        t_emb = t_emb.to(dtype=image_hidden_states.dtype)
        emb = self.time_embedding(t_emb, timestep_cond)

        emb_s = einops.rearrange(emb,'(n t) d -> n t d',n=n,t=t)# n,t,d
        emb_s = emb_s[:,:1,:].repeat(1,image_hidden_states.shape[1],1) # n,s,d
        emb_s = emb_s.flatten(0,1) # ns,d

        if camera_motion_target_hidden_states != None:
            Nc,Lc,Sc,D = camera_motion_target_hidden_states.shape
            # camera_motion_seq_length = Lc
            # camera_motion_seq_length = Sc
            # print("sc",Sc)
            # Ni = n*image_seq_length
            # if Sc != image_seq_length:
            #     # repeat的方式
            #     camera_motion_target_hidden_states = camera_motion_target_hidden_states.repeat_interleave((image_seq_length//Sc)+1,dim=2)
            #     camera_motion_target_hidden_states = camera_motion_target_hidden_states[:,:,:image_seq_length]
            
            # camera_motion_target_hidden_states = einops.rearrange(camera_motion_target_hidden_states,'n t s d -> (n s) t d')
            camera_motion_target_hidden_states = einops.rearrange(camera_motion_target_hidden_states,'n t s d -> (n t) s d')
            camera_motion_hidden_states = self.camera_motion_patch_embed(camera_motion_target_hidden_states) # [N,S,D]

            # Position embedding
            # print("camera_motion_hidden_states.shape",camera_motion_hidden_states.shape)
            # camera_motion_hidden_states = camera_motion_hidden_states + self.motion_pos_embedding[:, :camera_motion_seq_length]
            self.embedding_dropout(camera_motion_hidden_states)

        if object_motion_source_hidden_states != None:
            No,Lo,_ = object_motion_target_hidden_states.shape
            object_motion_seq_length = 2*Lo + 2

            source_token = self.source_token.repeat(N, 1, 1)
            target_token = self.target_token.repeat(N, 1, 1)

            object_motion_source_hidden_states = self.object_motion_patch_embed(object_motion_source_hidden_states) # [N,S,D]
            object_motion_target_hidden_states = self.object_motion_patch_embed(object_motion_target_hidden_states) # [N,S,D]
            # object_motion_source_hidden_states = object_motion_source_hidden_states + self.motion_pos_embedding[:, 1:L+1]
            # object_motion_target_hidden_states = object_motion_source_hidden_states + self.motion_pos_embedding[:, L+2:motion_seq_length]
            object_motion_hidden_states = torch.cat([source_token,object_motion_source_hidden_states,target_token,object_motion_target_hidden_states],dim=1) # [N,2S+2,D]
            object_motion_hidden_states = object_motion_hidden_states + self.motion_pos_embedding[:, :object_motion_seq_length]
            self.embedding_dropout(object_motion_hidden_states)


        # Position embedding
        # motion_hidden_states = motion_hidden_states + self.motion_pos_embedding[:, :motion_seq_length] # nt,2l+2,d
        image_hidden_states = image_hidden_states + self.pos_embedding[:, :image_seq_length]  # nt,s,d
        image_hidden_states = einops.rearrange(image_hidden_states,'(n t) s d -> (n s) t d',n=n)
        image_hidden_states = image_hidden_states + self.img_temporal_embedding[:,:t] # ns,t,d
        image_hidden_states = einops.rearrange(image_hidden_states,'(n s) t d -> (n t) s d',n=n) # nt,s,d
        # self.embedding_dropout(motion_hidden_states)

        # Transformer blocks        
            # motion_hidden_states赋值相机运动
        # for i,(camera_block,object_block,s_block) in enumerate(zip(self.camera_transformer_blocks,self.object_transformer_blocks,self.spatial_blocks)):
        for i,s_block in enumerate(self.spatial_blocks):
            
            if object_motion_source_hidden_states != None:
                object_block = self.object_transformer_blocks[i]
                object_motion_hidden_states = object_motion_hidden_states # [N,2S+2,D]
                object_motion_hidden_states, image_hidden_states = object_block(
                    hidden_states=object_motion_hidden_states,
                    encoder_hidden_states=image_hidden_states,
                    temb=emb,
                )
            # Q img(1 ref + video noise)(n*h*w,1+t,c); KV camera_motion (n*h*w,t,c)
            # (n*h*w, 2t, c)
            if camera_motion_target_hidden_states != None:
                camera_block = self.camera_transformer_blocks[i]
                # image_hidden_states = einops.rearrange(image_hidden_states,'(n t) s d -> (n s) t d',n=n)
                # camera_motion_hidden_states, image_hidden_states = camera_block(
                #     hidden_states=camera_motion_hidden_states,
                #     encoder_hidden_states=image_hidden_states,
                #     temb=emb_s,
                # )
                camera_motion_hidden_states, image_hidden_states = camera_block(
                    hidden_states=camera_motion_hidden_states,
                    encoder_hidden_states=image_hidden_states,
                    temb=emb,
                )
                # image_hidden_states = einops.rearrange(image_hidden_states,'(n s) t d -> (n t) s d',n=n)
            
            # Q img(t ref img + video noise)(n*h*w,2t,c);
            image_hidden_states = einops.rearrange(image_hidden_states,'(n t) s d -> (n s) t d',n=n)
            image_hidden_states = s_block(
                hidden_states = image_hidden_states,
                temb = emb_s,
            )

            image_hidden_states = einops.rearrange(image_hidden_states,'(n s) t d -> (n t) s d',n=n)

         

        image_hidden_states = self.norm_final(image_hidden_states)

        # 6. Final block
        image_hidden_states = self.norm_out(image_hidden_states, temb=emb)
        image_hidden_states = self.proj_out(image_hidden_states)

        # 7. Unpatchify
        p = self.image_patch_size
        output = image_hidden_states.reshape(N, 1, Hi // p, Wi // p, self.out_channels, p, p)
        output = output.permute(0, 1, 4, 2, 5, 3, 6).flatten(5, 6).flatten(3, 4).squeeze(1) # [N,C,H,W]

        return output


class AMDDiffusionTransformerModelImgSpatial(nn.Module):
    """
    Diffusion Transformer
    """

    _supports_gradient_checkpointing = True
    def __init__(
        self,
        num_attention_heads: int = 20,
        attention_head_dim: int = 64,
        out_channels: Optional[int] = 4,
        num_layers: int = 12,
        motion_type: str = 'plus',
        # ----- img
        image_width: int = 32,
        image_height: int = 32,
        image_patch_size: int = 2,
        image_in_channels: Optional[int] = 4, 
        # ----- motion
        motion_token_num:int = 12,
        motion_in_channels: Optional[int] = 128,
        motion_target_num_frame : int = 16,

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
        """
        
        Traning:
        Z(N,1,C,H,W)
        Motion(N,k,d,h,w)
        
        Inference:
        Z(N,1,C,H,W)
        Motion(N,k,d,h,w)
        """
        super().__init__()
        
        # 1. Setting
        hidden_dim = num_attention_heads * attention_head_dim
        iph = image_height // image_patch_size
        ipw = image_width // image_patch_size
        itl = iph * ipw # image token length
        self.image_patch_size = image_patch_size
        self.out_channels = out_channels
        self.target_frame = motion_target_num_frame
        self.motion_type = motion_type

        # Patch embedding (N,F,C,H,W) -> (B,S,D)
        self.image_patch_embed = PatchEmbed(patch_size=image_patch_size,in_channels=image_in_channels,embed_dim=hidden_dim, bias=True)
        self.motion_patch_embed = nn.Linear(motion_in_channels,hidden_dim)
        self.embedding_dropout = nn.Dropout(dropout)

        # 2D positional embeddings
        image_pos_embedding = get_2d_sincos_pos_embed(hidden_dim, (iph, ipw)) # (iph*ipw,D)
        image_pos_embedding = torch.from_numpy(image_pos_embedding) # (iph*ipw,D)
        pos_embedding = torch.zeros(1, itl, hidden_dim, requires_grad=False)
        pos_embedding.data[:, :itl].copy_(image_pos_embedding)
        self.register_buffer("pos_embedding", pos_embedding, persistent=False)

        # 1D position encoding
        temporal_embedding = get_1d_sincos_pos_embed_from_grid(hidden_dim,torch.arange(2+2*motion_token_num)) 
        motion_pos_embedding = torch.zeros(1,*temporal_embedding.shape,requires_grad=False)
        motion_pos_embedding.data.copy_(torch.from_numpy(temporal_embedding))
        self.register_buffer("motion_pos_embedding",motion_pos_embedding,persistent=False)

        # 1D img temporal position encoding
        temporal_embedding = get_1d_sincos_pos_embed_from_grid(hidden_dim,torch.arange(self.target_frame)) 
        img_pos_embedding = torch.zeros(1,*temporal_embedding.shape,requires_grad=False)
        img_pos_embedding.data.copy_(torch.from_numpy(temporal_embedding))
        self.register_buffer("img_temporal_embedding",img_pos_embedding,persistent=False)
        
        # Split Token
        self.source_token = nn.Parameter(torch.zeros(1, 1, hidden_dim),requires_grad=True)
        self.target_token = nn.Parameter(torch.zeros(1, 1, hidden_dim),requires_grad=True)

        # Time embeddings
        self.time_proj = Timesteps(hidden_dim, flip_sin_to_cos, freq_shift)
        self.time_embedding = TimestepEmbedding(hidden_dim, time_embed_dim, timestep_activation_fn)

        # Define spatio-temporal transformers blocks
        self.transformer_blocks = nn.ModuleList(
            [ 
                AMDTransformerBlock(
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
        self.spatial_blocks = nn.ModuleList(
            [ 
                BasicDiTBlock(
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

        # Output blocks
        self.norm_out = AdaLayerNorm(
            embedding_dim=time_embed_dim,
            output_dim=2 * hidden_dim,
            norm_elementwise_affine=norm_elementwise_affine,
            norm_eps=norm_eps,
            chunk_dim=1,
        )
        self.proj_out = nn.Linear(hidden_dim, image_patch_size * image_patch_size * out_channels)

        self.gradient_checkpointing = False

    def _set_gradient_checkpointing(self, module, value=False):
        self.gradient_checkpointing = value

    def forward(
        self,
        camera_motion_target_hidden_states: torch.Tensor,
        image_hidden_states: torch.Tensor,
        timestep: Union[int, float, torch.LongTensor], # Timesteps should be a 1d-array
        timestep_cond: Optional[torch.Tensor] = None,
        camera_motion_source_hidden_states: torch.Tensor = None,
        object_motion_source_hidden_states: torch.Tensor = None,
        object_motion_target_hidden_states: torch.Tensor = None,
        return_dict: bool = True,
    ):
        """
        motion_hidden_states : (b,d,h,w)
        image_hidden_states : (b,2c,H,W)
        time_step : (b,)

        """
        N,Ci,Hi,Wi = image_hidden_states.shape
        N,L,Cm = camera_motion_target_hidden_states.shape
        image_seq_length = Hi * Wi // (self.image_patch_size**2)
        
        camera_motion_seq_length = motion_seq_length = 2*L + 2
        if camera_motion_source_hidden_states == None:
            camera_motion_seq_length = L + 1

        n = N // self.target_frame
        t = self.target_frame
        l = L
        d = Cm

        # Patch embedding
        # motion_source_hidden_states = self.motion_patch_embed(motion_source_hidden_states) # [B,S,D]
        # motion_target_hidden_states = self.motion_patch_embed(motion_target_hidden_states) # [B,S,D]
        image_hidden_states = self.image_patch_embed(image_hidden_states)

        # Time embedding   # Timesteps should be a 1d-array
        t_emb = self.time_proj(timestep)
        t_emb = t_emb.to(dtype=camera_motion_target_hidden_states.dtype)
        emb = self.time_embedding(t_emb, timestep_cond)

        emb_s = einops.rearrange(emb,'(n t) d -> n t d',n=n,t=t)# n,t,d
        emb_s = emb_s[:,:1,:].repeat(1,image_hidden_states.shape[1],1) # n,s,d
        emb_s = emb_s.flatten(0,1) # ns,d

        # cat
        source_token = self.source_token.repeat(N, 1, 1)
        target_token = self.target_token.repeat(N, 1, 1)
        # motion_hidden_states = torch.cat([source_token,motion_source_hidden_states,target_token,motion_target_hidden_states],dim=1) # [B,2S+2,D]

        if self.motion_type == 'plus':
            # motion_source_hidden_states = camera_motion_source_hidden_states + object_motion_source_hidden_states
            # motion_target_hidden_states = camera_motion_target_hidden_states + object_motion_target_hidden_states

            motion_source_hidden_states = object_motion_source_hidden_states
            motion_target_hidden_states = object_motion_target_hidden_states

            motion_source_hidden_states = self.motion_patch_embed(motion_source_hidden_states) # [N,S,D]
            motion_target_hidden_states = self.motion_patch_embed(motion_target_hidden_states) # [N,S,D]
            motion_hidden_states = torch.cat([source_token,motion_source_hidden_states,target_token,motion_target_hidden_states],dim=1) # [N,2S+2,D]
            
            # Position embedding
            motion_hidden_states = motion_hidden_states + self.motion_pos_embedding[:, :motion_seq_length]
            self.embedding_dropout(motion_hidden_states)
        elif self.motion_type == 'decouple':
        
            camera_motion_target_hidden_states = self.motion_patch_embed(camera_motion_target_hidden_states) # [N,S,D]
            if camera_motion_source_hidden_states != None:
                camera_motion_source_hidden_states = self.motion_patch_embed(camera_motion_source_hidden_states) # [N,S,D]
                camera_motion_hidden_states = torch.cat([source_token,camera_motion_source_hidden_states,target_token,camera_motion_target_hidden_states],dim=1) # [N,2S+2,D]
            else:
                camera_motion_hidden_states = torch.cat([target_token,camera_motion_target_hidden_states],dim=1)
            # Position embedding
            camera_motion_hidden_states = camera_motion_hidden_states + self.motion_pos_embedding[:, :camera_motion_seq_length]
            self.embedding_dropout(camera_motion_hidden_states)

            if object_motion_source_hidden_states != None:
                object_motion_source_hidden_states = self.motion_patch_embed(object_motion_source_hidden_states) # [N,S,D]
                object_motion_target_hidden_states = self.motion_patch_embed(object_motion_target_hidden_states) # [N,S,D]
                object_motion_source_hidden_states = object_motion_source_hidden_states + self.motion_pos_embedding[:, 1:L+1]
                object_motion_target_hidden_states = object_motion_source_hidden_states + self.motion_pos_embedding[:, L+2:motion_seq_length]
                # object_motion_hidden_states = object_motion_hidden_states + self.motion_pos_embedding[:, :motion_seq_length]
                # self.embedding_dropout(object_motion_hidden_states)


        # Position embedding
        # motion_hidden_states = motion_hidden_states + self.motion_pos_embedding[:, :motion_seq_length] # nt,2l+2,d
        image_hidden_states = image_hidden_states + self.pos_embedding[:, :image_seq_length]  # nt,s,d
        image_hidden_states = einops.rearrange(image_hidden_states,'(n t) s d -> (n s) t d',n=n)
        image_hidden_states = image_hidden_states + self.img_temporal_embedding[:,:t] # ns,t,d
        image_hidden_states = einops.rearrange(image_hidden_states,'(n s) t d -> (n t) s d',n=n) # nt,s,d
        # self.embedding_dropout(motion_hidden_states)

        # Transformer blocks        
        if self.motion_type == 'decouple' and object_motion_source_hidden_states != None:
            # motion_hidden_states赋值相机运动
            motion_hidden_states = camera_motion_hidden_states
            for i,(block,s_block) in enumerate(zip(self.transformer_blocks,self.spatial_blocks)):
                if i < 6:
                    motion_hidden_states, image_hidden_states = block(
                        hidden_states=motion_hidden_states,
                        encoder_hidden_states=image_hidden_states,
                        temb=emb,
                    )

                    image_hidden_states = einops.rearrange(image_hidden_states,'(n t) s d -> (n s) t d',n=n)
                    image_hidden_states = s_block(
                        hidden_states = image_hidden_states,
                        temb = emb_s,
                    )

                    image_hidden_states = einops.rearrange(image_hidden_states,'(n s) t d -> (n t) s d',n=n)

            if camera_motion_source_hidden_states != None:
                # 将motion latents中,camera_motion部分替换成高频object_motion
                source_token = motion_hidden_states[:,0,:].unsqueeze(1)
                camera_motion_source_hidden_states = motion_hidden_states[:,1:1+L,:]
                target_token = motion_hidden_states[:,1+L,:].unsqueeze(1)
                camera_motion_target_hidden_states = motion_hidden_states[:,2+L:,:]
            else:
                target_token = motion_hidden_states[:,0,:].unsqueeze(1)
                camera_motion_target_hidden_states = motion_hidden_states[:,1:,:]
            
            motion_hidden_states = torch.cat([source_token,object_motion_source_hidden_states,target_token,object_motion_target_hidden_states],dim=1) # [N,2S+2,D]
            
            for i,(block,s_block) in enumerate(zip(self.transformer_blocks,self.spatial_blocks)):
                if i>=6:
                    motion_hidden_states, image_hidden_states = block(
                        hidden_states=motion_hidden_states,
                        encoder_hidden_states=image_hidden_states,
                        temb=emb,
                    )

                    image_hidden_states = einops.rearrange(image_hidden_states,'(n t) s d -> (n s) t d',n=n)
                    image_hidden_states = s_block(
                        hidden_states = image_hidden_states,
                        temb = emb_s,
                    )

                    image_hidden_states = einops.rearrange(image_hidden_states,'(n s) t d -> (n t) s d',n=n)

        else:
            if self.motion_type == 'decouple':
                motion_hidden_states = camera_motion_hidden_states
            for i,(block,s_block) in enumerate(zip(self.transformer_blocks,self.spatial_blocks)):
                motion_hidden_states, image_hidden_states = block(
                            hidden_states=motion_hidden_states,
                            encoder_hidden_states=image_hidden_states,
                            temb=emb,
                        )
                
                image_hidden_states = einops.rearrange(image_hidden_states,'(n t) s d -> (n s) t d',n=n)
                image_hidden_states = s_block(
                    hidden_states = image_hidden_states,
                    temb = emb_s,
                )

                image_hidden_states = einops.rearrange(image_hidden_states,'(n s) t d -> (n t) s d',n=n)
         

        image_hidden_states = self.norm_final(image_hidden_states)

        # 6. Final block
        image_hidden_states = self.norm_out(image_hidden_states, temb=emb)
        image_hidden_states = self.proj_out(image_hidden_states)

        # 7. Unpatchify
        p = self.image_patch_size
        output = image_hidden_states.reshape(N, 1, Hi // p, Wi // p, self.out_channels, p, p)
        output = output.permute(0, 1, 4, 2, 5, 3, 6).flatten(5, 6).flatten(3, 4).squeeze(1) # [N,C,H,W]

        return output


class AMDDiffusionTransformerModelSplitInput(nn.Module):
    """
    Diffusion Transformer
    """

    _supports_gradient_checkpointing = True
    def __init__(
        self,
        num_attention_heads: int = 30,
        attention_head_dim: int = 64,
        image_in_channels: Optional[int] = 4,
        motion_in_channels: Optional[int] = 16,
        out_channels: Optional[int] = 4,
        num_layers: int = 16,
        
        image_width: int = 64,
        image_height: int = 64,
        motion_width: int = 8,
        motion_height: int = 8,
        image_patch_size: int = 2,
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
        """
        
        Traning:
        Z(N,1,C,H,W)
        Motion(N,k,d,h,w)
        
        Inference:
        Z(N,1,C,H,W)
        Motion(N,k,d,h,w)
        """
        super().__init__()
        
        # 1. Setting
        hidden_dim = num_attention_heads * attention_head_dim
        iph = image_height // image_patch_size
        ipw = image_width // image_patch_size
        itl = 2*iph * ipw # image token length
        mph = motion_height // motion_patch_size
        mpw = motion_width // motion_patch_size
        mtl = mph * mpw * motion_frames # motion token num
        self.max_seq_length = itl + mtl
        
        self.image_patch_size = image_patch_size
        self.motion_patch_size = motion_patch_size
        self.out_channels = out_channels

        # 2. Patch embedding (N,F,C,H,W) -> (B,S,D)
        self.zi_patch_embed = PatchEmbed(patch_size=image_patch_size,in_channels=image_in_channels//2,embed_dim=hidden_dim, bias=True)
        self.zt_patch_embed = PatchEmbed(patch_size=image_patch_size,in_channels=image_in_channels//2,embed_dim=hidden_dim, bias=True)
        self.motion_patch_embed = PatchEmbed(patch_size=motion_patch_size,in_channels=motion_in_channels,embed_dim=hidden_dim, bias=True)

        self.embedding_dropout = nn.Dropout(dropout)

        # # 3. 2D&3D positional embeddings
        # image_pos_embedding = get_2d_sincos_pos_embed(hidden_dim, (iph, ipw)) # (iph*ipw,D)
        # image_pos_embedding = torch.from_numpy(image_pos_embedding) # (iph*ipw,D)
        # pos_embedding = torch.zeros(1, itl, hidden_dim, requires_grad=False)
        # pos_embedding.data[:, :itl].copy_(image_pos_embedding)
        # self.register_buffer("pos_embedding", pos_embedding, persistent=False)

        # 3. 3D positional embeddings
        spatial_pos_embedding = get_3d_sincos_pos_embed(  
            hidden_dim,
            (iph, ipw),
            2,
            spatial_interpolation_scale,
            temporal_interpolation_scale,
        ) 
        spatial_pos_embedding = torch.from_numpy(spatial_pos_embedding).flatten(0, 1) # [T*H*W, D]
        # pos_embedding = spatial_pos_embedding.unsqueeze(0)# [1,T*H*W, D]
        
        pos_embedding = torch.zeros(1,*spatial_pos_embedding.shape, requires_grad=False)
        pos_embedding.data.copy_(spatial_pos_embedding)
        self.register_buffer("pos_embedding", pos_embedding, persistent=False)
        
        # # 4. Token <pad>
        # self.pad = nn.Parameter(torch.zeros(1, 1, hidden_dim),requires_grad=False)

        # 5. Time embeddings
        self.time_proj = Timesteps(hidden_dim, flip_sin_to_cos, freq_shift)
        self.time_embedding = TimestepEmbedding(hidden_dim, time_embed_dim, timestep_activation_fn)

        # 6. Define spatio-temporal transformers blocks
        self.transformer_blocks = nn.ModuleList(
            [
                AMDTransformerBlock(
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
            output_dim=2 * hidden_dim,
            norm_elementwise_affine=norm_elementwise_affine,
            norm_eps=norm_eps,
            chunk_dim=1,
        )
        self.proj_out = nn.Linear(hidden_dim, image_patch_size * image_patch_size * out_channels)

        self.gradient_checkpointing = False

    def _set_gradient_checkpointing(self, module, value=False):
        self.gradient_checkpointing = value

    def forward(
        self,
        motion_hidden_states: torch.Tensor,
        image_hidden_states: torch.Tensor,
        timestep: Union[int, float, torch.LongTensor], # Timesteps should be a 1d-array
        timestep_cond: Optional[torch.Tensor] = None,
        return_dict: bool = True,
    ):
        """
        motion_hidden_states : (b,d,h,w)
        image_hidden_states : (b,2c,H,W)
        time_step : (b,)

        """
        N,Ci,Hi,Wi = image_hidden_states.shape
        N,Cm,Hm,Wm = motion_hidden_states.shape
        image_seq_length = 2 * Hi * Wi // (self.image_patch_size**2)
        motion_seq_length = Hm * Wm  // (self.motion_patch_size**2)

        zi = image_hidden_states[:,:Ci//2]
        zt = image_hidden_states[:,Ci//2:]

        # 1. Time embedding   # Timesteps should be a 1d-array
        t_emb = self.time_proj(timestep)
        t_emb = t_emb.to(dtype=motion_hidden_states.dtype)
        emb = self.time_embedding(t_emb, timestep_cond)

        # 2. Patch embedding
        motion_hidden_states = self.motion_patch_embed(motion_hidden_states) # [N,S,D]
        zi_hidden_states = self.zi_patch_embed(zi)
        zt_hidden_states = self.zt_patch_embed(zt)
        image_hidden_states = torch.cat((zi_hidden_states,zt_hidden_states),dim=1)

        assert image_seq_length == image_hidden_states.shape[1] , f"image_seq_length : {image_seq_length} != image_hidden_states.shape[1] : {image_hidden_states.shape[1]}"

        # 3. Position embedding
        image_hidden_states = image_hidden_states + self.pos_embedding[:, :image_seq_length]
        self.embedding_dropout(motion_hidden_states)

        # 4. Transformer blocks
        for i, block in enumerate(self.transformer_blocks):
            if self.training and self.gradient_checkpointing:

                def create_custom_forward(module):
                    def custom_forward(*inputs):
                        return module(*inputs)

                    return custom_forward

                ckpt_kwargs: Dict[str, Any] = {"use_reentrant": False} if is_torch_version(">=", "1.11.0") else {}
                motion_hidden_states, image_hidden_states = torch.utils.checkpoint.checkpoint(
                    create_custom_forward(block),
                    motion_hidden_states,
                    image_hidden_states,
                    emb,
                    **ckpt_kwargs,
                )
            else:
                motion_hidden_states, image_hidden_states = block(
                    hidden_states=motion_hidden_states,
                    encoder_hidden_states=image_hidden_states,
                    temb=emb,
                )

        pre = image_hidden_states[:,image_seq_length//2:] # (N,S,D)
        pre = self.norm_final(pre)

        # 6. Final block
        pre = self.norm_out(pre, temb=emb)
        pre = self.proj_out(pre)

        # 7. Unpatchify
        p = self.image_patch_size
        output = pre.reshape(N, 1, Hi // p, Wi // p, self.out_channels, p, p)
        output = output.permute(0, 1, 4, 2, 5, 3, 6).flatten(5, 6).flatten(3, 4).squeeze(1) # [N,C,H,W]

        return output


class DiffusionTransformerModel2Condition(nn.Module):
    """
    Diffusion Transformer
    """

    _supports_gradient_checkpointing = True
    def __init__(
        self,
        num_attention_heads: int = 30,
        attention_head_dim: int = 64,
        image_in_channels: Optional[int] = 4,
        motion_in_channels: Optional[int] = 16,
        out_channels: Optional[int] = 4,
        num_layers: int = 16,
        
        image_width: int = 32,
        image_height: int = 32,
        motion_width: int = 8,
        motion_height: int = 8,
        image_patch_size: int = 2,
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
        """
        
        Traning:
        Z(N,1,C,H,W)
        Motion(N,k,d,h,w)
        
        Inference:
        Z(N,1,C,H,W)
        Motion(N,k,d,h,w)
        """
        super().__init__()
        
        # 1. Setting
        hidden_dim = num_attention_heads * attention_head_dim
        iph = image_height // image_patch_size
        ipw = image_width // image_patch_size
        itl = iph * ipw # image token length

        mph = motion_height // motion_patch_size
        mpw = motion_width // motion_patch_size
        mtl = mph * mpw * motion_frames # motion token num

        self.max_seq_length = 2 * itl + mtl
        
        self.image_patch_size = image_patch_size
        self.motion_patch_size = motion_patch_size
        self.out_channels = out_channels

        # 2. Patch embedding (N,F,C,H,W) -> (B,S,D)
        self.image_patch_embed = PatchEmbed(patch_size=image_patch_size,in_channels=image_in_channels,embed_dim=hidden_dim, bias=True)
        self.refimg_patch_embed = PatchEmbed(patch_size=image_patch_size,in_channels=image_in_channels,embed_dim=hidden_dim, bias=True)
        self.motion_patch_embed = PatchEmbed(patch_size=motion_patch_size,in_channels=motion_in_channels,embed_dim=hidden_dim, bias=True)
        # self.patch_embed = CogVideoXPatchEmbed(patch_size, in_channels, inner_dim, text_embed_dim, bias=True)
        self.embedding_dropout = nn.Dropout(dropout)

        # 3. 3D positional embeddings for img
        spatial_pos_embedding = get_3d_sincos_pos_embed(  
            hidden_dim,
            (iph, iph),
            2,
            spatial_interpolation_scale,
            temporal_interpolation_scale,
        ) 
        spatial_pos_embedding = torch.from_numpy(spatial_pos_embedding).flatten(0, 1) # [T*H*W, D]
        pos_embedding = torch.zeros(1,*spatial_pos_embedding.shape, requires_grad=False)
        pos_embedding.data.copy_(spatial_pos_embedding)
        self.register_buffer("img_pos_embedding", pos_embedding, persistent=False)

        #  3D positional embeddings for motion
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
            output_dim=2 * hidden_dim,
            norm_elementwise_affine=norm_elementwise_affine,
            norm_eps=norm_eps,
            chunk_dim=1,
        )
        self.proj_out = nn.Linear(hidden_dim, image_patch_size * image_patch_size * out_channels)

        self.gradient_checkpointing = False

    def _set_gradient_checkpointing(self, module, value=False):
        self.gradient_checkpointing = value

    def forward(
        self,
        hidden_states: torch.Tensor,
        refimg_hidden_states: torch.Tensor,  # condition1
        motion_hidden_states: torch.Tensor,  # condition2
        timestep: Union[int, float, torch.LongTensor], # Timesteps should be a 1d-array
        timestep_cond: Optional[torch.Tensor] = None,
        return_dict: bool = True,
    ):
        """
        hidden_states : (b,c,H,W)
        motion_hidden_states : (b,d,h,w)
        refimg_hidden_states : (b,c,H,W)
        time_step : (b,)

        """

        N,Ci,Hi,Wi = hidden_states.shape
        N,Cm,Hm,Wm = motion_hidden_states.shape
        image_seq_length = Hi * Wi // (self.image_patch_size**2)
        motion_seq_length = Hm * Wm  // (self.motion_patch_size**2)
        
        # 1. Time embedding   # Timesteps should be a 1d-array
        t_emb = self.time_proj(timestep)
        t_emb = t_emb.to(dtype=motion_hidden_states.dtype)
        emb = self.time_embedding(t_emb, timestep_cond)

        # 2. Patch embedding
        hidden_states = self.image_patch_embed(hidden_states)    # [N,T,D]
        motion_hidden_states = self.motion_patch_embed(motion_hidden_states) # [N,S,D] condition
        refimg_hidden_states = self.refimg_patch_embed(refimg_hidden_states) # [N,S,D] condition

        # 3. Position embedding
        hidden_states = hidden_states + self.img_pos_embedding[:,:image_seq_length,:]
        refimg_hidden_states = refimg_hidden_states + self.img_pos_embedding[:, image_seq_length:,:]
        motion_hidden_states = motion_hidden_states + self.motion_pos_embedding[:, :motion_seq_length,:]
        self.embedding_dropout(motion_hidden_states)

        # 4. Transformer blocks
        for i, block in enumerate(self.transformer_blocks):
            if self.training and self.gradient_checkpointing:

                def create_custom_forward(module):
                    def custom_forward(*inputs):
                        return module(*inputs)

                    return custom_forward

                ckpt_kwargs: Dict[str, Any] = {"use_reentrant": False} if is_torch_version(">=", "1.11.0") else {}
                motion_hidden_states, refimg_hidden_states,motion_hidden_states = torch.utils.checkpoint.checkpoint(
                    create_custom_forward(block),
                    hidden_states,
                    refimg_hidden_states,
                    motion_hidden_states,
                    emb,
                    **ckpt_kwargs,
                )
            else:
                hidden_states, refimg_hidden_states,motion_hidden_states = block(
                    hidden_states,
                    refimg_hidden_states,
                    motion_hidden_states,
                    emb,
                )

        hidden_states = self.norm_final(hidden_states)

        # 6. Final block
        hidden_states = self.norm_out(hidden_states, temb=emb)
        hidden_states = self.proj_out(hidden_states)

        # 7. Unpatchify
        p = self.image_patch_size
        output = hidden_states.reshape(N, 1, Hi // p, Wi // p, self.out_channels, p, p)
        output = output.permute(0, 1, 4, 2, 5, 3, 6).flatten(5, 6).flatten(3, 4).squeeze(1) # [N,C,H,W]

        return output



# ------------ A2M ---------------

# Audio + mition_ref + motion_t
class AudioMitionref_LearnableToken(nn.Module):
    _supports_gradient_checkpointing = True
    def __init__(
        self,
        motion_num_token: int = 12,
        motion_inchannel: int = 128,
        motion_frames: int = 128,

        extra_in_channels: Optional[int] = 768,
        out_channels: Optional[int] = 128,

        num_attention_heads: int = 8,
        attention_head_dim: int = 64,
        num_layers: int = 16,

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
        self.out_channels = out_channels
        self.motion_frames = motion_frames
        self.motion_num_token = motion_num_token
        self.motion_num_tokens = motion_num_token * motion_frames

        # 2. Patch embedding (N,F,C,H,W) -> (B,S,D)
        self.refmotion_patch_embed = nn.Linear(motion_inchannel,hidden_dim)
        self.motion_patch_embed = nn.Linear(motion_inchannel,hidden_dim)
        self.extra_embed = nn.Linear(extra_in_channels,hidden_dim)
        self.embedding_dropout = nn.Dropout(dropout)

        # 3. 1d positional embeddings
        temporal_embedding = get_1d_sincos_pos_embed_from_grid(hidden_dim,torch.arange(self.motion_num_token + self.motion_num_tokens))
        motion_pos_embedding = torch.zeros(1,*temporal_embedding.shape,requires_grad=False)
        motion_pos_embedding.data.copy_(torch.from_numpy(temporal_embedding))
        self.register_buffer("motion_pos_embedding",motion_pos_embedding,persistent=False)

        temporal_embedding = get_1d_sincos_pos_embed_from_grid(hidden_dim,torch.arange(self.motion_frames))
        audio_pos_embedding = torch.zeros(1,*temporal_embedding.shape,requires_grad=False)
        audio_pos_embedding.data.copy_(torch.from_numpy(temporal_embedding))
        self.register_buffer("audio_pos_embedding",audio_pos_embedding,persistent=False)
            
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
        self.proj_out = nn.Linear(hidden_dim, out_channels)

        self.gradient_checkpointing = False

    def _set_gradient_checkpointing(self, module, value=False):
        self.gradient_checkpointing = value

    def forward(
        self,
        motion_hidden_states: torch.Tensor,
        refmotion_hidden_states: torch.Tensor,
        extra_hidden_states: torch.Tensor,
        timestep: Union[int, float, torch.LongTensor], # Timesteps should be a 1d-array
        timestep_cond: Optional[torch.Tensor] = None,
        return_dict: bool = True,
    ):
        """
        motion_hidden_states : (N,F,L,D)   
        refmotion_hidden_states : (N,L,D)  
        pose_hidden_states: (N,C,H,W) 
        extra_hidden_states  : (N,F,D)   这个需要提前做 position encoding if needed ,前面两个固定在这里做position encoding
        """
        assert motion_hidden_states.shape[1] == extra_hidden_states.shape[1],f"motion {motion_hidden_states.shape} ,audio{extra_hidden_states.shape}"

        N,L,D = refmotion_hidden_states.shape
        N,F,L,D = motion_hidden_states.shape
        
        # 1. Time embedding   # Timesteps should be a 1d-array
        t_emb = self.time_proj(timestep)
        t_emb = t_emb.to(dtype=motion_hidden_states.dtype)
        emb = self.time_embedding(t_emb, timestep_cond) # (batch,time_embed_dim) (8,512)

        # 2. Patch embedding
        motion_hidden_states = einops.rearrange(motion_hidden_states,'n f l d -> n (f l) d')
        motion_hidden_states = self.motion_patch_embed(motion_hidden_states) # [N,FL,D]
        ref_motion_hidden_states = self.refmotion_patch_embed(refmotion_hidden_states) # [N,L,D]
        extra_hidden_states  = self.extra_embed(extra_hidden_states) # [N,F,D]

        # 3. Position embedding
        ref_motion_hidden_states = ref_motion_hidden_states + self.motion_pos_embedding[:,:L,:]
        motion_hidden_states = motion_hidden_states + self.motion_pos_embedding[:,L:,:]
        extra_hidden_states = extra_hidden_states + self.audio_pos_embedding
        self.embedding_dropout(motion_hidden_states)

        # 4. Transformer blocks
        for i, block in enumerate(self.transformer_blocks):
            if self.training and self.gradient_checkpointing:

                def create_custom_forward(module):
                    def custom_forward(*inputs):
                        return module(*inputs)

                    return custom_forward

                ckpt_kwargs: Dict[str, Any] = {"use_reentrant": False} if is_torch_version(">=", "1.11.0") else {}
                motion_hidden_states,ref_motion_hidden_states,extra_hidden_states = torch.utils.checkpoint.checkpoint(
                    create_custom_forward(block),
                    motion_hidden_states,
                    ref_motion_hidden_states,
                    extra_hidden_states,
                    emb,
                    **ckpt_kwargs,
                )
            else:
                motion_hidden_states,ref_motion_hidden_states,extra_hidden_states = block(
                    motion_hidden_states,
                    ref_motion_hidden_states,
                    extra_hidden_states,
                    temb=emb,
                )

        motion_hidden_states = self.norm_final(motion_hidden_states)

        # 6. Final block
        motion_hidden_states = self.norm_out(motion_hidden_states, temb=emb)
        motion_hidden_states = self.proj_out(motion_hidden_states) # [N,L1,D]

        # 7. Unpatchify
        output = einops.rearrange(motion_hidden_states,'n (f l) d -> n f l d',f=F)
        
        return output # N,F,L,D


# Audio + mition_ref + motion_t
class AudioMitionref_LearnableToken_SimpleAdaLN(nn.Module):
    _supports_gradient_checkpointing = True
    def __init__(
        self,
        motion_num_token: int = 12,
        motion_inchannel: int = 128,
        motion_frames: int = 128,

        extra_in_channels: Optional[int] = 768,
        out_channels: Optional[int] = 128,

        num_attention_heads: int = 8,
        attention_head_dim: int = 64,
        num_layers: int = 16,

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
        self.out_channels = out_channels
        self.motion_frames = motion_frames
        self.motion_num_token = motion_num_token
        self.motion_num_tokens = motion_num_token * motion_frames

        # 2. Patch embedding (N,F,C,H,W) -> (B,S,D)
        self.refmotion_patch_embed = nn.Linear(motion_inchannel,hidden_dim)
        self.motion_patch_embed = nn.Linear(motion_inchannel,hidden_dim)
        self.extra_embed = nn.Linear(extra_in_channels,hidden_dim)
        self.embedding_dropout = nn.Dropout(dropout)

        # 3. 1d positional embeddings
        temporal_embedding = get_1d_sincos_pos_embed_from_grid(hidden_dim,torch.arange(self.motion_num_token + self.motion_num_tokens))
        motion_pos_embedding = torch.zeros(1,*temporal_embedding.shape,requires_grad=False)
        motion_pos_embedding.data.copy_(torch.from_numpy(temporal_embedding))
        self.register_buffer("motion_pos_embedding",motion_pos_embedding,persistent=False)

        temporal_embedding = get_1d_sincos_pos_embed_from_grid(hidden_dim,torch.arange(self.motion_frames))
        audio_pos_embedding = torch.zeros(1,*temporal_embedding.shape,requires_grad=False)
        audio_pos_embedding.data.copy_(torch.from_numpy(temporal_embedding))
        self.register_buffer("audio_pos_embedding",audio_pos_embedding,persistent=False)
            
        # 5. Time embeddings
        self.time_proj = Timesteps(hidden_dim, flip_sin_to_cos, freq_shift)
        self.time_embedding = TimestepEmbedding(hidden_dim, time_embed_dim, timestep_activation_fn)

        # 6. Define spatio-temporal transformers blocks
        self.transformer_blocks = nn.ModuleList(
            [
                TransformerBlock2Condition_SimpleAdaLN(
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
        self.proj_out = nn.Linear(hidden_dim, out_channels)

        self.gradient_checkpointing = False

    def _set_gradient_checkpointing(self, module, value=False):
        self.gradient_checkpointing = value

    def forward(
        self,
        motion_hidden_states: torch.Tensor,
        refmotion_hidden_states: torch.Tensor,
        extra_hidden_states: torch.Tensor,
        timestep: Union[int, float, torch.LongTensor], # Timesteps should be a 1d-array
        timestep_cond: Optional[torch.Tensor] = None,
        return_dict: bool = True,
    ):
        """
        motion_hidden_states : (N,F,L,D)   
        refmotion_hidden_states : (N,L,D)  
        pose_hidden_states: (N,C,H,W) 
        extra_hidden_states  : (N,F,D)   这个需要提前做 position encoding if needed ,前面两个固定在这里做position encoding
        """
        assert motion_hidden_states.shape[1] == extra_hidden_states.shape[1],f"motion {motion_hidden_states.shape} ,audio{extra_hidden_states.shape}"

        N,L,D = refmotion_hidden_states.shape
        N,F,L,D = motion_hidden_states.shape
        
        # 1. Time embedding   # Timesteps should be a 1d-array
        t_emb = self.time_proj(timestep)
        t_emb = t_emb.to(dtype=motion_hidden_states.dtype)
        emb = self.time_embedding(t_emb, timestep_cond) # (batch,time_embed_dim) (8,512)

        # 2. Patch embedding
        motion_hidden_states = einops.rearrange(motion_hidden_states,'n f l d -> n (f l) d')
        motion_hidden_states = self.motion_patch_embed(motion_hidden_states) # [N,FL,D]
        ref_motion_hidden_states = self.refmotion_patch_embed(refmotion_hidden_states) # [N,L,D]
        extra_hidden_states  = self.extra_embed(extra_hidden_states) # [N,F,D]

        # 3. Position embedding
        ref_motion_hidden_states = ref_motion_hidden_states + self.motion_pos_embedding[:,:L,:]
        motion_hidden_states = motion_hidden_states + self.motion_pos_embedding[:,L:,:]
        extra_hidden_states = extra_hidden_states + self.audio_pos_embedding
        self.embedding_dropout(motion_hidden_states)

        # 4. Transformer blocks
        for i, block in enumerate(self.transformer_blocks):
            if self.training and self.gradient_checkpointing:

                def create_custom_forward(module):
                    def custom_forward(*inputs):
                        return module(*inputs)

                    return custom_forward

                ckpt_kwargs: Dict[str, Any] = {"use_reentrant": False} if is_torch_version(">=", "1.11.0") else {}
                motion_hidden_states,ref_motion_hidden_states,extra_hidden_states = torch.utils.checkpoint.checkpoint(
                    create_custom_forward(block),
                    motion_hidden_states,
                    ref_motion_hidden_states,
                    extra_hidden_states,
                    emb,
                    **ckpt_kwargs,
                )
            else:
                motion_hidden_states,ref_motion_hidden_states,extra_hidden_states = block(
                    motion_hidden_states,
                    ref_motion_hidden_states,
                    extra_hidden_states,
                    temb=emb,
                )

        motion_hidden_states = self.norm_final(motion_hidden_states)

        # 6. Final block
        motion_hidden_states = self.norm_out(motion_hidden_states, temb=emb)
        motion_hidden_states = self.proj_out(motion_hidden_states) # [N,L1,D]

        # 7. Unpatchify
        output = einops.rearrange(motion_hidden_states,'n (f l) d -> n f l d',f=F)
        
        return output # N,F,L,D


# Audio 
class A2MTransformer_CrossAttn_Audio(nn.Module):
    _supports_gradient_checkpointing = True
    def __init__(
        self,
        motion_num_token: int = 12,
        motion_inchannel: int = 128,
        motion_frames: int = 128,

        audio_window : Optional[int] = 12,
        audio_in_channels: Optional[int] = 128,
        out_channels: Optional[int] = 128,

        num_attention_heads: int = 8,
        attention_head_dim: int = 64,
        num_layers: int = 16,

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
        self.out_channels = out_channels
        self.motion_frames = motion_frames
        self.motion_num_token = motion_num_token
        self.motion_num_tokens = motion_num_token * motion_frames

        # 2. Patch embedding (N,F,C,H,W) -> (B,S,D)
        self.refmotion_patch_embed = nn.Linear(motion_inchannel,hidden_dim)
        self.motion_patch_embed = nn.Linear(motion_inchannel,hidden_dim)
        self.audio_embed = nn.Linear(audio_in_channels,hidden_dim)
        self.embedding_dropout = nn.Dropout(dropout)

        # 3. 1d positional embeddings
        temporal_embedding = get_1d_sincos_pos_embed_from_grid(hidden_dim,torch.arange(self.motion_num_token + self.motion_num_tokens))
        motion_pos_embedding = torch.zeros(1,*temporal_embedding.shape,requires_grad=False)
        motion_pos_embedding.data.copy_(torch.from_numpy(temporal_embedding))
        self.register_buffer("motion_pos_embedding",motion_pos_embedding,persistent=False)
            
        # 5. Time embeddings
        self.time_proj = Timesteps(hidden_dim, flip_sin_to_cos, freq_shift)
        self.time_embedding = TimestepEmbedding(hidden_dim, time_embed_dim, timestep_activation_fn)

        # 6. Define spatio-temporal transformers blocks
        self.motion_blocks = nn.ModuleList(
            [
                A2MMotionSelfAttnBlock(
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

        self.audio_blocks = nn.ModuleList(
            [
                A2MCrossAttnBlock(
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
        self.proj_out = nn.Linear(hidden_dim, out_channels)

        self.gradient_checkpointing = False

    def _set_gradient_checkpointing(self, module, value=False):
        self.gradient_checkpointing = value

    def forward(
        self,
        motion_hidden_states: torch.Tensor,
        refmotion_hidden_states: torch.Tensor,
        audio_hidden_states: torch.Tensor,
        timestep: Union[int, float, torch.LongTensor], # Timesteps should be a 1d-array
        timestep_cond: Optional[torch.Tensor] = None,
        return_dict: bool = True,
    ):
        """
        motion_hidden_states : (N,F,L,D)   
        refmotion_hidden_states : (N,L,D)  
        audio_hidden_states  : (N,F+1,W,D)  
        """
        assert motion_hidden_states.shape[1]+1 == audio_hidden_states.shape[1],f"motion {motion_hidden_states.shape} ,audio{audio_hidden_states.shape}"

        N,L,D = refmotion_hidden_states.shape
        N,F,L,D = motion_hidden_states.shape
        
        # 1. Time embedding   # Timesteps should be a 1d-array
        t_emb = self.time_proj(timestep)
        t_emb = t_emb.to(dtype=motion_hidden_states.dtype)
        emb = self.time_embedding(t_emb, timestep_cond) # (batch,time_embed_dim) (8,512)

        # 2. Patch embedding
        motion_hidden_states = einops.rearrange(motion_hidden_states,'n f l d -> n (f l) d')
        motion_hidden_states = self.motion_patch_embed(motion_hidden_states) # [N,FL,D]
        ref_motion_hidden_states = self.refmotion_patch_embed(refmotion_hidden_states) # [N,L,D]
        audio_hidden_states  = self.audio_embed(audio_hidden_states) # [N,F+1,W,D]

        # 3. Position embedding
        ref_motion_hidden_states = ref_motion_hidden_states + self.motion_pos_embedding[:,:L,:]
        motion_hidden_states = motion_hidden_states + self.motion_pos_embedding[:,L:L+motion_hidden_states.shape[1],:]
        self.embedding_dropout(motion_hidden_states)

        # 4. Transformer blocks
        for motion_block,audio_block in zip(self.motion_blocks,self.audio_blocks):

            motion_hidden_states,ref_motion_hidden_states = motion_block(
                motion_hidden_states,
                ref_motion_hidden_states,
                temb=emb,
            )

            motion_hidden_states,ref_motion_hidden_states = audio_block(
                motion_hidden_states,
                ref_motion_hidden_states,
                audio_hidden_states,
                temb=emb,
            )            

        motion_hidden_states = self.norm_final(motion_hidden_states)

        # 6. Final block
        motion_hidden_states = self.norm_out(motion_hidden_states, temb=emb)
        motion_hidden_states = self.proj_out(motion_hidden_states) # [N,L1,D]

        # 7. Unpatchify
        output = einops.rearrange(motion_hidden_states,'n (f l) d -> n f l d',f=F)
        
        return output # N,F,L,D

# Audio + dwpose
class A2MTransformer_CrossAttn_Audio_Pose(nn.Module):
    _supports_gradient_checkpointing = True
    def __init__(
        self,
        motion_num_token: int = 12,
        motion_inchannel: int = 128,
        motion_frames: int = 128,

        audio_window : Optional[int] = 12,
        audio_in_channels: Optional[int] = 128,
        out_channels: Optional[int] = 128,

        pose_height : int = 32,
        pose_width : int = 32,
        pose_inchannel : int = 4,
        pose_patch_size : int = 2,

        num_attention_heads: int = 8,
        attention_head_dim: int = 64,
        num_layers: int = 16,

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
        self.out_channels = out_channels
        self.motion_frames = motion_frames
        self.motion_num_token = motion_num_token
        self.motion_num_tokens = motion_num_token * motion_frames

        # 2. Patch embedding (N,F,C,H,W) -> (B,S,D)
        self.refmotion_patch_embed = nn.Linear(motion_inchannel,hidden_dim)
        self.motion_patch_embed = nn.Linear(motion_inchannel,hidden_dim)
        self.audio_embed = nn.Linear(audio_in_channels,hidden_dim)
        self.pose_embed =  PatchEmbed(patch_size=pose_patch_size,in_channels=pose_inchannel,embed_dim=hidden_dim, bias=True)
        self.embedding_dropout = nn.Dropout(dropout)

        # 3. 1d positional embeddings
        temporal_embedding = get_1d_sincos_pos_embed_from_grid(hidden_dim,torch.arange(self.motion_num_token + self.motion_num_tokens))
        motion_pos_embedding = torch.zeros(1,*temporal_embedding.shape,requires_grad=False)
        motion_pos_embedding.data.copy_(torch.from_numpy(temporal_embedding))
        self.register_buffer("motion_pos_embedding",motion_pos_embedding,persistent=False)

        # 4. 2d positional embeddings
        iph = pose_height // pose_patch_size
        ipw = pose_width // pose_patch_size
        itl = iph * ipw
        image_pos_embedding = get_2d_sincos_pos_embed(hidden_dim, (iph, ipw)) # (iph*ipw,D)
        image_pos_embedding = torch.from_numpy(image_pos_embedding) # (iph*ipw,D)
        pos_embedding = torch.zeros(1, itl, hidden_dim, requires_grad=False)
        pos_embedding.data[:, :itl].copy_(image_pos_embedding)
        self.register_buffer("pose_pos_embedding", pos_embedding, persistent=False)
            
        # 5. Time embeddings
        self.time_proj = Timesteps(hidden_dim, flip_sin_to_cos, freq_shift)
        self.time_embedding = TimestepEmbedding(hidden_dim, time_embed_dim, timestep_activation_fn)

        # 6. Define spatio-temporal transformers blocks
        self.motion_blocks = nn.ModuleList(
            [
                A2MMotionSelfAttnBlock(
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

        self.audio_blocks = nn.ModuleList(
            [
                A2MCrossAttnBlock(
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

        self.pose_blocks = nn.ModuleList(
            [
                A2MCrossAttnBlock(
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
        self.proj_out = nn.Linear(hidden_dim, out_channels)

        self.gradient_checkpointing = False

    def _set_gradient_checkpointing(self, module, value=False):
        self.gradient_checkpointing = value

    def forward(
        self,
        motion_hidden_states: torch.Tensor,
        refmotion_hidden_states: torch.Tensor,
        audio_hidden_states: torch.Tensor,
        pose_hidden_states: torch.Tensor,
        timestep: Union[int, float, torch.LongTensor], # Timesteps should be a 1d-array
        timestep_cond: Optional[torch.Tensor] = None,
        return_dict: bool = True,
    ):
        """
        motion_hidden_states : (N,F,L,D)   
        refmotion_hidden_states : (N,L,D)  
        audio_hidden_states  : (N,F+1,W,D)  
        pose_hidden_states : (N,F+1,c,h,w)
        """
        assert motion_hidden_states.shape[1]+1 == audio_hidden_states.shape[1],f"motion {motion_hidden_states.shape} ,audio{audio_hidden_states.shape}"

        N,L,D = refmotion_hidden_states.shape
        N,F,L,D = motion_hidden_states.shape
        
        # 1. Time embedding   # Timesteps should be a 1d-array
        t_emb = self.time_proj(timestep)
        t_emb = t_emb.to(dtype=motion_hidden_states.dtype)
        emb = self.time_embedding(t_emb, timestep_cond) # (batch,time_embed_dim) (8,512)

        # 2. Patch embedding
        motion_hidden_states = einops.rearrange(motion_hidden_states,'n f l d -> n (f l) d')
        motion_hidden_states = self.motion_patch_embed(motion_hidden_states) # [N,FL,D]
        ref_motion_hidden_states = self.refmotion_patch_embed(refmotion_hidden_states) # [N,L,D]
        audio_hidden_states  = self.audio_embed(audio_hidden_states) # [N,F+1,W,D]
        pose_hidden_states = self.pose_embed(pose_hidden_states.flatten(0,1)) # [N(F+1),S,D]

        # 3. Position embedding
        ref_motion_hidden_states = ref_motion_hidden_states + self.motion_pos_embedding[:,:L,:]
        motion_hidden_states = motion_hidden_states + self.motion_pos_embedding[:,L:L+motion_hidden_states.shape[1],:]
        pose_hidden_states = pose_hidden_states + self.pose_pos_embedding
        pose_hidden_states = einops.rearrange(pose_hidden_states,'(n f) l d -> n f l d',n=N)
        self.embedding_dropout(motion_hidden_states)

        # 4. Transformer blocks
        for motion_block,audio_block,pose_block in zip(self.motion_blocks,self.audio_blocks,self.pose_blocks):

            motion_hidden_states,ref_motion_hidden_states = motion_block(
                motion_hidden_states,
                ref_motion_hidden_states,
                temb=emb,
            )

            motion_hidden_states,ref_motion_hidden_states = audio_block(
                motion_hidden_states,
                ref_motion_hidden_states,
                audio_hidden_states,
                temb=emb,
            )           

            motion_hidden_states,ref_motion_hidden_states = pose_block(
                motion_hidden_states,
                ref_motion_hidden_states,
                pose_hidden_states,
                temb=emb,
            )     

        motion_hidden_states = self.norm_final(motion_hidden_states)

        # 6. Final block
        motion_hidden_states = self.norm_out(motion_hidden_states, temb=emb)
        motion_hidden_states = self.proj_out(motion_hidden_states) # [N,L1,D]

        # 7. Unpatchify
        output = einops.rearrange(motion_hidden_states,'n (f l) d -> n f l d',f=F)
        
        return output # N,F,L,D


# dwpose
class A2MTransformer_CrossAttn_Pose(nn.Module):
    _supports_gradient_checkpointing = True
    def __init__(
        self,
        motion_num_token: int = 12,
        motion_inchannel: int = 128,
        motion_frames: int = 128,

        out_channels: Optional[int] = 128,

        pose_height : int = 32,
        pose_width : int = 32,
        pose_inchannel : int = 4,
        pose_patch_size : int = 2,

        num_attention_heads: int = 8,
        attention_head_dim: int = 64,
        num_layers: int = 16,

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
        self.out_channels = out_channels
        self.motion_frames = motion_frames
        self.motion_num_token = motion_num_token
        self.motion_num_tokens = motion_num_token * motion_frames

        # 2. Patch embedding (N,F,C,H,W) -> (B,S,D)
        self.refmotion_patch_embed = nn.Linear(motion_inchannel,hidden_dim)
        self.motion_patch_embed = nn.Linear(motion_inchannel,hidden_dim)
        self.pose_embed =  PatchEmbed(patch_size=pose_patch_size,in_channels=pose_inchannel,embed_dim=hidden_dim, bias=True)
        self.embedding_dropout = nn.Dropout(dropout)

        # 3. 1d positional embeddings
        temporal_embedding = get_1d_sincos_pos_embed_from_grid(hidden_dim,torch.arange(self.motion_num_token + self.motion_num_tokens))
        motion_pos_embedding = torch.zeros(1,*temporal_embedding.shape,requires_grad=False)
        motion_pos_embedding.data.copy_(torch.from_numpy(temporal_embedding))
        self.register_buffer("motion_pos_embedding",motion_pos_embedding,persistent=False)

        # 4. 2d positional embeddings
        iph = pose_height // pose_patch_size
        ipw = pose_width // pose_patch_size
        itl = iph * ipw
        image_pos_embedding = get_2d_sincos_pos_embed(hidden_dim, (iph, ipw)) # (iph*ipw,D)
        image_pos_embedding = torch.from_numpy(image_pos_embedding) # (iph*ipw,D)
        pos_embedding = torch.zeros(1, itl, hidden_dim, requires_grad=False)
        pos_embedding.data[:, :itl].copy_(image_pos_embedding)
        self.register_buffer("pose_pos_embedding", pos_embedding, persistent=False)
            
        # 5. Time embeddings
        self.time_proj = Timesteps(hidden_dim, flip_sin_to_cos, freq_shift)
        self.time_embedding = TimestepEmbedding(hidden_dim, time_embed_dim, timestep_activation_fn)

        # 6. Define spatio-temporal transformers blocks
        self.motion_blocks = nn.ModuleList(
            [
                A2MMotionSelfAttnBlock(
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

        self.pose_blocks = nn.ModuleList(
            [
                A2MCrossAttnBlock(
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
        self.proj_out = nn.Linear(hidden_dim, out_channels)

        self.gradient_checkpointing = False

    def _set_gradient_checkpointing(self, module, value=False):
        self.gradient_checkpointing = value

    def forward(
        self,
        motion_hidden_states: torch.Tensor,
        refmotion_hidden_states: torch.Tensor,
        pose_hidden_states: torch.Tensor,
        timestep: Union[int, float, torch.LongTensor], # Timesteps should be a 1d-array
        timestep_cond: Optional[torch.Tensor] = None,
        return_dict: bool = True,
    ):
        """
        motion_hidden_states : (N,F,L,D)   
        refmotion_hidden_states : (N,L,D)  
        pose_hidden_states : (N,F+1,c,h,w)
        """


        N,L,D = refmotion_hidden_states.shape
        N,F,L,D = motion_hidden_states.shape
        
        # 1. Time embedding   # Timesteps should be a 1d-array
        t_emb = self.time_proj(timestep)
        t_emb = t_emb.to(dtype=motion_hidden_states.dtype)
        emb = self.time_embedding(t_emb, timestep_cond) # (batch,time_embed_dim) (8,512)

        # 2. Patch embedding
        motion_hidden_states = einops.rearrange(motion_hidden_states,'n f l d -> n (f l) d')
        motion_hidden_states = self.motion_patch_embed(motion_hidden_states) # [N,FL,D]
        ref_motion_hidden_states = self.refmotion_patch_embed(refmotion_hidden_states) # [N,L,D]
        pose_hidden_states = self.pose_embed(pose_hidden_states.flatten(0,1)) # [N(F+1),S,D]

        # 3. Position embedding
        ref_motion_hidden_states = ref_motion_hidden_states + self.motion_pos_embedding[:,:L,:]
        motion_hidden_states = motion_hidden_states + self.motion_pos_embedding[:,L:L+motion_hidden_states.shape[1],:]
        pose_hidden_states = pose_hidden_states + self.pose_pos_embedding
        pose_hidden_states = einops.rearrange(pose_hidden_states,'(n f) l d -> n f l d',n=N)
        self.embedding_dropout(motion_hidden_states)

        # 4. Transformer blocks
        for motion_block,pose_block in zip(self.motion_blocks,self.pose_blocks):

            motion_hidden_states,ref_motion_hidden_states = motion_block(
                motion_hidden_states,
                ref_motion_hidden_states,
                temb=emb,
            )     

            motion_hidden_states,ref_motion_hidden_states = pose_block(
                motion_hidden_states,
                ref_motion_hidden_states,
                pose_hidden_states,
                temb=emb,
            )     

        motion_hidden_states = self.norm_final(motion_hidden_states)

        # 6. Final block
        motion_hidden_states = self.norm_out(motion_hidden_states, temb=emb)
        motion_hidden_states = self.proj_out(motion_hidden_states) # [N,L1,D]

        # 7. Unpatchify
        output = einops.rearrange(motion_hidden_states,'n (f l) d -> n f l d',f=F)
        
        return output # N,F,L,D


# ------------- A2P ---------------

class A2PTransformer(nn.Module):
    _supports_gradient_checkpointing = True
    def __init__(
        self,

        audio_window : Optional[int] = 12,
        audio_in_channels: Optional[int] = 128,

        pose_height : int = 32,
        pose_width : int = 32,
        pose_inchannel : int = 4,
        pose_patch_size : int = 4,
        pose_frame : int = 17,

        num_attention_heads: int = 8,
        attention_head_dim: int = 64,
        num_layers: int = 16,

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
        
        # Setting
        hidden_dim = num_attention_heads * attention_head_dim
        self.out_channel = pose_inchannel
        self.pose_patch_size = pose_patch_size

        # Patch embedding (N,F,C,H,W) -> (B,S,D)
        self.pose_embed =  PatchEmbed(patch_size=pose_patch_size,in_channels=pose_inchannel,embed_dim=hidden_dim, bias=True)
        self.audio_embed = nn.Linear(audio_in_channels,hidden_dim)

        # pose mask token
        iph = pose_height // pose_patch_size
        ipw = pose_width // pose_patch_size
        itl = iph * ipw
        INIT_CONST = 0.02
        self.pose_mask_token = nn.Parameter(torch.randn(1, itl, hidden_dim) * INIT_CONST) # 1,L,D

        # 3d position embedding
        spatial_pos_embedding = get_3d_sincos_pos_embed(  
            hidden_dim,
            (iph, ipw),
            pose_frame,
            spatial_interpolation_scale,
            temporal_interpolation_scale,
        ) 
        spatial_pos_embedding = torch.from_numpy(spatial_pos_embedding).flatten(0, 1) # [T*H*W, D]
        pos_embedding = torch.zeros(1,*spatial_pos_embedding.shape, requires_grad=False)
        pos_embedding.data.copy_(spatial_pos_embedding)
        self.register_buffer("pose_pos_embedding", pos_embedding, persistent=False)
            

        # blocks
        self.temporal_spatial_blocks = nn.ModuleList(
            [
                A2PTemporalSpatialBlock(
                    dim=hidden_dim,
                    num_attention_heads=num_attention_heads,
                    attention_head_dim=attention_head_dim,
                    dropout=dropout,
                    activation_fn=activation_fn,
                    attention_bias=attention_bias,
                    norm_elementwise_affine=norm_elementwise_affine,
                    norm_eps=norm_eps,
                )
                for _ in range(num_layers)
            ]
        )

        self.audio_blocks = nn.ModuleList(
            [
                A2PCrossAudioBlock(
                    dim=hidden_dim,
                    num_attention_heads=num_attention_heads,
                    attention_head_dim=attention_head_dim,
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

        # Output blocks
        self.proj_out = nn.Linear(hidden_dim, pose_patch_size * pose_patch_size * pose_inchannel)

        self.gradient_checkpointing = False

    def _set_gradient_checkpointing(self, module, value=False):
        self.gradient_checkpointing = value

    def forward(
        self,
        ref_pose_hidden_states: torch.Tensor,
        audio_hidden_states: torch.Tensor, # N,F,W,D
    ):
        """
        ref_pose_hidden_states : (N,C,H,W)   
        audio_hidden_states  : (N,F+1,W,D)  
        """

        N,C,H,W = ref_pose_hidden_states.shape
        N,F,L,D = audio_hidden_states.shape

        # Audio patch embedding
        audio_hidden_states = einops.rearrange(audio_hidden_states,'n f w d -> (n f) w d') # NF,W,d
        audio_hidden_states = self.audio_embed(audio_hidden_states) # NF,W,D
        audio_hidden_states = einops.rearrange(audio_hidden_states,'(n f) w d -> n f w d',n=N) # N,F,W,d
        
        # pose patch embedding
        ref_pose_hidden_states = self.pose_embed(ref_pose_hidden_states) # N,L,D
        ref_pose_hidden_states = ref_pose_hidden_states.unsqueeze(1) # N,1,L,D
        pose_mask_hidden_states = self.pose_mask_token.unsqueeze(0).repeat(N,F-1,1,1) # N,F-1,L,D
        pose_hidden_states = torch.cat((ref_pose_hidden_states,pose_mask_hidden_states),dim=1) # N,F,L,D

        # Transformer blocks
        for st_block,audio_block in zip(self.temporal_spatial_blocks,self.audio_blocks):

            pose_hidden_states = st_block(
                pose_hidden_states,
            )

            pose_hidden_states = audio_block(
                pose_hidden_states,
                audio_hidden_states,
            )           

        pose_hidden_states = self.norm_final(pose_hidden_states)

        # 6. Final block
        pose_hidden_states = self.proj_out(pose_hidden_states) # [N,F,iph*ipw,p*p*outchannel]

        # 7. Unpatchify
        p = self.pose_patch_size
        output = pose_hidden_states.reshape(N, F, H // p, W // p, self.out_channel, p, p)
        output = output.permute(0, 1, 4, 2, 5, 3, 6).flatten(5, 6).flatten(3, 4) # N,F,C,H,W
        
        return output # N,F,C,H,W
