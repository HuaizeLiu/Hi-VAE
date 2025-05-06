# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.

# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
# --------------------------------------------------------
# References:
# timm: https://github.com/rwightman/pytorch-image-models/tree/master/timm
# DeiT: https://github.com/facebookresearch/deit
# --------------------------------------------------------

from functools import partial

import torch
import torch.nn as nn
from .modules import PatchEmbed
from timm.models.vision_transformer import Block
from diffusers.models.embeddings import TimestepEmbedding, Timesteps, get_3d_sincos_pos_embed,get_2d_sincos_pos_embed
from diffusers.configuration_utils import ConfigMixin, register_to_config
from diffusers.models.modeling_utils import ModelMixin
import time
import itertools

def count_time(func):
    def wrapper(*args, **kwargs):
        start = time.time()
        result = func(*args, **kwargs)
        print(f'{func.__name__} cost time: {time.time()-start}')
        return result
    return wrapper

class MaskedAutoencoderViT(ModelMixin, ConfigMixin):
    """ Masked Autoencoder with VisionTransformer backbone
    """

    @register_to_config    
    def __init__(self, img_size=(32,32), patch_size=2, in_chans=4,
                 embed_dim=1024, depth=24, num_heads=16,
                 decoder_embed_dim=512, decoder_depth=8, decoder_num_heads=16,
                 mlp_ratio=4., norm_layer=nn.LayerNorm, norm_pix_loss=False):
        super().__init__()

        # --------------------------------------------------------------------------
        # MAE encoder specifics

        assert img_size[0] == img_size[1] , "Only square images are supported"

        self.patch_embed = PatchEmbed(patch_size, in_chans, embed_dim)
        self.patch_size = patch_size
        self.num_patch_h = img_size[0] // patch_size
        self.num_patch_w = img_size[1] // patch_size
        self.num_patches = self.num_patch_h * self.num_patch_w
        self.in_chans = in_chans
        self.embed_dim = embed_dim

        self.cls_token = nn.Parameter(torch.zeros(1, 1, embed_dim))
        self.pos_embed = nn.Parameter(torch.zeros(1, self.num_patches + 1, embed_dim), requires_grad=False)  # fixed sin-cos embedding w cls

        self.blocks = nn.ModuleList([
            Block(embed_dim, num_heads, mlp_ratio, qkv_bias=True, norm_layer=norm_layer)
            for i in range(depth)])
        self.norm = norm_layer(embed_dim)
        # --------------------------------------------------------------------------

        # --------------------------------------------------------------------------
        # MAE decoder specifics
        self.decoder_embed = nn.Linear(embed_dim, decoder_embed_dim, bias=True)

        self.mask_token = nn.Parameter(torch.zeros(1, 1, decoder_embed_dim))

        self.decoder_pos_embed = nn.Parameter(torch.zeros(1, self.num_patches + 1, decoder_embed_dim), requires_grad=False)  # (1,1+patch,dim)

        self.decoder_blocks = nn.ModuleList([
            Block(decoder_embed_dim, decoder_num_heads, mlp_ratio, qkv_bias=True, norm_layer=norm_layer)
            for i in range(decoder_depth)])

        self.decoder_norm = norm_layer(decoder_embed_dim)
        self.decoder_pred = nn.Linear(decoder_embed_dim, patch_size**2 * in_chans, bias=True) # decoder to patch
        # --------------------------------------------------------------------------

        self.norm_pix_loss = norm_pix_loss

        self.initialize_weights()

    
    def initialize_weights(self):
        # initialization
        # initialize (and freeze) pos_embed by sin-cos embedding
        pos_embed = get_2d_sincos_pos_embed(self.pos_embed.shape[-1], (self.num_patch_h,self.num_patch_w), cls_token=True,extra_tokens=1) #[1+grid_size*grid_size, embed_dim] 
        self.pos_embed.data.copy_(torch.from_numpy(pos_embed).float().unsqueeze(0))

        decoder_pos_embed = get_2d_sincos_pos_embed(self.decoder_pos_embed.shape[-1], (self.num_patch_h,self.num_patch_w), cls_token=True,extra_tokens=1) #[1+grid_size*grid_size, embed_dim]
        self.decoder_pos_embed.data.copy_(torch.from_numpy(decoder_pos_embed).float().unsqueeze(0))

        # initialize patch_embed like nn.Linear (instead of nn.Conv2d)
        w = self.patch_embed.proj.weight.data
        torch.nn.init.xavier_uniform_(w.view([w.shape[0], -1]))

        # timm's trunc_normal_(std=.02) is effectively normal_(std=0.02) as cutoff is too big (2.)
        torch.nn.init.normal_(self.cls_token, std=.02)
        torch.nn.init.normal_(self.mask_token, std=.02)

        # initialize nn.Linear and nn.LayerNorm
        self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            # we use xavier_uniform following official JAX ViT:
            torch.nn.init.xavier_uniform_(m.weight)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)

    def patchify(self, imgs):
        """
        imgs: (N, C, H, W)
        x: (N, S, patch_size**2 *C)
        """
        _,c,_,_, = imgs.shape
        p = self.patch_size
        assert imgs.shape[2] == imgs.shape[3] and imgs.shape[2] % p == 0 # make sure H == W

        h = w = imgs.shape[2] // p  # num_patch
        x = imgs.reshape(shape=(imgs.shape[0], c, h, p, w, p)) # (N, c, h, p, w, p)
        x = torch.einsum('nchpwq->nhwpqc', x) # (N, h, w, p, p, c)
        x = x.reshape(shape=(imgs.shape[0], h * w, p**2 * c)) # (N, h*w, p*p*c)  = (N,S,D)
        return x # (N,S,D)

    def unpatchify(self, x):
        """
        x: (N, S, patch_size**2 *C)
        imgs: (N, C, H, W)
        """
        p = self.patch_size
        h = w = int(x.shape[1]**.5)
        # c = self.in_chans
        c = x.shape[2] // (p**2)
        assert h * w == x.shape[1]
        
        x = x.reshape(shape=(x.shape[0], h, w, p, p, c)) # (N, h, w, p, p, c)
        x = torch.einsum('nhwpqc->nchpwq', x) # (N, c, h, p, w, p)
        imgs = x.reshape(shape=(x.shape[0], c, h * p, h * p))
        return imgs #(N,C,H,W)

    def amd_unpatchify(self, mae_output):
        """
        mae_output: (N, S+1, D)
        """

        imgs = mae_output[:, 1:, :]  # remove cls token
        imgs = self.unpatchify(imgs)
        return imgs
    
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

    def encode(self, x):
        # embed patches
        x = self.patch_embed(x) # (N,S,D)

        # add pos embed w/o cls token
        x = x + self.pos_embed[:, 1:, :] # (N,S,D)

        # append cls token
        cls_token = self.cls_token + self.pos_embed[:, :1, :] # (1,1,D)
        cls_tokens = cls_token.expand(x.shape[0], -1, -1)   # (N,1,D)
        x = torch.cat((cls_tokens, x), dim=1) # (N,L+1,D)

        # apply Transformer blocks
        for blk in self.blocks:
            x = blk(x)
        x = self.norm(x)

        return x # (N,L+1,D)

    def decode(self,x):
        # embed tokens
        x = self.decoder_embed(x) # (N,L+1,D)

        # add pos embed
        x = x + self.decoder_pos_embed

        # apply Transformer blocks
        for blk in self.decoder_blocks:
            x = blk(x)
        x = self.decoder_norm(x)

        # predictor projection
        x = self.decoder_pred(x)

        # remove cls token
        x = x[:, 1:, :]

        return x # (N,L,D)
    
    def forward_encoder(self, x, mask_ratio):
        # embed patches
        x = self.patch_embed(x) # (N,S,D)

        # add pos embed w/o cls token
        x = x + self.pos_embed[:, 1:, :]

        # masking: length -> length * mask_ratio
        x, mask, ids_restore = self.random_masking(x, mask_ratio) # (N,len_keep,D), (N,L), (N,L)

        # append cls token
        cls_token = self.cls_token + self.pos_embed[:, :1, :] # (1,1,D)
        cls_tokens = cls_token.expand(x.shape[0], -1, -1)   # (N,1,D)
        x = torch.cat((cls_tokens, x), dim=1) # (N,L+1,D)

        # apply Transformer blocks
        for blk in self.blocks:
            x = blk(x)
        x = self.norm(x)

        return x, mask, ids_restore # (N,L+1,D), (N,L), (N,L)

    
    def forward_decoder(self, x, ids_restore):
        # embed tokens
        x = self.decoder_embed(x) # (N,L+1,D)

        # append mask tokens to sequence
        mask_tokens = self.mask_token.repeat(x.shape[0], ids_restore.shape[1] + 1 - x.shape[1], 1)
        x_ = torch.cat([x[:, 1:, :], mask_tokens], dim=1)  # no cls token (N,S,D)
        x_ = torch.gather(x_, dim=1, index=ids_restore.unsqueeze(-1).repeat(1, 1, x.shape[2]))  # unshuffle
        x = torch.cat([x[:, :1, :], x_], dim=1)  # append cls token (N,S+1,D)

        # add pos embed
        x = x + self.decoder_pos_embed

        # apply Transformer blocks
        for blk in self.decoder_blocks:
            x = blk(x)
        x = self.decoder_norm(x)

        # predictor projection
        x = self.decoder_pred(x)

        # remove cls token
        x = x[:, 1:, :]

        return x

    
    def forward_loss(self, imgs, pred, mask):
        """
        imgs: [N, C, H, W]
        pred: [N, L, p*p*C]
        mask: [N, L], 0 is keep, 1 is remove, 
        """
        target = self.patchify(imgs)
        if self.norm_pix_loss:
            mean = target.mean(dim=-1, keepdim=True)
            var = target.var(dim=-1, keepdim=True)
            target = (target - mean) / (var + 1.e-6)**.5

        loss = (pred - target) ** 2
        loss = loss.mean(dim=-1)  # [N, L], mean loss per patch

        loss = (loss * mask).sum() / mask.sum()  # mean loss on removed patches
        return loss

    def forward(self, imgs, mask_ratio=0.75,mode='reconstruction'):
        # mode: 'reconstruction' or 'downstream'
        if mode == 'reconstruction':
            latent, mask, ids_restore = self.forward_encoder(imgs, mask_ratio)
            pred = self.forward_decoder(latent, ids_restore)  # [N, L, p*p*c]
            loss = self.forward_loss(imgs, pred, mask)
            return loss, pred, mask
        elif mode == 'downstream':
            x = self.encode(imgs)
            return x # (N,L+1,D)
        else:
            return 0
    
    def forward_unpatchify(self, x):
        # mask_ratio = 0
        x = self.encode(x)
        x = self.decode(x)
        x = self.unpatchify(x)
        return x # (N,C,H,W)

    def decoder_param(self):
        tensor_list = []
        for name, param in self.named_parameters():
            if 'decoder' in name:
                tensor_list.append(param)
        return itertools.chain(tensor_list)

    def encoder_param(self):
        tensor_list = []
        for name, param in self.named_parameters():
            if 'decoder' not in name:
                tensor_list.append(param)
        return itertools.chain(tensor_list)

# def mae_vit_base_patch1_dec512d8b(**kwargs):
#     model = MaskedAutoencoderViT(
#         patch_size=1, embed_dim=768, depth=12, num_heads=12,
#         decoder_embed_dim=512, decoder_depth=8, decoder_num_heads=16,
#         mlp_ratio=4, norm_layer=partial(nn.LayerNorm, eps=1e-6), **kwargs)
#     return model


# def mae_vit_large_patch1_dec512d8b(**kwargs):
#     model = MaskedAutoencoderViT(
#         patch_size=1, embed_dim=1024, depth=24, num_heads=16,
#         decoder_embed_dim=512, decoder_depth=8, decoder_num_heads=16,
#         mlp_ratio=4, norm_layer=partial(nn.LayerNorm, eps=1e-6), **kwargs)
#     return model


# def mae_vit_huge_patch14_dec512d8b(**kwargs):
#     model = MaskedAutoencoderViT(
#         patch_size=14, embed_dim=1280, depth=32, num_heads=16,
#         decoder_embed_dim=512, decoder_depth=8, decoder_num_heads=16,
#         mlp_ratio=4, norm_layer=partial(nn.LayerNorm, eps=1e-6), **kwargs)
#     return model


# # set recommended archs
# mae_vit_base_patch1 = mae_vit_base_patch1_dec512d8b  # decoder: 512 dim, 8 blocks
# mae_vit_large_patch1 = mae_vit_large_patch1_dec512d8b  # decoder: 512 dim, 8 blocks
# mae_vit_huge_patch14 = mae_vit_huge_patch14_dec512d8b  # decoder: 512 dim, 8 blocks

def MAE_S(**kwargs):
    return MaskedAutoencoderViT(   
        embed_dim=768, depth=12, num_heads=12,
        decoder_embed_dim=512, decoder_depth=8, decoder_num_heads=16,
        mlp_ratio=4, norm_layer=partial(nn.LayerNorm, eps=1e-6),**kwargs)

def MAE_L(**kwargs):
    return MaskedAutoencoderViT(   
        embed_dim=1024, depth=24, num_heads=16,
        decoder_embed_dim=512, decoder_depth=8, decoder_num_heads=16,
        mlp_ratio=4, norm_layer=partial(nn.LayerNorm, eps=1e-6),**kwargs)

MAE_models = {
    "MAE_S": MAE_S,  #       150M
    "MAE_L": MAE_L, #        500M
}
