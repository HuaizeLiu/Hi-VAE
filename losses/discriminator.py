# Modified from Opensora Plan
import functools
import torch.nn as nn
import torch
from .modules import ActNorm
from .modules import CausalConv3d
from einops import rearrange
from typing import Union,Optional,Dict,Any,List
from diffusers.models.attention import FeedForward
from .modules import PatchEmbed,DiscTransformer,AdaLayerNorm,LayerNormZero
from diffusers.models.embeddings import TimestepEmbedding, Timesteps,get_2d_sincos_pos_embed
from diffusers.utils import is_torch_version
from timm.layers import Mlp

def weights_init(m):
    classname = m.__class__.__name__
    if classname.find('Conv') != -1:
        nn.init.normal_(m.weight.data, 0.0, 0.02)
    elif classname.find('BatchNorm') != -1:
        nn.init.normal_(m.weight.data, 1.0, 0.02)
        nn.init.constant_(m.bias.data, 0)

def weights_init_conv(m):
    if hasattr(m, 'conv'):
        m = m.conv
    classname = m.__class__.__name__
    if classname.find('Conv') != -1:
        nn.init.normal_(m.weight.data, 0.0, 0.02)
    elif classname.find('BatchNorm') != -1:
        nn.init.normal_(m.weight.data, 1.0, 0.02)
        nn.init.constant_(m.bias.data, 0)

class Discriminator3DConv(nn.Module):
    def __init__(self,
                 latent_width: int = 32,
                 latent_height: int = 32,
                 input_nc=4,
                 ndf=64,
                 n_layers=3,
                 use_actnorm=False,
                 mlp_hidden_dim: int = 256,
                 dropout:float = 0.,
                 use_sigmoid:bool = False
                 ):
        super().__init__()
        if not use_actnorm:
            norm_layer = nn.BatchNorm3d
        else:
            raise NotImplemented
        if type(norm_layer) == functools.partial:  # no need to use bias as BatchNorm2d has affine parameters
            use_bias = norm_layer.func != nn.BatchNorm3d
        else:
            use_bias = norm_layer != nn.BatchNorm3d
        kw = 3
        padw = 1
        sequence = [nn.Conv3d(input_nc, ndf, kernel_size=kw, stride=2, padding=padw), nn.LeakyReLU(0.2, True)]
        nf_mult = 1
        nf_mult_prev = 1
        for n in range(1, n_layers):  # gradually increase the number of filters
            nf_mult_prev = nf_mult
            nf_mult = min(2 ** n, 8)
            sequence += [
                nn.Conv3d(ndf * nf_mult_prev, ndf * nf_mult, kernel_size=kw, stride=2, padding=padw, bias=use_bias),
                norm_layer(ndf * nf_mult),
                nn.LeakyReLU(0.2, True)
            ]

        nf_mult_prev = nf_mult
        nf_mult = min(2 ** n_layers, 4)
        sequence += [
            nn.Conv3d(ndf * nf_mult_prev, ndf * nf_mult, kernel_size=kw, stride=1, padding=padw, bias=use_bias),
            norm_layer(ndf * nf_mult),
            nn.LeakyReLU(0.2, True)
        ]
        self.conv = nn.Sequential(*sequence)
        self.pool = nn.AdaptiveAvgPool3d(1)
        self.hidden_dim = nf_mult * ndf
        self.mlp = Mlp(in_features=self.hidden_dim,hidden_features=mlp_hidden_dim,out_features=1,drop=dropout)
        self.use_sigmoid = use_sigmoid
        # self.proj_out = 
        
    def forward(self, x:torch.Tensor):
        """forward 

        Args:
            x (torch.Tensor): zj pred shape = (N,C,T,H,W)
        Returns:
            socre: shape = (N)
        """        
        # x = rearrange(x,"n t c h w -> n c t h w")
        b = x.shape[0]
        x = self.conv(x)
        x = self.pool(x)
        x = x.reshape(b,-1)
        x = self.mlp(x).squeeze()
        if self.use_sigmoid:
            x = torch.sigmoid(x)
        return x
class NLayerDiscriminator(nn.Module):
    """Defines a PatchGAN discriminator as in Pix2Pix
        --> see https://github.com/junyanz/pytorch-CycleGAN-and-pix2pix/blob/master/models/networks.py
    """
    def __init__(self, 
                 input_nc=3,
                 ndf=64,
                 n_layers=3,
                 use_actnorm=False):
        """Construct a PatchGAN discriminator
        Parameters:
            input_nc (int)  -- the number of channels in input images
            ndf (int)       -- the number of filters in the last conv layer
            n_layers (int)  -- the number of conv layers in the discriminator
            norm_layer      -- normalization layer
        """
        super(NLayerDiscriminator, self).__init__()
        if not use_actnorm:
            norm_layer = nn.BatchNorm2d
        else:
            norm_layer = ActNorm
        if type(norm_layer) == functools.partial:  # no need to use bias as BatchNorm2d has affine parameters
            use_bias = norm_layer.func != nn.BatchNorm2d
        else:
            use_bias = norm_layer != nn.BatchNorm2d

        kw = 4
        padw = 1
        sequence = [nn.Conv2d(input_nc, ndf, kernel_size=kw, stride=2, padding=padw), nn.LeakyReLU(0.2, True)]
        nf_mult = 1
        nf_mult_prev = 1
        for n in range(1, n_layers):  # gradually increase the number of filters
            nf_mult_prev = nf_mult
            nf_mult = min(2 ** n, 8)
            sequence += [
                nn.Conv2d(ndf * nf_mult_prev, ndf * nf_mult, kernel_size=kw, stride=2, padding=padw, bias=use_bias),
                norm_layer(ndf * nf_mult),
                nn.LeakyReLU(0.2, True)
            ]

        nf_mult_prev = nf_mult
        nf_mult = min(2 ** n_layers, 8)
        sequence += [
            nn.Conv2d(ndf * nf_mult_prev, ndf * nf_mult, kernel_size=kw, stride=1, padding=padw, bias=use_bias),
            norm_layer(ndf * nf_mult),
            nn.LeakyReLU(0.2, True)
        ]

        sequence += [
            nn.Conv2d(ndf * nf_mult, 1, kernel_size=kw, stride=1, padding=padw)]  # output 1 channel prediction map
        self.main = nn.Sequential(*sequence)

    def forward(self, input):
        """Standard forward."""
        return self.main(input)
class NLayerDiscriminator3D(nn.Module):
    """Defines a 3D PatchGAN discriminator as in Pix2Pix but for 3D inputs."""
    def __init__(self, 
                input_nc,
                lattent_h:int = 32,
                lattent_w:int = 32,
                ndf=64,
                n_layers=3,
                use_actnorm=False,
                # motion_nc=128,
                # motion_hidden_nc=16
                ):

        """
        Construct a 3D PatchGAN discriminator

        Parameters:
            input_nc (int)  -- the number of channels in input volumes
            ndf (int)       -- the number of filters in the last conv layer
            n_layers (int)  -- the number of conv layers in the discriminator
            use_actnorm (bool) -- flag to use actnorm instead of batchnorm
        """
        super().__init__()
        if not use_actnorm:
            norm_layer = nn.BatchNorm3d
        else:
            raise NotImplementedError("Not implemented.")
        if type(norm_layer) == functools.partial:
            use_bias = norm_layer.func != nn.BatchNorm3d
        else:
            use_bias = norm_layer != nn.BatchNorm3d
        
        kw = 3
        padw = 1
        # self.motion_conv_in = nn.Sequential(
            # nn.ConvTranspose2d(motion_nc,motion_hidden_nc,kernel_size=(4,4),stride=(4,4)),
            # nn.BatchNorm2d(motion_hidden_nc),
            # nn.LeakyReLU(0.2,True)
        # )
        # sequence = [nn.Conv3d(input_nc + motion_hidden_nc, ndf, kernel_size=kw, stride=2, padding=padw), nn.LeakyReLU(0.2, True)]
        sequence = [nn.Conv3d(input_nc, ndf, kernel_size=kw, stride=2, padding=padw), nn.LeakyReLU(0.2, True)]
        nf_mult = 1
        nf_mult_prev = 1
        for n in range(1, n_layers):  # gradually increase the number of filters
            nf_mult_prev = nf_mult
            nf_mult = min(2 ** n, 8)
            sequence += [
                nn.Conv3d(ndf * nf_mult_prev, ndf * nf_mult, kernel_size=(kw, kw, kw), stride=(2 if n==1 else 1,2,2), padding=padw, bias=use_bias),
                norm_layer(ndf * nf_mult),
                nn.LeakyReLU(0.2, True)
            ]

        nf_mult_prev = nf_mult
        nf_mult = min(2 ** n_layers, 8)
        sequence += [
            nn.Conv3d(ndf * nf_mult_prev, ndf * nf_mult, kernel_size=(kw, kw, kw), stride=1, padding=padw, bias=use_bias),
            norm_layer(ndf * nf_mult),
            nn.LeakyReLU(0.2, True)
        ]

        sequence += [nn.Conv3d(ndf * nf_mult, 1, kernel_size=kw, stride=1, padding=padw)]  # output 1 channel prediction map
        self.main = nn.Sequential(*sequence)

    def forward(self, zi:torch.Tensor,zj:torch.Tensor):
        """

        Args:
            zi (torch.Tensor): (B,C,1,H,W)
            zj (torch.Tensor): (B,C,T-1,H,W)
        """
        enc = torch.concat([zi,zj],dim=2)
        return self.main(enc)

class Discriminator2DConv(nn.Module):
    def __init__(self,
                 latent_width: int = 32,
                 latent_height: int = 32,
                 input_nc=4,
                 ndf=64,
                 n_layers=3,
                 use_actnorm=False,
                 mlp_hidden_dim: int = 256,
                 dropout:float = 0.,
                 use_sigmoid:bool = False
                 ):
        super().__init__()
        if not use_actnorm:
            norm_layer = nn.BatchNorm2d
        else:
            raise NotImplemented
        if type(norm_layer) == functools.partial:  # no need to use bias as BatchNorm2d has affine parameters
            use_bias = norm_layer.func != nn.BatchNorm2d
        else:
            use_bias = norm_layer != nn.BatchNorm2d
        kw = 3
        padw = 1
        sequence = [nn.Conv2d(input_nc, ndf, kernel_size=kw, stride=2, padding=padw), nn.LeakyReLU(0.2, True)]
        nf_mult = 1
        nf_mult_prev = 1
        for n in range(1, n_layers):  # gradually increase the number of filters
            nf_mult_prev = nf_mult
            nf_mult = min(2 ** n, 8)
            sequence += [
                nn.Conv2d(ndf * nf_mult_prev, ndf * nf_mult, kernel_size=kw, stride=2, padding=padw, bias=use_bias),
                norm_layer(ndf * nf_mult),
                nn.LeakyReLU(0.2, True)
            ]

        nf_mult_prev = nf_mult
        nf_mult = min(2 ** n_layers, 4)
        sequence += [
            nn.Conv2d(ndf * nf_mult_prev, ndf * nf_mult, kernel_size=kw, stride=1, padding=padw, bias=use_bias),
            norm_layer(ndf * nf_mult),
            nn.LeakyReLU(0.2, True)
        ]
        self.conv = nn.Sequential(*sequence)
        self.pool = nn.AdaptiveAvgPool2d(1)
        self.hidden_dim = nf_mult * ndf
        self.mlp = Mlp(in_features=self.hidden_dim,hidden_features=mlp_hidden_dim,out_features=1,drop=dropout)
        self.use_sigmoid = use_sigmoid
        # self.proj_out = 
        
    def forward(self, x:torch.Tensor):
        """forward 

        Args:
            x (torch.Tensor): zj pred shape = (N,T,C,H,W)
        Returns:
            socre: shape = (N)
        """        
        # x = rearrange(x,"n t c h w -> n c t h w")
        b = x.shape[0]
        x = self.conv(x)
        x = self.pool(x)
        x = x.reshape(b,-1)
        x = self.mlp(x).squeeze()
        if self.use_sigmoid:
            x = torch.sigmoid(x)
        return x

    
class Discriminator2DConvVel(nn.Module):
    def __init__(self,
                 latent_width: int = 32,
                 latent_height: int = 32,
                 input_nc=8,
                 ndf=64,
                 n_layers=3,
                 use_actnorm=False,
                 norm_elementwise_affine: bool = True,
                 norm_eps: float = 1e-5,
                 mlp_hidden_dim: int = 256,
                 time_embed_dim:int = 256,
                 freq_shift:int = 0,
                 flip_sin_to_cos:bool = True,
                 timestep_activation_fn: str = "silu",
                 dropout:float = 0.,
                 use_sigmoid:bool = False
                 ):
        super().__init__()
        if not use_actnorm:
            norm_layer = nn.BatchNorm2d
        else:
            norm_layer = ActNorm
        if type(norm_layer) == functools.partial:  # no need to use bias as BatchNorm2d has affine parameters
            use_bias = norm_layer.func != nn.BatchNorm2d
        else:
            use_bias = norm_layer != nn.BatchNorm2d

        kw = 3
        padw = 1
        sequence = [nn.Conv2d(input_nc, ndf, kernel_size=kw, stride=2, padding=padw), nn.LeakyReLU(0.2, True)]
        nf_mult = 1
        nf_mult_prev = 1
        for n in range(1, n_layers):  # gradually increase the number of filters
            nf_mult_prev = nf_mult
            nf_mult = min(2 ** n, 8)
            sequence += [
                nn.Conv2d(ndf * nf_mult_prev, ndf * nf_mult, kernel_size=kw, stride=2, padding=padw, bias=use_bias),
                norm_layer(ndf * nf_mult),
                nn.LeakyReLU(0.2, True)
            ]

        nf_mult_prev = nf_mult
        nf_mult = min(2 ** n_layers, 4)
        sequence += [
            nn.Conv2d(ndf * nf_mult_prev, ndf * nf_mult, kernel_size=kw, stride=1, padding=padw, bias=use_bias),
            norm_layer(ndf * nf_mult),
            nn.LeakyReLU(0.2, True)
        ]
        self.conv = nn.Sequential(*sequence)
        self.pool = nn.AdaptiveAvgPool2d(1)
        self.hidden_dim = nf_mult * ndf
        self.time_proj = Timesteps(time_embed_dim, flip_sin_to_cos, freq_shift)
        self.time_embedding = TimestepEmbedding(time_embed_dim, self.hidden_dim, timestep_activation_fn)
        self.mlp = Mlp(in_features=self.hidden_dim,hidden_features=mlp_hidden_dim,out_features=self.hidden_dim,drop=dropout)

        self.norm = LayerNormZero(self.hidden_dim, self.hidden_dim, norm_elementwise_affine, norm_eps, bias=True)
        self.ff = FeedForward(self.hidden_dim,1,mult=2)
        self.use_sigmoid = use_sigmoid
        # self.proj_out = 
        
    def forward(self, x:torch.Tensor,timestep:torch.Tensor,timestep_cond: Optional[torch.Tensor] = None):
        """Standard forward."""
        b = x.shape[0]
        t_emb = self.time_proj(timestep)
        t_emb = t_emb.to(dtype=x.dtype)
        emb = self.time_embedding(t_emb, timestep_cond)
        x = self.pool(self.conv(x)).reshape(b,-1)
        x = self.mlp(x)
        norm_x,_ = self.norm(x,emb)
        x = self.ff(norm_x).squeeze()
        if self.use_sigmoid:
            x = torch.sigmoid(x)
        return x

    # def forward(self,
                # image_hidden_states: torch.Tensor,
                # timestep: Union[int, float, torch.LongTensor], # Timesteps should be a 1d-array
                # timestep_cond: Optional[torch.Tensor] = None,
                # ):
    #    pass 
class Discriminator2DAttn(nn.Module):
    def __init__(self,
                 latent_width: int = 32,
                 latent_height: int = 32,
                 latten_chs:int = 8,

                 patch_size:int = 2,
                 attention_head_dim:int = 64,
                 num_attention_heads:int = 12,
                 num_layers:int = 8,
                 attention_bias: bool = True,
                 activation_fn: str = "gelu-approximate",
                 norm_elementwise_affine: bool = True,
                 norm_eps: float = 1e-5,
                 mlp_hidden_dim: int = 512,
                 time_embed_dim:int = 512,
                 freq_shift:int = 0,
                 flip_sin_to_cos:bool = True,
                 timestep_activation_fn: str = "silu",
                 
                 dropout:float = 0.,
                 use_sigmoid:bool = False
                 ):
        super().__init__()
        hidden_dim = num_attention_heads * attention_head_dim
        iph = latent_height // patch_size
        ipw = latent_width // patch_size
        self.seq_len = iph * ipw # image token length
        
        self.patch_size = patch_size

        # 2. Patch embedding (N,F,C,H,W) -> (B,S,D)
        self.image_patch_embed = PatchEmbed(patch_size=patch_size,in_channels=latten_chs,embed_dim=hidden_dim, bias=True)
        # self.patch_embed = CogVideoXPatchEmbed(patch_size, in_channels, inner_dim, text_embed_dim, bias=True)
        self.embedding_dropout = nn.Dropout(dropout)

        # 3. 2D&3D positional embeddings
        image_pos_embedding = get_2d_sincos_pos_embed(hidden_dim, (iph, ipw)) # (iph*ipw,D)
        image_pos_embedding = torch.from_numpy(image_pos_embedding) # (iph*ipw,D)
        pos_embedding = torch.zeros(1, self.seq_len, hidden_dim, requires_grad=False)
        pos_embedding.data.copy_(image_pos_embedding)
        self.register_buffer("pos_embedding", pos_embedding, persistent=False)

        self.time_proj = Timesteps(hidden_dim, flip_sin_to_cos, freq_shift)
        self.time_embedding = TimestepEmbedding(hidden_dim, time_embed_dim, timestep_activation_fn)
        self.use_sigmoid = use_sigmoid
        self.transformer_blocks = nn.ModuleList(
            [
                DiscTransformer(
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
        self.mlp = Mlp(hidden_dim*self.seq_len,mlp_hidden_dim,1)

        self.gradient_checkpointing = False

    def forward(
        self,
        image_hidden_states: torch.Tensor,
        timestep: Union[int, float, torch.LongTensor], # Timesteps should be a 1d-array
        timestep_cond: Optional[torch.Tensor] = None,
    ):
        """
        image_hidden_states : (b,2c,H,W)
        time_step : (b,)

        """
        N = image_hidden_states.shape[0]
        
        # 1. Time embedding   # Timesteps should be a 1d-array
        t_emb = self.time_proj(timestep)
        t_emb = t_emb.to(dtype=image_hidden_states.dtype)
        emb = self.time_embedding(t_emb, timestep_cond)

        # 2. Patch embedding
        image_hidden_states = self.image_patch_embed(image_hidden_states)

        # 3. Position embedding
        image_hidden_states = image_hidden_states + self.pos_embedding

        # 4. Transformer blocks
        for i, block in enumerate(self.transformer_blocks):
            if self.training and self.gradient_checkpointing:

                def create_custom_forward(module):
                    def custom_forward(*inputs):
                        return module(*inputs)

                    return custom_forward

                ckpt_kwargs: Dict[str, Any] = {"use_reentrant": False} if is_torch_version(">=", "1.11.0") else {}
                image_hidden_states = torch.utils.checkpoint.checkpoint(
                    create_custom_forward(block),
                    image_hidden_states,
                    emb,
                    **ckpt_kwargs,
                )
            else:
                image_hidden_states = block(
                    image_hidden_states,
                    temb=emb,
                )

        image_hidden_states = self.norm_final(image_hidden_states)

        # 6. Final block
        image_hidden_states = self.norm_out(image_hidden_states, temb=emb)

        image_hidden_states = image_hidden_states.reshape(N,-1)
        score = self.mlp(image_hidden_states).squeeze()
        if self.use_sigmoid:
            score = torch.sigmoid(score)
        return score

