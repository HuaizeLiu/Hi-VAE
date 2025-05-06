import torch
from torch import nn
from .base_model import BaseDiffusionModel,BaseDecoder
from diffusers.configuration_utils import register_to_config
from timm.models.layers import Mlp
class Label2MotionDecoder(BaseDecoder):
    pass
class Label2MotionDiffusionDecoder(BaseDiffusionModel):

    @register_to_config
    def __init__(
        self,
        
        label_dim:int = 512,
        motion_dim:int = 256,
        motion_width:int = 4,
        motion_height:int = 4,
        refimg_width: int = 32,
        refimg_height: int = 32,
        refimg_patch_size: int = 4,
        refimg_dim:int = 4,
        num_frames: int = 16,

        num_steps:int = 1000,
        flip_sin_to_cos: bool = True,
        freq_shift: int = 0,
        time_embed_dim: int = 768,
        timestep_activation_fn: str = "silu",

        attention_head_dim : int = 80,
        num_attention_heads: int = 12,
        num_layers: int = 16,
        dropout: float = 0.0,
        norm_elementwise_affine: bool = True,
        norm_eps: float = 1e-5,
        spatial_interpolation_scale: float = 1.0,
        temporal_interpolation_scale: float = 1.0,
        **kwargs 
    ):
        super().__init__( motion_dim, motion_width, motion_height,
                                        refimg_width, refimg_height, refimg_patch_size, refimg_dim,
                                        num_frames,
                                        num_steps, flip_sin_to_cos, freq_shift, time_embed_dim, timestep_activation_fn,
                                        attention_head_dim, num_attention_heads, num_layers, dropout, norm_elementwise_affine, norm_eps,
                                        spatial_interpolation_scale, temporal_interpolation_scale,
                                        **kwargs)

        
        hidden_dim = attention_head_dim * num_attention_heads
        self.label_proj_in = Mlp(label_dim,hidden_dim,hidden_dim)
    def cond_injection(self,
                       cond:torch.Tensor,
                       image_hidden_state:torch.Tensor,
                       t_emb:torch.Tensor
                       ):

        label_emb = self.label_proj_in(cond)
        return image_hidden_state,t_emb + label_emb

class Label2VideoPipe(nn.Module):
    def __init__(self,
                  *args, 
                  **kwargs):
        super().__init__()
    def generate(self,):
        pass