import torch
from tqdm import tqdm
from torch import nn
from diffusers.utils import is_torch_version
from diffusers.configuration_utils import ConfigMixin, register_to_config
from diffusers.models.modeling_utils import ModelMixin
from typing import Optional,Union,Dict,Any
from einops import rearrange
from .modules import Any2MotionDiffusionTransformer
from .utils import get_sample_t_schedule
class BaseModel(ModelMixin, ConfigMixin):
    def __init__(
        self,
        **config 
    ):
        super().__init__()
    def _set_gradient_checkpointing(self, module, value=False):
        self.gradient_checkpointing = value
class BaseDecoder(BaseModel):
    @register_to_config
    def __init__(
        self,
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
        super().__init__(
            motion_dim, motion_width, motion_height, refimg_width,
            refimg_height, refimg_patch_size, refimg_dim,
            num_frames,
            num_steps, flip_sin_to_cos, freq_shift, time_embed_dim, timestep_activation_fn,
            attention_head_dim, num_attention_heads, num_layers, dropout, norm_elementwise_affine, norm_eps,
            spatial_interpolation_scale, temporal_interpolation_scale, **kwargs
        )
    def forward(self,
                cond:torch.Tensor,
                ref_img:torch.Tensor,
                return_meta_info=False):
        """
        Args:
            cond (torch.Tensor): ()
            ref_img (torch.Tensor): (B,C,H,W)
        """
        # embeddings include: 
        # reference image embedding
        # audio embedding
        # timestep embedding
        device = ref_img.device
        dtype = ref_img.dtype
        image_hidden_state = self.patch_embed(ref_img)
        image_hidden_state = self.embedding_dropout(image_hidden_state + self.img_pos_embedding)
        # embedding = t_emb + Optional(cond), token = t_emb + Optional(cond)
        hidden_state,emb = self.cond_injection(cond,image_hidden_state)

        for i, block in enumerate(self.transformer_blocks):
            if self.training and self.gradient_checkpointing:
                def create_custom_forward(module):
                    def custom_forward(*inputs):
                        return module(*inputs)

                    return custom_forward

                ckpt_kwargs: Dict[str, Any] = {"use_reentrant": False} if is_torch_version(">=", "1.11.0") else {}
                hidden_state = torch.utils.checkpoint.checkpoint(
                    create_custom_forward(block),
                    hidden_state,
                    emb,
                    **ckpt_kwargs,
                )
            else:
                hidden_state = block(
                    hidden_state,
                    emb,
                )
        hidden_state = self.norm_final(hidden_state)
        hidden_state = self.proj_out(hidden_state)
        motion_pred = hidden_state[:,:self.motion_seq_len]

        return motion_pred
    def cond_injection(self,cond):
        pass

class BaseDiffusionModel(BaseModel):
    
    @register_to_config
    def __init__(
        self,
        **config
    ):
        super().__init__(**config)
        self.num_steps = config.get("num_steps",1000)
        self.motion_seq_len = config.get("motion_seq_len",30)
        self.motion_channels = config.get('motion_channels',256)
        self.motion_height = config.get("motion_height",4)
        self.motion_width = config.get("motion_width",4)

    def forward(self,
                motion_gt:torch.Tensor, 
                ref_img:torch.Tensor,
                audio:torch.Tensor,
                pose:torch.Tensor,
                ref_pose:torch.Tensor,
                timestep: Union[int, float, torch.LongTensor], # Timesteps should be a 1d-array
                timestep_cond: Optional[torch.Tensor] = None,
                return_meta_info=False):
        """
        Args:
            motion_gt (torch.Tensor): (N,F,C,h,w)
            ref_img (torch.Tensor): (N,C,H,W)
            audio (torch.Tensor): (N,M,D)
            pose (torch.Tensor): (N,F,C,W,H)
            ref_pose (torch.Tensor): (N,C,W,H)
            timestep (torch.Tensor): (N,) <= num_steps
        """
        
        # special
        extra,ref_img,timestep = self.cond_injection(audio,pose,ref_pose)

        # prepare motion
        t = (1 - timestep / self.num_steps)[:,None,None,None,None]
        noise = torch.randn_like(motion_gt)
        vel_gt = motion_gt - noise
        motion_with_noise = t * motion_gt +  (1 - t) * noise #  (B,F,C,h,w)

        # forward
        vel_pred = self.model(
            motion_with_noise,
            ref_img,
            extra,
            timestep,
            timestep_cond
        )
        motion_pred = motion_with_noise + (1 - t) * vel_pred

        if return_meta_info:
            return {
                    'motion_with_noise' : motion_with_noise,  # (b f c h w) 
                    'motion_pred' : motion_pred,              # (b f c h w)
                    'vel_pred' : vel_pred,                    # (b (f h w) c)
                    'vel_gt' : vel_gt,                        # (b (f h w) c)
                    }
        else:
            return motion_with_noise,motion_pred,vel_pred,vel_gt 
        
    @torch.no_grad()
    def sample(self,
               ref_img:torch.Tensor,
               extra:torch.Tensor,
               timestep_cond=None,
               sample_steps:int=10,
               t_schedule:Optional[Dict] = None
               ):
        """
        Args:
            motion_gt (torch.Tensor): (N,F,C,h,w)
            ref_img (torch.Tensor): (N,C,H,W)
            extra (torch.Tensor): (N,F,M,D)
            timestep (torch.Tensor): (N,) <= num_steps
        """
        n = extra.shape[0]
        device = ref_img.device
        timestep = torch.ones(n).to(device) * self.num_steps
        motion_with_noise = torch.randn(n,self.motion_seq_len,self.motion_channels,self.motion_height,self.motion_width).to(device)

        # t_schedule
        if t_schedule is not None:
            steps = get_sample_t_schedule(t_schedule,sample_steps)
        else:
            steps = [1. / sample_steps] * sample_steps
        
        # denoise
        extra,ref_img,timestep = self.cond_injection(extra,ref_img,timestep)
        for i,dt in tqdm(enumerate(steps)):
            vel_pred = self.model(
                motion_with_noise,
                ref_img,
                extra,
                timestep,
                timestep_cond
            ) # (NF,C,h,w)
            # if vel_pred.dim() == 4:
                # vel_pred = rearrange(vel_pred,'(n f) c h w -> b f c h w',f=self.motion_seq_len)
            motion_with_noise = motion_with_noise + dt * vel_pred
            timestep = timestep - dt * self.num_steps

        return motion_with_noise
    
    def cond_injection(self,
                       extra:torch.Tensor,
                       refimg:torch.Tensor,
                       timestep:torch.Tensor
                       ):
        pass 