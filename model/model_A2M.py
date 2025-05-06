import torch
from torch import nn
import einops
from typing import Tuple
import random
import numpy as np
from tqdm import tqdm
from .modules import (AudioFeatureMlp,
                      AudioFeatureWindowMlp,
                      Audio2Pose,
                      AudioToImageShapeMlp,
                      AudioFeatureWindowMlp)
from .loss import l1,l2
from .transformer import (AudioMitionref_LearnableToken,
                          AudioMitionref_LearnableToken_SimpleAdaLN,
                          A2MTransformer_CrossAttn_Audio,
                          A2MTransformer_CrossAttn_Audio_Pose,
                          A2PTransformer,
                          A2MTransformer_CrossAttn_Pose)
from .rectified_flow import RectifiedFlow
from diffusers.configuration_utils import ConfigMixin, register_to_config
from diffusers.models.modeling_utils import ModelMixin
from diffusers.models.resnet import ResnetBlock2D
import einops
import torch.nn.functional as F
from typing import Optional,Union,Dict,Any

from diffusers.utils import export_to_gif

class A2MModel_PosePre(ModelMixin, ConfigMixin):
    _supports_gradient_checkpointing = True
    
    @register_to_config
    def __init__(self,
                 image_inchannel :int  = 4,
                 image_height  :int = 32,
                 image_width  :int  = 32,
                 image_patch_size : int  = 2,

                 audio_inchannel :int  = 384,
                 audio_block : int = 50,
                 
                 motion_height  :int = 4,
                 motion_width  :int = 4,
                 motion_frames  :int = 30,
                 motion_in_channel :int = 256,
                 motion_patch_size : int = 1,
                 num_step :int = 1000,
                 
                 # ----------- Audio feature encoder -----------
                 encoder_out_dim :int = 512,
                 encoder_num_attention_heads  = 8,
                 encoder_attention_dim  = 64,
                 
                 # ----------- Diffusion Transformer -----------
                 diffusion_attn_head_dim : int  = 64,
                 diffusion_attn_num_heads : int = 16,
                 diffusion_out_channels : int = 256,
                 diffusion_num_layers : int = 8,

                 **kwargs,
                 ):
        super().__init__()

        # setting
        self.num_step = num_step
        self.scheduler = RectifiedFlow(num_steps=num_step)
        self.motion_frame = motion_frames
        self.motion_in_channel = motion_in_channel
        self.motion_height = motion_height
        self.motion_width = motion_width


        # audio encoder
        self.audio_encoder = Audio2Pose(audio_dim = audio_inchannel,
                                        audio_block = audio_block,
                                        pose_width = image_height,
                                        pose_height = image_width,
                                        pose_dim = image_inchannel,
                                        num_frames = motion_frames,
                                        outdim = encoder_out_dim,
                                        num_attention_heads = encoder_num_attention_heads,
                                        attention_dim = encoder_attention_dim,) # layer = 4
        
        # diffusion transformer
        self.diffusion = Audio2MotionAllSequence(num_attention_heads = diffusion_attn_num_heads,
                                                attention_head_dim = diffusion_attn_head_dim,
                                                motion_in_channels = motion_in_channel,
                                                refimg_in_channels = image_inchannel,
                                                extra_in_channels = encoder_out_dim,
                                                out_channels = motion_in_channel,
                                                num_layers = diffusion_num_layers,
                                                
                                                image_width = image_width,
                                                image_height = image_height,
                                                image_patch_size = image_patch_size,

                                                motion_width = motion_width,
                                                motion_height = motion_height,
                                                motion_patch_size = motion_patch_size,
                                                motion_frames = motion_frames,)


    def forward(self,
                motion_gt:torch.Tensor, 
                ref_img:torch.Tensor,
                audio:torch.Tensor,
                pose:torch.Tensor,
                ref_pose:torch.Tensor,
                timestep: Union[int, float, torch.LongTensor] = None, # Timesteps should be a 1d-array
                timestep_cond: Optional[torch.Tensor] = None,
                return_meta_info=False,
                **kwargs):
        """
        Args:
            motion_gt (torch.Tensor): (N,F,C,h,w)
            ref_img (torch.Tensor): (N,C,H,W)
            audio (torch.Tensor): (N,M,D)
            pose (torch.Tensor): (N,F,C,W,H)
            ref_pose (torch.Tensor): (N,C,W,H)
            timestep (torch.Tensor): (N,) <= num_steps
        """

        device = motion_gt.device
        n,t,c,h,w = motion_gt.shape

        pose_pred , mix_extra = self.audio_encoder(audio,ref_pose) # (n,t,c,h,w) (n,t,d)
        
        # add noise
        if timestep is None:
            timestep = torch.randint(0,self.num_step+1,(n,)).to(device)
        t = (1 - timestep / self.num_step)[:,None,None,None,None]
        noise = torch.randn_like(motion_gt)
        vel_gt = motion_gt - noise
        motion_with_noise = t * motion_gt +  (1 - t) * noise #  (n,t,c,h,w)

        # forward
        vel_pred = self.diffusion(
            motion_hidden_states = motion_with_noise,
            refimg_hidden_states = ref_img,
            pose_hidden_states = ref_pose,
            extra_hidden_states = mix_extra,
            timestep = timestep,
        )
        
        # loss
        diff_loss = l2(vel_pred,vel_gt)
        pose_loss = F.mse_loss(pose_pred,pose)
        
        loss = diff_loss + pose_loss

        loss_dict = {'loss':loss,'diff_loss':diff_loss,'pose_loss':pose_loss}

        return loss_dict
    
    @torch.no_grad()
    def sample(self,
                ref_img:torch.Tensor,
                audio:torch.Tensor,
                ref_pose:torch.Tensor,
                timestep: Union[int, float, torch.LongTensor] = None, # Timesteps should be a 1d-array
                start_step:int = None,
                sample_step:int = 10,
                timestep_cond: Optional[torch.Tensor] = None,
                return_meta_info=False):
        """
        Args:
            ref_img (torch.Tensor): (N,C,H,W)
            audio (torch.Tensor): (N,M,D)
            ref_pose (torch.Tensor): (N,C,W,H)
            timestep (torch.Tensor): (N,) <= num_steps
        """

        device = ref_img.device
        n,c,h,w = ref_img.shape

        audio_hidden_state, pose_pred = self.audio_encoder.prepare_extra(audio,ref_pose) # (n,t,c,h,w) (n,t,d)
        
        # get noise 
        zt = torch.randn(n,self.motion_frame,self.motion_in_channel,self.motion_height,self.motion_width).to(device)

        # start_step : 1000
        start_step = self.num_step if start_step is None else start_step
        
        # step_seq [1000,995 ....]
        step_seq = np.linspace(0, start_step, num=sample_step+1, endpoint=True,dtype=int) # [0,5,10,15,....,start_step]
        step_seq = list(reversed(step_seq[1:])) # delete step:0  [start_step,.....,15,10,5]

        # Euler step
        dt = 1./sample_step
        for i in tqdm(step_seq):
            # time_step
            time_step = torch.ones((zt.shape[0],)).to(zt.device)  
            time_step = time_step * i
            
            # input
            zt = zt.to(ref_img.dtype)
            
            # forward
            pre = self.diffusion(
                motion_hidden_states = zt,
                refimg_hidden_states = ref_img,
                pose_hidden_states = ref_pose,
                extra_hidden_states = mix_extra,
                timestep = time_step,
            )
            zt = zt + pre * dt

        return zt


class A2MModel_Mlp(ModelMixin, ConfigMixin):
    _supports_gradient_checkpointing = True
    
    @register_to_config
    def __init__(self,
                 image_inchannel :int  = 4,
                 image_height  :int = 32,
                 image_width  :int  = 32,
                 image_patch_size : int  = 2,

                 audio_inchannel :int  = 384,
                 audio_block : int = 50,
                 
                 motion_height  :int = 4,
                 motion_width  :int = 4,
                 motion_frames  :int = 30,
                 motion_in_channel :int = 256,
                 motion_patch_size : int = 1,
                 num_step :int = 1000,
                 
                 # ----------- Audio feature encoder -----------
                 encoder_out_dim :int = 1024,
                 
                 # ----------- Diffusion Transformer -----------
                 diffusion_attn_head_dim : int  = 64,
                 diffusion_attn_num_heads : int = 16,
                 diffusion_out_channels : int = 256,
                 diffusion_num_layers : int = 8,

                 **kwargs,
                 ):
        super().__init__()

        # setting
        self.num_step = num_step
        self.scheduler = RectifiedFlow(num_steps=num_step)
        self.motion_frame = motion_frames
        self.motion_in_channel = motion_in_channel
        self.motion_height = motion_height
        self.motion_width = motion_width


        # audio encoder
        self.audio_encoder = AudioFeatureMlp(audio_dim = audio_inchannel,
                                            audio_block = audio_block,
                                            outdim = encoder_out_dim,)
        
        # diffusion transformer
        self.diffusion = Audio2MotionAllSequence(num_attention_heads = diffusion_attn_num_heads,
                                                attention_head_dim = diffusion_attn_head_dim,
                                                motion_in_channels = motion_in_channel,
                                                refimg_in_channels = image_inchannel,
                                                extra_in_channels = encoder_out_dim,
                                                out_channels = motion_in_channel,
                                                num_layers = diffusion_num_layers,
                                                
                                                image_width = image_width,
                                                image_height = image_height,
                                                image_patch_size = image_patch_size,

                                                motion_width = motion_width,
                                                motion_height = motion_height,
                                                motion_patch_size = motion_patch_size,
                                                motion_frames = motion_frames,)


    def forward(self,
                motion_gt:torch.Tensor, 
                ref_img:torch.Tensor,
                audio:torch.Tensor,
                pose:torch.Tensor,
                ref_pose:torch.Tensor,
                timestep: Union[int, float, torch.LongTensor] = None, # Timesteps should be a 1d-array
                timestep_cond: Optional[torch.Tensor] = None,
                return_meta_info=False,
                **kwargs):
        """
        Args:
            motion_gt (torch.Tensor): (N,F,C,h,w)
            ref_img (torch.Tensor): (N,C,H,W)
            audio (torch.Tensor): (N,F,M,D)
            pose (torch.Tensor): (N,F,C,W,H)
            ref_pose (torch.Tensor): (N,C,W,H)
            timestep (torch.Tensor): (N,) <= num_steps
        """

        device = motion_gt.device
        n,f,c,h,w = motion_gt.shape

        audio_feature = self.audio_encoder(audio) # (n,t,d)
        
        # add noise
        if timestep is None:
            timestep = torch.randint(0,self.num_step+1,(n,)).to(device)
        t = (1 - timestep / self.num_step)[:,None,None,None,None]
        noise = torch.randn_like(motion_gt)
        vel_gt = motion_gt - noise
        motion_with_noise = t * motion_gt +  (1 - t) * noise #  (n,t,c,h,w)

        # forward
        vel_pred = self.diffusion(
            motion_hidden_states = motion_with_noise,
            refimg_hidden_states = ref_img,
            pose_hidden_states = ref_pose,
            extra_hidden_states = audio_feature,
            timestep = timestep,
        )
        
        # loss
        diff_loss = l2(vel_pred,vel_gt)
        
        loss = diff_loss 

        loss_dict = {'loss':loss,'diff_loss':diff_loss}

        return loss_dict
    
    @torch.no_grad()
    def sample(self,
                ref_img:torch.Tensor,
                audio:torch.Tensor,
                ref_pose:torch.Tensor,
                timestep: Union[int, float, torch.LongTensor] = None, # Timesteps should be a 1d-array
                start_step:int = None,
                sample_step:int = 10,
                timestep_cond: Optional[torch.Tensor] = None,
                return_meta_info=False):
        """
        Args:
            ref_img (torch.Tensor): (N,C,H,W)
            audio (torch.Tensor): (N,M,D)
            ref_pose (torch.Tensor): (N,C,W,H)
            timestep (torch.Tensor): (N,) <= num_steps
        """

        device = ref_img.device
        n,c,h,w = ref_img.shape

        audio_feature = self.audio_encoder(audio,ref_pose) # (n,t,c,h,w) (n,t,d)
        
        # get noise 
        zt = torch.randn(n,self.motion_frame,self.motion_in_channel,self.motion_height,self.motion_width).to(device)

        # start_step : 1000
        start_step = self.num_step if start_step is None else start_step
        
        # step_seq [1000,995 ....]
        step_seq = np.linspace(0, start_step, num=sample_step+1, endpoint=True,dtype=int) # [0,5,10,15,....,start_step]
        step_seq = list(reversed(step_seq[1:])) # delete step:0  [start_step,.....,15,10,5]

        # Euler step
        dt = 1./sample_step
        for i in tqdm(step_seq):
            # time_step
            time_step = torch.ones((zt.shape[0],)).to(zt.device)  
            time_step = time_step * i
            
            # input
            zt = zt.to(ref_img.dtype)
            
            # forward
            pre = self.diffusion(
                motion_hidden_states = zt,
                refimg_hidden_states = ref_img,
                pose_hidden_states = ref_pose,
                extra_hidden_states = audio_feature,
                timestep = time_step,
            )
            zt = zt + pre * dt

        return zt



class A2MModel_MotionrefOnly(ModelMixin, ConfigMixin):
    _supports_gradient_checkpointing = True
    
    @register_to_config
    def __init__(self,
                 image_inchannel :int  = 4,
                 image_height  :int = 32,
                 image_width  :int  = 32,
                 image_patch_size : int  = 2,

                 audio_inchannel :int  = 384,
                 audio_block : int = 50,
                 
                 motion_frames  :int = 30,
                 motion_in_channel :int = 256,
                 num_step :int = 1000,
                 
                 # ----------- Audio feature encoder -----------
                 encoder_out_dim :int = 1024,
                 
                 # ----------- Diffusion Transformer -----------
                 diffusion_attn_head_dim : int  = 64,
                 diffusion_attn_num_heads : int = 16,
                 diffusion_out_channels : int = 256,
                 diffusion_num_layers : int = 8,

                 **kwargs,
                 ):
        super().__init__()

        # setting
        self.num_step = num_step
        self.scheduler = RectifiedFlow(num_steps=num_step)
        self.motion_frame = motion_frames
        self.motion_in_channel = motion_in_channel
        self.motion_height = motion_height
        self.motion_width = motion_width


        # audio encoder
        self.audio_encoder = AudioFeatureMlp(audio_dim = audio_inchannel,
                                            audio_block = audio_block,
                                            outdim = encoder_out_dim,)
        
        # diffusion transformer
        self.diffusion = AudioMitionrefAllSequence(num_attention_heads = diffusion_attn_num_heads,
                                                attention_head_dim = diffusion_attn_head_dim,
                                                motion_in_channels = motion_in_channel,
                                                refimg_in_channels = image_inchannel,
                                                extra_in_channels = encoder_out_dim,
                                                out_channels = motion_in_channel,
                                                num_layers = diffusion_num_layers,
                                                
                                                image_width = image_width,
                                                image_height = image_height,
                                                image_patch_size = image_patch_size,

                                                motion_width = motion_width,
                                                motion_height = motion_height,
                                                motion_patch_size = motion_patch_size,
                                                motion_frames = motion_frames,)


    def forward(self,
                motion_gt:torch.Tensor, 
                ref_motion:torch.Tensor,
                audio:torch.Tensor ,
                ref_img:torch.Tensor = None,
                pose:torch.Tensor = None,
                ref_pose:torch.Tensor = None,
                timestep: Union[int, float, torch.LongTensor] = None, # Timesteps should be a 1d-array
                timestep_cond: Optional[torch.Tensor] = None,
                return_meta_info=False,
                **kwargs):
        """
        Args:
            motion_gt (torch.Tensor): (N,F,C,h,w)
            ref_img (torch.Tensor): (N,C,H,W)
            audio (torch.Tensor): (N,F,M,D)
            pose (torch.Tensor): (N,F,C,W,H)
            ref_pose (torch.Tensor): (N,C,W,H)
            timestep (torch.Tensor): (N,) <= num_steps
        """

        device = motion_gt.device
        n,f,c,h,w = motion_gt.shape

        audio_feature = self.audio_encoder(audio) # (n,t,d)
        
        # add noise
        if timestep is None:
            timestep = torch.randint(0,self.num_step+1,(n,)).to(device)
        t = (1 - timestep / self.num_step)[:,None,None,None,None]
        noise = torch.randn_like(motion_gt)
        vel_gt = motion_gt - noise
        motion_with_noise = t * motion_gt +  (1 - t) * noise #  (n,t,c,h,w)

        # forward
        vel_pred = self.diffusion(
            motion_hidden_states = motion_with_noise,
            refmotion_hidden_states = ref_motion,
            extra_hidden_states = audio_feature,
            timestep = timestep,
        )
        
        # loss
        diff_loss = l2(vel_pred,vel_gt)
        
        loss = diff_loss 

        loss_dict = {'loss':loss,'diff_loss':diff_loss}

        return loss_dict
    
    @torch.no_grad()
    def sample(self,
                ref_motion:torch.Tensor,
                audio:torch.Tensor ,
                ref_img:torch.Tensor = None,
                pose:torch.Tensor = None,
                ref_pose:torch.Tensor = None,
                timestep: Union[int, float, torch.LongTensor] = None, # Timesteps should be a 1d-array
                start_step:int = None,
                sample_step:int = 10,
                timestep_cond: Optional[torch.Tensor] = None,
                return_meta_info=False,
                **kwargs):
        """
        Args:
            ref_img (torch.Tensor): (N,C,H,W)
            audio (torch.Tensor): (N,M,D)
            ref_pose (torch.Tensor): (N,C,W,H)
            timestep (torch.Tensor): (N,) <= num_steps
        """

        device = ref_motion.device
        n,c,h,w = ref_motion.shape

        audio_feature = self.audio_encoder(audio) # (n,t,c,h,w) (n,t,d)
        
        # get noise 
        zt = torch.randn(n,self.motion_frame,self.motion_in_channel,self.motion_height,self.motion_width).to(device)

        # start_step : 1000
        if start_step is None:
            start_step = self.num_step 
        else:
            start_step = start_step
            zt = ref_motion.unsqueeze(1).repeat(1,self.motion_frame,1,1,1)
            timestep = torch.ones((n,)).to(device)
            timestep = timestep * start_step
            t = (1 - timestep / self.num_step)[:,None,None,None,None]
            noise = torch.randn_like(zt)
            zt = t * zt +  (1 - t) * noise #  (n,t,c,h,w)
        
        # step_seq [1000,995 ....]
        step_seq = np.linspace(0, start_step, num=sample_step+1, endpoint=True,dtype=int) # [0,5,10,15,....,start_step]
        step_seq = list(reversed(step_seq[1:])) # delete step:0  [start_step,.....,15,10,5]

        # Euler step
        dt = 1./sample_step
        for i in tqdm(step_seq):
            # time_step
            time_step = torch.ones((zt.shape[0],)).to(zt.device)  
            time_step = time_step * i
            
            # input
            zt = zt.to(ref_motion.dtype)
            
            # forward
            pre = self.diffusion(
                motion_hidden_states = zt,
                refmotion_hidden_states = ref_motion,
                extra_hidden_states = audio_feature,
                timestep = time_step,
            )
            zt = zt + pre * dt

        return zt



class A2MModel_MotionrefOnly_LearnableToken(ModelMixin, ConfigMixin):
    _supports_gradient_checkpointing = True
    
    @register_to_config
    def __init__(self,
                 audio_inchannel :int  = 384,
                 audio_block : int = 50,          
                 motion_num_token  :int = 12,
                 motion_in_channel :int = 128,
                 motion_frames : int = 128,
                 num_step :int = 1000,
                 # ----------- Audio feature encoder -----------
                 encoder_out_dim :int = 1024,
                 # ----------- Diffusion Transformer -----------
                 diffusion_attn_head_dim : int  = 64,
                 diffusion_attn_num_heads : int = 16,
                 diffusion_num_layers : int = 8,

                 **kwargs,
                 ):
        super().__init__()

        # setting
        self.num_step = num_step
        self.scheduler = RectifiedFlow(num_steps=num_step)
        self.motion_frames = motion_frames
        self.motion_num_token = motion_num_token
        self.motion_in_channel = motion_in_channel

        # audio encoder
        self.audio_encoder = AudioFeatureMlp(audio_dim = audio_inchannel,
                                            audio_block = audio_block,
                                            outdim = encoder_out_dim,)
        
        # diffusion transformer
        self.diffusion = AudioMitionref_LearnableToken(motion_num_token = motion_num_token,
                                                        motion_inchannel = motion_in_channel,
                                                        motion_frames = motion_frames,
                                                        extra_in_channels = encoder_out_dim,
                                                        out_channels = motion_in_channel,
                                                        num_attention_heads = diffusion_attn_num_heads,
                                                        attention_head_dim = diffusion_attn_head_dim,
                                                        num_layers = diffusion_num_layers,)

    def forward(self,
                motion_gt:torch.Tensor, 
                ref_motion:torch.Tensor,
                audio:torch.Tensor ,
                ref_audio:torch.Tensor = None,
                pose:torch.Tensor = None ,
                ref_pose:torch.Tensor = None,
                mask:torch.Tensor = None,
                timestep: Union[int, float, torch.LongTensor] = None, # Timesteps should be a 1d-array
                timestep_cond: Optional[torch.Tensor] = None,
                return_meta_info=False,
                **kwargs):
        """
        Args:
            motion_gt (torch.Tensor): (N,F,L,D)
            ref_motion (torch.Tensor): (N,L,D)
            audio (torch.Tensor): (N,F,M,D)
            pose (torch.Tensor): (N,F,C,W,H)
            ref_pose (torch.Tensor): (N,C,W,H)
            timestep (torch.Tensor): (N,) <= num_steps
        """

        device = motion_gt.device
        n,f,l,d = motion_gt.shape

        audio_feature = self.audio_encoder(audio) # (n,f,d)
        
        # add noise
        if timestep is None:
            timestep = torch.randint(0,self.num_step+1,(n,)).to(device)
        motion_with_noise,vel_gt = self.scheduler.get_train_tuple(z1=motion_gt,time_step=timestep)

        # forward
        vel_pred = self.diffusion(
            motion_hidden_states = motion_with_noise,
            refmotion_hidden_states = ref_motion,
            extra_hidden_states = audio_feature,
            timestep = timestep,
        )
        
        # mask
        if mask is None:
            mask = torch.ones((n,f)).to(device)

        # loss
        diff_loss = (vel_pred - vel_gt) ** 2 # [N,F,L,D]
        diff_loss = diff_loss.mean(dim=(2,3))  # [N,F], mean loss per frame
        diff_loss = (diff_loss * mask).sum() / mask.sum() 
        
        loss = diff_loss 

        loss_dict = {'loss':loss,'diff_loss':diff_loss}

        return loss_dict
    
    @torch.no_grad()
    def sample(self,
                ref_motion:torch.Tensor,
                audio:torch.Tensor ,
                ref_audio:torch.Tensor = None,
                pose:torch.Tensor = None ,
                ref_pose:torch.Tensor = None,
                timestep: Union[int, float, torch.LongTensor] = None, # Timesteps should be a 1d-array
                start_step:int = None,
                sample_step:int = 10,
                timestep_cond: Optional[torch.Tensor] = None,
                return_meta_info=False,
                **kwargs):
        """
        Args:
            ref_motion (torch.Tensor): (N,L,D)
            audio (torch.Tensor): (N,F,M,D)
            timestep (torch.Tensor): (N,) <= num_steps
        """

        device = ref_motion.device
        n,l,d = ref_motion.shape
        n,f,_,_ = audio.shape

        audio_feature = self.audio_encoder(audio) #  (n,f,d)
        
        # get noise
        start_step = self.num_step 
        zt = torch.randn(n,f,l,d).to(device)
        
        # step_seq [1000,995 ....]
        step_seq = np.linspace(0, start_step, num=sample_step+1, endpoint=True,dtype=int) # [0,5,10,15,....,start_step]
        step_seq = list(reversed(step_seq[1:])) # delete step:0  [start_step,.....,15,10,5]

        # Euler step
        dt = 1./sample_step
        for i in tqdm(step_seq):
            # time_step
            time_step = torch.ones((zt.shape[0],)).to(zt.device)  
            time_step = time_step * i
            
            # input
            zt = zt.to(ref_motion.dtype)
            
            # forward
            pre = self.diffusion(
                motion_hidden_states = zt,
                refmotion_hidden_states = ref_motion,
                extra_hidden_states = audio_feature,
                timestep = time_step,
            )
            zt = zt + pre * dt

        return zt


class A2MModel_MotionrefOnly_LearnableToken_SimpleAdaLN(ModelMixin, ConfigMixin):
    _supports_gradient_checkpointing = True
    
    @register_to_config
    def __init__(self,
                 audio_inchannel :int  = 384,
                 audio_block : int = 50,          
                 motion_num_token  :int = 12,
                 motion_in_channel :int = 128,
                 motion_frames : int = 128,
                 num_step :int = 1000,
                 # ----------- Audio feature encoder -----------
                 encoder_out_dim :int = 1024,
                 # ----------- Diffusion Transformer -----------
                 diffusion_attn_head_dim : int  = 64,
                 diffusion_attn_num_heads : int = 16,
                 diffusion_num_layers : int = 8,

                 **kwargs,
                 ):
        super().__init__()

        # setting
        self.num_step = num_step
        self.scheduler = RectifiedFlow(num_steps=num_step)
        self.motion_frames = motion_frames
        self.motion_num_token = motion_num_token
        self.motion_in_channel = motion_in_channel

        # audio encoder
        self.audio_encoder = AudioFeatureMlp(audio_dim = audio_inchannel,
                                            audio_block = audio_block,
                                            outdim = encoder_out_dim,)
        
        # diffusion transformer
        self.diffusion = AudioMitionref_LearnableToken_SimpleAdaLN(motion_num_token = motion_num_token,
                                                        motion_inchannel = motion_in_channel,
                                                        motion_frames = motion_frames,
                                                        extra_in_channels = encoder_out_dim,
                                                        out_channels = motion_in_channel,
                                                        num_attention_heads = diffusion_attn_num_heads,
                                                        attention_head_dim = diffusion_attn_head_dim,
                                                        num_layers = diffusion_num_layers,)

    def forward(self,
                motion_gt:torch.Tensor, 
                ref_motion:torch.Tensor,
                audio:torch.Tensor ,
                mask:torch.Tensor = None,
                timestep: Union[int, float, torch.LongTensor] = None, # Timesteps should be a 1d-array
                timestep_cond: Optional[torch.Tensor] = None,
                return_meta_info=False,
                **kwargs):
        """
        Args:
            motion_gt (torch.Tensor): (N,F,L,D)
            ref_motion (torch.Tensor): (N,L,D)
            audio (torch.Tensor): (N,F,M,D)
            pose (torch.Tensor): (N,F,C,W,H)
            ref_pose (torch.Tensor): (N,C,W,H)
            timestep (torch.Tensor): (N,) <= num_steps
        """

        device = motion_gt.device
        n,f,l,d = motion_gt.shape

        audio_feature = self.audio_encoder(audio) # (n,f,d)
        
        # add noise
        if timestep is None:
            timestep = torch.randint(0,self.num_step+1,(n,)).to(device)
        motion_with_noise,vel_gt = self.scheduler.get_train_tuple(z1=motion_gt,time_step=timestep)

        # forward
        vel_pred = self.diffusion(
            motion_hidden_states = motion_with_noise,
            refmotion_hidden_states = ref_motion,
            extra_hidden_states = audio_feature,
            timestep = timestep,
        )
        
        # mask
        if mask is None:
            mask = torch.ones((n,f)).to(device)

        # loss
        diff_loss = (vel_pred - vel_gt) ** 2 # [N,F,L,D]
        diff_loss = diff_loss.mean(dim=(2,3))  # [N,F], mean loss per frame
        diff_loss = (diff_loss * mask).sum() / mask.sum() 
        
        loss = diff_loss 

        loss_dict = {'loss':loss,'diff_loss':diff_loss}

        return loss_dict
    
    @torch.no_grad()
    def sample(self,
                ref_motion:torch.Tensor,
                audio:torch.Tensor ,
                ref_audio:torch.Tensor =None,
                pose:torch.Tensor = None ,
                ref_pose:torch.Tensor = None,
                timestep: Union[int, float, torch.LongTensor] = None, # Timesteps should be a 1d-array
                start_step:int = None,
                sample_step:int = 10,
                timestep_cond: Optional[torch.Tensor] = None,
                return_meta_info=False,
                **kwargs):
        """
        Args:
            ref_motion (torch.Tensor): (N,L,D)
            audio (torch.Tensor): (N,F,M,D)
            timestep (torch.Tensor): (N,) <= num_steps
        """

        device = ref_motion.device
        n,l,d = ref_motion.shape
        n,f,_,_ = audio.shape

        audio_feature = self.audio_encoder(audio) #  (n,f,d)
        
        # get noise
        start_step = self.num_step 
        zt = torch.randn(n,f,l,d).to(device)
        
        # step_seq [1000,995 ....]
        step_seq = np.linspace(0, start_step, num=sample_step+1, endpoint=True,dtype=int) # [0,5,10,15,....,start_step]
        step_seq = list(reversed(step_seq[1:])) # delete step:0  [start_step,.....,15,10,5]

        # Euler step
        dt = 1./sample_step
        for i in tqdm(step_seq):
            # time_step
            time_step = torch.ones((zt.shape[0],)).to(zt.device)  
            time_step = time_step * i
            
            # input
            zt = zt.to(ref_motion.dtype)
            
            # forward
            pre = self.diffusion(
                motion_hidden_states = zt,
                refmotion_hidden_states = ref_motion,
                extra_hidden_states = audio_feature,
                timestep = time_step,
            )
            zt = zt + pre * dt

        return zt


class A2MModel_CrossAtten_Audio(ModelMixin, ConfigMixin):
    _supports_gradient_checkpointing = True
    
    @register_to_config
    def __init__(self,
                 audio_inchannel :int  = 384,
                 audio_block : int = 50,          
                 motion_num_token  :int = 12,
                 motion_in_channel :int = 128,
                 motion_frames : int = 128,
                 num_step :int = 1000,
                 # ----------- Audio feature encoder -----------
                 intermediate_dim = 1024,
                 window_size = 32,
                 encoder_out_dim :int = 768,
                 # ----------- Diffusion Transformer -----------
                 diffusion_attn_head_dim : int  = 64,
                 diffusion_attn_num_heads : int = 16,
                 diffusion_num_layers : int = 8,

                 **kwargs,
                 ):
        super().__init__()

        # setting
        self.num_step = num_step
        self.scheduler = RectifiedFlow(num_steps=num_step)
        self.motion_frames = motion_frames
        self.motion_num_token = motion_num_token
        self.motion_in_channel = motion_in_channel

        # audio encoder
        self.audio_encoder = AudioFeatureWindowMlp(audio_dim = audio_inchannel,
                                                    audio_block = audio_block,
                                                    intermediate_dim = intermediate_dim,
                                                    window_size=window_size,
                                                    outdim = encoder_out_dim,)
        
        # diffusion transformer
        self.diffusion = A2MTransformer_CrossAttn_Audio(motion_num_token = motion_num_token,
                                                        motion_inchannel = motion_in_channel,
                                                        motion_frames = motion_frames,
                                                        audio_in_channels = encoder_out_dim,
                                                        out_channels = motion_in_channel,
                                                        num_attention_heads = diffusion_attn_num_heads,
                                                        attention_head_dim = diffusion_attn_head_dim,
                                                        num_layers = diffusion_num_layers,)

    def forward(self,
                motion_gt:torch.Tensor, 
                ref_motion:torch.Tensor,
                audio:torch.Tensor ,
                ref_audio:torch.Tensor,
                pose:torch.Tensor = None ,
                ref_pose:torch.Tensor = None,
                mask:torch.Tensor = None,
                timestep: Union[int, float, torch.LongTensor] = None, # Timesteps should be a 1d-array
                timestep_cond: Optional[torch.Tensor] = None,
                return_meta_info=False,
                **kwargs):
        """
        Args:
            motion_gt (torch.Tensor): (N,F,L,D)
            ref_motion (torch.Tensor): (N,L,D)
            audio (torch.Tensor): (N,F,M,D)
            ref_audio (torch.Tensor) : (N,M,D)
            pose (torch.Tensor): (N,F,C,W,H)
            ref_pose (torch.Tensor): (N,C,W,H)
            timestep (torch.Tensor): (N,) <= num_steps
        """

        device = motion_gt.device
        n,f,l,d = motion_gt.shape

        mix_audio = torch.cat((ref_audio.unsqueeze(1),audio),dim=1) # (N,F+1,M,D)
        audio_feature = self.audio_encoder(mix_audio) # (N,F+1,W,D)
        
        # add noise
        if timestep is None:
            timestep = torch.randint(0,self.num_step+1,(n,)).to(device)
        motion_with_noise,vel_gt = self.scheduler.get_train_tuple(z1=motion_gt,time_step=timestep)

        # forward
        vel_pred = self.diffusion(
            motion_hidden_states = motion_with_noise,
            refmotion_hidden_states = ref_motion,
            audio_hidden_states = audio_feature,
            timestep = timestep,
        )
        
        # mask
        if mask is None:
            mask = torch.ones((n,f)).to(device)

        # loss
        diff_loss = (vel_pred - vel_gt) ** 2 # [N,F,L,D]
        diff_loss = diff_loss.mean(dim=(2,3))  # [N,F], mean loss per frame
        diff_loss = (diff_loss * mask).sum() / mask.sum() 
        
        loss = diff_loss 

        loss_dict = {'loss':loss,'diff_loss':diff_loss}

        return loss_dict
    
    @torch.no_grad()
    def sample(self,
                ref_motion:torch.Tensor,
                audio:torch.Tensor ,
                ref_audio:torch.Tensor ,
                pose:torch.Tensor = None ,
                ref_pose:torch.Tensor = None,
                timestep: Union[int, float, torch.LongTensor] = None, # Timesteps should be a 1d-array
                start_step:int = None,
                sample_step:int = 10,
                timestep_cond: Optional[torch.Tensor] = None,
                return_meta_info=False,
                **kwargs):
        """
        Args:
            ref_motion (torch.Tensor): (N,L,D)
            audio (torch.Tensor): (N,F,M,D)
            timestep (torch.Tensor): (N,) <= num_steps
        """

        device = ref_motion.device
        n,l,d = ref_motion.shape
        n,f,_,_ = audio.shape

        mix_audio = torch.cat((ref_audio.unsqueeze(1),audio),dim=1) # (N,F+1,M,D)
        audio_feature = self.audio_encoder(mix_audio) # (N,F+1,W,D)
        
        # get noise
        start_step = self.num_step 
        zt = torch.randn(n,f,l,d).to(device)
        
        # step_seq [1000,995 ....]
        step_seq = np.linspace(0, start_step, num=sample_step+1, endpoint=True,dtype=int) # [0,5,10,15,....,start_step]
        step_seq = list(reversed(step_seq[1:])) # delete step:0  [start_step,.....,15,10,5]

        # Euler step
        dt = 1./sample_step
        for i in tqdm(step_seq):
            # time_step
            time_step = torch.ones((zt.shape[0],)).to(zt.device)  
            time_step = time_step * i
            
            # input
            zt = zt.to(ref_motion.dtype)
            
            # forward
            pre = self.diffusion(
                motion_hidden_states = zt,
                refmotion_hidden_states = ref_motion,
                audio_hidden_states = audio_feature,
                timestep = time_step,
            )
            zt = zt + pre * dt

        return zt


class A2MModel_CrossAtten_Audio_Pose(ModelMixin, ConfigMixin):
    _supports_gradient_checkpointing = True
    
    @register_to_config
    def __init__(self,
                 audio_inchannel :int  = 384,
                 audio_block : int = 50,          
                 motion_num_token  :int = 12,
                 motion_in_channel :int = 128,
                 motion_frames : int = 128,
                 num_step :int = 1000,
                 # ----------- pose -----------
                 pose_height :int = 32,
                 pose_width : int = 32, 
                 pose_inchannel : int = 4,
                 pose_patch_size : int = 2,
                 # ----------- Audio feature encoder -----------
                 intermediate_dim = 1024,
                 window_size = 32,
                 encoder_out_dim :int = 768,
                 # ----------- Diffusion Transformer -----------
                 diffusion_attn_head_dim : int  = 64,
                 diffusion_attn_num_heads : int = 16,
                 diffusion_num_layers : int = 8,

                 **kwargs,
                 ):
        super().__init__()

        # setting
        self.num_step = num_step
        self.scheduler = RectifiedFlow(num_steps=num_step)
        self.motion_frames = motion_frames
        self.motion_num_token = motion_num_token
        self.motion_in_channel = motion_in_channel

        # audio encoder
        self.audio_encoder = AudioFeatureWindowMlp(audio_dim = audio_inchannel,
                                                    audio_block = audio_block,
                                                    intermediate_dim = intermediate_dim,
                                                    window_size=window_size,
                                                    outdim = encoder_out_dim,)
        
        # diffusion transformer
        self.diffusion = A2MTransformer_CrossAttn_Audio_Pose(motion_num_token = motion_num_token,
                                                        motion_inchannel = motion_in_channel,
                                                        motion_frames = motion_frames,
                                                        pose_height = pose_height,
                                                        pose_width = pose_width,
                                                        pose_inchannel = pose_inchannel,
                                                        pose_patch_size = pose_patch_size,
                                                        audio_in_channels = encoder_out_dim,
                                                        out_channels = motion_in_channel,
                                                        num_attention_heads = diffusion_attn_num_heads,
                                                        attention_head_dim = diffusion_attn_head_dim,
                                                        num_layers = diffusion_num_layers,)

    def forward(self,
                motion_gt:torch.Tensor, 
                ref_motion:torch.Tensor,
                audio:torch.Tensor ,
                ref_audio:torch.Tensor,
                pose:torch.Tensor ,
                ref_pose:torch.Tensor ,
                mask:torch.Tensor = None,
                timestep: Union[int, float, torch.LongTensor] = None, # Timesteps should be a 1d-array
                timestep_cond: Optional[torch.Tensor] = None,
                return_meta_info=False,
                **kwargs):
        """
        Args:
            motion_gt (torch.Tensor): (N,F,L,D)
            ref_motion (torch.Tensor): (N,L,D)
            audio (torch.Tensor): (N,F,M,D)
            ref_audio (torch.Tensor) : (N,M,D)
            pose (torch.Tensor): (N,F,C,W,H)
            ref_pose (torch.Tensor): (N,C,W,H)
            timestep (torch.Tensor): (N,) <= num_steps
        """

        device = motion_gt.device
        n,f,l,d = motion_gt.shape

        # audio
        mix_audio = torch.cat((ref_audio.unsqueeze(1),audio),dim=1) # (N,F+1,M,D)
        audio_feature = self.audio_encoder(mix_audio) # (N,F+1,W,D)

        # pose
        mixpose = torch.cat((ref_pose.unsqueeze(1),pose),dim=1) # (N,F+1,C,H,W)
        
        # add noise
        if timestep is None:
            timestep = torch.randint(0,self.num_step+1,(n,)).to(device)
        motion_with_noise,vel_gt = self.scheduler.get_train_tuple(z1=motion_gt,time_step=timestep)

        # forward
        vel_pred = self.diffusion(
            motion_hidden_states = motion_with_noise,
            refmotion_hidden_states = ref_motion,
            audio_hidden_states = audio_feature,
            pose_hidden_states = mixpose,
            timestep = timestep,
        )
        
        # mask
        if mask is None:
            mask = torch.ones((n,f)).to(device)

        # loss
        diff_loss = (vel_pred - vel_gt) ** 2 # [N,F,L,D]
        diff_loss = diff_loss.mean(dim=(2,3))  # [N,F], mean loss per frame
        diff_loss = (diff_loss * mask).sum() / mask.sum() 
        
        loss = diff_loss 

        loss_dict = {'loss':loss,'diff_loss':diff_loss}

        return loss_dict
    
    @torch.no_grad()
    def sample(self,
                ref_motion:torch.Tensor,
                audio:torch.Tensor ,
                ref_audio:torch.Tensor ,
                pose:torch.Tensor ,
                ref_pose:torch.Tensor ,
                timestep: Union[int, float, torch.LongTensor] = None, # Timesteps should be a 1d-array
                start_step:int = None,
                sample_step:int = 10,
                timestep_cond: Optional[torch.Tensor] = None,
                return_meta_info=False,
                **kwargs):
        """
        Args:
            ref_motion (torch.Tensor): (N,L,D)
            audio (torch.Tensor): (N,F,M,D)
            timestep (torch.Tensor): (N,) <= num_steps
        """

        device = ref_motion.device
        n,l,d = ref_motion.shape
        n,f,_,_ = audio.shape

        # audio
        mix_audio = torch.cat((ref_audio.unsqueeze(1),audio),dim=1) # (N,F+1,M,D)
        audio_feature = self.audio_encoder(mix_audio) # (N,F+1,W,D)

        # pose
        mixpose = torch.cat((ref_pose.unsqueeze(1),pose),dim=1) # (N,F+1,C,H,W)
        
        # get noise
        start_step = self.num_step 
        zt = torch.randn(n,f,l,d).to(device)
        
        # step_seq [1000,995 ....]
        step_seq = np.linspace(0, start_step, num=sample_step+1, endpoint=True,dtype=int) # [0,5,10,15,....,start_step]
        step_seq = list(reversed(step_seq[1:])) # delete step:0  [start_step,.....,15,10,5]

        # Euler step
        dt = 1./sample_step
        for i in tqdm(step_seq):
            # time_step
            time_step = torch.ones((zt.shape[0],)).to(zt.device)  
            time_step = time_step * i
            
            # input
            zt = zt.to(ref_motion.dtype)
            
            # forward
            pre = self.diffusion(
                motion_hidden_states = zt,
                refmotion_hidden_states = ref_motion,
                audio_hidden_states = audio_feature,
                pose_hidden_states = mixpose,
                timestep = time_step,
            )
            zt = zt + pre * dt

        return zt


class A2MModel_CrossAtten_Audio_PosePre(ModelMixin, ConfigMixin):
    _supports_gradient_checkpointing = True
    
    @register_to_config
    def __init__(self,
                 audio_inchannel :int  = 384,
                 audio_block : int = 50,          
                 motion_num_token  :int = 12,
                 motion_in_channel :int = 128,
                 motion_frames : int = 128,
                 num_step :int = 1000,
                 # ----------- pose -----------
                 pose_height :int = 32,
                 pose_width : int = 32, 
                 pose_inchannel : int = 4,
                 pose_patch_size : int = 2,
                 # ----------- pose predictor -----------
                 pose_predictor_attn_head_dim :int = 64,
                 pose_predictor_attn_num_heads : int = 8,
                 pose_predictor_attn_num_layers : int = 4,
                 # ----------- Audio feature encoder -----------
                 intermediate_dim = 1024,
                 window_size = 32,
                 encoder_out_dim :int = 768,
                 # ----------- Diffusion Transformer -----------
                 diffusion_attn_head_dim : int  = 64,
                 diffusion_attn_num_heads : int = 16,
                 diffusion_num_layers : int = 8,

                 **kwargs,
                 ):
        super().__init__()

        # setting
        self.num_step = num_step
        self.scheduler = RectifiedFlow(num_steps=num_step)
        self.motion_frames = motion_frames
        self.motion_num_token = motion_num_token
        self.motion_in_channel = motion_in_channel

        # audio encoder
        self.audio_encoder = AudioFeatureWindowMlp(audio_dim = audio_inchannel,
                                                    audio_block = audio_block,
                                                    intermediate_dim = intermediate_dim,
                                                    window_size=window_size,
                                                    outdim = encoder_out_dim,)

        # pose pre
        self.pose_predictor = A2PTransformer(audio_window = window_size,
                                            audio_in_channels = encoder_out_dim,

                                            pose_height = pose_height,
                                            pose_width = pose_width,
                                            pose_inchannel = pose_inchannel,
                                            pose_patch_size = pose_patch_size,
                                            pose_frame = motion_frames + 1,

                                            num_attention_heads = pose_predictor_attn_head_dim,
                                            attention_head_dim = pose_predictor_attn_num_heads,
                                            num_layers = pose_predictor_attn_num_layers)
        
        # diffusion transformer
        self.diffusion = A2MTransformer_CrossAttn_Audio_Pose(motion_num_token = motion_num_token,
                                                        motion_inchannel = motion_in_channel,
                                                        motion_frames = motion_frames,
                                                        pose_height = pose_height,
                                                        pose_width = pose_width,
                                                        pose_inchannel = pose_inchannel,
                                                        pose_patch_size = pose_patch_size,
                                                        audio_in_channels = encoder_out_dim,
                                                        out_channels = motion_in_channel,
                                                        num_attention_heads = diffusion_attn_num_heads,
                                                        attention_head_dim = diffusion_attn_head_dim,
                                                        num_layers = diffusion_num_layers,)

    def forward(self,
                motion_gt:torch.Tensor, 
                ref_motion:torch.Tensor,
                audio:torch.Tensor ,
                ref_audio:torch.Tensor,
                pose:torch.Tensor ,
                ref_pose:torch.Tensor ,
                mask:torch.Tensor = None,
                timestep: Union[int, float, torch.LongTensor] = None, # Timesteps should be a 1d-array
                timestep_cond: Optional[torch.Tensor] = None,
                return_meta_info=False,
                **kwargs):
        """
        Args:
            motion_gt (torch.Tensor): (N,F,L,D)
            ref_motion (torch.Tensor): (N,L,D)
            audio (torch.Tensor): (N,F,M,D)
            ref_audio (torch.Tensor): (N,M,D)
            pose (torch.Tensor): (N,F,C,H,W)
            ref_pose (torch.Tensor): (N,C,H,W)
            timestep (torch.Tensor): (N,) <= num_steps
        """

        device = motion_gt.device
        n,f,l,d = motion_gt.shape

        # audio
        mix_audio = torch.cat((ref_audio.unsqueeze(1),audio),dim=1) # (N,F+1,M,D)
        audio_feature = self.audio_encoder(mix_audio) # (N,F+1,W,D)

        # pose
        mix_pose_pre = self.pose_predictor(ref_pose,audio_feature) # N,F+1,C,H,W
        pose_pre = mix_pose_pre[:,1:,:]
        
        # add noise
        if timestep is None:
            timestep = torch.randint(0,self.num_step+1,(n,)).to(device)
        motion_with_noise,vel_gt = self.scheduler.get_train_tuple(z1=motion_gt,time_step=timestep)

        # forward
        vel_pred = self.diffusion(
            motion_hidden_states = motion_with_noise,
            refmotion_hidden_states = ref_motion,
            audio_hidden_states = audio_feature,
            pose_hidden_states = mix_pose_pre,
            timestep = timestep,
        )
        
        # mask
        if mask is None:
            mask = torch.ones((n,f)).to(device)

        # loss
        diff_loss = (vel_pred - vel_gt) ** 2 # [N,F,L,D]
        diff_loss = diff_loss.mean(dim=(2,3))  # [N,F], mean loss per frame
        diff_loss = (diff_loss * mask).sum() / mask.sum() 

        pose_loss = (pose_pre - pose) ** 2  # N,F,C,H,W
        pose_loss = pose_loss.mean(dim=(2,3,4)) # N,F
        pose_loss = (pose_loss * mask).sum() / mask.sum()
        
        loss = diff_loss + pose_loss

        loss_dict = {'loss':loss,'diff_loss':diff_loss,'pose_loss':pose_loss}

        return loss_dict
    
    @torch.no_grad()
    def sample(self,
                ref_motion:torch.Tensor,
                audio:torch.Tensor ,
                ref_audio:torch.Tensor ,
                pose:torch.Tensor ,
                ref_pose:torch.Tensor ,
                timestep: Union[int, float, torch.LongTensor] = None, # Timesteps should be a 1d-array
                start_step:int = None,
                sample_step:int = 2,
                timestep_cond: Optional[torch.Tensor] = None,
                return_meta_info=False,
                **kwargs):
        """
        Args:
            ref_motion (torch.Tensor): (N,L,D)
            audio (torch.Tensor): (N,F,M,D)
            timestep (torch.Tensor): (N,) <= num_steps
        """

        device = ref_motion.device
        n,l,d = ref_motion.shape
        n,f,_,_ = audio.shape

        # audio
        mix_audio = torch.cat((ref_audio.unsqueeze(1),audio),dim=1) # (N,F+1,M,D)
        audio_feature = self.audio_encoder(mix_audio) # (N,F+1,W,D)

        # pose
        mix_pose_pre = self.pose_predictor(ref_pose,audio_feature) # N,F+1,C,H,W
        
        # get noise
        start_step = self.num_step 
        zt = torch.randn(n,f,l,d).to(device)
        
        # step_seq [1000,995 ....]
        step_seq = np.linspace(0, start_step, num=sample_step+1, endpoint=True,dtype=int) # [0,5,10,15,....,start_step]
        step_seq = list(reversed(step_seq[1:])) # delete step:0  [start_step,.....,15,10,5]

        # Euler step
        dt = 1./sample_step
        for i in tqdm(step_seq):
            # time_step
            time_step = torch.ones((zt.shape[0],)).to(zt.device)  
            time_step = time_step * i
            
            # input
            zt = zt.to(ref_motion.dtype)
            
            # forward
            pre = self.diffusion(
                motion_hidden_states = zt,
                refmotion_hidden_states = ref_motion,
                audio_hidden_states = audio_feature,
                pose_hidden_states = mix_pose_pre,
                timestep = time_step,
            )
            zt = zt + pre * dt

        return zt

class A2MModel_CrossAtten_Pose(ModelMixin, ConfigMixin):
    _supports_gradient_checkpointing = True
    
    @register_to_config
    def __init__(self,       
                 motion_num_token  :int = 12,
                 motion_in_channel :int = 128,
                 motion_frames : int = 128,
                 num_step :int = 1000,
                 # ----------- pose -----------
                 pose_height :int = 32,
                 pose_width : int = 32, 
                 pose_inchannel : int = 4,
                 pose_patch_size : int = 2,
                 # ----------- Diffusion Transformer -----------
                 diffusion_attn_head_dim : int  = 64,
                 diffusion_attn_num_heads : int = 16,
                 diffusion_num_layers : int = 8,

                 **kwargs,
                 ):
        super().__init__()

        # setting
        self.num_step = num_step
        self.scheduler = RectifiedFlow(num_steps=num_step)
        self.motion_frames = motion_frames
        self.motion_num_token = motion_num_token
        self.motion_in_channel = motion_in_channel

        
        # diffusion transformer
        self.diffusion = A2MTransformer_CrossAttn_Pose(motion_num_token = motion_num_token,
                                                        motion_inchannel = motion_in_channel,
                                                        motion_frames = motion_frames,
                                                        pose_height = pose_height,
                                                        pose_width = pose_width,
                                                        pose_inchannel = pose_inchannel,
                                                        pose_patch_size = pose_patch_size,
                                                        out_channels = motion_in_channel,
                                                        num_attention_heads = diffusion_attn_num_heads,
                                                        attention_head_dim = diffusion_attn_head_dim,
                                                        num_layers = diffusion_num_layers,)

    def forward(self,
                motion_gt:torch.Tensor, 
                ref_motion:torch.Tensor,
                pose:torch.Tensor ,
                ref_pose:torch.Tensor ,
                audio:torch.Tensor = None,
                ref_audio:torch.Tensor = None,
                mask:torch.Tensor = None,
                timestep: Union[int, float, torch.LongTensor] = None, # Timesteps should be a 1d-array
                timestep_cond: Optional[torch.Tensor] = None,
                return_meta_info=False,
                **kwargs):
        """
        Args:
            motion_gt (torch.Tensor): (N,F,L,D)
            ref_motion (torch.Tensor): (N,L,D)
            audio (torch.Tensor): (N,F,M,D)
            ref_audio (torch.Tensor) : (N,M,D)
            pose (torch.Tensor): (N,F,C,W,H)
            ref_pose (torch.Tensor): (N,C,W,H)
            timestep (torch.Tensor): (N,) <= num_steps
        """

        device = motion_gt.device
        n,f,l,d = motion_gt.shape

        # pose
        mixpose = torch.cat((ref_pose.unsqueeze(1),pose),dim=1) # (N,F+1,C,H,W)
        
        # add noise
        if timestep is None:
            timestep = torch.randint(0,self.num_step+1,(n,)).to(device)
        motion_with_noise,vel_gt = self.scheduler.get_train_tuple(z1=motion_gt,time_step=timestep)

        # forward
        vel_pred = self.diffusion(
            motion_hidden_states = motion_with_noise,
            refmotion_hidden_states = ref_motion,
            pose_hidden_states = mixpose,
            timestep = timestep,
        )
        
        # mask
        if mask is None:
            mask = torch.ones((n,f)).to(device)

        # loss
        diff_loss = (vel_pred - vel_gt) ** 2 # [N,F,L,D]
        diff_loss = diff_loss.mean(dim=(2,3))  # [N,F], mean loss per frame
        diff_loss = (diff_loss * mask).sum() / mask.sum() 
        
        loss = diff_loss 

        loss_dict = {'loss':loss,'diff_loss':diff_loss}

        return loss_dict
    
    @torch.no_grad()
    def sample(self,
                ref_motion:torch.Tensor,
                pose:torch.Tensor ,
                ref_pose:torch.Tensor ,
                audio:torch.Tensor =None,
                ref_audio:torch.Tensor =None,
                timestep: Union[int, float, torch.LongTensor] = None, # Timesteps should be a 1d-array
                start_step:int = None,
                sample_step:int = 10,
                timestep_cond: Optional[torch.Tensor] = None,
                return_meta_info=False,
                **kwargs):
        """
        Args:
            ref_motion (torch.Tensor): (N,L,D)
            audio (torch.Tensor): (N,F,M,D)
            timestep (torch.Tensor): (N,) <= num_steps
        """

        device = ref_motion.device
        n,l,d = ref_motion.shape
        n,f,_,_,_ = pose.shape

        # pose
        mixpose = torch.cat((ref_pose.unsqueeze(1),pose),dim=1) # (N,F+1,C,H,W)
        
        # get noise
        start_step = self.num_step 
        zt = torch.randn(n,f,l,d).to(device)
        
        # step_seq [1000,995 ....]
        step_seq = np.linspace(0, start_step, num=sample_step+1, endpoint=True,dtype=int) # [0,5,10,15,....,start_step]
        step_seq = list(reversed(step_seq[1:])) # delete step:0  [start_step,.....,15,10,5]

        # Euler step
        dt = 1./sample_step
        for i in tqdm(step_seq):
            # time_step
            time_step = torch.ones((zt.shape[0],)).to(zt.device)  
            time_step = time_step * i
            
            # input
            zt = zt.to(ref_motion.dtype)
            
            # forward
            pre = self.diffusion(
                motion_hidden_states = zt,
                refmotion_hidden_states = ref_motion,
                pose_hidden_states = mixpose,
                timestep = time_step,
            )
            zt = zt + pre * dt

        return zt
