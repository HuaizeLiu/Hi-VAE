import torch
from torch import nn
import einops
from typing import Tuple
import random
import numpy as np
from tqdm import tqdm
from .modules import DuoFrameDownEncoder,Upsampler,MapConv,MotionDownEncoder
from .frequency_utils import gaussian_low_pass_filter, freq_3d_filter
from .loss import l1,l2
from .regularizers import DiagonalGaussianRegularizer
from .transformer import (MotionTransformer,
                        AMDDiffusionTransformerModel,
                        AMDDiffusionTransformerModelTempMotion,
                        MotionEncoderLearnTokenTransformer,
                        AMDReconstructTransformerModel,
                        AMDDiffusionTransformerModelDualStream,
                        AMDDiffusionTransformerModelImgSpatial,
                        AMDDiffusionTransformerModelImgSpatialTempMotion,
                        MotionEncoderLearnTokenTemporalTransformer,
                        MotionEncoderLearnTokenOnlyTemporalTransformer)
from .rectified_flow import RectifiedFlow
from diffusers.configuration_utils import ConfigMixin, register_to_config
from diffusers.models.modeling_utils import ModelMixin
from diffusers.models.resnet import ResnetBlock2D
import einops
import torch.nn.functional as F
from torchsummary import summary

from diffusers.utils import export_to_gif

class AMDModel(ModelMixin, ConfigMixin):
    _supports_gradient_checkpointing = True
    
    @register_to_config
    def __init__(self,
                 image_inchannel :int  = 4,
                 image_height  :int = 32,
                 image_width  :int  = 32,
                 video_frames  :int = 16,
                 scheduler_num_step :int = 1000,
                 use_filter:bool = False,
                 filter_num:float = 0.4,
                 high_filter_num:float = 0.6,
                 use_grey:bool = False,
                 use_camera_down:bool = False,
                 use_regularizers:bool = False,
                 use_motiontemporal:bool = True,
                 klloss_weight: float = 0.005, # 比diffloss小一个量级
                 use_mask:bool = False,
                 motion_type:str='plus',
                 
                 # 理论上来说,高频信号在transformer中的深层,低频信号在transformer的浅层,所以不同支路的layer_num也需要做实验
                 # -----------Object MotionEncoder -----------
                 object_motion_token_num:int = 12,
                 object_motion_token_channel: int = 128,
                 object_enc_num_layers:int = 8,
                 enc_nhead:int = 8,
                 enc_ndim:int = 64,
                 enc_dropout:float = 0.0,
                 motion_need_norm_out:bool = False,

                 # -----------Camera MotionEncoder -----------
                 camera_motion_token_num:int = 12,
                 camera_motion_token_channel: int = 128,
                 camera_enc_num_layers:int = 8,
                
                 # -----------MotionTransformer ---------
                 motion_token_num:int = 12,
                 motion_token_channel: int = 128, 
                 need_motion_transformer :bool = False,
                 motion_transformer_attn_head_dim:int = 64,
                 motion_transformer_attn_num_heads:int = 16,
                 motion_transformer_num_layers:int = 4, 
                
                 # ----------- Diffusion Transformer -----------
                 diffusion_model_type : str = 'default', # or dual
                 diffusion_attn_head_dim : int  = 64,
                 diffusion_attn_num_heads : int = 16,
                 diffusion_out_channels : int = 4,
                 diffusion_num_layers : int = 16,
                 image_patch_size : int  = 2,
                 motion_patch_size : int = 1,
                 motion_drop_ratio: float = 0.0,

                 # ----------- Sample --------------
                 extract_motion_with_motion_transformer = False,
                 **kwargs,
                 ):
        super().__init__()

        # setting
        self.num_step = scheduler_num_step
        self.scheduler = RectifiedFlow(num_steps=scheduler_num_step)
        self.need_motion_transformer = need_motion_transformer
        self.extract_motion_with_motion_transformer = extract_motion_with_motion_transformer
        self.diffusion_model_type = diffusion_model_type
        self.target_frame = video_frames
        self.use_motiontemporal = use_motiontemporal

        self.camera_motion_token_channel = camera_motion_token_channel
        self.object_motion_token_channel = object_motion_token_channel
        self.motion_token_channel = motion_token_channel

        # self.use_motion = use_motion
        self.filter_num = filter_num
        self.high_filter_num = high_filter_num

        # frepuency filter
        self.use_filter = use_filter 
        self.use_grey = use_grey
        self.use_camera_down = use_camera_down
        self.use_mask = use_mask
        # self.mask_video_ratio = mask_video_ratio

        # use regularizers:
        self.use_regularizers = use_regularizers
        self.klloss_weight = klloss_weight
        if use_regularizers:
            self.regularization = DiagonalGaussianRegularizer(sample=True)

            self.camera_target_motion_map = nn.Linear(camera_motion_token_channel//2, motion_token_channel)
            self.camera_source_motion_map = nn.Linear(camera_motion_token_channel, motion_token_channel)
            self.object_target_motion_map = nn.Linear(object_motion_token_channel//2, motion_token_channel)
            self.object_source_motion_map = nn.Linear(object_motion_token_channel, motion_token_channel)
        else:

            if camera_motion_token_channel != motion_token_channel:
                self.camera_motion_map = nn.Linear(camera_motion_token_channel, motion_token_channel)
            if object_motion_token_channel != motion_token_channel:
                self.object_motion_map = nn.Linear(object_motion_token_channel, motion_token_channel)

        if use_motiontemporal:
            # object motion Encoder 
            self.object_motion_encoder = MotionEncoderLearnTokenTemporalTransformer(img_height = image_height,
                                                                    img_width=image_width,
                                                                    img_inchannel=image_inchannel,
                                                                    img_patch_size = image_patch_size,
                                                                    motion_token_num =  object_motion_token_num,
                                                                    motion_channel = object_motion_token_channel,
                                                                    need_norm_out = motion_need_norm_out,
                                                                    video_frames=self.target_frame,
                                                                    # ----- attention
                                                                    num_attention_heads=enc_nhead,
                                                                    attention_head_dim=enc_ndim,
                                                                    num_layers=object_enc_num_layers,      
                                                                    dropout=enc_dropout,
                                                                    attention_bias= True,)
            
            # camera motion Encoder 
            self.camera_motion_encoder = MotionEncoderLearnTokenTemporalTransformer(img_height = image_height,
                                                                    img_width=image_width,
                                                                    img_inchannel=image_inchannel,
                                                                    img_patch_size = image_patch_size,
                                                                    motion_token_num =  camera_motion_token_num,
                                                                    motion_channel = camera_motion_token_channel,
                                                                    need_norm_out = motion_need_norm_out,
                                                                    video_frames=self.target_frame,
                                                                    # ----- attention
                                                                    num_attention_heads=enc_nhead,
                                                                    attention_head_dim=enc_ndim,
                                                                    num_layers=camera_enc_num_layers,      
                                                                    dropout=enc_dropout,
                                                                    attention_bias= True,)
        else:
            # object motion Encoder 
            self.object_motion_encoder = MotionEncoderLearnTokenTransformer(img_height = image_height,
                                                                    img_width=image_width,
                                                                    img_inchannel=image_inchannel,
                                                                    img_patch_size = image_patch_size,
                                                                    motion_token_num =  object_motion_token_num,
                                                                    motion_channel = object_motion_token_channel,
                                                                    need_norm_out = motion_need_norm_out,
                                                                    # ----- attention
                                                                    num_attention_heads=enc_nhead,
                                                                    attention_head_dim=enc_ndim,
                                                                    num_layers=object_enc_num_layers,      
                                                                    dropout=enc_dropout,
                                                                    attention_bias= True,)
            
            # camera motion Encoder 
            self.camera_motion_encoder = MotionEncoderLearnTokenTransformer(img_height = image_height // 4, # down_camera: image_height // 4
                                                                    img_width=image_width // 4, # down_camera: image_height // 4
                                                                    img_inchannel=image_inchannel,
                                                                    img_patch_size = image_patch_size,
                                                                    motion_token_num =  camera_motion_token_num,
                                                                    motion_channel = camera_motion_token_channel,
                                                                    need_norm_out = motion_need_norm_out,
                                                                    # ----- attention
                                                                    num_attention_heads=enc_nhead,
                                                                    attention_head_dim=enc_ndim,
                                                                    num_layers=camera_enc_num_layers,      
                                                                    dropout=enc_dropout,
                                                                    attention_bias= True,)
        if self.use_camera_down:
            self.camera_down = nn.Sequential(
                nn.Conv2d(in_channels=4, out_channels=16, kernel_size=3, stride=2, padding=1),
                nn.Conv2d(in_channels=16, out_channels=4, kernel_size=3, stride=1, padding=1),
                nn.MaxPool2d(kernel_size=2, stride=2, padding=0)
            )
        
        # motion transformer
        if need_motion_transformer:
            self.motion_transformer = MotionTransformer(motion_token_num=motion_token_num,
                                                        motion_token_channel=motion_token_channel,
                                                        attention_head_dim=motion_transformer_attn_head_dim,
                                                        num_attention_heads=motion_transformer_attn_num_heads,
                                                        num_layers=motion_transformer_num_layers,)
        
        # diffusion transformer
        dit_image_inchannel = image_inchannel * 2 # zi + zt
        if diffusion_model_type == 'default':
            self.diffusion_transformer = AMDDiffusionTransformerModel(num_attention_heads= diffusion_attn_num_heads,
                                                                    attention_head_dim= diffusion_attn_head_dim,
                                                                    out_channels = diffusion_out_channels,
                                                                    num_layers= diffusion_num_layers,
                                                                    motion_type = motion_type,
                                                                    # ----- img
                                                                    image_width= image_width,
                                                                    image_height= image_height,
                                                                    image_patch_size= image_patch_size,
                                                                    image_in_channels = dit_image_inchannel, 
                                                                    # ----- motion
                                                                    motion_token_num = motion_token_num,
                                                                    motion_in_channels = motion_token_channel,)
        elif diffusion_model_type == 'dual':
            self.diffusion_transformer = AMDDiffusionTransformerModelDualStream(num_attention_heads= diffusion_attn_num_heads,
                                                                attention_head_dim= diffusion_attn_head_dim,
                                                                out_channels = diffusion_out_channels,
                                                                num_layers= diffusion_num_layers,
                                                                # ----- img
                                                                image_width= image_width,
                                                                image_height= image_height,
                                                                image_patch_size= image_patch_size,
                                                                image_in_channels = dit_image_inchannel, 
                                                                # ----- motion
                                                                motion_token_num = motion_token_num,
                                                                motion_in_channels = motion_token_channel,
                                                                motion_target_num_frame = self.target_frame)
        elif diffusion_model_type == 'spatial':
            self.diffusion_transformer = AMDDiffusionTransformerModelImgSpatial(num_attention_heads= diffusion_attn_num_heads,
                                                                attention_head_dim= diffusion_attn_head_dim,
                                                                out_channels = diffusion_out_channels,
                                                                num_layers= diffusion_num_layers,
                                                                motion_type = motion_type,
                                                                # ----- img
                                                                image_width= image_width,
                                                                image_height= image_height,
                                                                image_patch_size= image_patch_size,
                                                                image_in_channels = dit_image_inchannel, 
                                                                # ----- motion
                                                                motion_token_num = motion_token_num,
                                                                motion_in_channels = motion_token_channel,
                                                                motion_target_num_frame = self.target_frame)
        else:
            raise IndexError

    def forward(self,video:torch.Tensor,ref_img:torch.Tensor,video_grey=None,ref_img_grey=None,camera_mask=None,time_step:torch.tensor = None,return_meta_info=False,mask_ratio=None):
        """
        Args:
            video: (N,T,C,H,W)
            ref_img: (N,T,C,H,W)
        """

        device = video.device
        n,t,c,h,w = video.shape

        assert video.shape == ref_img.shape ,f'video.shape:{video.shape}should be equal to ref_img.shape:{ref_img.shape}'
        
        # motion encoder
        if mask_ratio is not None:
            mask_ratio = torch.rand(1).item() * mask_ratio

        gt_video = video
        # gt_video = video_grey
        # if self.mask_video_ratio != 0:
                
        #     b_z, f_z, c_z, h_z, w_z = video.shape
        #     rand_mask = torch.rand(h_z, w_z).to(device=video.device)
        #     mask = rand_mask > self.mask_video_ratio
        #     mask = mask.unsqueeze(0).unsqueeze(0).unsqueeze(0)  
        #     mask = mask.expand(b_z, f_z, c_z, h_z, w_z) 
        #     video = video * mask

        #     if self.use_grey:
        #         video_grey = video_grey * mask
        
        refimg_and_video = torch.cat([ref_img,video],dim=1)# (n,t+t,C,H,W)

        # frequency filter
        if self.use_filter:
            freq_filter = gaussian_low_pass_filter([2*t, h, w],self.filter_num,self.filter_num)
            freq_filter = freq_filter.unsqueeze(0).unsqueeze(0).repeat(n, c, 1, 1, 1)
            freq_filter = freq_filter.to(device=refimg_and_video.device)

            high_freq_filter = gaussian_low_pass_filter([2*t, h, w],self.high_filter_num,self.high_filter_num)
            high_freq_filter = high_freq_filter.unsqueeze(0).unsqueeze(0).repeat(n, c, 1, 1, 1)
            high_freq_filter = high_freq_filter.to(device=refimg_and_video.device)            

            if self.use_grey:
                refimg_and_video_grey = torch.cat([ref_img_grey,video_grey],dim=1)# (n,t+t,C,H,W) (1,32,4,32,32)
                refimg_and_video_grey = einops.rearrange(refimg_and_video_grey, "n t c h w -> n c t h w")
                LF_refimg_and_video, _ = freq_3d_filter(refimg_and_video_grey, freq_filter)
                _,HF_refimg_and_video = freq_3d_filter(refimg_and_video_grey, high_freq_filter)
            else:
                refimg_and_video = einops.rearrange(refimg_and_video, "n t c h w -> n c t h w")
                LF_refimg_and_video, _ = freq_3d_filter(refimg_and_video_grey, freq_filter)
                _,HF_refimg_and_video = freq_3d_filter(refimg_and_video_grey, high_freq_filter)

            LF_refimg_and_video = einops.rearrange(LF_refimg_and_video, "n c t h w -> n t c h w")
            HF_refimg_and_video = einops.rearrange(HF_refimg_and_video, "n c t h w -> n t c h w")

            if self.use_mask:
                LF_refimg_and_video = LF_refimg_and_video * camera_mask

            if self.use_camera_down:
                LF_refimg_and_video = einops.rearrange(LF_refimg_and_video, "n t c h w -> (n t) c h w")
                LF_refimg_and_video = self.camera_down(LF_refimg_and_video)
                LF_refimg_and_video = einops.rearrange(LF_refimg_and_video, "(n t) c h w -> n t c h w",n=n)

            object_motion = self.object_motion_encoder(HF_refimg_and_video,mask_ratio) # (n,t+t,l,d)
            camera_motion = self.camera_motion_encoder(LF_refimg_and_video,mask_ratio) # (n,t+t,l,d)

        else:
            if self.use_camera_down:
                LF_refimg_and_video = einops.rearrange(LF_refimg_and_video, "n t c h w -> (n t) c h w")
                LF_refimg_and_video = self.camera_down(LF_refimg_and_video)
                LF_refimg_and_video = einops.rearrange(LF_refimg_and_video, "(n t) c h w -> n t c h w",n=n)
            object_motion = self.object_motion_encoder(refimg_and_video,mask_ratio) # (n,t+t,l,d)
            camera_motion = self.camera_motion_encoder(refimg_and_video,mask_ratio) # (n,t+t,l,d)

        # object motion encoder
        object_source_motion = object_motion[:,:t].flatten(0,1) # (NT,motion_token,d)
        object_target_motion = object_motion[:,t:].flatten(0,1) # (NT,motion_token,d)
        assert object_source_motion.shape == object_target_motion.shape , f'object_source_motion.shape {object_source_motion.shape} != object_target_motion.shape {object_target_motion.shape}'

        # camera motion encoder
        camera_source_motion = camera_motion[:,:t].flatten(0,1) # (NT,motion_token,d)
        camera_target_motion = camera_motion[:,t:].flatten(0,1) # (NT,motion_token,d)
        assert camera_source_motion.shape == camera_target_motion.shape , f'camera_source_motion.shape {camera_source_motion.shape} != camera_target_motion.shape {camera_target_motion.shape}'

        # KL Loss
        if self.use_regularizers:
            object_target_motion = einops.rearrange(object_target_motion, "n t d -> n d t")
            camera_target_motion = einops.rearrange(camera_target_motion, "n t d -> n d t")
            object_target_motion, object_target_KLloss = self.regularization(object_target_motion)
            camera_target_motion, camera_target_KLloss = self.regularization(camera_target_motion)
            object_target_motion = einops.rearrange(object_target_motion, "n d t -> n t d")
            camera_target_motion = einops.rearrange(camera_target_motion, "n d t -> n t d")


        # motion fusion
        if self.use_regularizers:
            camera_source_motion = self.camera_source_motion_map(camera_source_motion)
            camera_target_motion = self.camera_target_motion_map(camera_target_motion)

            object_source_motion = self.object_source_motion_map(object_source_motion)
            object_target_motion = self.object_target_motion_map(object_target_motion)
        else:
            if self.camera_motion_token_channel != self.motion_token_channel:
                camera_source_motion = self.camera_motion_map(camera_source_motion)
                camera_target_motion = self.camera_motion_map(camera_target_motion)
            if self.object_motion_token_channel != self.motion_token_channel:
                object_source_motion = self.object_motion_map(object_source_motion)
                object_target_motion = self.object_motion_map(object_target_motion)

        # source_motion = object_source_motion + camera_source_motion # (NT,motion_token,d)
        # target_motion = object_target_motion + camera_target_motion # (NT,motion_token,d)

        # motion transformer
        if self.need_motion_transformer:
            target_motion = einops.rearrange(target_motion,'(n f) l d -> n f l d',n=n)
            target_motion = self.motion_transformer(target_motion)
            target_motion = einops.rearrange(target_motion,'n f l d -> (n f) l d',n=n)

        
        # prepare for Diffusion Transformer
        zi = ref_img.flatten(0,1) # (NT,C,H,W)
        # zi = ref_img_grey.flatten(0,1) # (NT,C,H,W)

        zj = gt_video.flatten(0,1) # (NT,C,H,W)
        if time_step is None:
            time_step = self.prepare_timestep(batch_size= zj.shape[0],device= device) #(b,)
        if self.diffusion_model_type != 'default':
            time_step = self.prepare_timestep(batch_size= n,device= device) # (n,)
            time_step = time_step.repeat_interleave(t) # (b,)
        zt,vel = self.scheduler.get_train_tuple(z1=zj,time_step=time_step)  # (NT,C,H,W),(NT,C,H,W)
        
        # dit forward
        image_hidden_states = torch.cat((zi,zt),dim=1) # (b,2C,H,W)
        pre = self.diffusion_transformer(camera_motion_source_hidden_states = camera_source_motion,
                                        camera_motion_target_hidden_states = camera_target_motion,
                                        object_motion_source_hidden_states = object_source_motion,
                                        object_motion_target_hidden_states = object_target_motion,
                                        image_hidden_states = image_hidden_states,
                                        timestep = time_step,)

        # loss
        diff_loss = l2(pre,vel)

        rec_zj = self.scheduler.get_target_with_zt_vel(zt,pre,time_step)
        rec_loss = l2(rec_zj,zj)

        
        if self.use_regularizers:
            KLloss = self.klloss_weight * (object_target_KLloss + camera_target_KLloss)/2
            loss = diff_loss + KLloss
            loss_dict = {'loss':loss,'diff_loss':diff_loss,'rec_loss':rec_loss,'KLloss':KLloss}
        else:
            loss = diff_loss
            loss_dict = {'loss':loss,'diff_loss':diff_loss,'rec_loss':rec_loss}
    
        if return_meta_info:
            return {'camera_motion' : camera_motion,               # (,t,motion_out_channels,h,w) , output of camera motion encoder 
                    'object_motion' : object_motion,               # (,t,motion_out_channels,h,w) , output of object motion encoder 
                    'zi' : zi,                       # (b,C,H,W) | b = n * t
                    'zj' : zj,                       # (b,C,H,W)
                    'zt' : zt,                       # (b,C,H,W)
                    'pre': pre,                      # (b,C,H,W)
                    'time_step': time_step,          # (b,)
                    }
        else:
            return pre,vel,loss_dict  # (b,C,H,W)

    @torch.no_grad()
    def sample(self,video:torch.Tensor,ref_img:torch.Tensor,video_grey=None,ref_img_grey=None,camera_mask=None,sample_step:int = 50,mask_ratio = None,start_step:int = None,return_meta_info=False):

        device = video.device
        n,t,c,h,w = video.shape

        if start_step is None:
            start_step = self.scheduler.num_step
        assert start_step <= self.scheduler.num_step , 'start_step cant be larger than scheduler.num_step'

        # motion encoder
        if mask_ratio is not None:
            print(f'* Sampling with Mask_Ratio = {mask_ratio}')
            mask_ratio =  mask_ratio
        
        refimg_and_video = torch.cat([ref_img,video],dim=1)# (n,t+t,C,H,W) (1,32,4,32,32)
        # if not self.use_grey:
        #     ref_img_grey = torch.zeros_like
        # refimg_and_video_grey = torch.cat([ref_img_grey,video_grey],dim=1)# (n,t+t,C,H,W) (1,32,4,32,32)

        if self.use_filter:
            freq_filter = gaussian_low_pass_filter([2*t, h, w],self.filter_num,self.filter_num)
            freq_filter = freq_filter.unsqueeze(0).unsqueeze(0).repeat(n, c, 1, 1, 1)
            freq_filter = freq_filter.to(device=refimg_and_video.device)

            high_freq_filter = gaussian_low_pass_filter([2*t, h, w],self.high_filter_num,self.high_filter_num)
            high_freq_filter = high_freq_filter.unsqueeze(0).unsqueeze(0).repeat(n, c, 1, 1, 1)
            high_freq_filter = high_freq_filter.to(device=refimg_and_video.device)

            if self.use_grey:
                refimg_and_video_grey = torch.cat([ref_img_grey,video_grey],dim=1)# (n,t+t,C,H,W) (1,32,4,32,32)
                refimg_and_video_grey = einops.rearrange(refimg_and_video_grey, "n t c h w -> n c t h w")
                LF_refimg_and_video, _ = freq_3d_filter(refimg_and_video_grey, freq_filter)
                _,HF_refimg_and_video = freq_3d_filter(refimg_and_video_grey, high_freq_filter)

            else:
                refimg_and_video = einops.rearrange(refimg_and_video, "n t c h w -> n c t h w")
                LF_refimg_and_video, _ = freq_3d_filter(refimg_and_video_grey, freq_filter)
                _,HF_refimg_and_video = freq_3d_filter(refimg_and_video_grey, high_freq_filter)

            LF_refimg_and_video = einops.rearrange(LF_refimg_and_video, "n c t h w -> n t c h w")
            HF_refimg_and_video = einops.rearrange(HF_refimg_and_video, "n c t h w -> n t c h w")

            # if self.use_mask:
            #     LF_refimg_and_video = LF_refimg_and_video * camera_mask

            if self.use_camera_down:
                LF_refimg_and_video = einops.rearrange(LF_refimg_and_video, "n t c h w -> (n t) c h w")
                LF_refimg_and_video = self.camera_down(LF_refimg_and_video)
                LF_refimg_and_video = einops.rearrange(LF_refimg_and_video, "(n t) c h w -> n t c h w",n=n)

            object_motion = self.object_motion_encoder(HF_refimg_and_video,mask_ratio) # (n,t+t,l,d)
            camera_motion = self.camera_motion_encoder(LF_refimg_and_video,mask_ratio) # (n,t+t,l,d)

        else:
            if self.use_camera_down:
                LF_refimg_and_video = einops.rearrange(LF_refimg_and_video, "n t c h w -> (n t) c h w")
                LF_refimg_and_video = self.camera_down(LF_refimg_and_video)
                LF_refimg_and_video = einops.rearrange(LF_refimg_and_video, "(n t) c h w -> n t c h w",n=n)
            object_motion = self.object_motion_encoder(refimg_and_video,mask_ratio) # (n,t+t,l,d)
            camera_motion = self.camera_motion_encoder(refimg_and_video,mask_ratio) # (n,t+t,l,d)

        # object motion encoder
        object_source_motion = object_motion[:,:t].flatten(0,1) # (NT,motion_token,d)
        object_target_motion = object_motion[:,t:].flatten(0,1) # (NT,motion_token,d)
        assert object_source_motion.shape == object_target_motion.shape , f'object_source_motion.shape {object_source_motion.shape} != object_target_motion.shape {object_target_motion.shape}'

        # camera motion encoder
        camera_source_motion = camera_motion[:,:t].flatten(0,1) # (NT,motion_token,d)
        camera_target_motion = camera_motion[:,t:].flatten(0,1) # (NT,motion_token,d)
        assert camera_source_motion.shape == camera_target_motion.shape , f'camera_source_motion.shape {camera_source_motion.shape} != camera_target_motion.shape {camera_target_motion.shape}'

         # KL Loss
        if self.use_regularizers:
            object_target_motion = einops.rearrange(object_target_motion, "n t d -> n d t")
            camera_target_motion = einops.rearrange(camera_target_motion, "n t d -> n d t")
            object_target_motion, object_target_KLloss = self.regularization(object_target_motion)
            camera_target_motion, camera_target_KLloss = self.regularization(camera_target_motion)
            object_target_motion = einops.rearrange(object_target_motion, "n d t -> n t d")
            camera_target_motion = einops.rearrange(camera_target_motion, "n d t -> n t d")

        # motion fusion
        # camera motion average
        if self.use_regularizers:
            camera_source_motion = self.camera_source_motion_map(camera_source_motion)
            camera_target_motion = self.camera_target_motion_map(camera_target_motion)

            object_source_motion = self.object_source_motion_map(object_source_motion)
            object_target_motion = self.object_target_motion_map(object_target_motion)
        else:
            if self.camera_motion_token_channel != self.motion_token_channel:
                camera_source_motion = self.camera_motion_map(camera_source_motion)
                camera_target_motion = self.camera_motion_map(camera_target_motion)
            if self.object_motion_token_channel != self.motion_token_channel:
                object_source_motion = self.object_motion_map(object_source_motion)
                object_target_motion = self.object_motion_map(object_target_motion)

        # if self.use_motion == 'camera':
        #     source_motion = camera_source_motion # (NT,motion_token,d)
        #     target_motion = camera_target_motion # (NT,motion_token,d)
        # elif self.use_motion == 'object':
        #     source_motion = object_source_motion # (NT,motion_token,d)
        #     target_motion = object_target_motion # (NT,motion_token,d)
        # else:
        #     source_motion = object_source_motion + camera_source_motion # (NT,motion_token,d)
        #     target_motion = object_target_motion + camera_target_motion # (NT,motion_token,d)

        # motion transformer
        if self.need_motion_transformer:
            target_motion = einops.rearrange(target_motion,'(n f) l d -> n f l d',n=n)
            target_motion = self.motion_transformer(target_motion)
            target_motion = einops.rearrange(target_motion,'n f l d -> (n f) l d',n=n)
        
        # prepare for Diffusion Transformer
        time_step = torch.ones((camera_source_motion.shape[0],)).to(device) 
        time_step = time_step * start_step

        zi = ref_img.flatten(0,1) # (NT,C,H,W)
        zj = video.flatten(0,1) # (NT,C,H,W)
        # zi = ref_img_grey.flatten(0,1) # (NT,C,H,W)
        # zj = video_grey.flatten(0,1) # (NT,C,H,W)

        zt,vel = self.scheduler.get_train_tuple(z1=zj,time_step=time_step)  # (NT,C,H,W),(NT,C,H,W)
        
        # Sample Loop
        pre_cache = []
        sample_cache = []
        
        # 1.step_seq
        step_seq = np.linspace(0, start_step, num=sample_step+1, endpoint=True,dtype=int) # [0,5,10,15,....,start_step]
        step_seq = list(reversed(step_seq[1:])) # delete step:0  [start_step,.....,15,10,5]

        # 2.Euler step
        dt = 1./sample_step

        for i in tqdm(step_seq):
            # time_step
            time_step = torch.ones((zt.shape[0],)).to(zt.device)  
            time_step = time_step * i
            
            # input
            zt = zt.to(video.dtype)
            image_hidden_states = torch.cat((zi,zt),dim=1) # (b,2C,H,W)
            
            # forward
            pre = self.diffusion_transformer(camera_motion_source_hidden_states = camera_source_motion,
                                            camera_motion_target_hidden_states = camera_target_motion,
                                            object_motion_source_hidden_states = object_source_motion,
                                            object_motion_target_hidden_states = object_target_motion,
                                            image_hidden_states = image_hidden_states,
                                            timestep = time_step,) 
            zt = zt + pre * dt
            pre_cache.append(pre)
            sample_cache.append(zt)
        
        zi = einops.rearrange(zi,'(n t) c h w -> n t c h w',n=n)
        zt = einops.rearrange(zt,'(n t) c h w -> n t c h w',n=n)
        zj = einops.rearrange(zj,'(n t) c h w -> n t c h w',n=n)
        
        if return_meta_info:
            return {'zi' : zi, # (b,1,c,h,w)
                    'zj' : zj,             # (b,1,c,h,w)
                    'sample' : zt,           # (b,1,c,h,w)
                    'pre_cache' : pre_cache,           # [(b,c,h,w),....]
                    'sample_cache' : sample_cache,   # [(b,c,h,w),....]
                    'step_seq' : step_seq,
                    'motion' : target_motion, # (b,C,H,W)
                    }
        else:
            return zi,zt,zj # (n,t,c,h,w)
        

    @torch.no_grad()
    def sample_diff_motion(self,video:torch.Tensor,ref_img:torch.Tensor,video_grey=None,ref_img_grey=None,camera_video_grey=None,sample_step:int = 50,mask_ratio = None,start_step:int = None,return_meta_info=False):

        device = video.device
        n,t,c,h,w = video.shape

        if start_step is None:
            start_step = self.scheduler.num_step
        assert start_step <= self.scheduler.num_step , 'start_step cant be larger than scheduler.num_step'

        # motion encoder
        if mask_ratio is not None:
            print(f'* Sampling with Mask_Ratio = {mask_ratio}')
            mask_ratio =  mask_ratio
        
        refimg_and_video = torch.cat([ref_img,video],dim=1)# (n,t+t,C,H,W) (1,32,4,32,32)
        
        # if not self.use_grey:
        #     ref_img_grey = torch.zeros_like
        # refimg_and_video_grey = torch.cat([ref_img_grey,video_grey],dim=1)# (n,t+t,C,H,W) (1,32,4,32,32)

        if self.use_filter:
            freq_filter = gaussian_low_pass_filter([2*t, h, w],d_s=0.4,d_t=0.4)
            freq_filter = freq_filter.unsqueeze(0).unsqueeze(0).repeat(n, c, 1, 1, 1)
            freq_filter = freq_filter.to(device=refimg_and_video.device)

            if self.use_grey:
                refimg_and_video_grey = torch.cat([ref_img_grey,video_grey],dim=1)# (n,t+t,C,H,W) (1,32,4,32,32)
                refimg_and_video_grey = einops.rearrange(refimg_and_video_grey, "n t c h w -> n c t h w")
                refimg_and_camera_video_grey = torch.cat([ref_img,camera_video_grey],dim=1)# (n,t+t,C,H,W) (1,32,4,32,32)
                refimg_and_camera_video_grey = einops.rearrange(refimg_and_camera_video_grey, "n t c h w -> n c t h w")

                _, HF_refimg_and_video = freq_3d_filter(refimg_and_video_grey, freq_filter)
                LF_refimg_and_video, _ = freq_3d_filter(refimg_and_camera_video_grey, freq_filter)


            else:
                refimg_and_video = einops.rearrange(refimg_and_video, "n t c h w -> n c t h w")
                LF_refimg_and_video, HF_refimg_and_video = freq_3d_filter(refimg_and_video, freq_filter)

            LF_refimg_and_video = einops.rearrange(LF_refimg_and_video, "n c t h w -> n t c h w")
            HF_refimg_and_video = einops.rearrange(HF_refimg_and_video, "n c t h w -> n t c h w")

            if self.use_camera_down:
                LF_refimg_and_video = einops.rearrange(LF_refimg_and_video, "n t c h w -> (n t) c h w")
                LF_refimg_and_video = self.camera_down(LF_refimg_and_video)
                LF_refimg_and_video = einops.rearrange(LF_refimg_and_video, "(n t) c h w -> n t c h w",n=n)

            object_motion = self.object_motion_encoder(HF_refimg_and_video,mask_ratio) # (n,t+t,l,d)
            camera_motion = self.camera_motion_encoder(LF_refimg_and_video,mask_ratio) # (n,t+t,l,d)

        else:
            if self.use_camera_down:
                LF_refimg_and_video = einops.rearrange(LF_refimg_and_video, "n t c h w -> (n t) c h w")
                LF_refimg_and_video = self.camera_down(LF_refimg_and_video)
                LF_refimg_and_video = einops.rearrange(LF_refimg_and_video, "(n t) c h w -> n t c h w",n=n)
            object_motion = self.object_motion_encoder(refimg_and_video,mask_ratio) # (n,t+t,l,d)
            camera_motion = self.camera_motion_encoder(refimg_and_video,mask_ratio) # (n,t+t,l,d)

        # object motion encoder
        object_source_motion = object_motion[:,:t].flatten(0,1) # (NT,motion_token,d)
        object_target_motion = object_motion[:,t:].flatten(0,1) # (NT,motion_token,d)
        assert object_source_motion.shape == object_target_motion.shape , f'object_source_motion.shape {object_source_motion.shape} != object_target_motion.shape {object_target_motion.shape}'

        # camera motion encoder
        camera_source_motion = camera_motion[:,:t].flatten(0,1) # (NT,motion_token,d)
        camera_target_motion = camera_motion[:,t:].flatten(0,1) # (NT,motion_token,d)
        assert camera_source_motion.shape == camera_target_motion.shape , f'camera_source_motion.shape {camera_source_motion.shape} != camera_target_motion.shape {camera_target_motion.shape}'

         # KL Loss
        if self.use_regularizers:
            object_target_motion = einops.rearrange(object_target_motion, "n t d -> n d t")
            camera_target_motion = einops.rearrange(camera_target_motion, "n t d -> n d t")
            object_target_motion, object_target_KLloss = self.regularization(object_target_motion)
            camera_target_motion, camera_target_KLloss = self.regularization(camera_target_motion)
            object_target_motion = einops.rearrange(object_target_motion, "n d t -> n t d")
            camera_target_motion = einops.rearrange(camera_target_motion, "n d t -> n t d")

        # motion fusion
        # camera motion average
        if self.use_regularizers:
            camera_source_motion = self.camera_source_motion_map(camera_source_motion)
            camera_target_motion = self.camera_target_motion_map(camera_target_motion)

            object_source_motion = self.object_source_motion_map(object_source_motion)
            object_target_motion = self.object_target_motion_map(object_target_motion)
        else:
            if self.camera_motion_token_channel != self.motion_token_channel:
                camera_source_motion = self.camera_motion_map(camera_source_motion)
                camera_target_motion = self.camera_motion_map(camera_target_motion)
            if self.object_motion_token_channel != self.motion_token_channel:
                object_source_motion = self.object_motion_map(object_source_motion)
                object_target_motion = self.object_motion_map(object_target_motion)

        if self.use_motion == 'camera':
            source_motion = camera_source_motion # (NT,motion_token,d)
            target_motion = camera_target_motion # (NT,motion_token,d)
        elif self.use_motion == 'object':
            source_motion = object_source_motion # (NT,motion_token,d)
            target_motion = object_target_motion # (NT,motion_token,d)
        else:
            source_motion = object_source_motion + camera_source_motion # (NT,motion_token,d)
            target_motion = object_target_motion + camera_target_motion # (NT,motion_token,d)

        # motion transformer
        if self.need_motion_transformer:
            target_motion = einops.rearrange(target_motion,'(n f) l d -> n f l d',n=n)
            target_motion = self.motion_transformer(target_motion)
            target_motion = einops.rearrange(target_motion,'n f l d -> (n f) l d',n=n)
        
        # prepare for Diffusion Transformer
        time_step = torch.ones((source_motion.shape[0],)).to(device) 
        time_step = time_step * start_step

        zi = ref_img.flatten(0,1) # (NT,C,H,W)
        zj = video.flatten(0,1) # (NT,C,H,W)
        zt,vel = self.scheduler.get_train_tuple(z1=zj,time_step=time_step)  # (NT,C,H,W),(NT,C,H,W)
        
        # Sample Loop
        pre_cache = []
        sample_cache = []
        
        # 1.step_seq
        step_seq = np.linspace(0, start_step, num=sample_step+1, endpoint=True,dtype=int) # [0,5,10,15,....,start_step]
        step_seq = list(reversed(step_seq[1:])) # delete step:0  [start_step,.....,15,10,5]

        # 2.Euler step
        dt = 1./sample_step

        for i in tqdm(step_seq):
            # time_step
            time_step = torch.ones((zt.shape[0],)).to(zt.device)  
            time_step = time_step * i
            
            # input
            zt = zt.to(video.dtype)
            image_hidden_states = torch.cat((zi,zt),dim=1) # (b,2C,H,W)
            
            # forward
            pre = self.diffusion_transformer(motion_source_hidden_states = source_motion,
                                            motion_target_hidden_states = target_motion,
                                            image_hidden_states = image_hidden_states,
                                            timestep = time_step,) 
            zt = zt + pre * dt
            pre_cache.append(pre)
            sample_cache.append(zt)
        
        zi = einops.rearrange(zi,'(n t) c h w -> n t c h w',n=n)
        zt = einops.rearrange(zt,'(n t) c h w -> n t c h w',n=n)
        zj = einops.rearrange(zj,'(n t) c h w -> n t c h w',n=n)
        
        if return_meta_info:
            return {'zi' : zi, # (b,1,c,h,w)
                    'zj' : zj,             # (b,1,c,h,w)
                    'sample' : zt,           # (b,1,c,h,w)
                    'pre_cache' : pre_cache,           # [(b,c,h,w),....]
                    'sample_cache' : sample_cache,   # [(b,c,h,w),....]
                    'step_seq' : step_seq,
                    'motion' : target_motion, # (b,C,H,W)
                    }
        else:
            return zi,zt,zj # (n,t,c,h,w)

    def sample_with_refimg_motion(self,ref_img:torch.Tensor,motion=torch.Tensor,sample_step:int = 10,return_meta_info=False):
        """
        Args:
            ref_img : (N,C,H,W)                             
            motion : (N,F,L,D)                           
        Return:
            video : (N,T,C,H,W)                             
        """
        device = motion.device
        n,t,l,d = motion.shape

        start_step = self.scheduler.num_step
        
        # motion encoder
        refimg = ref_img.unsqueeze(1) # (N,1,C,H,W)
        source_motion = self.motion_encoder(refimg) # (n,1,motion_token,d)

        source_motion = source_motion.repeat(1,t,1,1).flatten(0,1) # (NT,l,d)
        target_motion = motion.flatten(0,1) # (NT,l,d)

        assert source_motion.shape == target_motion.shape , f'source_motion.shape {source_motion.shape} != target_motion.shape {target_motion.shape}'

        # motion transformer
        if self.need_motion_transformer and not self.extract_motion_with_motion_transformer:
            target_motion = einops.rearrange(target_motion,'(n f) l d -> n f l d',n=n)
            target_motion = self.motion_transformer(target_motion)
            target_motion = einops.rearrange(target_motion,'n f l d -> (n f) l d',n=n)
        
        # prepare for Diffusion Transformer
        time_step = torch.ones((source_motion.shape[0],)).to(device) 
        time_step = time_step * start_step

        zi = refimg.repeat(1,t,1,1,1).flatten(0,1) # (NT,C,H,W)
        zj = zi
        zt,vel = self.scheduler.get_train_tuple(z1=zj,time_step=time_step)  # (NT,C,H,W),(NT,C,H,W)
        
        # Sample Loop
        pre_cache = []
        sample_cache = []
        
        # 1.step_seq
        step_seq = np.linspace(0, start_step, num=sample_step+1, endpoint=True,dtype=int) # [0,5,10,15,....,start_step]
        step_seq = list(reversed(step_seq[1:])) # delete step:0  [start_step,.....,15,10,5]

        # 2.Euler step
        dt = 1./sample_step

        for i in tqdm(step_seq):
            # time_step
            time_step = torch.ones((zt.shape[0],)).to(zt.device)  
            time_step = time_step * i
            
            # input
            zt = zt.to(ref_img.dtype)
            image_hidden_states = torch.cat((zi,zt),dim=1) # (b,2C,H,W)
            
            # forward
            pre = self.diffusion_transformer(motion_source_hidden_states = source_motion,
                                            motion_target_hidden_states = target_motion,
                                            image_hidden_states = image_hidden_states,
                                            timestep = time_step,) 
            zt = zt + pre * dt
            pre_cache.append(pre)
            sample_cache.append(zt)
        
        # unsqueeze (n,1,c,h,w) means images, (n,t,c,h,w) means video t>1 .
        zi = einops.rearrange(zi,'(n t) c h w -> n t c h w',n=n,t=t)
        zt = einops.rearrange(zt,'(n t) c h w -> n t c h w',n=n,t=t)
        
        if return_meta_info:
            return {'zi' : zi, # (b,1,c,h,w)
                    'zj' : zj,             # (b,1,c,h,w)
                    'sample' : zt,           # (b,1,c,h,w)
                    'pre_cache' : pre_cache,           # [(b,c,h,w),....]
                    'sample_cache' : sample_cache,   # [(b,c,h,w),....]
                    'step_seq' : step_seq,
                    'motion' : target_motion, # (b,C,H,W)
                    }
        else:
            return zi,zt,zj # (b,1,c,h,w)
    
    def extract_motion(self,video:torch.tensor):
        # video : (N,T,C,H,W)
        n,t,c,h,w = video.shape
        
        motion = self.motion_encoder(video) # (N,T,L,D)

        if self.need_motion_transformer and self.extract_motion_with_motion_transformer:
            motion = self.motion_transformer(motion) # (N,T,L,D)

        return motion
     
    def prepare_timestep(self,batch_size:int,device,time_step = None):
        if time_step is not None:
            return time_step.to(device)
        else:
            return torch.randint(0,self.num_step+1,(batch_size,)).to(device)
  
    def prepare_encoder_input(self,video:torch.tensor):
        assert len(video.shape) == 5 , f'only support video data : 5D tensor , but got {video.shape}'
        
        # cat
        pre = video[:,:-1,:,:,:] 
        post= video[:,1:,:,:,:]
        duo_frame_mix = torch.cat([pre,post],dim=2)    # (b,t-1,2c,h,w)
        duo_frame_mix = einops.rearrange(duo_frame_mix,'b t c h w -> (b t) c h w')
        
        return duo_frame_mix # (b*f-1,2c,h,w)


    def unpatchify(self, x ,patch_size):
        """
        x: (N, S, patch_size**2 *C)
        imgs: (N, C, H, W)
        """
        p = patch_size
        h = w = int(x.shape[1]**.5)
        # c = self.in_chans
        c = x.shape[2] // (p**2)
        assert h * w == x.shape[1]
        
        x = x.reshape(shape=(x.shape[0], h, w, p, p, c)) # (N, h, w, p, p, c)
        x = torch.einsum('nhwpqc->nchpwq', x) # (N, c, h, p, w, p)
        imgs = x.reshape(shape=(x.shape[0], c, h * p, h * p))
        return imgs #(N,C,H,W)    

    def reset_infer_num_frame(self, num:int):
        old_num = self.diffusion_transformer.target_frame
        self.diffusion_transformer.target_frame = num
        if self.use_motiontemporal:
            self.camera_motion_encoder.video_frames = num
            self.object_motion_encoder.video_frames = num
        print(f'* Reset infer frame from {old_num} to {self.diffusion_transformer.target_frame} *')

class AMDModel_Camera(ModelMixin, ConfigMixin):
    _supports_gradient_checkpointing = True
    
    @register_to_config
    def __init__(self,
                 image_inchannel :int  = 4,
                 image_height  :int = 32,
                 image_width  :int  = 32,
                 video_frames  :int = 16,
                 scheduler_num_step :int = 1000,
                 use_filter:bool = False,
                 filter_num:float = 0.4,
                 use_grey:bool = False,
                 use_camera_down:bool = False,
                 use_regularizers:bool = False,
                 use_motiontemporal:bool = True,
                 klloss_weight: float = 0.005, # 比diffloss小一个量级
                 use_mask:bool = True,
                 motion_type:str='plus',
                 
                 # 理论上来说,高频信号在transformer中的深层,低频信号在transformer的浅层,所以不同支路的layer_num也需要做实验
                 # -----------Object MotionEncoder -----------
                 object_motion_token_num:int = 12,
                 object_motion_token_channel: int = 128,
                 object_enc_num_layers:int = 8,
                 enc_nhead:int = 8,
                 enc_ndim:int = 64,
                 enc_dropout:float = 0.0,
                 motion_need_norm_out:bool = False,

                 # -----------Camera MotionEncoder -----------
                 camera_motion_token_num:int = 12,
                 camera_motion_token_channel: int = 128,
                 camera_enc_num_layers:int = 8,
                
                 # -----------MotionTransformer ---------
                 motion_token_num:int = 12,
                 motion_token_channel: int = 128, 
                 need_motion_transformer :bool = False,
                 motion_transformer_attn_head_dim:int = 64,
                 motion_transformer_attn_num_heads:int = 16,
                 motion_transformer_num_layers:int = 4, 
                
                 # ----------- Diffusion Transformer -----------
                 diffusion_model_type : str = 'default', # or dual
                 diffusion_attn_head_dim : int  = 64,
                 diffusion_attn_num_heads : int = 16,
                 diffusion_out_channels : int = 4,
                 diffusion_num_layers : int = 16,
                 image_patch_size : int  = 2,
                 motion_patch_size : int = 1,
                 motion_drop_ratio: float = 0.0,

                 # ----------- Sample --------------
                 extract_motion_with_motion_transformer = False,
                 **kwargs,
                 ):
        super().__init__()

        # setting
        self.num_step = scheduler_num_step
        self.scheduler = RectifiedFlow(num_steps=scheduler_num_step)
        self.need_motion_transformer = need_motion_transformer
        self.extract_motion_with_motion_transformer = extract_motion_with_motion_transformer
        self.diffusion_model_type = diffusion_model_type
        self.target_frame = video_frames
        self.use_motiontemporal = use_motiontemporal

        self.camera_motion_token_channel = camera_motion_token_channel
        self.object_motion_token_channel = object_motion_token_channel
        self.motion_token_channel = motion_token_channel

        # self.use_motion = use_motion
        self.filter_num = filter_num

        # frepuency filter
        self.use_filter = use_filter 
        self.use_grey = use_grey
        self.use_camera_down = use_camera_down
        self.use_mask = use_mask

        # use regularizers:
        self.use_regularizers = use_regularizers
        self.klloss_weight = klloss_weight
        # if use_regularizers:
        #     self.regularization = DiagonalGaussianRegularizer(sample=True)

        #     self.camera_target_motion_map = nn.Linear(camera_motion_token_channel//2, motion_token_channel)
        #     self.camera_source_motion_map = nn.Linear(camera_motion_token_channel, motion_token_channel)
        #     # self.object_target_motion_map = nn.Linear(object_motion_token_channel//2, motion_token_channel)
        #     # self.object_source_motion_map = nn.Linear(object_motion_token_channel, motion_token_channel)
        # else:

        if camera_motion_token_channel != motion_token_channel:
            self.camera_motion_map = nn.Linear(camera_motion_token_channel, motion_token_channel)
            # if object_motion_token_channel != motion_token_channel:
            #     self.object_motion_map = nn.Linear(object_motion_token_channel, motion_token_channel)

        if self.use_motiontemporal:
            # camera motion Encoder 
            self.camera_motion_encoder = MotionEncoderLearnTokenOnlyTemporalTransformer(img_height = image_height,
                                                                    img_width=image_width,
                                                                    img_inchannel=image_inchannel,
                                                                    img_patch_size = image_patch_size,
                                                                    motion_token_num =  camera_motion_token_num,
                                                                    motion_channel = camera_motion_token_channel,
                                                                    need_norm_out = motion_need_norm_out,
                                                                    video_frames=self.target_frame,
                                                                    # ----- attention
                                                                    num_attention_heads=enc_nhead,
                                                                    attention_head_dim=enc_ndim // 4, # 原始dim为8*128, camera降低dim为8*32
                                                                    num_layers=camera_enc_num_layers,      
                                                                    dropout=enc_dropout,
                                                                    attention_bias= True,)
        else:         
            # camera motion Encoder 
            self.camera_motion_encoder = MotionEncoderLearnTokenTransformer(img_height = image_height, # down_camera: image_height // 4
                                                                    img_width=image_width, # down_camera: image_height // 4
                                                                    img_inchannel=image_inchannel,
                                                                    img_patch_size = image_patch_size,
                                                                    motion_token_num =  camera_motion_token_num,
                                                                    motion_channel = camera_motion_token_channel,
                                                                    need_norm_out = motion_need_norm_out,
                                                                    # ----- attention
                                                                    num_attention_heads=enc_nhead,
                                                                    attention_head_dim=enc_ndim, # 原始dim为8*128, camera降低dim为8*32
                                                                    num_layers=camera_enc_num_layers,      
                                                                    dropout=enc_dropout,
                                                                    attention_bias= True,)
        if self.use_camera_down:
            self.camera_down = nn.Sequential(
                nn.Conv2d(in_channels=4, out_channels=16, kernel_size=3, stride=2, padding=1),
                nn.Conv2d(in_channels=16, out_channels=4, kernel_size=3, stride=1, padding=1),
                nn.MaxPool2d(kernel_size=2, stride=2, padding=0)
            )
        
        # motion transformer
        if need_motion_transformer:
            self.motion_transformer = MotionTransformer(motion_token_num=motion_token_num,
                                                        motion_token_channel=motion_token_channel,
                                                        attention_head_dim=motion_transformer_attn_head_dim,
                                                        num_attention_heads=motion_transformer_attn_num_heads,
                                                        num_layers=motion_transformer_num_layers,)
        
        # diffusion transformer
        dit_image_inchannel = image_inchannel * 2 # zi + zt
        if diffusion_model_type == 'default':
            self.diffusion_transformer = AMDDiffusionTransformerModel(num_attention_heads= diffusion_attn_num_heads,
                                                                    attention_head_dim= diffusion_attn_head_dim,
                                                                    out_channels = diffusion_out_channels,
                                                                    num_layers= diffusion_num_layers,
                                                                    motion_type=motion_type, # 'decouple','plus'
                                                                    # ----- img
                                                                    image_width= image_width,
                                                                    image_height= image_height,
                                                                    image_patch_size= image_patch_size,
                                                                    image_in_channels = dit_image_inchannel, 
                                                                    # ----- motion
                                                                    motion_token_num = motion_token_num,
                                                                    motion_in_channels = motion_token_channel,)
        elif diffusion_model_type == 'dual':
            self.diffusion_transformer = AMDDiffusionTransformerModelDualStream(num_attention_heads= diffusion_attn_num_heads,
                                                                attention_head_dim= diffusion_attn_head_dim,
                                                                out_channels = diffusion_out_channels,
                                                                num_layers= diffusion_num_layers,
                                                                # ----- img
                                                                image_width= image_width,
                                                                image_height= image_height,
                                                                image_patch_size= image_patch_size,
                                                                image_in_channels = dit_image_inchannel, 
                                                                # ----- motion
                                                                motion_token_num = motion_token_num,
                                                                motion_in_channels = motion_token_channel,
                                                                motion_target_num_frame = self.target_frame)
        elif diffusion_model_type == 'spatial':
            self.diffusion_transformer = AMDDiffusionTransformerModelImgSpatial(num_attention_heads= diffusion_attn_num_heads,
                                                                attention_head_dim= diffusion_attn_head_dim,
                                                                out_channels = diffusion_out_channels,
                                                                num_layers= diffusion_num_layers,
                                                                motion_type = motion_type,
                                                                # ----- img
                                                                image_width= image_width,
                                                                image_height= image_height,
                                                                image_patch_size= image_patch_size,
                                                                image_in_channels = dit_image_inchannel, 
                                                                # ----- motion
                                                                motion_token_num = motion_token_num,
                                                                motion_in_channels = motion_token_channel,
                                                                motion_target_num_frame = self.target_frame)
        else:
            raise IndexError

    def forward(self,video:torch.Tensor,ref_img:torch.Tensor,video_grey=None,ref_img_grey=None,camera_mask=None,time_step:torch.tensor = None,return_meta_info=False,mask_ratio=None):
        """
        Args:
            video: (N,T,C,H,W)
            ref_img: (N,T,C,H,W)
        """

        device = video.device
        n,t,c,h,w = video.shape

        assert video.shape == ref_img.shape ,f'video.shape:{video.shape}should be equal to ref_img.shape:{ref_img.shape}'
        
        # motion encoder
        if mask_ratio is not None:
            mask_ratio = torch.rand(1).item() * mask_ratio

        # gt_video = video
        # if self.mask_video_ratio != 0:
                
        #     b_z, f_z, c_z, h_z, w_z = video.shape
        #     rand_mask = torch.rand(h_z, w_z).to(device=video.device)
        #     mask = rand_mask > self.mask_video_ratio
        #     mask = mask.unsqueeze(0).unsqueeze(0).unsqueeze(0)  
        #     mask = mask.expand(b_z, f_z, c_z, h_z, w_z) 
        #     video = video * mask

        #     if self.use_grey:
        #         video_grey = video_grey * mask
        
        refimg_and_video = torch.cat([ref_img,video],dim=1)# (n,t+t,C,H,W)
        

        # frequency filter
        if self.use_filter:
            freq_filter = gaussian_low_pass_filter([2*t, h, w],self.filter_num,self.filter_num)
            freq_filter = freq_filter.unsqueeze(0).unsqueeze(0).repeat(n, c, 1, 1, 1)
            freq_filter = freq_filter.to(device=refimg_and_video.device)

            if self.use_grey:
                refimg_and_video_grey = torch.cat([ref_img_grey,video_grey],dim=1)# (n,t+t,C,H,W) (1,32,4,32,32)
                refimg_and_video_grey = einops.rearrange(refimg_and_video_grey, "n t c h w -> n c t h w")
                LF_refimg_and_video, HF_refimg_and_video = freq_3d_filter(refimg_and_video_grey, freq_filter)
            else:
                refimg_and_video = einops.rearrange(refimg_and_video, "n t c h w -> n c t h w")
                LF_refimg_and_video, HF_refimg_and_video = freq_3d_filter(refimg_and_video, freq_filter)

            LF_refimg_and_video = einops.rearrange(LF_refimg_and_video, "n c t h w -> n t c h w")
            gt_LF_video = LF_refimg_and_video[:,t:] # 灰度图作为GT


            if self.use_mask:
                LF_refimg_and_video = LF_refimg_and_video * camera_mask

            if self.use_camera_down:
                LF_refimg_and_video = einops.rearrange(LF_refimg_and_video, "n t c h w -> (n t) c h w")
                LF_refimg_and_video = self.camera_down(LF_refimg_and_video)
                LF_refimg_and_video = einops.rearrange(LF_refimg_and_video, "(n t) c h w -> n t c h w",n=n)

            # object_motion = self.object_motion_encoder(HF_refimg_and_video,mask_ratio) # (n,t+t,l,d)
            if self.use_motiontemporal:
                LF_video = LF_refimg_and_video[:,t:]
                camera_motion = self.camera_motion_encoder(LF_video,mask_ratio)
            else:
                camera_motion = self.camera_motion_encoder(LF_refimg_and_video,mask_ratio) # (n,t+t,l,d)

        else:
            if self.use_camera_down:
                LF_refimg_and_video = einops.rearrange(LF_refimg_and_video, "n t c h w -> (n t) c h w")
                LF_refimg_and_video = self.camera_down(LF_refimg_and_video)
                LF_refimg_and_video = einops.rearrange(LF_refimg_and_video, "(n t) c h w -> n t c h w",n=n)
            # object_motion = self.object_motion_encoder(refimg_and_video,mask_ratio) # (n,t+t,l,d)
            camera_motion = self.camera_motion_encoder(refimg_and_video,mask_ratio) # (n,t+t,l,d)

        # camera motion encoder
        camera_source_motion = camera_motion[:,:t].flatten(0,1) # (NT,motion_token,d)
        camera_target_motion = camera_motion[:,t:].flatten(0,1) # (NT,motion_token,d)
        assert camera_source_motion.shape == camera_target_motion.shape , f'camera_source_motion.shape {camera_source_motion.shape} != camera_target_motion.shape {camera_target_motion.shape}'


        if self.camera_motion_token_channel != self.motion_token_channel:
            camera_source_motion = self.camera_motion_map(camera_source_motion)
            camera_target_motion = self.camera_motion_map(camera_target_motion)
        # if self.object_motion_token_channel != self.motion_token_channel:
        #     object_source_motion = self.object_motion_map(object_source_motion)
        #     object_target_motion = self.object_motion_map(object_target_motion)

        source_motion = camera_source_motion # (NT,motion_token,d)
        target_motion = camera_target_motion # (NT,motion_token,d)

        # motion transformer
        if self.need_motion_transformer:
            target_motion = einops.rearrange(target_motion,'(n f) l d -> n f l d',n=n)
            target_motion = self.motion_transformer(target_motion)
            target_motion = einops.rearrange(target_motion,'n f l d -> (n f) l d',n=n)

        
        # prepare for Diffusion Transformer
        zi = ref_img_grey.flatten(0,1) # (NT,C,H,W)
        zj = gt_LF_video.flatten(0,1) # (NT,C,H,W)
        # zi = ref_img.flatten(0,1) # (NT,C,H,W)
        # zj = video.flatten(0,1) # (NT,C,H,W)


        if time_step is None:
            time_step = self.prepare_timestep(batch_size= zj.shape[0],device= device) #(b,)
        if self.diffusion_model_type != 'default':
            time_step = self.prepare_timestep(batch_size= n,device= device) # (n,)
            time_step = time_step.repeat_interleave(t) # (b,)
        zt,vel = self.scheduler.get_train_tuple(z1=zj,time_step=time_step)  # (NT,C,H,W),(NT,C,H,W)
        
        # dit forward
        image_hidden_states = torch.cat((zi,zt),dim=1) # (b,2C,H,W)
        pre = self.diffusion_transformer(camera_motion_source_hidden_states = camera_source_motion,
                                            camera_motion_target_hidden_states = camera_target_motion,
                                            image_hidden_states = image_hidden_states,
                                            timestep = time_step,)

        # loss
        diff_loss = l2(pre,vel)

        rec_zj = self.scheduler.get_target_with_zt_vel(zt,pre,time_step)
        rec_loss = l2(rec_zj,zj)


        loss = diff_loss
        loss_dict = {'loss':loss,'diff_loss':diff_loss,'rec_loss':rec_loss}
    
        if return_meta_info:
            return {'camera_motion' : camera_motion,               # (,t,motion_out_channels,h,w) , output of camera motion encoder 
                    # 'object_motion' : object_motion,               # (,t,motion_out_channels,h,w) , output of object motion encoder 
                    'zi' : zi,                       # (b,C,H,W) | b = n * t
                    'zj' : zj,                       # (b,C,H,W)
                    'zt' : zt,                       # (b,C,H,W)
                    'pre': pre,                      # (b,C,H,W)
                    'time_step': time_step,          # (b,)
                    }
        else:
            return pre,vel,loss_dict  # (b,C,H,W)

    @torch.no_grad()
    def sample(self,video:torch.Tensor,ref_img:torch.Tensor,video_grey=None,ref_img_grey=None,camera_mask=None,sample_step:int = 50,mask_ratio = None,start_step:int = None,return_meta_info=False):

        device = video.device
        n,t,c,h,w = video.shape

        if start_step is None:
            start_step = self.scheduler.num_step
        assert start_step <= self.scheduler.num_step , 'start_step cant be larger than scheduler.num_step'

        # motion encoder
        if mask_ratio is not None:
            print(f'* Sampling with Mask_Ratio = {mask_ratio}')
            mask_ratio =  mask_ratio
        
        refimg_and_video = torch.cat([ref_img,video],dim=1)# (n,t+t,C,H,W)
        

        # frequency filter
        if self.use_filter:
            freq_filter = gaussian_low_pass_filter([2*t, h, w],self.filter_num,self.filter_num)
            freq_filter = freq_filter.unsqueeze(0).unsqueeze(0).repeat(n, c, 1, 1, 1)
            freq_filter = freq_filter.to(device=refimg_and_video.device)

            if self.use_grey:
                refimg_and_video_grey = torch.cat([ref_img_grey,video_grey],dim=1)# (n,t+t,C,H,W) (1,32,4,32,32)
                refimg_and_video_grey = einops.rearrange(refimg_and_video_grey, "n t c h w -> n c t h w")
                LF_refimg_and_video, HF_refimg_and_video = freq_3d_filter(refimg_and_video_grey, freq_filter)
            else:
                refimg_and_video = einops.rearrange(refimg_and_video, "n t c h w -> n c t h w")
                LF_refimg_and_video, HF_refimg_and_video = freq_3d_filter(refimg_and_video, freq_filter)

            LF_refimg_and_video = einops.rearrange(LF_refimg_and_video, "n c t h w -> n t c h w")
            gt_LF_video = LF_refimg_and_video[:,t:]

            if self.use_mask:
                LF_refimg_and_video = LF_refimg_and_video*camera_mask

            if self.use_camera_down:
                LF_refimg_and_video = einops.rearrange(LF_refimg_and_video, "n t c h w -> (n t) c h w")
                LF_refimg_and_video = self.camera_down(LF_refimg_and_video)
                LF_refimg_and_video = einops.rearrange(LF_refimg_and_video, "(n t) c h w -> n t c h w",n=n)

            if self.use_motiontemporal:
                LF_video = LF_refimg_and_video[:,t:]
                camera_motion = self.camera_motion_encoder(LF_video,mask_ratio)
            else:
                camera_motion = self.camera_motion_encoder(LF_refimg_and_video,mask_ratio) # (n,t+t,l,d)

        else:
            if self.use_camera_down:
                LF_refimg_and_video = einops.rearrange(LF_refimg_and_video, "n t c h w -> (n t) c h w")
                LF_refimg_and_video = self.camera_down(LF_refimg_and_video)
                LF_refimg_and_video = einops.rearrange(LF_refimg_and_video, "(n t) c h w -> n t c h w",n=n)

            if self.use_motiontemporal:
                LF_video = LF_refimg_and_video[:,t:]
                camera_motion = self.camera_motion_encoder(LF_video,mask_ratio)
            else:
                camera_motion = self.camera_motion_encoder(LF_refimg_and_video,mask_ratio) # (n,t+t,l,d)

        # camera motion encoder
        camera_source_motion = camera_motion[:,:t].flatten(0,1) # (NT,motion_token,d)
        camera_target_motion = camera_motion[:,t:].flatten(0,1) # (NT,motion_token,d)
        assert camera_source_motion.shape == camera_target_motion.shape , f'camera_source_motion.shape {camera_source_motion.shape} != camera_target_motion.shape {camera_target_motion.shape}'

        if self.camera_motion_token_channel != self.motion_token_channel:
            camera_source_motion = self.camera_motion_map(camera_source_motion)
            camera_target_motion = self.camera_motion_map(camera_target_motion)

        source_motion = camera_source_motion # (NT,motion_token,d)
        target_motion = camera_target_motion # (NT,motion_token,d)

        # motion transformer
        if self.need_motion_transformer:
            target_motion = einops.rearrange(target_motion,'(n f) l d -> n f l d',n=n)
            target_motion = self.motion_transformer(target_motion)
            target_motion = einops.rearrange(target_motion,'n f l d -> (n f) l d',n=n)
        
        # prepare for Diffusion Transformer
        time_step = torch.ones((source_motion.shape[0],)).to(device) 
        time_step = time_step * start_step

        # zi = ref_img.flatten(0,1) # (NT,C,H,W)
        # zj = video.flatten(0,1) # (NT,C,H,W)
        zi = ref_img_grey.flatten(0,1) # (NT,C,H,W)
        zj = gt_LF_video.flatten(0,1) # (NT,C,H,W)
        
        zt,vel = self.scheduler.get_train_tuple(z1=zj,time_step=time_step)  # (NT,C,H,W),(NT,C,H,W)
        
        # Sample Loop
        pre_cache = []
        sample_cache = []
        
        # 1.step_seq
        step_seq = np.linspace(0, start_step, num=sample_step+1, endpoint=True,dtype=int) # [0,5,10,15,....,start_step]
        step_seq = list(reversed(step_seq[1:])) # delete step:0  [start_step,.....,15,10,5]

        # 2.Euler step
        dt = 1./sample_step

        for i in tqdm(step_seq):
            # time_step
            time_step = torch.ones((zt.shape[0],)).to(zt.device)  
            time_step = time_step * i
            
            # input
            zt = zt.to(video.dtype)
            image_hidden_states = torch.cat((zi,zt),dim=1) # (b,2C,H,W)
            
            # forward
            # pre = self.diffusion_transformer(motion_source_hidden_states = source_motion,
            #                                 motion_target_hidden_states = target_motion,
            #                                 image_hidden_states = image_hidden_states,
            #                                 timestep = time_step,) 
            pre = self.diffusion_transformer(camera_motion_source_hidden_states = camera_source_motion,
                                            camera_motion_target_hidden_states = camera_target_motion,
                                            image_hidden_states = image_hidden_states,
                                            timestep = time_step,)

            zt = zt + pre * dt
            pre_cache.append(pre)
            sample_cache.append(zt)
        
        zi = einops.rearrange(zi,'(n t) c h w -> n t c h w',n=n)
        zt = einops.rearrange(zt,'(n t) c h w -> n t c h w',n=n)
        zj = einops.rearrange(zj,'(n t) c h w -> n t c h w',n=n)
        
        if return_meta_info:
            return {'zi' : zi, # (b,1,c,h,w)
                    'zj' : zj,             # (b,1,c,h,w)
                    'sample' : zt,           # (b,1,c,h,w)
                    'pre_cache' : pre_cache,           # [(b,c,h,w),....]
                    'sample_cache' : sample_cache,   # [(b,c,h,w),....]
                    'step_seq' : step_seq,
                    'motion' : target_motion, # (b,C,H,W)
                    }
        else:
            return zi,zt,zj # (n,t,c,h,w)
        

    @torch.no_grad()
    def sample_diff_motion(self,video:torch.Tensor,ref_img:torch.Tensor,video_grey=None,ref_img_grey=None,camera_video_grey=None,sample_step:int = 50,mask_ratio = None,start_step:int = None,return_meta_info=False):

        device = video.device
        n,t,c,h,w = video.shape

        if start_step is None:
            start_step = self.scheduler.num_step
        assert start_step <= self.scheduler.num_step , 'start_step cant be larger than scheduler.num_step'

        # motion encoder
        if mask_ratio is not None:
            print(f'* Sampling with Mask_Ratio = {mask_ratio}')
            mask_ratio =  mask_ratio
        
        refimg_and_video = torch.cat([ref_img,video],dim=1)# (n,t+t,C,H,W) (1,32,4,32,32)
        
        # if not self.use_grey:
        #     ref_img_grey = torch.zeros_like
        # refimg_and_video_grey = torch.cat([ref_img_grey,video_grey],dim=1)# (n,t+t,C,H,W) (1,32,4,32,32)

        if self.use_filter:
            freq_filter = gaussian_low_pass_filter([2*t, h, w],d_s=0.4,d_t=0.4)
            freq_filter = freq_filter.unsqueeze(0).unsqueeze(0).repeat(n, c, 1, 1, 1)
            freq_filter = freq_filter.to(device=refimg_and_video.device)

            if self.use_grey:
                refimg_and_video_grey = torch.cat([ref_img_grey,video_grey],dim=1)# (n,t+t,C,H,W) (1,32,4,32,32)
                refimg_and_video_grey = einops.rearrange(refimg_and_video_grey, "n t c h w -> n c t h w")
                refimg_and_camera_video_grey = torch.cat([ref_img,camera_video_grey],dim=1)# (n,t+t,C,H,W) (1,32,4,32,32)
                refimg_and_camera_video_grey = einops.rearrange(refimg_and_camera_video_grey, "n t c h w -> n c t h w")

                _, HF_refimg_and_video = freq_3d_filter(refimg_and_video_grey, freq_filter)
                LF_refimg_and_video, _ = freq_3d_filter(refimg_and_camera_video_grey, freq_filter)


            else:
                refimg_and_video = einops.rearrange(refimg_and_video, "n t c h w -> n c t h w")
                LF_refimg_and_video, HF_refimg_and_video = freq_3d_filter(refimg_and_video, freq_filter)

            LF_refimg_and_video = einops.rearrange(LF_refimg_and_video, "n c t h w -> n t c h w")
            HF_refimg_and_video = einops.rearrange(HF_refimg_and_video, "n c t h w -> n t c h w")

            if self.use_camera_down:
                LF_refimg_and_video = einops.rearrange(LF_refimg_and_video, "n t c h w -> (n t) c h w")
                LF_refimg_and_video = self.camera_down(LF_refimg_and_video)
                LF_refimg_and_video = einops.rearrange(LF_refimg_and_video, "(n t) c h w -> n t c h w",n=n)

            object_motion = self.object_motion_encoder(HF_refimg_and_video,mask_ratio) # (n,t+t,l,d)
            camera_motion = self.camera_motion_encoder(LF_refimg_and_video,mask_ratio) # (n,t+t,l,d)

        else:
            if self.use_camera_down:
                LF_refimg_and_video = einops.rearrange(LF_refimg_and_video, "n t c h w -> (n t) c h w")
                LF_refimg_and_video = self.camera_down(LF_refimg_and_video)
                LF_refimg_and_video = einops.rearrange(LF_refimg_and_video, "(n t) c h w -> n t c h w",n=n)
            object_motion = self.object_motion_encoder(refimg_and_video,mask_ratio) # (n,t+t,l,d)
            camera_motion = self.camera_motion_encoder(refimg_and_video,mask_ratio) # (n,t+t,l,d)

        # object motion encoder
        object_source_motion = object_motion[:,:t].flatten(0,1) # (NT,motion_token,d)
        object_target_motion = object_motion[:,t:].flatten(0,1) # (NT,motion_token,d)
        assert object_source_motion.shape == object_target_motion.shape , f'object_source_motion.shape {object_source_motion.shape} != object_target_motion.shape {object_target_motion.shape}'

        # camera motion encoder
        camera_source_motion = camera_motion[:,:t].flatten(0,1) # (NT,motion_token,d)
        camera_target_motion = camera_motion[:,t:].flatten(0,1) # (NT,motion_token,d)
        assert camera_source_motion.shape == camera_target_motion.shape , f'camera_source_motion.shape {camera_source_motion.shape} != camera_target_motion.shape {camera_target_motion.shape}'

         # KL Loss
        if self.use_regularizers:
            object_target_motion = einops.rearrange(object_target_motion, "n t d -> n d t")
            camera_target_motion = einops.rearrange(camera_target_motion, "n t d -> n d t")
            object_target_motion, object_target_KLloss = self.regularization(object_target_motion)
            camera_target_motion, camera_target_KLloss = self.regularization(camera_target_motion)
            object_target_motion = einops.rearrange(object_target_motion, "n d t -> n t d")
            camera_target_motion = einops.rearrange(camera_target_motion, "n d t -> n t d")

        # motion fusion
        # camera motion average
        if self.use_regularizers:
            camera_source_motion = self.camera_source_motion_map(camera_source_motion)
            camera_target_motion = self.camera_target_motion_map(camera_target_motion)

            object_source_motion = self.object_source_motion_map(object_source_motion)
            object_target_motion = self.object_target_motion_map(object_target_motion)
        else:
            if self.camera_motion_token_channel != self.motion_token_channel:
                camera_source_motion = self.camera_motion_map(camera_source_motion)
                camera_target_motion = self.camera_motion_map(camera_target_motion)
            if self.object_motion_token_channel != self.motion_token_channel:
                object_source_motion = self.object_motion_map(object_source_motion)
                object_target_motion = self.object_motion_map(object_target_motion)

        # if self.use_motion == 'camera':
        #     source_motion = camera_source_motion # (NT,motion_token,d)
        #     target_motion = camera_target_motion # (NT,motion_token,d)
        # elif self.use_motion == 'object':
        #     source_motion = object_source_motion # (NT,motion_token,d)
        #     target_motion = object_target_motion # (NT,motion_token,d)
        # else:
        source_motion = object_source_motion + camera_source_motion # (NT,motion_token,d)
        target_motion = object_target_motion + camera_target_motion # (NT,motion_token,d)

        # motion transformer
        if self.need_motion_transformer:
            target_motion = einops.rearrange(target_motion,'(n f) l d -> n f l d',n=n)
            target_motion = self.motion_transformer(target_motion)
            target_motion = einops.rearrange(target_motion,'n f l d -> (n f) l d',n=n)
        
        # prepare for Diffusion Transformer
        time_step = torch.ones((source_motion.shape[0],)).to(device) 
        time_step = time_step * start_step

        zi = ref_img.flatten(0,1) # (NT,C,H,W)
        zj = video.flatten(0,1) # (NT,C,H,W)
        zt,vel = self.scheduler.get_train_tuple(z1=zj,time_step=time_step)  # (NT,C,H,W),(NT,C,H,W)
        
        # Sample Loop
        pre_cache = []
        sample_cache = []
        
        # 1.step_seq
        step_seq = np.linspace(0, start_step, num=sample_step+1, endpoint=True,dtype=int) # [0,5,10,15,....,start_step]
        step_seq = list(reversed(step_seq[1:])) # delete step:0  [start_step,.....,15,10,5]

        # 2.Euler step
        dt = 1./sample_step

        for i in tqdm(step_seq):
            # time_step
            time_step = torch.ones((zt.shape[0],)).to(zt.device)  
            time_step = time_step * i
            
            # input
            zt = zt.to(video.dtype)
            image_hidden_states = torch.cat((zi,zt),dim=1) # (b,2C,H,W)
            
            # forward
            pre = self.diffusion_transformer(motion_source_hidden_states = source_motion,
                                            motion_target_hidden_states = target_motion,
                                            image_hidden_states = image_hidden_states,
                                            timestep = time_step,) 
            zt = zt + pre * dt
            pre_cache.append(pre)
            sample_cache.append(zt)
        
        zi = einops.rearrange(zi,'(n t) c h w -> n t c h w',n=n)
        zt = einops.rearrange(zt,'(n t) c h w -> n t c h w',n=n)
        zj = einops.rearrange(zj,'(n t) c h w -> n t c h w',n=n)
        
        if return_meta_info:
            return {'zi' : zi, # (b,1,c,h,w)
                    'zj' : zj,             # (b,1,c,h,w)
                    'sample' : zt,           # (b,1,c,h,w)
                    'pre_cache' : pre_cache,           # [(b,c,h,w),....]
                    'sample_cache' : sample_cache,   # [(b,c,h,w),....]
                    'step_seq' : step_seq,
                    'motion' : target_motion, # (b,C,H,W)
                    }
        else:
            return zi,zt,zj # (n,t,c,h,w)

    def sample_with_refimg_motion(self,ref_img:torch.Tensor,motion=torch.Tensor,sample_step:int = 10,return_meta_info=False):
        """
        Args:
            ref_img : (N,C,H,W)                             
            motion : (N,F,L,D)                           
        Return:
            video : (N,T,C,H,W)                             
        """
        device = motion.device
        n,t,l,d = motion.shape

        start_step = self.scheduler.num_step
        
        # motion encoder
        refimg = ref_img.unsqueeze(1) # (N,1,C,H,W)
        source_motion = self.motion_encoder(refimg) # (n,1,motion_token,d)

        source_motion = source_motion.repeat(1,t,1,1).flatten(0,1) # (NT,l,d)
        target_motion = motion.flatten(0,1) # (NT,l,d)

        assert source_motion.shape == target_motion.shape , f'source_motion.shape {source_motion.shape} != target_motion.shape {target_motion.shape}'

        # motion transformer
        if self.need_motion_transformer and not self.extract_motion_with_motion_transformer:
            target_motion = einops.rearrange(target_motion,'(n f) l d -> n f l d',n=n)
            target_motion = self.motion_transformer(target_motion)
            target_motion = einops.rearrange(target_motion,'n f l d -> (n f) l d',n=n)
        
        # prepare for Diffusion Transformer
        time_step = torch.ones((source_motion.shape[0],)).to(device) 
        time_step = time_step * start_step

        zi = refimg.repeat(1,t,1,1,1).flatten(0,1) # (NT,C,H,W)
        zj = zi
        zt,vel = self.scheduler.get_train_tuple(z1=zj,time_step=time_step)  # (NT,C,H,W),(NT,C,H,W)
        
        # Sample Loop
        pre_cache = []
        sample_cache = []
        
        # 1.step_seq
        step_seq = np.linspace(0, start_step, num=sample_step+1, endpoint=True,dtype=int) # [0,5,10,15,....,start_step]
        step_seq = list(reversed(step_seq[1:])) # delete step:0  [start_step,.....,15,10,5]

        # 2.Euler step
        dt = 1./sample_step

        for i in tqdm(step_seq):
            # time_step
            time_step = torch.ones((zt.shape[0],)).to(zt.device)  
            time_step = time_step * i
            
            # input
            zt = zt.to(ref_img.dtype)
            image_hidden_states = torch.cat((zi,zt),dim=1) # (b,2C,H,W)
            
            # forward
            pre = self.diffusion_transformer(motion_source_hidden_states = source_motion,
                                            motion_target_hidden_states = target_motion,
                                            image_hidden_states = image_hidden_states,
                                            timestep = time_step,) 
            zt = zt + pre * dt
            pre_cache.append(pre)
            sample_cache.append(zt)
        
        # unsqueeze (n,1,c,h,w) means images, (n,t,c,h,w) means video t>1 .
        zi = einops.rearrange(zi,'(n t) c h w -> n t c h w',n=n,t=t)
        zt = einops.rearrange(zt,'(n t) c h w -> n t c h w',n=n,t=t)
        
        if return_meta_info:
            return {'zi' : zi, # (b,1,c,h,w)
                    'zj' : zj,             # (b,1,c,h,w)
                    'sample' : zt,           # (b,1,c,h,w)
                    'pre_cache' : pre_cache,           # [(b,c,h,w),....]
                    'sample_cache' : sample_cache,   # [(b,c,h,w),....]
                    'step_seq' : step_seq,
                    'motion' : target_motion, # (b,C,H,W)
                    }
        else:
            return zi,zt,zj # (b,1,c,h,w)
    
    def extract_motion(self,video:torch.tensor):
        # video : (N,T,C,H,W)
        n,t,c,h,w = video.shape
        
        motion = self.motion_encoder(video) # (N,T,L,D)

        if self.need_motion_transformer and self.extract_motion_with_motion_transformer:
            motion = self.motion_transformer(motion) # (N,T,L,D)

        return motion
     
    def prepare_timestep(self,batch_size:int,device,time_step = None):
        if time_step is not None:
            return time_step.to(device)
        else:
            return torch.randint(0,self.num_step+1,(batch_size,)).to(device)
  
    def prepare_encoder_input(self,video:torch.tensor):
        assert len(video.shape) == 5 , f'only support video data : 5D tensor , but got {video.shape}'
        
        # cat
        pre = video[:,:-1,:,:,:] 
        post= video[:,1:,:,:,:]
        duo_frame_mix = torch.cat([pre,post],dim=2)    # (b,t-1,2c,h,w)
        duo_frame_mix = einops.rearrange(duo_frame_mix,'b t c h w -> (b t) c h w')
        
        return duo_frame_mix # (b*f-1,2c,h,w)


    def unpatchify(self, x ,patch_size):
        """
        x: (N, S, patch_size**2 *C)
        imgs: (N, C, H, W)
        """
        p = patch_size
        h = w = int(x.shape[1]**.5)
        # c = self.in_chans
        c = x.shape[2] // (p**2)
        assert h * w == x.shape[1]
        
        x = x.reshape(shape=(x.shape[0], h, w, p, p, c)) # (N, h, w, p, p, c)
        x = torch.einsum('nhwpqc->nchpwq', x) # (N, c, h, p, w, p)
        imgs = x.reshape(shape=(x.shape[0], c, h * p, h * p))
        return imgs #(N,C,H,W)    

    def reset_infer_num_frame(self, num:int):
        old_num = self.diffusion_transformer.target_frame
        self.diffusion_transformer.target_frame = num
        if self.use_motiontemporal:
            self.camera_motion_encoder.video_frames = num
            self.object_motion_encoder.video_frames = num
        print(f'* Reset infer frame from {old_num} to {self.diffusion_transformer.target_frame} *')

class AMDModel_New(ModelMixin, ConfigMixin):
    _supports_gradient_checkpointing = True
    
    @register_to_config
    def __init__(self,
                 image_inchannel :int  = 4,
                 image_height  :int = 32,
                 image_width  :int  = 32,
                 video_frames  :int = 16,
                 scheduler_num_step :int = 1000,
                 use_filter:bool = False,
                 filter_num:float = 0.4,
                 use_grey:bool = False,
                 use_camera_down:bool = False,
                 use_regularizers:bool = False,
                 use_motiontemporal:bool = True,
                 klloss_weight: float = 0.005, # 比diffloss小一个量级
                 use_mask:bool = False,
                 motion_type:str='plus',
                 use_camera:bool = True,
                 use_object:bool = True,
                 
                 # 理论上来说,高频信号在transformer中的深层,低频信号在transformer的浅层,所以不同支路的layer_num也需要做实验
                 # -----------Object MotionEncoder -----------
                 object_motion_token_num:int = 12,
                 object_motion_token_channel: int = 128,
                 object_enc_num_layers:int = 8,
                 enc_nhead:int = 8,
                 enc_ndim:int = 64,
                 enc_dropout:float = 0.0,
                 motion_need_norm_out:bool = False,

                 # -----------Camera MotionEncoder -----------
                 camera_motion_token_num:int = 12,
                 camera_motion_token_channel: int = 128,
                 camera_enc_num_layers:int = 8,
                
                 # -----------MotionTransformer ---------
                 motion_token_num:int = 12,
                 motion_token_channel: int = 128, 
                 need_motion_transformer :bool = False,
                 motion_transformer_attn_head_dim:int = 64,
                 motion_transformer_attn_num_heads:int = 16,
                 motion_transformer_num_layers:int = 4, 
                
                 # ----------- Diffusion Transformer -----------
                 diffusion_model_type : str = 'default', # or dual
                 diffusion_attn_head_dim : int  = 64,
                 diffusion_attn_num_heads : int = 16,
                 diffusion_out_channels : int = 4,
                 diffusion_num_layers : int = 16,
                 image_patch_size : int  = 2,
                 motion_patch_size : int = 1,
                 motion_drop_ratio: float = 0.0,

                 # ----------- Sample --------------
                 extract_motion_with_motion_transformer = False,
                 **kwargs,
                 ):
        super().__init__()

        # setting
        self.num_step = scheduler_num_step
        self.scheduler = RectifiedFlow(num_steps=scheduler_num_step)
        self.need_motion_transformer = need_motion_transformer
        self.extract_motion_with_motion_transformer = extract_motion_with_motion_transformer
        self.diffusion_model_type = diffusion_model_type
        self.target_frame = video_frames

        self.camera_motion_token_channel = camera_motion_token_channel
        self.object_motion_token_channel = object_motion_token_channel
        self.motion_token_channel = motion_token_channel

        # self.use_motion = use_motion
        self.filter_num = filter_num
        self.use_camera = use_camera
        self.use_object = use_object

        # frepuency filter
        self.use_filter = use_filter 
        self.use_grey = use_grey
        self.use_camera_down = use_camera_down


        # if camera_motion_token_channel != motion_token_channel:
        #     self.camera_motion_map = nn.Linear(camera_motion_token_channel, motion_token_channel)
        # if object_motion_token_channel != motion_token_channel:
        #     self.object_motion_map = nn.Linear(object_motion_token_channel, motion_token_channel)

        if use_camera:
            # camera motion Encoder 
            if self.use_camera_down:
                camera_image_height = image_height//4
                camera_image_width = image_width//4
            else:
                camera_image_height = image_height
                camera_image_width = image_width
            self.camera_motion_encoder = MotionEncoderLearnTokenOnlyTemporalTransformer(img_height = camera_image_height, # 如果采用下采样,则img_height = image_height\\4
                                                                    img_width=camera_image_width,   # 如果采用下采样,则image_width = image_width\\4
                                                                    img_inchannel=image_inchannel,
                                                                    img_patch_size = image_patch_size,
                                                                    motion_token_num =  camera_motion_token_num,
                                                                    motion_channel = camera_motion_token_channel,
                                                                    need_norm_out = motion_need_norm_out,
                                                                    video_frames=self.target_frame,
                                                                    # ----- attention
                                                                    num_attention_heads=enc_nhead,
                                                                    attention_head_dim=enc_ndim, # 原始dim为8*128, camera降低dim为8*32
                                                                    num_layers=camera_enc_num_layers,      
                                                                    dropout=enc_dropout,
                                                                    attention_bias= True,)
        if use_object:
            self.object_motion_encoder = MotionEncoderLearnTokenTransformer(img_height = image_height,
                                                                        img_width=image_width,
                                                                        img_inchannel=image_inchannel,
                                                                        img_patch_size = image_patch_size,
                                                                        motion_token_num =  object_motion_token_num,
                                                                        motion_channel = object_motion_token_channel,
                                                                        need_norm_out = motion_need_norm_out,
                                                                        # ----- attention
                                                                        num_attention_heads=enc_nhead,
                                                                        attention_head_dim=enc_ndim,
                                                                        num_layers=object_enc_num_layers,      
                                                                        dropout=enc_dropout,
                                                                        attention_bias= True,)
        if self.use_camera_down:
            self.camera_down = nn.Sequential(
                nn.Conv2d(in_channels=4, out_channels=16, kernel_size=3, stride=2, padding=1),
                nn.Conv2d(in_channels=16, out_channels=4, kernel_size=3, stride=1, padding=1),
                nn.MaxPool2d(kernel_size=2, stride=2, padding=0)
            )
        
        # motion transformer
        if need_motion_transformer:
            self.motion_transformer = MotionTransformer(motion_token_num=motion_token_num,
                                                        motion_token_channel=motion_token_channel,
                                                        attention_head_dim=motion_transformer_attn_head_dim,
                                                        num_attention_heads=motion_transformer_attn_num_heads,
                                                        num_layers=motion_transformer_num_layers,)
        
        # diffusion transformer
        dit_image_inchannel = image_inchannel * 2 # zi + zt
        # dit_image_inchannel = image_inchannel

        if diffusion_model_type == 'default':
            self.diffusion_transformer = AMDDiffusionTransformerModelTempMotion(num_attention_heads= diffusion_attn_num_heads,
                                                                    attention_head_dim= diffusion_attn_head_dim,
                                                                    out_channels = diffusion_out_channels,
                                                                    num_layers= diffusion_num_layers,
                                                                    use_camera=use_camera,
                                                                    use_object=use_object,
                                                                    # ----- img
                                                                    image_width= image_width,
                                                                    image_height= image_height,
                                                                    image_patch_size= image_patch_size,
                                                                    image_in_channels = dit_image_inchannel, 
                                                                    # ----- motion
                                                                    motion_token_num = motion_token_num,
                                                                    camera_motion_in_channels = camera_motion_token_channel,
                                                                    object_motion_in_channels = object_motion_token_channel,
                                                                    motion_target_num_frame = self.target_frame)
        elif diffusion_model_type == 'spatial':
            self.diffusion_transformer = AMDDiffusionTransformerModelImgSpatialTempMotion(num_attention_heads= diffusion_attn_num_heads,
                                                                attention_head_dim= diffusion_attn_head_dim,
                                                                out_channels = diffusion_out_channels,
                                                                num_layers= diffusion_num_layers,
                                                                use_camera=use_camera,
                                                                use_object=use_object,
                                                                # ----- img
                                                                image_width= image_width,
                                                                image_height= image_height,
                                                                image_patch_size= image_patch_size,
                                                                image_in_channels = dit_image_inchannel, 
                                                                # ----- motion
                                                                motion_token_num = motion_token_num,
                                                                camera_motion_in_channels = camera_motion_token_channel,
                                                                object_motion_in_channels = object_motion_token_channel,
                                                                motion_target_num_frame = self.target_frame)
        else:
            raise IndexError

    def forward(self,video:torch.Tensor,ref_img:torch.Tensor,video_grey=None,ref_img_grey=None,time_step:torch.tensor = None,return_meta_info=False,camera_mask_ratio=None,object_mask_ratio=None):
        """
        Args:
            video: (N,T,C,H,W)
            ref_img: (N,T,C,H,W)
        """

        device = video.device
        n,t,c,h,w = video.shape

        assert video.shape == ref_img.shape ,f'video.shape:{video.shape}should be equal to ref_img.shape:{ref_img.shape}'
        
        # # motion encoder
        if camera_mask_ratio is not None:
            camera_mask_ratio = (0.6+0.4*torch.rand(1)).item() * camera_mask_ratio
            # print("camera_mask_ratio",camera_mask_ratio)

        if object_mask_ratio is not None:
            object_mask_ratio = (0.5*torch.rand(1)).item() * object_mask_ratio
            # print("object_mask_ratio",object_mask_ratio)
        # gt_video = video
        # if self.mask_video_ratio != 0:
                
        #     b_z, f_z, c_z, h_z, w_z = video.shape
        #     rand_mask = torch.rand(h_z, w_z).to(device=video.device)
        #     mask = rand_mask > self.mask_video_ratio
        #     mask = mask.unsqueeze(0).unsqueeze(0).unsqueeze(0)  
        #     mask = mask.expand(b_z, f_z, c_z, h_z, w_z) 
        #     video = video * mask

        #     if self.use_grey:
        #         video_grey = video_grey * mask
        
        refimg_and_video = torch.cat([ref_img,video],dim=1)# (n,t+t,C,H,W)
        

        # frequency filter
        if self.use_filter:
            freq_filter = gaussian_low_pass_filter([2*t, h, w],0.6,0.6)
            freq_filter = freq_filter.unsqueeze(0).unsqueeze(0).repeat(n, c, 1, 1, 1)
            freq_filter = freq_filter.to(device=refimg_and_video.device)

            high_freq_filter = gaussian_low_pass_filter([2*t, h, w],0.5,0.5)
            high_freq_filter = high_freq_filter.unsqueeze(0).unsqueeze(0).repeat(n, c, 1, 1, 1)
            high_freq_filter = high_freq_filter.to(device=refimg_and_video.device)            

            if self.use_grey:
                refimg_and_video_grey = torch.cat([ref_img_grey,video_grey],dim=1)# (n,t+t,C,H,W) (1,32,4,32,32)
                refimg_and_video_grey = einops.rearrange(refimg_and_video_grey, "n t c h w -> n c t h w")
                LF_refimg_and_video, _ = freq_3d_filter(refimg_and_video_grey, freq_filter)
                _,HF_refimg_and_video = freq_3d_filter(refimg_and_video_grey, high_freq_filter)

                refimg_and_video = einops.rearrange(refimg_and_video, "n t c h w -> n c t h w")
                LF_refimg_and_video_RGB, _ = freq_3d_filter(refimg_and_video, freq_filter)
                refimg_and_video = einops.rearrange(refimg_and_video, "n c t h w -> n t c h w")
                LF_refimg_and_video_RGB = einops.rearrange(LF_refimg_and_video_RGB, "n c t h w -> n t c h w")
                gt_LF_video_RGB = LF_refimg_and_video_RGB[:,t:]
            else:
                refimg_and_video = einops.rearrange(refimg_and_video, "n t c h w -> n c t h w")
                LF_refimg_and_video, _ = freq_3d_filter(refimg_and_video, freq_filter)
                _,HF_refimg_and_video = freq_3d_filter(refimg_and_video, high_freq_filter)
                refimg_and_video = einops.rearrange(refimg_and_video, "n c t h w -> n t c h w")

            LF_refimg_and_video = einops.rearrange(LF_refimg_and_video, "n c t h w -> n t c h w")
            HF_refimg_and_video = einops.rearrange(HF_refimg_and_video, "n c t h w -> n t c h w")

            gt_LF_video = LF_refimg_and_video[:,t:] # 灰度图作为GT
            
            LF_video = gt_LF_video

            if self.use_camera:
                if self.use_camera_down:
                    LF_video = einops.rearrange(LF_video, "n t c h w -> (n t) c h w")
                    LF_video = self.camera_down(LF_video)
                    LF_video = einops.rearrange(LF_video, "(n t) c h w -> n t c h w",n=n)
                
                camera_target_motion = self.camera_motion_encoder(LF_video,camera_mask_ratio) # video (n,t,l,d) 

                # if self.camera_motion_token_channel != self.motion_token_channel:
                #     camera_target_motion = self.camera_motion_map(camera_target_motion)

            if self.use_object:
                object_motion = self.object_motion_encoder(refimg_and_video,object_mask_ratio) # (n,t+t,l,d)
                # object motion encoder
                object_source_motion = object_motion[:,:t].flatten(0,1) # (NT,motion_token,d)
                object_target_motion = object_motion[:,t:].flatten(0,1) # (NT,motion_token,d)
                assert object_source_motion.shape == object_target_motion.shape , f'object_source_motion.shape {object_source_motion.shape} != object_target_motion.shape {object_target_motion.shape}'

                # if self.object_motion_token_channel != self.motion_token_channel:
                #     object_source_motion = self.object_motion_map(object_source_motion)
                #     object_target_motion = self.object_motion_map(object_target_motion)

        else:
            if self.use_camera_down:
                LF_refimg_and_video = einops.rearrange(LF_refimg_and_video, "n t c h w -> (n t) c h w")
                LF_refimg_and_video = self.camera_down(LF_refimg_and_video)
                LF_refimg_and_video = einops.rearrange(LF_refimg_and_video, "(n t) c h w -> n t c h w",n=n)
            # object_motion = self.object_motion_encoder(refimg_and_video,mask_ratio) # (n,t+t,l,d)
            LF_video = LF_refimg_and_video[:,t:]
            camera_target_motion = self.camera_motion_encoder(LF_video,camera_mask_ratio) # (n,t+t,l,d)

        
        # prepare for Diffusion Transformer
        # zi = ref_img_grey[:,0] # (N,C,H,W)
        # zi = ref_img_grey[:,0:2] # (N,T,C,H,W)
        # zj = gt_LF_video.flatten(0,1) # (N,T,C,H,W)
        # zi = ref_img[:,0]  # (NT,C,H,W)
        # zj = video.flatten(0,1) # (NT,C,H,W)

        # zi = ref_img_grey.flatten(0,1) # (NT,C,H,W)
        # zj = gt_LF_video.flatten(0,1) # (NT,C,H,W)
        # zj = gt_LF_video_RGB.flatten(0,1) # (NT,C,H,W)
        zi = ref_img.flatten(0,1) # (NT,C,H,W)
        zj = video.flatten(0,1) # (NT,C,H,W)


        if time_step is None:
            time_step = self.prepare_timestep(batch_size= zj.shape[0],device= device) #(b,)
        if self.diffusion_model_type != 'default':
            time_step = self.prepare_timestep(batch_size= n,device= device) # (n,)
            time_step = time_step.repeat_interleave(t) # (b,)
        zt,vel = self.scheduler.get_train_tuple(z1=zj,time_step=time_step)  # (NT,C,H,W),(NT,C,H,W)
        
        # dit forward
        # zt = zt.to(video.dtype)
        # zt = einops.rearrange(zt,'(n t) c h w -> n t c h w',n=n)
        # # z1 = (zt[:,0] + zi).unsqueeze(1)
        # z1 = (zt[:,0:2] + zi)
        # # zt = torch.cat((z1,zt[:,1:]),dim=1).flatten(0,1) # (b,C,H,W)
        # zt = torch.cat((z1,zt[:,2:]),dim=1).flatten(0,1) # (b,C,H,W)
        # image_hidden_states = zt
        image_hidden_states = torch.cat((zi,zt),dim=1) # (b,2C,H,W)

        
        if self.use_object and not self.use_camera:
            pre = self.diffusion_transformer(object_motion_source_hidden_states = object_source_motion,
                                            object_motion_target_hidden_states = object_target_motion,
                                            image_hidden_states = image_hidden_states,
                                            timestep = time_step,)
        elif not self.use_object and self.use_camera:
            pre = self.diffusion_transformer(camera_motion_target_hidden_states = camera_target_motion,
                                            image_hidden_states = image_hidden_states,
                                            timestep = time_step,)
        else:
            pre = self.diffusion_transformer(object_motion_source_hidden_states = object_source_motion,
                                            object_motion_target_hidden_states = object_target_motion,
                                            camera_motion_target_hidden_states = camera_target_motion,
                                            image_hidden_states = image_hidden_states,
                                            timestep = time_step,)

        # loss
        diff_loss = l2(pre,vel)

        rec_zj = self.scheduler.get_target_with_zt_vel(zt,pre,time_step)
        rec_loss = l2(rec_zj,zj)

        
        # if self.use_regularizers:
        #     KLloss = self.klloss_weight * (object_target_KLloss + camera_target_KLloss)/2
        #     loss = diff_loss + KLloss
        #     loss_dict = {'loss':loss,'diff_loss':diff_loss,'rec_loss':rec_loss,'KLloss':KLloss}
        # else:
        loss = diff_loss
        loss_dict = {'loss':loss,'diff_loss':diff_loss,'rec_loss':rec_loss}
    
        if return_meta_info:
            return {'zi' : zi,                       # (b,C,H,W) | b = n * t
                    'zj' : zj,                       # (b,C,H,W)
                    'zt' : zt,                       # (b,C,H,W)
                    'pre': pre,                      # (b,C,H,W)
                    'time_step': time_step,          # (b,)
                    }
        else:
            return pre,vel,loss_dict  # (b,C,H,W)

    @torch.no_grad()
    def sample(self,video:torch.Tensor,ref_img:torch.Tensor,video_grey=None,ref_img_grey=None,sample_step:int = 50,object_mask_ratio = None,camera_mask_ratio = None,start_step:int = None,return_meta_info=False):

        device = video.device
        n,t,c,h,w = video.shape

        if start_step is None:
            start_step = self.scheduler.num_step
        assert start_step <= self.scheduler.num_step , 'start_step cant be larger than scheduler.num_step'

        # motion encoder
        if object_mask_ratio is not None:
            # print(f'* Sampling with object Mask_Ratio = {object_mask_ratio}')
            object_mask_ratio =  object_mask_ratio
        if camera_mask_ratio is not None:
            # print(f'* Sampling with camera Mask_Ratio = {camera_mask_ratio}')
            camera_mask_ratio =  camera_mask_ratio

        refimg_and_video = torch.cat([ref_img,video],dim=1)# (n,t+t,C,H,W)
        

        # frequency filter
        if self.use_filter:
            freq_filter = gaussian_low_pass_filter([2*t, h, w],0.6,0.6)
            freq_filter = freq_filter.unsqueeze(0).unsqueeze(0).repeat(n, c, 1, 1, 1)
            freq_filter = freq_filter.to(device=refimg_and_video.device)

            high_freq_filter = gaussian_low_pass_filter([2*t, h, w],0.6,0.6)
            high_freq_filter = high_freq_filter.unsqueeze(0).unsqueeze(0).repeat(n, c, 1, 1, 1)
            high_freq_filter = high_freq_filter.to(device=refimg_and_video.device)            

            if self.use_grey:
                refimg_and_video_grey = torch.cat([ref_img_grey,video_grey],dim=1)# (n,t+t,C,H,W) (1,32,4,32,32)
                refimg_and_video_grey = einops.rearrange(refimg_and_video_grey, "n t c h w -> n c t h w")
                LF_refimg_and_video, _ = freq_3d_filter(refimg_and_video_grey, freq_filter)
                _,HF_refimg_and_video = freq_3d_filter(refimg_and_video_grey, high_freq_filter)

                refimg_and_video = einops.rearrange(refimg_and_video, "n t c h w -> n c t h w")
                LF_refimg_and_video_RGB, _ = freq_3d_filter(refimg_and_video, freq_filter)
                refimg_and_video = einops.rearrange(refimg_and_video, "n c t h w -> n t c h w")
                LF_refimg_and_video_RGB = einops.rearrange(LF_refimg_and_video_RGB, "n c t h w -> n t c h w")
                gt_LF_video_RGB = LF_refimg_and_video_RGB[:,t:]
            
            else:
                refimg_and_video = einops.rearrange(refimg_and_video, "n t c h w -> n c t h w")
                LF_refimg_and_video, _ = freq_3d_filter(refimg_and_video, freq_filter)
                _,HF_refimg_and_video = freq_3d_filter(refimg_and_video, high_freq_filter)
                refimg_and_video = einops.rearrange(refimg_and_video, "n c t h w -> n t c h w")

            LF_refimg_and_video = einops.rearrange(LF_refimg_and_video, "n c t h w -> n t c h w")
            HF_refimg_and_video = einops.rearrange(HF_refimg_and_video, "n c t h w -> n t c h w")

            gt_LF_video = LF_refimg_and_video[:,t:] # 灰度图作为GT
            
            LF_video = gt_LF_video

            if self.use_camera:
                if self.use_camera_down:
                    LF_video = einops.rearrange(LF_video, "n t c h w -> (n t) c h w")
                    LF_video = self.camera_down(LF_video)
                    LF_video = einops.rearrange(LF_video, "(n t) c h w -> n t c h w",n=n)

                camera_target_motion = self.camera_motion_encoder(LF_video,camera_mask_ratio) # video (n,t,l,d) 

                # if self.camera_motion_token_channel != self.motion_token_channel:
                #     camera_target_motion = self.camera_motion_map(camera_target_motion)

            if self.use_object:
                object_motion = self.object_motion_encoder(refimg_and_video,object_mask_ratio) # (n,t+t,l,d)
                # object motion encoder
                object_source_motion = object_motion[:,:t].flatten(0,1) # (NT,motion_token,d)
                object_target_motion = object_motion[:,t:].flatten(0,1) # (NT,motion_token,d)
                assert object_source_motion.shape == object_target_motion.shape , f'object_source_motion.shape {object_source_motion.shape} != object_target_motion.shape {object_target_motion.shape}'

                # if self.object_motion_token_channel != self.motion_token_channel:
                #     object_source_motion = self.object_motion_map(object_source_motion)
                #     object_target_motion = self.object_motion_map(object_target_motion)

        else:
            if self.use_camera_down:
                LF_refimg_and_video = einops.rearrange(LF_refimg_and_video, "n t c h w -> (n t) c h w")
                LF_refimg_and_video = self.camera_down(LF_refimg_and_video)
                LF_refimg_and_video = einops.rearrange(LF_refimg_and_video, "(n t) c h w -> n t c h w",n=n)
            # object_motion = self.object_motion_encoder(refimg_and_video,mask_ratio) # (n,t+t,l,d)
            LF_video = LF_refimg_and_video[:,t:]
            camera_target_motion = self.camera_motion_encoder(LF_video,camera_mask_ratio) # (n,t+t,l,d)



        # zi = ref_img[:,0]  # (N,C,H,W)
        # zj = video.flatten(0,1) # (N,T,C,H,W)
        # zi = ref_img_grey[:,0] # (N,T,C,H,W)
        # zi = ref_img_grey[:,0:2] # (N,T,C,H,W)
        # zj = gt_LF_video.flatten(0,1) # (NT,C,H,W)
        
        # zi = ref_img_grey.flatten(0,1) # (NT,C,H,W)
        zi = ref_img.flatten(0,1) # (NT,C,H,W)
        zj = video.flatten(0,1) # (N,T,C,H,W)
        # zj = gt_LF_video.flatten(0,1) # (NT,C,H,W)
        # zj = gt_LF_video_RGB.flatten(0,1)


        # prepare for Diffusion Transformer
        # time_step = torch.ones((zj.shape[0],)).to(device) 
        time_step = torch.ones((zj.shape[0],)).to(device) # 补充第一帧
        time_step = time_step * start_step
        

        zt,vel = self.scheduler.get_train_tuple(z1=zj,time_step=time_step)  # (NT,C,H,W),(NT,C,H,W)
        
        # Sample Loop
        pre_cache = []
        sample_cache = []
        
        # 1.step_seq
        step_seq = np.linspace(0, start_step, num=sample_step+1, endpoint=True,dtype=int) # [0,5,10,15,....,start_step]
        step_seq = list(reversed(step_seq[1:])) # delete step:0  [start_step,.....,15,10,5]

        # 2.Euler step
        dt = 1./sample_step

        for i in tqdm(step_seq):

            # time_step
            # print("zt.shape",zt.shape)
            time_step = torch.ones((zt.shape[0],)).to(zt.device)  
            time_step = time_step * i
            
            # input
            # zt = zt.to(video.dtype)
            # zt = einops.rearrange(zt,'(n t) c h w -> n t c h w',n=n)
            # z1 = (zt[:,0] + zi).unsqueeze(1)
            # # z1 = (zt[:,0:2] + zi)
            # zt = torch.cat((z1,zt[:,1:]),dim=1).flatten(0,1) # (b,C,H,W)
            # # zt = torch.cat((z1,zt[:,2:]),dim=1).flatten(0,1)
            # image_hidden_states = zt

            image_hidden_states = torch.cat((zi,zt),dim=1) # (b,2C,H,W)

            # pre = self.diffusion_transformer(camera_motion_target_hidden_states = camera_target_motion,
            #                                 image_hidden_states = image_hidden_states,
            #                                 timestep = time_step,)
            if self.use_object and not self.use_camera:
                pre = self.diffusion_transformer(object_motion_source_hidden_states = object_source_motion,
                                                object_motion_target_hidden_states = object_target_motion,
                                                image_hidden_states = image_hidden_states,
                                                timestep = time_step,)
            elif not self.use_object and self.use_camera:
                pre = self.diffusion_transformer(camera_motion_target_hidden_states = camera_target_motion,
                                                image_hidden_states = image_hidden_states,
                                                timestep = time_step,)
            else:
                pre = self.diffusion_transformer(object_motion_source_hidden_states = object_source_motion,
                                                object_motion_target_hidden_states = object_target_motion,
                                                camera_motion_target_hidden_states = camera_target_motion,
                                                image_hidden_states = image_hidden_states,
                                                timestep = time_step,)


            zt = zt + pre * dt
            pre_cache.append(pre)
            sample_cache.append(zt)
        
        zi = einops.rearrange(zi,'(n t) c h w -> n t c h w',n=n)
        zt = einops.rearrange(zt,'(n t) c h w -> n t c h w',n=n)
        zj = einops.rearrange(zj,'(n t) c h w -> n t c h w',n=n)
        
        if return_meta_info:
            return {'zi' : zi, # (b,1,c,h,w)
                    'zj' : zj,             # (b,1,c,h,w)
                    'sample' : zt,           # (b,1,c,h,w)
                    'pre_cache' : pre_cache,           # [(b,c,h,w),....]
                    'sample_cache' : sample_cache,   # [(b,c,h,w),....]
                    'step_seq' : step_seq,
                    }
        else:
            return zi,zt,zj # (n,t,c,h,w)

    @torch.no_grad()
    def sample_cross(self,video_1:torch.Tensor,video_2:torch.Tensor, ref_img:torch.Tensor,video_grey_1=None,video_grey_2=None,ref_img_grey=None,sample_step:int = 50,object_mask_ratio = None,camera_mask_ratio = None,start_step:int = None,return_meta_info=False):
        '''
            video_1提供camera_motion
            video_2提供object_motion和refimg
            decoder那里的加噪GT是video_2
        '''
        device = video_1.device
        n,t,c,h,w = video_1.shape

        if start_step is None:
            start_step = self.scheduler.num_step
        assert start_step <= self.scheduler.num_step , 'start_step cant be larger than scheduler.num_step'

        # motion encoder
        if object_mask_ratio is not None:
            # print(f'* Sampling with object Mask_Ratio = {object_mask_ratio}')
            object_mask_ratio =  object_mask_ratio
        if camera_mask_ratio is not None:
            # print(f'* Sampling with camera Mask_Ratio = {camera_mask_ratio}')
            camera_mask_ratio =  camera_mask_ratio

        refimg_and_video_2 = torch.cat([ref_img,video_2],dim=1)# (n,t+t,C,H,W)
        

        # frequency filter
        if self.use_filter:
            freq_filter = gaussian_low_pass_filter([t, h, w],0.5,0.5)
            freq_filter = freq_filter.unsqueeze(0).unsqueeze(0).repeat(n, c, 1, 1, 1)
            freq_filter = freq_filter.to(device=refimg_and_video_2.device)

            high_freq_filter = gaussian_low_pass_filter([2*t, h, w],0.6,0.6)
            high_freq_filter = high_freq_filter.unsqueeze(0).unsqueeze(0).repeat(n, c, 1, 1, 1)
            high_freq_filter = high_freq_filter.to(device=refimg_and_video_2.device)            

            if self.use_grey:
                refimg_and_video_grey_2 = torch.cat([ref_img_grey,video_grey_2],dim=1)# (n,t+t,C,H,W) (1,32,4,32,32)
                refimg_and_video_grey_2 = einops.rearrange(refimg_and_video_grey_2, "n t c h w -> n c t h w")
                video_grey_1 = einops.rearrange(video_grey_1, "n t c h w -> n c t h w")
                LF_video, _ = freq_3d_filter(video_grey_1, freq_filter)
                _,HF_refimg_and_video = freq_3d_filter(refimg_and_video_grey_2, high_freq_filter)
                LF_video = einops.rearrange(LF_video, "n c t h w -> n t c h w")

            else:
                refimg_and_video_2 = einops.rearrange(refimg_and_video_2, "n t c h w -> n c t h w")
                video_1 = einops.rearrange(video_1, "n t c h w -> n c t h w")
                LF_video, _ = freq_3d_filter(video_1, freq_filter)
                _,HF_refimg_and_video = freq_3d_filter(refimg_and_video_2, high_freq_filter)
                refimg_and_video_2 = einops.rearrange(refimg_and_video_2, "n c t h w -> n t c h w")
                LF_video = einops.rearrange(LF_video, "n c t h w -> n t c h w")

            # LF_refimg_and_video = einops.rearrange(LF_refimg_and_video, "n c t h w -> n t c h w")
            HF_refimg_and_video = einops.rearrange(HF_refimg_and_video, "n c t h w -> n t c h w")

            # gt_LF_video = LF_refimg_and_video[:,t:] # 灰度图作为GT
            
            # LF_video = gt_LF_video

            if self.use_camera:
                if self.use_camera_down:
                    LF_video = einops.rearrange(LF_video, "n t c h w -> (n t) c h w")
                    LF_video = self.camera_down(LF_video)
                    LF_video = einops.rearrange(LF_video, "(n t) c h w -> n t c h w",n=n)

                camera_target_motion = self.camera_motion_encoder(LF_video,camera_mask_ratio) # video (n,t,l,d) 

                # if self.camera_motion_token_channel != self.motion_token_channel:
                #     camera_target_motion = self.camera_motion_map(camera_target_motion)

            if self.use_object:
                object_motion = self.object_motion_encoder(refimg_and_video_2,object_mask_ratio) # (n,t+t,l,d)
                # object motion encoder
                object_source_motion = object_motion[:,:t].flatten(0,1) # (NT,motion_token,d)
                object_target_motion = object_motion[:,t:].flatten(0,1) # (NT,motion_token,d)
                assert object_source_motion.shape == object_target_motion.shape , f'object_source_motion.shape {object_source_motion.shape} != object_target_motion.shape {object_target_motion.shape}'

                # if self.object_motion_token_channel != self.motion_token_channel:
                #     object_source_motion = self.object_motion_map(object_source_motion)
                #     object_target_motion = self.object_motion_map(object_target_motion)

        else:
            if self.use_camera_down:
                LF_refimg_and_video = einops.rearrange(LF_refimg_and_video, "n t c h w -> (n t) c h w")
                LF_refimg_and_video = self.camera_down(LF_refimg_and_video)
                LF_refimg_and_video = einops.rearrange(LF_refimg_and_video, "(n t) c h w -> n t c h w",n=n)
            # object_motion = self.object_motion_encoder(refimg_and_video,mask_ratio) # (n,t+t,l,d)
            LF_video = LF_refimg_and_video[:,t:]
            camera_target_motion = self.camera_motion_encoder(LF_video,camera_mask_ratio) # (n,t+t,l,d)



        # zi = ref_img[:,0]  # (N,C,H,W)
        # zj = video.flatten(0,1) # (N,T,C,H,W)
        # zi = ref_img_grey[:,0] # (N,T,C,H,W)
        # zi = ref_img_grey[:,0:2] # (N,T,C,H,W)
        # zj = gt_LF_video.flatten(0,1) # (NT,C,H,W)
        
        # zi = ref_img_grey.flatten(0,1) # (NT,C,H,W)
        zi = ref_img.flatten(0,1) # (NT,C,H,W)
        zj = video_2.flatten(0,1) # (N,T,C,H,W)
        # zj = gt_LF_video.flatten(0,1) # (NT,C,H,W)
        # zj = gt_LF_video_RGB.flatten(0,1)


        # prepare for Diffusion Transformer
        # time_step = torch.ones((zj.shape[0],)).to(device) 
        time_step = torch.ones((zj.shape[0],)).to(device) # 补充第一帧
        time_step = time_step * start_step
        

        zt,vel = self.scheduler.get_train_tuple(z1=zj,time_step=time_step)  # (NT,C,H,W),(NT,C,H,W)
        
        # Sample Loop
        pre_cache = []
        sample_cache = []
        
        # 1.step_seq
        step_seq = np.linspace(0, start_step, num=sample_step+1, endpoint=True,dtype=int) # [0,5,10,15,....,start_step]
        step_seq = list(reversed(step_seq[1:])) # delete step:0  [start_step,.....,15,10,5]

        # 2.Euler step
        dt = 1./sample_step

        for i in tqdm(step_seq):

            # time_step
            # print("zt.shape",zt.shape)
            time_step = torch.ones((zt.shape[0],)).to(zt.device)  
            time_step = time_step * i
            
            # input
            # zt = zt.to(video.dtype)
            # zt = einops.rearrange(zt,'(n t) c h w -> n t c h w',n=n)
            # z1 = (zt[:,0] + zi).unsqueeze(1)
            # # z1 = (zt[:,0:2] + zi)
            # zt = torch.cat((z1,zt[:,1:]),dim=1).flatten(0,1) # (b,C,H,W)
            # # zt = torch.cat((z1,zt[:,2:]),dim=1).flatten(0,1)
            # image_hidden_states = zt

            image_hidden_states = torch.cat((zi,zt),dim=1) # (b,2C,H,W)

            pre = self.diffusion_transformer(camera_motion_target_hidden_states = camera_target_motion,
                                            image_hidden_states = image_hidden_states,
                                            timestep = time_step,)
            # if self.use_object and not self.use_camera:
            #     pre = self.diffusion_transformer(object_motion_source_hidden_states = object_source_motion,
            #                                     object_motion_target_hidden_states = object_target_motion,
            #                                     image_hidden_states = image_hidden_states,
            #                                     timestep = time_step,)
            # elif not self.use_object and self.use_camera:
            #     pre = self.diffusion_transformer(camera_motion_target_hidden_states = camera_target_motion,
            #                                     image_hidden_states = image_hidden_states,
            #                                     timestep = time_step,)
            # else:
            #     pre = self.diffusion_transformer(object_motion_source_hidden_states = object_source_motion,
            #                                     object_motion_target_hidden_states = object_target_motion,
            #                                     camera_motion_target_hidden_states = camera_target_motion,
            #                                     image_hidden_states = image_hidden_states,
            #                                     timestep = time_step,)


            zt = zt + pre * dt
            pre_cache.append(pre)
            sample_cache.append(zt)
        
        zi = einops.rearrange(zi,'(n t) c h w -> n t c h w',n=n)
        zt = einops.rearrange(zt,'(n t) c h w -> n t c h w',n=n)
        zj = einops.rearrange(zj,'(n t) c h w -> n t c h w',n=n)
        
        if return_meta_info:
            return {'zi' : zi, # (b,1,c,h,w)
                    'zj' : zj,             # (b,1,c,h,w)
                    'sample' : zt,           # (b,1,c,h,w)
                    'pre_cache' : pre_cache,           # [(b,c,h,w),....]
                    'sample_cache' : sample_cache,   # [(b,c,h,w),....]
                    'step_seq' : step_seq,
                    }
        else:
            return zi,zt,zj # (n,t,c,h,w)


    @torch.no_grad()
    def encode(self,video:torch.Tensor,ref_img:torch.Tensor,video_grey=None,ref_img_grey=None,camera_mask_ratio=None,object_mask_ratio=None):
        device = video.device
        n,t,c,h,w = video.shape

        # motion encoder
        if object_mask_ratio is not None:
            # print(f'* Sampling with object Mask_Ratio = {object_mask_ratio}')
            object_mask_ratio =  object_mask_ratio
        if camera_mask_ratio is not None:
            # print(f'* Sampling with camera Mask_Ratio = {camera_mask_ratio}')
            camera_mask_ratio =  camera_mask_ratio
            
        refimg_and_video = torch.cat([ref_img,video],dim=1)# (n,t+t,C,H,W)
        

        # frequency filter
        if self.use_filter:
            freq_filter = gaussian_low_pass_filter([2*t, h, w],0.6,0.6)
            freq_filter = freq_filter.unsqueeze(0).unsqueeze(0).repeat(n, c, 1, 1, 1)
            freq_filter = freq_filter.to(device=refimg_and_video.device)

            high_freq_filter = gaussian_low_pass_filter([2*t, h, w],0.6,0.6)
            high_freq_filter = high_freq_filter.unsqueeze(0).unsqueeze(0).repeat(n, c, 1, 1, 1)
            high_freq_filter = high_freq_filter.to(device=refimg_and_video.device)            

            if self.use_grey:
                refimg_and_video_grey = torch.cat([ref_img_grey,video_grey],dim=1)# (n,t+t,C,H,W) (1,32,4,32,32)
                refimg_and_video_grey = einops.rearrange(refimg_and_video_grey, "n t c h w -> n c t h w")
                LF_refimg_and_video, _ = freq_3d_filter(refimg_and_video_grey, freq_filter)
                _,HF_refimg_and_video = freq_3d_filter(refimg_and_video_grey, high_freq_filter)

                refimg_and_video = einops.rearrange(refimg_and_video, "n t c h w -> n c t h w")
                LF_refimg_and_video_RGB, _ = freq_3d_filter(refimg_and_video, freq_filter)
                refimg_and_video = einops.rearrange(refimg_and_video, "n c t h w -> n t c h w")
                LF_refimg_and_video_RGB = einops.rearrange(LF_refimg_and_video_RGB, "n c t h w -> n t c h w")
                gt_LF_video_RGB = LF_refimg_and_video_RGB[:,t:]
            else:
                refimg_and_video = einops.rearrange(refimg_and_video, "n t c h w -> n c t h w")
                LF_refimg_and_video, _ = freq_3d_filter(refimg_and_video, freq_filter)
                _,HF_refimg_and_video = freq_3d_filter(refimg_and_video, high_freq_filter)
                refimg_and_video = einops.rearrange(refimg_and_video, "n c t h w -> n t c h w")

            LF_refimg_and_video = einops.rearrange(LF_refimg_and_video, "n c t h w -> n t c h w")
            HF_refimg_and_video = einops.rearrange(HF_refimg_and_video, "n c t h w -> n t c h w")

            gt_LF_video = LF_refimg_and_video[:,t:] # 灰度图作为GT
            LF_video = gt_LF_video

            if self.use_camera:
                if self.use_camera_down:
                    LF_video = einops.rearrange(LF_video, "n t c h w -> (n t) c h w")
                    LF_video = self.camera_down(LF_video)
                    LF_video = einops.rearrange(LF_video, "(n t) c h w -> n t c h w",n=n)

                camera_target_motion = self.camera_motion_encoder(LF_video,camera_mask_ratio) # video (n,t,l,d) 

                # if self.camera_motion_token_channel != self.motion_token_channel:
                #     camera_target_motion = self.camera_motion_map(camera_target_motion)

            if self.use_object:
                object_motion = self.object_motion_encoder(refimg_and_video,object_mask_ratio) # (n,t+t,l,d)
                # object motion encoder
                object_source_motion = object_motion[:,:t].flatten(0,1) # (NT,motion_token,d)
                object_target_motion = object_motion[:,t:].flatten(0,1) # (NT,motion_token,d)
                assert object_source_motion.shape == object_target_motion.shape , f'object_source_motion.shape {object_source_motion.shape} != object_target_motion.shape {object_target_motion.shape}'

                # if self.object_motion_token_channel != self.motion_token_channel:
                #     object_source_motion = self.object_motion_map(object_source_motion)
                #     object_target_motion = self.object_motion_map(object_target_motion)

        else:
            if self.use_camera_down:
                LF_refimg_and_video = einops.rearrange(LF_refimg_and_video, "n t c h w -> (n t) c h w")
                LF_refimg_and_video = self.camera_down(LF_refimg_and_video)
                LF_refimg_and_video = einops.rearrange(LF_refimg_and_video, "(n t) c h w -> n t c h w",n=n)
            # object_motion = self.object_motion_encoder(refimg_and_video,mask_ratio) # (n,t+t,l,d)
            LF_video = LF_refimg_and_video[:,t:]
            camera_target_motion = self.camera_motion_encoder(LF_video,camera_mask_ratio) # (n,t+t,l,d)
        
        # print(camera_target_motion.shape)
        # print(object_source_motion.shape)
        # print(object_target_motion.shape)
        
        return camera_target_motion, object_source_motion, object_target_motion
        
    def decode(self, video:torch.Tensor,ref_img:torch.Tensor, camera_target_motion, object_source_motion, object_target_motion,time_step:torch.tensor = None,start_step:int = None,sample_step:int = 50):
        device = video.device
        n,t,c,h,w = video.shape
        zi = ref_img.flatten(0,1) # (NT,C,H,W)
        zj = video.flatten(0,1) # (N,T,C,H,W)
        # zj = gt_LF_video.flatten(0,1) # (NT,C,H,W)
        # zj = gt_LF_video_RGB.flatten(0,1)


        # prepare for Diffusion Transformer
        # time_step = torch.ones((zj.shape[0],)).to(device) 
        time_step = torch.ones((zj.shape[0],)).to(device) # 补充第一帧
        time_step = time_step * start_step
        

        zt,vel = self.scheduler.get_train_tuple(z1=zj,time_step=time_step)  # (NT,C,H,W),(NT,C,H,W)
        
        # Sample Loop
        pre_cache = []
        sample_cache = []
        
        # 1.step_seq
        step_seq = np.linspace(0, start_step, num=sample_step+1, endpoint=True,dtype=int) # [0,5,10,15,....,start_step]
        step_seq = list(reversed(step_seq[1:])) # delete step:0  [start_step,.....,15,10,5]

        # 2.Euler step
        dt = 1./sample_step

        for i in tqdm(step_seq):

            # time_step
            # print("zt.shape",zt.shape)
            time_step = torch.ones((zt.shape[0],)).to(zt.device)  
            time_step = time_step * i
            
            # input
            # zt = zt.to(video.dtype)
            # zt = einops.rearrange(zt,'(n t) c h w -> n t c h w',n=n)
            # z1 = (zt[:,0] + zi).unsqueeze(1)
            # # z1 = (zt[:,0:2] + zi)
            # zt = torch.cat((z1,zt[:,1:]),dim=1).flatten(0,1) # (b,C,H,W)
            # # zt = torch.cat((z1,zt[:,2:]),dim=1).flatten(0,1)
            # image_hidden_states = zt

            image_hidden_states = torch.cat((zi,zt),dim=1) # (b,2C,H,W)

            
            if self.use_object and not self.use_camera:
                pre = self.diffusion_transformer(object_motion_source_hidden_states = object_source_motion,
                                                object_motion_target_hidden_states = object_target_motion,
                                                image_hidden_states = image_hidden_states,
                                                timestep = time_step,)
            elif not self.use_object and self.use_camera:
                pre = self.diffusion_transformer(camera_motion_target_hidden_states = camera_target_motion,
                                                image_hidden_states = image_hidden_states,
                                                timestep = time_step,)
            else:
                pre = self.diffusion_transformer(object_motion_source_hidden_states = object_source_motion,
                                                object_motion_target_hidden_states = object_target_motion,
                                                camera_motion_target_hidden_states = camera_target_motion,
                                                image_hidden_states = image_hidden_states,
                                                timestep = time_step,)


            zt = zt + pre * dt
            pre_cache.append(pre)
            sample_cache.append(zt)
        
        zi = einops.rearrange(zi,'(n t) c h w -> n t c h w',n=n)
        zt = einops.rearrange(zt,'(n t) c h w -> n t c h w',n=n)
        zj = einops.rearrange(zj,'(n t) c h w -> n t c h w',n=n)
            
        return zi,zt,zj # (n,t,c,h,w)

    @torch.no_grad()
    def sample_diff_motion(self,video:torch.Tensor,ref_img:torch.Tensor,video_grey=None,ref_img_grey=None,camera_video_grey=None,sample_step:int = 50,mask_ratio = None,start_step:int = None,return_meta_info=False):

        device = video.device
        n,t,c,h,w = video.shape

        if start_step is None:
            start_step = self.scheduler.num_step
        assert start_step <= self.scheduler.num_step , 'start_step cant be larger than scheduler.num_step'

        # motion encoder
        if mask_ratio is not None:
            print(f'* Sampling with Mask_Ratio = {mask_ratio}')
            mask_ratio =  mask_ratio
        
        refimg_and_video = torch.cat([ref_img,video],dim=1)# (n,t+t,C,H,W) (1,32,4,32,32)
        
        # if not self.use_grey:
        #     ref_img_grey = torch.zeros_like
        # refimg_and_video_grey = torch.cat([ref_img_grey,video_grey],dim=1)# (n,t+t,C,H,W) (1,32,4,32,32)

        if self.use_filter:
            freq_filter = gaussian_low_pass_filter([2*t, h, w],d_s=0.4,d_t=0.4)
            freq_filter = freq_filter.unsqueeze(0).unsqueeze(0).repeat(n, c, 1, 1, 1)
            freq_filter = freq_filter.to(device=refimg_and_video.device)

            if self.use_grey:
                refimg_and_video_grey = torch.cat([ref_img_grey,video_grey],dim=1)# (n,t+t,C,H,W) (1,32,4,32,32)
                refimg_and_video_grey = einops.rearrange(refimg_and_video_grey, "n t c h w -> n c t h w")
                refimg_and_camera_video_grey = torch.cat([ref_img,camera_video_grey],dim=1)# (n,t+t,C,H,W) (1,32,4,32,32)
                refimg_and_camera_video_grey = einops.rearrange(refimg_and_camera_video_grey, "n t c h w -> n c t h w")

                _, HF_refimg_and_video = freq_3d_filter(refimg_and_video_grey, freq_filter)
                LF_refimg_and_video, _ = freq_3d_filter(refimg_and_camera_video_grey, freq_filter)


            else:
                refimg_and_video = einops.rearrange(refimg_and_video, "n t c h w -> n c t h w")
                LF_refimg_and_video, HF_refimg_and_video = freq_3d_filter(refimg_and_video, freq_filter)

            LF_refimg_and_video = einops.rearrange(LF_refimg_and_video, "n c t h w -> n t c h w")
            HF_refimg_and_video = einops.rearrange(HF_refimg_and_video, "n c t h w -> n t c h w")

            if self.use_camera_down:
                LF_refimg_and_video = einops.rearrange(LF_refimg_and_video, "n t c h w -> (n t) c h w")
                LF_refimg_and_video = self.camera_down(LF_refimg_and_video)
                LF_refimg_and_video = einops.rearrange(LF_refimg_and_video, "(n t) c h w -> n t c h w",n=n)

            object_motion = self.object_motion_encoder(HF_refimg_and_video,mask_ratio) # (n,t+t,l,d)
            camera_motion = self.camera_motion_encoder(LF_refimg_and_video,mask_ratio) # (n,t+t,l,d)

        else:
            if self.use_camera_down:
                LF_refimg_and_video = einops.rearrange(LF_refimg_and_video, "n t c h w -> (n t) c h w")
                LF_refimg_and_video = self.camera_down(LF_refimg_and_video)
                LF_refimg_and_video = einops.rearrange(LF_refimg_and_video, "(n t) c h w -> n t c h w",n=n)
            object_motion = self.object_motion_encoder(refimg_and_video,mask_ratio) # (n,t+t,l,d)
            camera_motion = self.camera_motion_encoder(refimg_and_video,mask_ratio) # (n,t+t,l,d)

        # object motion encoder
        object_source_motion = object_motion[:,:t].flatten(0,1) # (NT,motion_token,d)
        object_target_motion = object_motion[:,t:].flatten(0,1) # (NT,motion_token,d)
        assert object_source_motion.shape == object_target_motion.shape , f'object_source_motion.shape {object_source_motion.shape} != object_target_motion.shape {object_target_motion.shape}'

        # camera motion encoder
        camera_source_motion = camera_motion[:,:t].flatten(0,1) # (NT,motion_token,d)
        camera_target_motion = camera_motion[:,t:].flatten(0,1) # (NT,motion_token,d)
        assert camera_source_motion.shape == camera_target_motion.shape , f'camera_source_motion.shape {camera_source_motion.shape} != camera_target_motion.shape {camera_target_motion.shape}'

         # KL Loss
        if self.use_regularizers:
            object_target_motion = einops.rearrange(object_target_motion, "n t d -> n d t")
            camera_target_motion = einops.rearrange(camera_target_motion, "n t d -> n d t")
            object_target_motion, object_target_KLloss = self.regularization(object_target_motion)
            camera_target_motion, camera_target_KLloss = self.regularization(camera_target_motion)
            object_target_motion = einops.rearrange(object_target_motion, "n d t -> n t d")
            camera_target_motion = einops.rearrange(camera_target_motion, "n d t -> n t d")

        # motion fusion
        # camera motion average
        if self.use_regularizers:
            camera_source_motion = self.camera_source_motion_map(camera_source_motion)
            camera_target_motion = self.camera_target_motion_map(camera_target_motion)

            object_source_motion = self.object_source_motion_map(object_source_motion)
            object_target_motion = self.object_target_motion_map(object_target_motion)
        else:
            if self.camera_motion_token_channel != self.motion_token_channel:
                camera_source_motion = self.camera_motion_map(camera_source_motion)
                camera_target_motion = self.camera_motion_map(camera_target_motion)
            if self.object_motion_token_channel != self.motion_token_channel:
                object_source_motion = self.object_motion_map(object_source_motion)
                object_target_motion = self.object_motion_map(object_target_motion)

        # if self.use_motion == 'camera':
        #     source_motion = camera_source_motion # (NT,motion_token,d)
        #     target_motion = camera_target_motion # (NT,motion_token,d)
        # elif self.use_motion == 'object':
        #     source_motion = object_source_motion # (NT,motion_token,d)
        #     target_motion = object_target_motion # (NT,motion_token,d)
        # else:
        source_motion = object_source_motion + camera_source_motion # (NT,motion_token,d)
        target_motion = object_target_motion + camera_target_motion # (NT,motion_token,d)

        # motion transformer
        if self.need_motion_transformer:
            target_motion = einops.rearrange(target_motion,'(n f) l d -> n f l d',n=n)
            target_motion = self.motion_transformer(target_motion)
            target_motion = einops.rearrange(target_motion,'n f l d -> (n f) l d',n=n)
        
        # prepare for Diffusion Transformer
        time_step = torch.ones((source_motion.shape[0],)).to(device) 
        time_step = time_step * start_step

        zi = ref_img.flatten(0,1) # (NT,C,H,W)
        zj = video.flatten(0,1) # (NT,C,H,W)
        zt,vel = self.scheduler.get_train_tuple(z1=zj,time_step=time_step)  # (NT,C,H,W),(NT,C,H,W)
        
        # Sample Loop
        pre_cache = []
        sample_cache = []
        
        # 1.step_seq
        step_seq = np.linspace(0, start_step, num=sample_step+1, endpoint=True,dtype=int) # [0,5,10,15,....,start_step]
        step_seq = list(reversed(step_seq[1:])) # delete step:0  [start_step,.....,15,10,5]

        # 2.Euler step
        dt = 1./sample_step

        for i in tqdm(step_seq):
            # time_step
            time_step = torch.ones((zt.shape[0],)).to(zt.device)  
            time_step = time_step * i
            
            # input
            zt = zt.to(video.dtype)
            image_hidden_states = torch.cat((zi,zt),dim=1) # (b,2C,H,W)
            
            # forward
            pre = self.diffusion_transformer(motion_source_hidden_states = source_motion,
                                            motion_target_hidden_states = target_motion,
                                            image_hidden_states = image_hidden_states,
                                            timestep = time_step,) 
            zt = zt + pre * dt
            pre_cache.append(pre)
            sample_cache.append(zt)
        
        # zi = einops.rearrange(zi,'(n t) c h w -> n t c h w',n=n)
        zt = einops.rearrange(zt,'(n t) c h w -> n t c h w',n=n)
        zj = einops.rearrange(zj,'(n t) c h w -> n t c h w',n=n)
        
        if return_meta_info:
            return {'zi' : zi, # (b,1,c,h,w)
                    'zj' : zj,             # (b,1,c,h,w)
                    'sample' : zt,           # (b,1,c,h,w)
                    'pre_cache' : pre_cache,           # [(b,c,h,w),....]
                    'sample_cache' : sample_cache,   # [(b,c,h,w),....]
                    'step_seq' : step_seq,
                    'motion' : target_motion, # (b,C,H,W)
                    }
        else:
            return zi,zt,zj # (n,t,c,h,w)

    def sample_with_refimg_motion(self,ref_img:torch.Tensor,camera_target_motion=torch.Tensor,object_target_motion=torch.Tensor,sample_step:int = 10,return_meta_info=False,ref_img_grey=None,):
        """
        Args:
            ref_img : (N,C,H,W)                             
            motion : (N,F,L,D)                           
        Return:
            video : (N,T,C,H,W)                             
        """
        device = camera_target_motion.device
        n,t,c,h,w = ref_img.shape
        start_step = self.scheduler.num_step
        
        ref_img_grey = einops.rearrange(ref_img_grey, "n t c h w -> n c t h w")
        high_freq_filter = gaussian_low_pass_filter([t, h, w],0.6,0.6)
        high_freq_filter = high_freq_filter.unsqueeze(0).unsqueeze(0).repeat(n, c, 1, 1, 1)
        high_freq_filter = high_freq_filter.to(device=ref_img.device)
        # print("ref_img_grey.shape",ref_img_grey.shape)
        # print("high_freq_filter.shape",high_freq_filter.shape)
        _,HF_refimg = freq_3d_filter(ref_img_grey, high_freq_filter)
        HF_refimg = einops.rearrange(HF_refimg, "n c t h w -> n t c h w")

        # motion encoder
        object_source_motion = self.object_motion_encoder(HF_refimg).flatten(0,1) # (n,1,motion_token,d)

        assert object_source_motion.shape == object_target_motion.shape , f'object_source_motion.shape {object_source_motion.shape} != object_target_motion.shape {object_target_motion.shape}'

        # motion transformer
        if self.need_motion_transformer and not self.extract_motion_with_motion_transformer:
            target_motion = einops.rearrange(target_motion,'(n f) l d -> n f l d',n=n)
            target_motion = self.motion_transformer(target_motion)
            target_motion = einops.rearrange(target_motion,'n f l d -> (n f) l d',n=n)
        
        # prepare for Diffusion Transformer
        time_step = torch.ones((object_source_motion.shape[0],)).to(device) 
        time_step = time_step * start_step

        zi = ref_img.flatten(0,1) # (NT,C,H,W)
        zj = zi
        zt,vel = self.scheduler.get_train_tuple(z1=zj,time_step=time_step)  # (NT,C,H,W),(NT,C,H,W)
        
        # Sample Loop
        pre_cache = []
        sample_cache = []
        
        # 1.step_seq
        step_seq = np.linspace(0, start_step, num=sample_step+1, endpoint=True,dtype=int) # [0,5,10,15,....,start_step]
        step_seq = list(reversed(step_seq[1:])) # delete step:0  [start_step,.....,15,10,5]

        # 2.Euler step
        dt = 1./sample_step

        for i in tqdm(step_seq):
            # time_step
            time_step = torch.ones((zt.shape[0],)).to(zt.device)  
            time_step = time_step * i
            
            # input
            zt = zt.to(ref_img.dtype)
            image_hidden_states = torch.cat((zi,zt),dim=1).to(device)  # (b,2C,H,W)
            
            # forward
            pre = self.diffusion_transformer(object_motion_source_hidden_states = object_source_motion,
                                                object_motion_target_hidden_states = object_target_motion,
                                                camera_motion_target_hidden_states = camera_target_motion,
                                                image_hidden_states = image_hidden_states,
                                                timestep = time_step,)
            zt = zt + pre * dt
            pre_cache.append(pre)
            sample_cache.append(zt)
        
        # unsqueeze (n,1,c,h,w) means images, (n,t,c,h,w) means video t>1 .
        zi = einops.rearrange(zi,'(n t) c h w -> n t c h w',n=n,t=t)
        zt = einops.rearrange(zt,'(n t) c h w -> n t c h w',n=n,t=t)
        
        if return_meta_info:
            return {'zi' : zi, # (b,1,c,h,w)
                    'zj' : zj,             # (b,1,c,h,w)
                    'sample' : zt,           # (b,1,c,h,w)
                    'pre_cache' : pre_cache,           # [(b,c,h,w),....]
                    'sample_cache' : sample_cache,   # [(b,c,h,w),....]
                    'step_seq' : step_seq,
                    'motion' : target_motion, # (b,C,H,W)
                    }
        else:
            return zi,zt,zj # (b,1,c,h,w)
    
    def extract_motion(self,video:torch.tensor):
        # video : (N,T,C,H,W)
        n,t,c,h,w = video.shape
        
        motion = self.motion_encoder(video) # (N,T,L,D)

        if self.need_motion_transformer and self.extract_motion_with_motion_transformer:
            motion = self.motion_transformer(motion) # (N,T,L,D)

        return motion
     
    def prepare_timestep(self,batch_size:int,device,time_step = None):
        if time_step is not None:
            return time_step.to(device)
        else:
            return torch.randint(0,self.num_step+1,(batch_size,)).to(device)
  
    def prepare_encoder_input(self,video:torch.tensor):
        assert len(video.shape) == 5 , f'only support video data : 5D tensor , but got {video.shape}'
        
        # cat
        pre = video[:,:-1,:,:,:] 
        post= video[:,1:,:,:,:]
        duo_frame_mix = torch.cat([pre,post],dim=2)    # (b,t-1,2c,h,w)
        duo_frame_mix = einops.rearrange(duo_frame_mix,'b t c h w -> (b t) c h w')
        
        return duo_frame_mix # (b*f-1,2c,h,w)


    def unpatchify(self, x ,patch_size):
        """
        x: (N, S, patch_size**2 *C)
        imgs: (N, C, H, W)
        """
        p = patch_size
        h = w = int(x.shape[1]**.5)
        # c = self.in_chans
        c = x.shape[2] // (p**2)
        assert h * w == x.shape[1]
        
        x = x.reshape(shape=(x.shape[0], h, w, p, p, c)) # (N, h, w, p, p, c)
        x = torch.einsum('nhwpqc->nchpwq', x) # (N, c, h, p, w, p)
        imgs = x.reshape(shape=(x.shape[0], c, h * p, h * p))
        return imgs #(N,C,H,W)    

    def reset_infer_num_frame(self, num:int):
        old_num = self.diffusion_transformer.target_frame
        self.diffusion_transformer.target_frame = num
        if self.use_motiontemporal:
            self.camera_motion_encoder.video_frames = num
            self.object_motion_encoder.video_frames = num
        print(f'* Reset infer frame from {old_num} to {self.diffusion_transformer.target_frame} *')

class AMDModel_Rec(ModelMixin, ConfigMixin):
    _supports_gradient_checkpointing = True
    
    @register_to_config
    def __init__(self,
                 image_inchannel :int  = 4,
                 image_height  :int = 32,
                 image_width  :int  = 32,
                 video_frames  :int = 16,
                 scheduler_num_step :int = 1000,

                 # -----------camera MotionEncoder -----------
                 motion_token_num:int = 12,
                 motion_token_channel: int = 128,
                 enc_num_layers:int = 8,
                 enc_nhead:int = 8,
                 enc_ndim:int = 64,
                 enc_dropout:float = 0.0,
                 motion_need_norm_out:bool = True,

                 # ----------- MotionTransformer  目前的版本用不到---------
                 need_motion_transformer :bool = False,
                 motion_transformer_attn_head_dim:int = 64,
                 motion_transformer_attn_num_heads:int = 16,
                 motion_transformer_num_layers:int = 4,

                 # ----------- Diffusion Transformer -----------
                 diffusion_attn_head_dim : int  = 64,
                 diffusion_attn_num_heads : int = 16,
                 diffusion_out_channels : int = 4,
                 diffusion_num_layers : int = 16,
                 image_patch_size : int  = 2,
                 motion_patch_size : int = 1,
                 motion_drop_ratio: float = 0.0,
                 **kwargs,
                 ):
        super().__init__()

        # setting
        self.num_step = scheduler_num_step
        self.scheduler = RectifiedFlow(num_steps=scheduler_num_step)
        self.need_motion_transformer = need_motion_transformer

        # zt token
        INIT_CONST = 0.02
        self.zt_token = nn.Parameter(torch.randn(1, image_inchannel, image_height,image_width) * INIT_CONST)

        # object_motion Encoder 
        self.object_motion_encoder = MotionEncoderLearnTokenTransformer(img_height = image_height,
                                                                img_width=image_width,
                                                                img_inchannel=image_inchannel,
                                                                img_patch_size = image_patch_size,
                                                                motion_token_num =  motion_token_num,
                                                                motion_channel = motion_token_channel,
                                                                need_norm_out = motion_need_norm_out,
                                                                # ----- attention
                                                                num_attention_heads=enc_nhead,
                                                                attention_head_dim=enc_ndim,
                                                                num_layers=enc_num_layers,      
                                                                dropout=enc_dropout,
                                                                attention_bias= True,)

        # camera_motion Encoder 
        self.camera_motion_encoder = MotionEncoderLearnTokenTransformer(img_height = image_height,
                                                                img_width=image_width,
                                                                img_inchannel=image_inchannel,
                                                                img_patch_size = image_patch_size,
                                                                motion_token_num =  motion_token_num,
                                                                motion_channel = motion_token_channel,
                                                                need_norm_out = motion_need_norm_out,
                                                                # ----- attention
                                                                num_attention_heads=enc_nhead,
                                                                attention_head_dim=enc_ndim,
                                                                num_layers=enc_num_layers,      
                                                                dropout=enc_dropout,
                                                                attention_bias= True,)

        # motion transformer
        if need_motion_transformer:
            self.motion_transformer = MotionTransformer(motion_token_num=motion_token_num,
                                                        motion_token_channel=motion_token_channel,
                                                        attention_head_dim=motion_transformer_attn_head_dim,
                                                        num_attention_heads=motion_transformer_attn_num_heads,
                                                        num_layers=motion_transformer_num_layers,)
        
        # diffusion transformer
        dit_image_inchannel = image_inchannel * 2 # zi + zt
        self.transformer = AMDReconstructTransformerModel(num_attention_heads= diffusion_attn_num_heads,
                                                                attention_head_dim= diffusion_attn_head_dim,
                                                                out_channels = diffusion_out_channels,
                                                                num_layers= diffusion_num_layers,
                                                                # ----- img
                                                                image_width= image_width,
                                                                image_height= image_height,
                                                                image_patch_size= image_patch_size,
                                                                image_in_channels = dit_image_inchannel, 
                                                                # ----- motion
                                                                motion_token_num = motion_token_num,
                                                                motion_in_channels = motion_token_channel,)

    def forward(self,video:torch.tensor,ref_img:torch.Tensor ,time_step:torch.tensor = None,return_meta_info=False):
        """
        Args:
            video: (N,T,C,H,W)
            ref_img: (N,T,C,H,W)
        """

        device = video.device
        n,t,c,h,w = video.shape

        assert video.shape == ref_img.shape ,f'video.shape:{video.shape}should be equal to ref_img.shape:{ref_img.shape}'
        
        # motion encoder
        refimg_and_video = torch.cat([ref_img,video],dim=1)# (n,t+t,C,H,W)
        # object motion encoder
        object_motion = self.object_motion_encoder(refimg_and_video) # (n,t+t,l,d)
        object_source_motion = object_motion[:,:t].flatten(0,1) # (NT,motion_token,d)
        object_target_motion = object_motion[:,t:].flatten(0,1) # (NT,motion_token,d)
        assert object_source_motion.shape == object_target_motion.shape , f'object_source_motion.shape {object_source_motion.shape} != object_target_motion.shape {object_target_motion.shape}'

        # camera motion encoder
        camera_motion = self.camera_motion_encoder(refimg_and_video) # (n,t+t,l,d)
        camera_source_motion = camera_motion[:,:t].flatten(0,1) # (NT,motion_token,d)
        camera_target_motion = camera_motion[:,t:].flatten(0,1) # (NT,motion_token,d)
        assert camera_source_motion.shape == camera_target_motion.shape , f'camera_source_motion.shape {camera_source_motion.shape} != camera_target_motion.shape {camera_target_motion.shape}'


        # motion transformer
        if self.need_motion_transformer:
            target_motion = einops.rearrange(target_motion,'(n f) l d -> n f l d',n=n)
            target_motion = self.motion_transformer(target_motion)
            target_motion = einops.rearrange(target_motion,'n f l d -> (n f) l d',n=n)

        
        # prepare for Diffusion Transformer
        zi = ref_img.flatten(0,1) # (NT,C,H,W)
        zj = video.flatten(0,1) # (NT,C,H,W)
        zt = self.zt_token.repeat(zj.shape[0],1,1,1) # (NT,C,H,W)
        
        # motion fusion
        source_motion = object_source_motion + camera_source_motion
        target_motion = object_target_motion + camera_target_motion

        # dit forward
        image_hidden_states = torch.cat((zi,zt),dim=1) # (b,2C,H,W)
        pre = self.transformer(motion_source_hidden_states = source_motion,
                                        motion_target_hidden_states = target_motion,
                                        image_hidden_states = image_hidden_states,) 

        # loss
        rec_loss = l2(pre,zj)

        loss = rec_loss

        loss_dict = {'loss':loss,'rec_loss':rec_loss}
    
        if return_meta_info:
            return {'camera_motion' : camera_motion,               # (,t,motion_out_channels,h,w) , output of camera motion encoder 
                    'object_motion' : object_motion,               # (,t,motion_out_channels,h,w) , output of object motion encoder 
                    'zi' : zi,                       # (b,C,H,W) | b = n * t
                    'zj' : zj,                       # (b,C,H,W)
                    'zt' : zt,                       # (b,C,H,W)
                    'pre': pre,                      # (b,C,H,W)
                    'time_step': time_step,          # (b,)
                    }
        else:
            return pre,zj,loss_dict  # (b,C,H,W)

    @torch.no_grad()
    def sample(self,video:torch.tensor,ref_img:torch.Tensor ,sample_step:int = 50,start_step:int = None,return_meta_info=False):

        device = video.device
        n,t,c,h,w = video.shape

        if start_step is None:
            start_step = self.scheduler.num_step
        assert start_step <= self.scheduler.num_step , 'start_step cant be larger than scheduler.num_step'

        # motion encoder
        refimg_and_video = torch.cat([ref_img,video],dim=1)# (n,t+t,C,H,W)
        motion = self.motion_encoder(refimg_and_video) # (n,t+t,motion_out_channels,h,w)

        source_motion = motion[:,:t].flatten(0,1) # (NT,motion_token,d)
        target_motion = motion[:,t:].flatten(0,1) # (NT,motion_token,d)

        assert source_motion.shape == target_motion.shape , f'source_motion.shape {source_motion.shape} != target_motion.shape {target_motion.shape}'

        # motion transformer
        if self.need_motion_transformer:
            target_motion = einops.rearrange(target_motion,'(n f) l d -> n f l d',n=n)
            target_motion = self.motion_transformer(target_motion)
            target_motion = einops.rearrange(target_motion,'n f l d -> (n f) l d',n=n)
        
        zi = ref_img.flatten(0,1) # (NT,C,H,W)
        zj = video.flatten(0,1) # (NT,C,H,W)
        zt = self.zt_token.repeat(zj.shape[0],1,1,1) # (NT,C,H,W)
        

        # input
        zt = zt.to(video.dtype)
        image_hidden_states = torch.cat((zi,zt),dim=1) # (b,2C,H,W)
        
        # forward
        pre = self.transformer(motion_source_hidden_states = source_motion,
                                        motion_target_hidden_states = target_motion,
                                        image_hidden_states = image_hidden_states,) 
        
        zi = einops.rearrange(zi,'(n t) c h w -> n t c h w',n=n)
        zt = einops.rearrange(pre,'(n t) c h w -> n t c h w',n=n)
        zj = einops.rearrange(zj,'(n t) c h w -> n t c h w',n=n)
        
        if return_meta_info:
            return {'zi' : zi, # (b,1,c,h,w)
                    'zj' : zj,             # (b,1,c,h,w)
                    }
        else:
            return zi,zt,zj # (n,t,c,h,w)

    def sample_with_refimg_motion(self,ref_img:torch.Tensor,motion=torch.Tensor,sample_step:int = 10,return_meta_info=False):
        """
        Args:
            ref_img : (N,C,H,W)                             
            motion : (N,F,L,D)                           
        Return:
            video : (N,T,C,H,W)                             
        """
        device = motion.device
        n,t,l,d = motion.shape

        start_step = self.scheduler.num_step
        
        # motion encoder
        refimg = ref_img.unsqueeze(1) # (N,1,C,H,W)
        source_motion = self.motion_encoder(refimg) # (n,1,motion_token,d)

        source_motion = source_motion.repeat(1,t,1,1).flatten(0,1) # (NT,l,d)
        target_motion = motion.flatten(0,1) # (NT,l,d)

        assert source_motion.shape == target_motion.shape , f'source_motion.shape {source_motion.shape} != target_motion.shape {target_motion.shape}'

        # motion transformer
        if self.need_motion_transformer:
            target_motion = einops.rearrange(target_motion,'(n f) l d -> n f l d',n=n)
            target_motion = self.motion_transformer(target_motion)
            target_motion = einops.rearrange(target_motion,'n f l d -> (n f) l d',n=n)
        
        # prepare for Diffusion Transformer
        time_step = torch.ones((source_motion.shape[0],)).to(device) 
        time_step = time_step * start_step

        zi = refimg.repeat(1,t,1,1,1).flatten(0,1) # (NT,C,H,W)
        zj = zi
        zt = self.zt_token.repeat(zj.shape[0],1,1,1) # (NT,C,H,W)
        
        # input
        zt = zt.to(zj.dtype)
        image_hidden_states = torch.cat((zi,zt),dim=1) # (b,2C,H,W)
        
        # forward
        pre = self.transformer(motion_source_hidden_states = source_motion,
                                        motion_target_hidden_states = target_motion,
                                        image_hidden_states = image_hidden_states,) 
        
        zi = einops.rearrange(zi,'(n t) c h w -> n t c h w',n=n)
        zt = einops.rearrange(pre,'(n t) c h w -> n t c h w',n=n)
        zj = einops.rearrange(zj,'(n t) c h w -> n t c h w',n=n)
        
        if return_meta_info:
            return {'zi' : zi, # (b,1,c,h,w)
                    'zj' : zj,             # (b,1,c,h,w)
                    }
        else:
            return zi,zt,zj # (b,1,c,h,w)
    
    def extract_motion(self,video:torch.tensor):
        # video : (N,T,C,H,W)

        # motion Encoder
        motion = self.motion_encoder(video) # (N,T,L,D)

        if self.need_motion_transformer:
            motion = self.motion_transformer(motion) # (N,T,L,D)

    
        return motion
     

def AMD_S(**kwargs) -> AMDModel:
    return AMDModel( 
                    # ----------- motion encoder -----------
                    enc_nhead = 8,
                    enc_ndim = 64,
                    # ----------- Diffusion Transformer -----------
                    diffusion_attn_head_dim  = 64,
                    diffusion_attn_num_heads = 16,
                    diffusion_out_channels = 4,
                    diffusion_num_layers = 12,
                    **kwargs)

def AMD_S_Camera(**kwargs) -> AMDModel:
    return AMDModel_Camera( 
                    # ----------- motion encoder -----------
                    enc_nhead = 8,
                    enc_ndim = 64,
                    # ----------- Diffusion Transformer -----------
                    diffusion_attn_head_dim  = 64,
                    diffusion_attn_num_heads = 16,
                    diffusion_out_channels = 4,
                    diffusion_num_layers = 12,
                    **kwargs)

def AMD_N(**kwargs) -> AMDModel:
    return AMDModel_New( 
                    # ----------- motion encoder -----------
                    enc_nhead = 8,
                    enc_ndim = 64,
                    # ----------- Diffusion Transformer -----------
                    diffusion_attn_head_dim  = 64,
                    diffusion_attn_num_heads = 16,
                    diffusion_out_channels = 4,
                    diffusion_num_layers = 12,
                    **kwargs)

def AMD_L(**kwargs) -> AMDModel:
    return AMDModel(   
                    # ----------- motion encoder -----------
                    enc_num_layers = 8,
                    enc_nhead = 16,
                    enc_ndim = 64,
                    # ----------- Diffusion Transformer -----------
                    diffusion_attn_head_dim  = 96,
                    diffusion_attn_num_heads = 16,
                    diffusion_out_channels = 4,
                    diffusion_num_layers = 16,
                    **kwargs)

def AMD_S_Rec(**kwargs) -> AMDModel:
    return AMDModel_Rec( 
                    # ----------- motion encoder -----------
                    enc_num_layers = 8,
                    enc_nhead = 8,
                    enc_ndim = 64,
                    # ----------- Diffusion Transformer -----------
                    diffusion_attn_head_dim  = 64,
                    diffusion_attn_num_heads = 16,
                    diffusion_out_channels = 4,
                    diffusion_num_layers = 12,
                    **kwargs)

def AMD_S_RecSplit(**kwargs) -> AMDModel:
    return AMDModel_Rec( 
                    # ----------- motion encoder -----------
                    enc_num_layers = 8,
                    enc_nhead = 8,
                    enc_ndim = 64,
                    # ----------- Diffusion Transformer -----------
                    diffusion_attn_head_dim  = 64,
                    diffusion_attn_num_heads = 16,
                    diffusion_out_channels = 4,
                    diffusion_num_layers = 12,
                    is_split = True,
                    **kwargs)


AMD_models = {
    "AMD_S": AMD_S,  # 250M
    "AMD_S_Camera": AMD_S_Camera,
    "AMD_N": AMD_N,
    "AMD_L": AMD_L,  # 700M
    "AMD_S_Rec": AMD_S_Rec,  # 250M
    "AMD_S_RecSplit" : AMD_S_RecSplit, # 250M
} # S 206 B 333  M 642 L 1053 