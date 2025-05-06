import torch
from torch import nn
import einops

from .modules import DuoFrameDownEncoder,Upsampler,MapConv
from .loss import l1,l2
import torch
from torch import nn
import einops
from typing import Tuple
import random
import numpy as np
from tqdm import tqdm
from .modules import DuoFrameDownEncoder,Upsampler,MapConv,MotionDownEncoder
from .loss import l1,l2
from .transformer import MotionTransformer,AMDDiffusionTransformerModel,AMDDiffusionTransformerModelSplitInput
from .rectified_flow import RectifiedFlow
from diffusers.configuration_utils import ConfigMixin, register_to_config
from diffusers.models.modeling_utils import ModelMixin
from diffusers.models.resnet import ResnetBlock2D

from diffusers.utils import export_to_gif


# m_t = CNN(z_{t-1},z_t) 
# z_t = F(z_{t-1},m_t)
class AMDModel1(nn.Module):
    def __init__(self,
                 inchannel : int = 4,
                 upsampler_outchannel : int = 4,
                 block_out_channels_down = (64,128,256,256),
                 ):
        super().__init__()

        # setting
        self.block_out_channels_down = block_out_channels_down
        self.block_out_channels_up = list(reversed(block_out_channels_down))
        self.latent_dim = block_out_channels_down[-1]

        # model
        self.dfd_encoder = DuoFrameDownEncoder(in_channel=inchannel*2,
                                               block_out_channels=self.block_out_channels_down)
        self.upsampler = Upsampler(in_channel=self.block_out_channels_down[-1],
                                   out_channel=upsampler_outchannel,
                                   block_out_channels=self.block_out_channels_up)
        mapconv_inchannel = inchannel + upsampler_outchannel
        self.mapconv = MapConv(in_channel= mapconv_inchannel,
                               out_channel=inchannel)

    def forward(self,video:torch.tensor):
        """
        Args:
            * video : torch.tensor (b,t,c,h,w)
        """
        
        # shift_video -> Frame [1,1,2,3,...,t-1]
        # video       -> Frame [1,2,3,4,...,t]
        # motion      -> Motion[1,2,3,4,...,t]  
        # z_t = F(z_{t-1},m_t)

        device = video.device
        b,t,c,h,w = video.shape

        # Data transform
        ff = video[:,:1,:,:,:]                               # fisrt frame (b,1,c,h,w)
        sf = video[:,:-1,:,:,:]                             # ! last frame (b,t-1,c,h,w)
        assert ff.shape == (b,1,c,h,w) , f'ff shape {ff.shape}'
        shift_video = torch.cat([ff,sf],dim=1)              # (b,t,c,h,w)  [1,1,2,3,....,t-1]
        duo_frame_mix = torch.cat([shift_video,video],dim=2)    # (b,t,2c,h,w)
        input = einops.rearrange(duo_frame_mix,'b t c h w -> (b t) c h w')

        # DuoFrameDownEncoder
        motion = self.dfd_encoder(input)                         # (b*t,latent_dim,h/ds,w/ds)

        # Upsampler
        motion = self.upsampler(motion)                               # (b*t,c,h,w)


        # MapConv
        motion = einops.rearrange(motion,'(b t) c h w -> b t c h w',b=b,t=t)    # (b,t,c,h,w)
        frame_motion_mix = torch.cat([shift_video,motion],dim=2)                # (b,t,2c,h,w)
        frame_motion_mix = einops.rearrange(frame_motion_mix,'b t c h w -> (b t) c h w')
        predict = self.mapconv(frame_motion_mix)                               # (b,t,c,h,w)
        predict = einops.rearrange(predict,'(b t) c h w -> b t c h w',b=b,t=t)

        return predict  # (b,t,c,h,w)
    
    def forward_loss(self,pre:torch.tensor,gt:torch.tensor):
        """
        Args:
            * video : torch.tensor (b,t,c,h,w)
            * gt    : torch.tensor (b,t,c,h,w)
        """
        loss = l2(pre[:,1:,:],gt[:,1:,:])
        return loss


class AMDModel(ModelMixin, ConfigMixin):
    """example
    Data:
    * video : (n,F,C,H,W)
    
    Process:
    1. DuoFrameDownEncoder : 
        * video : (n,F,C,H,W)
        ------------------------------------------------------------------------------------
        * mix_video : (n,T,C,H,W)     | T = F-1 
        
    2. Motion Transformer :
        * motion : (n,T,c,h,w)        | T = F-1 
        ------------------------------------------------------------------------------------
        * motion : (n,T,c,h,w)        | T = F-1 
    
    3. Diffusion Transformer :
        * video_start : (b,C,H,W)     | b = n * groups_per_batch
        * motion : (b,l,c,h,w)        | b = n * groups_per_batch    | l random motion length
        * video_end(gt) : (b,C,H,W)   | b = n * groups_per_batch
        ------------------------------------------------------------------------------------
        * pre : (b,C,H,W)             | b = n * groups_per_batch
    """
    _supports_gradient_checkpointing = True
    
    @register_to_config
    def __init__(self,
                 mae_patch_size :int = 2,
                 mae_inchannel :int = 768,
                 image_inchannel :int  = 4,
                 image_height  :int = 32,
                 image_width  :int  = 32,
                 video_frames  :int = 16,
                 scheduler_num_step :int = 1000,
                 
                 # ----------- DownEncoder -----------
                 block_out_channels_down : Tuple = (64,128,256,256),
                 
                 # ----------- Motion Transformer -----------
                 no_motion_transformer = False,
                 motion_attn_head_dim  = 64,
                 motion_attn_num_heads  = 4,
                 motion_num_layers  = 4,
                 
                 mae_output_with_img : bool = False,
                 **kwargs,
                 ):
        super().__init__()

        # setting
        self.block_out_channels_down = block_out_channels_down
        self.block_out_channels_up = list(reversed(block_out_channels_down))
        self.encoder_outdim = block_out_channels_down[-1]
        self.num_step = scheduler_num_step
        self.scheduler = RectifiedFlow(num_steps=scheduler_num_step)
        self.mae_output_with_img = mae_output_with_img
        self.mae_patch_size = mae_patch_size

        # mae_linear
        mae_in_c = mae_inchannel + mae_inchannel // (mae_patch_size**2) # with cls token
        self.mae_conv_in = ResnetBlock2D(in_channels=mae_in_c,out_channels=image_inchannel,temb_channels=None,groups=1)

        # motion Encoder 
        if mae_output_with_img:
            dfd_inchannel = 4 * image_inchannel
        else:
            dfd_inchannel = 2 * image_inchannel
        self.dfd_encoder = DuoFrameDownEncoder(in_channel=dfd_inchannel,
                                               block_out_channels=self.block_out_channels_down)
        
        # motion Transformer
        downsample_ratio = 2 ** (len(self.block_out_channels_down) - 1 )
        motion_inchannels = self.encoder_outdim
        motion_height = image_height // downsample_ratio
        motion_width = image_width // downsample_ratio
        motion_frames = video_frames - 1

        # motion
        motion_outchannels = motion_inchannels
        if no_motion_transformer:
            self.motion_transformer = nn.Identity()
        else:
            self.motion_transformer = MotionTransformer(attention_head_dim = motion_attn_head_dim,
                                            num_attention_heads = motion_attn_num_heads,
                                            in_channels = motion_inchannels,
                                            out_channels = motion_outchannels,
                                            num_layers = motion_num_layers,
                                            sample_width = motion_width,
                                            sample_height = motion_height,
                                            sample_frames = motion_frames,)
        
        # diffusion transformer
        self.upsampler = Upsampler(in_channel=self.block_out_channels_down[-1],
                            out_channel=4,
                            block_out_channels=self.block_out_channels_up)
        mapconv_inchannel = 8
        self.mapconv = MapConv(in_channel= mapconv_inchannel,
                               out_channel=4,
                               block_layer=8)
        


    def extract_motion(self,video:torch.tensor,mae_output:torch.Tensor=None):
        # mae_output : (b,C,H,W)

        device = video.device
        n,t,c,h,w = video.shape

        if self.mae_output_with_img:
            # transform mae_output
            if len(mae_output.shape) == 5:
                mae_output = einops.rearrange(mae_output,'n t c h w -> (n t) c h w') # (b,C,H,W)
            mae_output = self.prepare_mae_output(mae_output) # (b,C,H,W)
            mae_output = einops.rearrange(mae_output,'(n t) c h w -> n t c h w',n=n) # (n,t,C,H,W)
            assert video.shape == mae_output.shape , f'video shape {video.shape} != mae_output shape {mae_output.shape}'
            # mix video
            video_with_mae = torch.cat([video,mae_output],dim=2) # (n,t,2C,H,W)
        else:
            video_with_mae = video

        # DuoFrameDownEncoder
        mix_duo_frame = self.prepare_encoder_input(video_with_mae) # (n*(t-1),4C,H,W)
        motion = self.dfd_encoder(mix_duo_frame)          # (n*(t-1),encoder_outdim,h,w)

        # Motion Transformer
        motion_input = einops.rearrange(motion,'(b t) c h w -> b t c h w',b=n) # (n,t-1,encoder_outdim,h,w)
        motion = self.motion_transformer(motion_input) # (n,t-1,motion_out_channels,h,w)

        return motion
        

    def forward(self,video:torch.tensor,mae_output:torch.Tensor,ref_img:torch.Tensor = None,time_step:torch.tensor = None,return_meta_info=False):
        """
        Args:
            video: (b,t,c,H,W)
            mae_output : (b,L+1,D)
            ref_img : (b,c,H,W)
        """

        device = video.device
        n,t,c,h,w = video.shape

        motion = self.extract_motion(video,mae_output)
        
        # Upsampler
        if len(motion.shape) == 5:
            motion = motion.flatten(0,1)
        motion = self.upsampler(motion)                               # (b*t,c,h,w)

        # MapConv
        motion = einops.rearrange(motion,'(b t) c h w -> b t c h w',b=n)    # (b,t,c,h,w)
        ref_img = ref_img.unsqueeze(1).repeat(1,motion.shape[1],1,1,1)
        frame_motion_mix = torch.cat([ref_img,motion],dim=2)                # (b,t,2c,h,w)
        frame_motion_mix = einops.rearrange(frame_motion_mix,'b t c h w -> (b t) c h w')
        predict = self.mapconv(frame_motion_mix)                               # (b,t,c,h,w)
        predict = einops.rearrange(predict,'(b t) c h w -> b t c h w',b=n)

        gt = video[:,1:,:]
    
        return predict, gt # (b,C,H,W)
    


    @torch.no_grad()
    def sample(self,video:torch.tensor,mae_output:torch.Tensor,sample_step:int=None,ref_img:torch.Tensor = None,start_step:int = None,return_meta_info=False,test_zi_idx = None):
        """
        mae_output : (b,t,embed//patch_size**2,H,W)
        """
        #         device = video.device
        # n,t,c,h,w = video.shape

        predict, gt = self.forward(video,mae_output,ref_img)

        pre = predict.flatten(0,1).unsqueeze(1) # (b,1,c,h,w) 
        gt = gt.flatten(0,1).unsqueeze(1) # (b,1,c,h,w) 


        return pre,gt # (b,1,c,h,w)
    
    def prepare_diffusion_transformer(self,video:torch.Tensor,motion:torch.Tensor,time_step:torch.Tensor = None,ref_img:torch.Tensor = None):
        """prepare diffusion input

        Args:
            video (torch.Tensor): video tensor, shape = (n,f,c,h,w)
            motion (torch.Tensor): motion transformer output, shape = (n,f-1,c,h',w') 
            time_step (torch.Tensor, optional): diffusion timestep, range = (0,1), shape = (n,). If input timestep is None, 
            timestep will be randomly sampled. Defaults to None.

        Returns:
            zi (torch.Tensor): initial frames, shape = (n*(f-1),c,h,w). All the frames is the same as video[:,0]
            zj (torch.Tensor): target frames, shape = (n*(f-1),c,h,w).
            zt (torch.Tensor): zj + noise, shape = (n*(f-1),c,h,w).
            vel (torch.Tensor): velocity tensor. Ground truth
            motion (torch.Tensor): shape = (n*(f-1),c,h',w')
            time_step (torch.Tensor): sampled timestep. shape = (n*(f-1),)

        """        
        n,f,C,H,W = video.shape
        _,t,c,h,w = motion.shape

        device = video.device
        
        assert t == f-1 , f'motion frames {t} != video frames {f} - 1'
        
        # extract frame
        zi = video[:,:1,:].expand(-1,t,-1,-1,-1) # video_start:(n,t,C,H,W), repeat over `frame dimension`
        zj = video[:,1:,:] # video_target:(n,t,C,H,W)
        
        zi = zi.flatten(0,1).contiguous() # (b,C,H,W)  | b = n*t
        zj = zj.flatten(0,1).contiguous() # (b,C,H,W)  | b = n*t
        motion = motion.flatten(0,1).contiguous() # (b,C,H,W)  | b = n*t
        
        # time_step
        if time_step is None:
            time_step = self.prepare_timestep(batch_size= zj.shape[0],device= device) #(b,)
        
        # train tuple
        # default z0 : gaussian noise | z1:target distribution
        zt,vel = self.scheduler.get_train_tuple(z1=zj,time_step=time_step)  # (b,C,H,W),(b,C,H,W)

        # ref_img (N,C,H,W)
        if ref_img is not None:
            zi = ref_img.unsqueeze(1).expand(-1,t,-1,-1,-1) # (n,t,C,H,W)
            zi = zi.flatten(0,1).contiguous()
        
        return zi,zj,zt,vel,motion,time_step # (b,C,H,W),(b,C,H,W),(b,C,H,W),(b,C,H,W),(b,c,h,w),(b,)
        
    def prepare_timestep(self,batch_size:int,device,time_step = None):
        if time_step is not None:
            return time_step.to(device)
        else:
            return torch.randint(0,self.num_step+1,(batch_size,)).to(device)
  
    def forward_loss(self,pre:torch.tensor,gt:torch.tensor):
        """
        Args:
            * video : torch.tensor (b,c,h,w)
            * gt    : torch.tensor (b,c,h,w)
        """
        loss = l2(pre,gt)
        return loss

    def prepare_encoder_input(self,video:torch.tensor):
        assert len(video.shape) == 5 , f'only support video data : 5D tensor , but got {video.shape}'
        
        # cat
        pre = video[:,:-1,:,:,:] 
        post= video[:,1:,:,:,:]
        duo_frame_mix = torch.cat([pre,post],dim=2)    # (b,t-1,2c,h,w)
        duo_frame_mix = einops.rearrange(duo_frame_mix,'b t c h w -> (b t) c h w')
        
        return duo_frame_mix # (b*f-1,2c,h,w)

    def prepare_mae_output(self,mae_output:torch.tensor):
        p = self.mae_patch_size

        # imgpart 
        img_part = mae_output[:,1:,:] # (b,L,D)
        img_pix = self.unpatchify(img_part,patch_size=self.mae_patch_size) # (b,D,H,W) H=L**0.5*p

        # cls part
        cls_part = mae_output[:,:1,:] # (b,1,D)
        cls_part = cls_part.repeat(1,img_part.shape[1]*p*p,1) # (b,L*p*p,D)
        h = int(img_part.shape[1]**.5) * p
        w = h
        cls_pix = einops.rearrange(cls_part,'b (h w) d -> b d h w',h=h,w=w) # (b,4D,H,W)

        # cat
        mix_pix = torch.cat([cls_pix,img_pix],dim=1) # (b,D+4D,H,W)
        mix_pix = self.mae_conv_in(mix_pix,None) # (b,C,H,W)

        return mix_pix # (b,C,H,W)

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

def AMD_S(**kwargs) -> AMDModel:
    return AMDModel( 
                    
                    # ----------- Motion Transformer -----------
                    motion_attn_head_dim  = 64,
                    motion_attn_num_heads  = 8,
                    motion_num_layers  = 6,
                    **kwargs)
def AMD_B(**kwargs) -> AMDModel:
    return AMDModel(  
                    # ----------- Motion Transformer -----------
                    motion_attn_head_dim  = 64,
                    motion_attn_num_heads  = 4,
                    motion_num_layers  = 4,
                    
                    **kwargs)

def AMD_M(**kwargs) -> AMDModel:
    return AMDModel (   
                    # ----------- Motion Transformer -----------
                    motion_attn_head_dim  = 64,
                    motion_attn_num_heads  = 8,
                    motion_num_layers  = 8,
                    **kwargs)

def AMD_L(**kwargs) -> AMDModel:
    return AMDModel(   
                    # ----------- Motion Transformer -----------
                    motion_attn_head_dim  = 96,
                    motion_attn_num_heads  = 10,
                    motion_num_layers  = 10,
                    **kwargs)


AMD_models = {
    "AMD_S": AMD_S,  
    "AMD_B": AMD_B,        
    "AMD_M": AMD_M,
    "AMD_L": AMD_L,
} # S 206 B 333  M 642 L 1053 

