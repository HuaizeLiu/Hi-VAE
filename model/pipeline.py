
import torch
from torch import nn
import einops
from typing import Tuple
import random
import numpy as np
from tqdm import tqdm
from .modules import DuoFrameDownEncoder,Upsampler,MapConv,MotionDownEncoder
from .loss import l1,l2
from .transformer import (MotionTransformer,
                        AMDDiffusionTransformerModel,
                        MotionEncoderLearnTokenTransformer,
                        AMDReconstructTransformerModel,
                        AMDDiffusionTransformerModelDualStream,
                        AMDDiffusionTransformerModelImgSpatial)
from .rectified_flow import RectifiedFlow
from diffusers.configuration_utils import ConfigMixin, register_to_config
from diffusers.models.modeling_utils import ModelMixin
from diffusers.models.resnet import ResnetBlock2D
import einops
import torch.nn.functional as F
from dataclasses import dataclass
from .utils import vae_encode,vae_decode
import torch
import subprocess
import tempfile
import os
import cv2
from torchvision.io import write_video
import torchvision.transforms as transforms
from torchvision.io import write_video, read_video, read_image
from pathlib import Path
from decord import VideoReader
from decord import cpu, gpu


@dataclass
class Block:
    idx : int = 0
    start_frame : int = 0
    end_frame : int = 0

    ref_img: torch.Tensor = None
    randomref_img: torch.Tensor = None  # N,C,H,W
    ref_audio: torch.Tensor = None
    audio: torch.Tensor = None

    motion_pre : torch.Tensor = None
    video_pre : torch.Tensor = None
    
    def __str__(self):
        return  f"Block {self.idx} \n"\
                f"Start:{self.start_frame : <4} End:{self.end_frame:<4}\n"\
                f"Ref_img:          {self.ref_img.shape if self.ref_img is not None else None}\n"\
                f"RandomRef_img:    {self.randomref_img.shape if self.randomref_img is not None else None}\n"\
                f"Ref_audio:        {self.ref_audio.shape if self.ref_audio is not None else None}\n"\
                f"Audio:            {self.audio.shape if self.audio is not None else None}\n"\
                f"Motion_pre:       {self.motion_pre.shape if self.motion_pre is not None else None}\n"\
                f"Video_pre:        {self.video_pre.shape if self.video_pre is not None else None}\n"



class A2VPipeLine(ModelMixin, ConfigMixin):
    @register_to_config
    def __init__(self,
                 vae,
                 amd_model,
                 a2m_model,
                 a2m_ref_num_frame:int = 8,
                 window:int = 16, # inference num of frames per loop
                 need_motion_extract_model: bool = False,
                 **kwargs,
                 ):
        super().__init__()

        self.vae = vae
        self.amd_model = amd_model
        self.a2m_model = a2m_model
        self.window = window
        self.a2m_ref_num_frame = a2m_ref_num_frame
        self.need_motion_extract_model = need_motion_extract_model

        if window != amd_model.target_frame:
            amd_model.reset_infer_num_frame(window)

        assert self.window >= self.a2m_ref_num_frame , "ref frame should be shorter than infer frame"
    
    @torch.no_grad()
    def forward(self,
                ref_img:torch.Tensor,
                ref_audio:torch.Tensor,
                audio:torch.Tensor,
                motion_sample_step:int = 4,
                video_sample_step:int = 4,
                **kwargs):
        """
        Args:
            ref_img: N,F,C,H,W
            ref_audio: N,F,M,D
            audio: N,T,M,D
        Return:
            video: N,T+1,C,H,W
        """
        W = self.window
        R = self.a2m_ref_num_frame
        print(f'sample window {W}')
        print(f'ref num frame {R}')

        # vae
        vae = self.vae
        raw_ref_img = ref_img
        pad_length = R-ref_img.shape[1]
        ref_img_pad = torch.zeros((ref_img.shape[0],pad_length, *ref_img.shape[2:]), dtype=ref_img.dtype).to(ref_img.device)
        ref_img = torch.cat([ref_img_pad,ref_img],dim=1)
        ref_img = vae_encode(vae,ref_img)
        print(f'ref_img,{ref_img.shape}')

        initial_blocks = self.initial_blocks(ref_img = ref_img,
                                             randomref_img = ref_img[:,-1,:],
                                             ref_audio = ref_audio,
                                             audio = audio ,
                                             window = W)
        pre_blocks = []

        # pre motion
        for i,block in enumerate(initial_blocks):
            if block.audio.shape[1]<W:
                break

            # get ref_img from last Block
            if block.ref_img is None and len(pre_blocks) > 0:
                block.ref_img = pre_blocks[-1].video_pre[:,-R:,:] # n f c h w
            print(f'* Sample Loop {i} *')
            print(block)
            
            # # get ref_motion
            if self.need_motion_extract_model:
                ref_motion = self.amd_model.extract_motion(block.ref_img) # n,f,l,d
            else:
                if len(pre_blocks) == 0:
                    ref_motion = self.amd_model.extract_motion(block.ref_img) # n,f,l,d
                else:
                    ref_motion = pre_blocks[-1].motion_pre[:,-R:,]

            # pre motion
            motion_pre = self.a2m_model.sample(  ref_motion = ref_motion,
                                                audio =block.audio.to(ref_motion.dtype),
                                                ref_audio =block.ref_audio.to(ref_motion.dtype) ,
                                                sample_step=motion_sample_step) # n f d h w
            
            # pre video
            m2v_ref_img = block.ref_img[:,-1,:]
            _,video_pre,_  = self.amd_model.sample_with_refimg_motion(ref_img = m2v_ref_img,
                                                                      motion = motion_pre,
                                                                      randomref_img = block.randomref_img.to(ref_motion.dtype),
                                                                      sample_step=video_sample_step) # n f d h w

            # save
            block.motion_pre = motion_pre # n t l d
            block.video_pre = video_pre # n t c h w
            assert block.video_pre.shape[1] == W , f'video_pre length {block.video_pre.shape[1]} should be equal to window {W}'
            pre_blocks.append(block)
        
        videos = [ref_img[:,-1:,:]] + [x.video_pre for x in pre_blocks]
        videos = torch.cat(videos,dim=1) # n s*t+1 c h w

        del initial_blocks, pre_blocks
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        return videos


    def initial_blocks(self,ref_img,randomref_img,ref_audio,audio,window):
        N,T,_,_= audio.shape
        W = window
        R = self.a2m_ref_num_frame
        blocks = []

        idx = 0
        for i in range(0,T-1,W):
            block = Block(idx = idx,
                          start_frame=i,
                          end_frame=i+W-1,

                          ref_img = ref_img if i == 0 else None,
                          randomref_img= randomref_img,
                          ref_audio = self.transform_a2m_ref(ref_audio) if i == 0 else audio[:,i-R:i,:],
                          audio = audio[:,i:i+W])

            blocks.append(block)
            idx +=1 
        return blocks

    def transform_a2m_ref(self,ref:torch.Tensor):

        R = self.a2m_ref_num_frame
        
        if ref.shape[1] >= R:
            result = ref[:,-R:,:]
        else:
            pad_length = R-ref.shape[1]
            pad = torch.zeros((ref.shape[0], pad_length,*ref.shape[2:]), dtype=ref.dtype).to(ref.device)
            result = torch.cat([pad,ref],dim=1)

        assert result.shape[1] == R, f"padding result.shape{result.shape} should be equal to {R}"
        return result


    def export_video_with_audio(self,video_tensor, audio_path, start_time, fps, output_path):
        """
        将生成的视频张体与音频文件合并，生成带音频的MP4文件。

        参数：
        video_tensor (torch.Tensor): 形状为 ( F, C, H, W) 的视频张量，值范围为0-255，uint8类型。
        audio_path (str): 输入的.wav音频文件路径。
        start_time (float): 音频开始时间（秒）。
        fps (int): 视频的帧率。
        output_path (str): 输出文件路径，应以.mp4结尾。
        """
        # 确认批次大小为1
        assert video_tensor.dim() == 4, "仅支持批次大小为1的视频"
        F, C, H, W = video_tensor.shape
        duration = F / fps  # 计算视频时长

        # 创建临时文件保存视频
        with tempfile.NamedTemporaryFile(suffix='.mp4', delete=False) as tmp_vid:
            temp_video = tmp_vid.name
            # 调整张量维度并保存视频
            video_frames = video_tensor.permute(0, 2, 3, 1)  # 转换为 (F, H, W, C)
            write_video(temp_video, video_frames, fps=fps, video_codec='libx264')

        # 创建临时文件保存截取的音频
        with tempfile.NamedTemporaryFile(suffix='.wav', delete=False) as tmp_aud:
            temp_audio = tmp_aud.name
            # 使用FFmpeg截取音频
            cmd_extract = [
                'ffmpeg',
                '-i', audio_path,        # 输入音频
                '-y',  # 覆盖输出文件
                '-ss', str(start_time),  # 开始时间
                '-t', str(duration),     # 持续时间
                '-acodec', 'copy',       # 直接复制音频流
                temp_audio
            ]
            subprocess.run(cmd_extract, check=True)

        # 合并音视频
        cmd_merge = [
            'ffmpeg',
            '-y',
            '-i', temp_video,    # 输入视频
            '-i', temp_audio,    # 输入音频
            '-c:v', 'copy',      # 复制视频流
            '-c:a', 'aac',       # 编码音频为AAC
            '-strict', 'experimental',
            output_path
        ]
        subprocess.run(cmd_merge, check=True)

        # 清理临时文件
        os.remove(temp_video)
        os.remove(temp_audio)


class AMDTestPipeLine(ModelMixin, ConfigMixin):
    @register_to_config
    def __init__(self,
                 vae,
                 amd_model,
                 a2m_model,
                 a2m_ref_num_frame:int = 8,
                 window:int = 16, # inference num of frames per loop
                 need_motion_extract_model: bool = False,
                 mask_ratio:float = 0.0,
                 **kwargs,
                 ):
        super().__init__()

        self.vae = vae
        self.amd_model = amd_model
        self.a2m_model = a2m_model
        self.window = window
        self.a2m_ref_num_frame = a2m_ref_num_frame
        self.need_motion_extract_model = need_motion_extract_model
        self.mask_ratio = mask_ratio
        print(f"********** MASK RATIO {self.mask_ratio}**********")

        if window != amd_model.target_frame:
            amd_model.reset_infer_num_frame(window)

        assert self.window >= self.a2m_ref_num_frame , "ref frame should be shorter than infer frame"
    
    @torch.no_grad()
    def forward(self,
                ref_img:torch.Tensor,
                ref_audio:torch.Tensor,
                audio:torch.Tensor,
                motion_sample_step:int = 4,
                video_sample_step:int = 4,
                gt_video_z:torch.Tensor = None,
                **kwargs):
        """
        Args:
            ref_img: N,F,C,H,W
            ref_audio: N,F,M,D
            audio: N,T,M,D
        Return:
            video: N,T+1,C,H,W
        """
        W = self.window
        R = self.a2m_ref_num_frame
        print(f'sample window {W}')
        print(f'ref num frame {R}')

        assert gt_video_z is not None

        # vae
        vae = self.vae
        raw_ref_img = ref_img
        pad_length = R-ref_img.shape[1]
        ref_img_pad = torch.zeros((ref_img.shape[0],pad_length, *ref_img.shape[2:]), dtype=ref_img.dtype).to(ref_img.device)
        ref_img = torch.cat([ref_img_pad,ref_img],dim=1)
        ref_img = vae_encode(vae,ref_img)
        print(f'ref_img,{ref_img.shape}')

        initial_blocks = self.initial_blocks(ref_img = ref_img,
                                             randomref_img = ref_img[:,-1,:],
                                             ref_audio = ref_audio,
                                             audio = audio ,
                                             window = W)
        pre_blocks = []

        # pre motion
        for i,block in enumerate(initial_blocks):

            # get ref_img from last Block
            if block.ref_img is None and len(pre_blocks) > 0:
                block.ref_img = pre_blocks[-1].video_pre[:,-R:,:] # n f c h w
            print(f'* Sample Loop {i} *')
            print(block)
            
            cur_gt_video = gt_video_z[:,block.start_frame:block.end_frame+1,:]
            cur_target_motion = self.amd_model.extract_motion(cur_gt_video,self.mask_ratio) # n,f,l,d

            # pre video
            m2v_ref_img = block.ref_img[:,-1,:]
            _,video_pre,_  = self.amd_model.sample_with_refimg_motion(ref_img = m2v_ref_img,
                                                                      motion = cur_target_motion,
                                                                      randomref_img = block.randomref_img.to(m2v_ref_img.dtype),
                                                                      sample_step=video_sample_step,
                                                                      mask_ratio=self.mask_ratio) # n f d h w

            # save
            block.video_pre = video_pre # n t c h w
            assert block.video_pre.shape[1] == W , f'video_pre length {block.video_pre.shape[1]} should be equal to window {W}'
            pre_blocks.append(block)
        
        videos = [ref_img[:,-1:,:]] + [x.video_pre for x in pre_blocks]
        videos = torch.cat(videos,dim=1) # n s*t+1 c h w

        del initial_blocks, pre_blocks
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        return videos


    def initial_blocks(self,ref_img,randomref_img,ref_audio,audio,window):
        N,T,_,_= audio.shape
        W = window
        blocks = []

        idx = 0
        for i in range(0,T-1,W):
            block = Block(idx = idx,
                          start_frame=i,
                          end_frame=i+W-1,

                          ref_img = self.transform_a2m_ref(ref_img) if i == 0 else None,
                          randomref_img= randomref_img,
                          ref_audio = self.transform_a2m_ref(ref_audio) if i == 0 else self.transform_a2m_ref(audio[:,i:i+1,:]),
                          audio = audio[:,i:i+W])

            blocks.append(block)
            idx +=1 
        return blocks

    def transform_a2m_ref(self,ref:torch.Tensor):

        R = self.a2m_ref_num_frame
        
        if ref.shape[1] >= R:
            result = ref[:,-R:,:]
        else:
            pad_length = R-ref.shape[1]
            pad = torch.zeros((ref.shape[0], pad_length,*ref.shape[2:]), dtype=ref.dtype).to(ref.device)
            result = torch.cat([pad,ref],dim=1)

        assert result.shape[1] == R, f"padding result.shape{result.shape} should be equal to {R}"
        return result


    def export_video_with_audio(self,video_tensor, audio_path, start_time, fps, output_path):
        """
        将生成的视频张体与音频文件合并，生成带音频的MP4文件。

        参数：
        video_tensor (torch.Tensor): 形状为 ( F, C, H, W) 的视频张量，值范围为0-255，uint8类型。
        audio_path (str): 输入的.wav音频文件路径。
        start_time (float): 音频开始时间（秒）。
        fps (int): 视频的帧率。
        output_path (str): 输出文件路径，应以.mp4结尾。
        """
        # 确认批次大小为1
        assert video_tensor.dim() == 4, "仅支持批次大小为1的视频"
        F, C, H, W = video_tensor.shape
        duration = F / fps  # 计算视频时长

        # 创建临时文件保存视频
        with tempfile.NamedTemporaryFile(suffix='.mp4', delete=False) as tmp_vid:
            temp_video = tmp_vid.name
            # 调整张量维度并保存视频
            video_frames = video_tensor.permute(0, 2, 3, 1)  # 转换为 (F, H, W, C)
            write_video(temp_video, video_frames, fps=fps, video_codec='libx264')

        # 创建临时文件保存截取的音频
        with tempfile.NamedTemporaryFile(suffix='.wav', delete=False) as tmp_aud:
            temp_audio = tmp_aud.name
            # 使用FFmpeg截取音频
            cmd_extract = [
                'ffmpeg',
                '-i', audio_path,        # 输入音频
                '-y',  # 覆盖输出文件
                '-ss', str(start_time),  # 开始时间
                '-t', str(duration),     # 持续时间
                '-acodec', 'copy',       # 直接复制音频流
                temp_audio
            ]
            subprocess.run(cmd_extract, check=True)

        # 合并音视频
        cmd_merge = [
            'ffmpeg',
            '-y',
            '-i', temp_video,    # 输入视频
            '-i', temp_audio,    # 输入音频
            '-c:v', 'copy',      # 复制视频流
            '-c:a', 'aac',       # 编码音频为AAC
            '-strict', 'experimental',
            output_path
        ]
        subprocess.run(cmd_merge, check=True)

        # 清理临时文件
        os.remove(temp_video)
        os.remove(temp_audio)


class AMDPipeLine(ModelMixin, ConfigMixin):
    @register_to_config
    def __init__(self,
                 vae,
                 amd_model,
                 window:int = 16, # inference num of frames per loop
                 device = "cuda:0",
                 use_grey = False,
                 **kwargs,
                 ):
        super().__init__()

        self.vae = vae
        self.amd_model = amd_model
        self.window = window
        self.max_infer_length = 256
        self.ddevice = device
        self.use_grey = use_grey

        # if window != amd_model.target_frame:
        #     amd_model.reset_infer_num_frame(window)

        self.pixel_transforms = transforms.Compose([
            transforms.Resize(256),
            transforms.CenterCrop(256),
            transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5], inplace=True),
        ])

    @torch.no_grad()
    def sample(self,
                video_path: torch.Tensor,
                output_path: str,
                video_sample_step:int = 4,
                mask_ratio:float = 0.0,
                fps:int = 30,
                drop_prev_img:bool = False,
                drop_ref_img:bool = False,
                **kwargs):
        """
        Args:
            ref_img: N,F,C,H,W
            ref_audio: N,F,M,D
            audio: N,T,M,D
        Return:
            video: N,T+1,C,H,W
        """
        W = self.window
        print(f'sample window {W}')

        # load video
        video_reader = VideoReader(video_path, ctx=cpu(0))
        video_length = min(len(video_reader),self.max_infer_length+1) # 1 for refimg
        batch_index = np.arange(video_length)
        videos = torch.from_numpy(video_reader.get_batch(batch_index).asnumpy()).permute(0, 3, 1, 2).contiguous() #(N,H,W,C)->(N,C,H,W)
        videos = videos / 255.0 
        videos = self.pixel_transforms(videos).to(self.ddevice)

        # vae encode
        videos_z = vae_encode(self.vae,videos) # F,C,H,W
        refimg_z = videos_z[:1,:] # N,C,H,W
        gt_video_z = videos_z[1:,:].unsqueeze(0) # N,F,C,H,W

        if self.use_grey:
            frames = video_reader.get_batch(batch_index).asnumpy()
            grey_frames = np.zeros((frames.shape[0], frames.shape[1], frames.shape[2]))
            for i in range(frames.shape[0]):
                # 将 RGB 转换为 BGR，因为 OpenCV 默认使用 BGR 格式
                bgr_frame = cv2.cvtColor(frames[i, ...], cv2.COLOR_RGB2BGR)
                # 将 BGR 转换为灰度
                grey_frames[i, ...] = cv2.cvtColor(bgr_frame, cv2.COLOR_BGR2GRAY)
            grey_videos = torch.from_numpy(grey_frames).unsqueeze(1).contiguous()
            grey_videos = grey_videos.repeat(1,3,1,1)
            grey_videos = grey_videos / 255.0
            grey_videos = self.pixel_transforms(grey_videos).to(dtype=torch.float32,device=self.ddevice)

            # vae encode
            grey_videos_z = vae_encode(self.vae,grey_videos) # F,C,H,W
            grey_refimg_z = grey_videos_z[:1,:] # N,C,H,W
            grey_gt_video_z = grey_videos_z[1:,:].unsqueeze(0) # N,F,C,H,W


        # sample setting
        sample_num = gt_video_z.shape[1] // W
        offset = gt_video_z.shape[1] % W
        pre_video = None

        for i in range(sample_num):
            S = i*W
            E = (i+1)*W -1 
            cur_gt_video = gt_video_z[:,S:E+1,:] # n f c h w
            cur_prev_img = pre_video[:,-1,:] if pre_video is not None else refimg_z # n c h w
            # cur_prev_img = gt_video_z[:,S-1,:] if pre_video is not None else refimg_z # n c h w
            cur_prev_img = cur_prev_img.unsqueeze(1).repeat(1,W,1,1,1)
            if self.use_grey:
                grey_cur_gt_video = grey_gt_video_z[:,S:E+1,:] # n f c h w
                grey_cur_prev_img = grey_gt_video_z[:,S-1,:] if pre_video is not None else refimg_z # n c h w
                grey_cur_prev_img = grey_cur_prev_img.unsqueeze(1).repeat(1,W,1,1,1)

                _,video_pre,_ = self.amd_model.sample(video = cur_gt_video,
                                                    ref_img = cur_prev_img if not drop_prev_img else torch.zeros_like(cur_prev_img).to(cur_prev_img.device),
                                                    video_grey=grey_cur_gt_video,
                                                    ref_img_grey=grey_cur_prev_img,
                                                    sample_step=video_sample_step,
                                                    mask_ratio=mask_ratio)
            else:
                _,video_pre,_ = self.amd_model.sample(video = cur_gt_video,
                                                    ref_img = cur_prev_img if not drop_prev_img else torch.zeros_like(cur_prev_img).to(cur_prev_img.device),
                                                    sample_step=video_sample_step,
                                                    mask_ratio=mask_ratio) # n f c h w
            pre_video = video_pre if pre_video is None else torch.cat([pre_video,video_pre],dim=1)

        if offset > 0:
            E = gt_video_z.shape[1] - 1
            S = E - W + 1
            cur_gt_video = gt_video_z[:,S:E+1,:] # n f c h w
            # cur_prev_img = gt_video_z[:,S-1,:] if len(pre_video) > 0 else refimg_z # n c h w
            cur_prev_img = pre_video[:,-1,:] if len(pre_video) > 0 else refimg_z # n c h w
            cur_prev_img = cur_prev_img.unsqueeze(1).repeat(1,W,1,1,1)

            if self.use_grey:
                grey_cur_gt_video = grey_gt_video_z[:,S:E+1,:] # n f c h w
                grey_cur_prev_img = grey_gt_video_z[:,S-1,:] if pre_video is not None else refimg_z # n c h w
                grey_cur_prev_img = grey_cur_prev_img.unsqueeze(1).repeat(1,W,1,1,1)

                _,video_pre,_ = self.amd_model.sample(video = cur_gt_video,
                                                    ref_img = cur_prev_img if not drop_prev_img else torch.zeros_like(cur_prev_img).to(cur_prev_img.device),
                                                    video_grey=grey_cur_gt_video,
                                                    ref_img_grey=grey_cur_prev_img,
                                                    sample_step=video_sample_step,
                                                    mask_ratio=mask_ratio)
            else:
                _,video_pre,_ = self.amd_model.sample(video = cur_gt_video,
                                                    ref_img = cur_prev_img if not drop_prev_img else torch.zeros_like(cur_prev_img).to(cur_prev_img.device),
                                                    sample_step=video_sample_step,
                                                    mask_ratio=mask_ratio) # n f c h w

            pre_video = video_pre if pre_video is None else torch.cat([pre_video,video_pre],dim=1)

        # decode
        result = torch.cat([refimg_z.unsqueeze(1),pre_video],dim=1) # n s*t+1 c h w
        result = vae_decode(self.vae,result)
        result = ((result / 2.0 + 0.5).clamp(0, 1) * 255).to(dtype=torch.uint8).cpu().contiguous()
        result=result[0,:]

        # save
        self.export_video(result,fps=fps,output_path=output_path)

        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    def export_video(self,video_tensor, fps, output_path):
        # 确认批次大小为1
        assert video_tensor.dim() == 4, "仅支持批次大小为1的视频"
        F, C, H, W = video_tensor.shape

        video_frames = video_tensor.permute(0, 2, 3, 1)  # 转换为 (F, H, W, C)
        write_video(output_path, video_frames, fps=fps, video_codec='libx264')

        print("success save video to ",output_path)

class AMDPipeLine_single(ModelMixin, ConfigMixin):
    @register_to_config
    def __init__(self,
                 vae,
                 amd_model,
                 window:int = 16, # inference num of frames per loop
                 device = "cuda:0",
                 use_grey = False,
                 **kwargs,
                 ):
        super().__init__()

        self.vae = vae
        self.amd_model = amd_model
        self.window = window
        self.ddevice = device
        self.use_grey = use_grey

        # if window != amd_model.target_frame:
        #     amd_model.reset_infer_num_frame(window)

        self.pixel_transforms = transforms.Compose([
            transforms.Resize(256),
            transforms.CenterCrop(256),
            transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5], inplace=True),
        ])

    @torch.no_grad()
    def sample(self,
                video_path: torch.Tensor,
                output_path: str,
                output_path_gt: str,
                video_sample_step:int = 20,
                object_mask_ratio:float = None,
                camera_mask_ratio:float = None,
                fps:int = 8,
                drop_prev_img:bool = False,
                drop_ref_img:bool = False,
                **kwargs):
        """
        Args:
            ref_img: N,F,C,H,W
            ref_audio: N,F,M,D
            audio: N,T,M,D
        Return:
            video: N,T+1,C,H,W
        """
        W = self.window
        print(f'sample window {W}')

        # load video
        video_reader = VideoReader(video_path, ctx=cpu(0))
        video_length = len(video_reader)
        video_fps = video_reader.get_avg_fps()
        batch_index = self.sample_frames_with_fps(video_length,video_fps,self.window+1,fps,start_index=0)
        # video_length = min(len(video_reader),self.window+1) # 1 for refimg
        # batch_index = np.arange(video_length)



        videos = torch.from_numpy(video_reader.get_batch(batch_index).asnumpy()).permute(0, 3, 1, 2).contiguous() #(N,H,W,C)->(N,C,H,W)
        videos = videos / 255.0 
        videos = self.pixel_transforms(videos).to(self.ddevice)

        # vae encode
        videos_z = vae_encode(self.vae,videos) # F,C,H,W
        refimg_z = videos_z[:1,:] # N,C,H,W
        gt_video_z = videos_z[1:,:].unsqueeze(0) # N,F,C,H,W

        if self.use_grey:
            frames = video_reader.get_batch(batch_index).asnumpy()
            grey_frames = np.zeros((frames.shape[0], frames.shape[1], frames.shape[2]))
            for i in range(frames.shape[0]):
                # 将 RGB 转换为 BGR，因为 OpenCV 默认使用 BGR 格式
                bgr_frame = cv2.cvtColor(frames[i, ...], cv2.COLOR_RGB2BGR)
                # 将 BGR 转换为灰度
                grey_frames[i, ...] = cv2.cvtColor(bgr_frame, cv2.COLOR_BGR2GRAY)
            grey_videos = torch.from_numpy(grey_frames).unsqueeze(1).contiguous()
            grey_videos = grey_videos.repeat(1,3,1,1)
            grey_videos = grey_videos / 255.0
            grey_videos = self.pixel_transforms(grey_videos).to(dtype=torch.float32,device=self.ddevice)

            # vae encode
            grey_videos_z = vae_encode(self.vae,grey_videos) # F,C,H,W
            grey_refimg_z = grey_videos_z[:1,:] # N,C,H,W
            grey_gt_video_z = grey_videos_z[1:,:].unsqueeze(0) # N,F,C,H,W


        cur_gt_video = gt_video_z # n f c h w
        cur_prev_img = refimg_z.unsqueeze(1).repeat(1,W,1,1,1) # n c h w
        # cur_prev_img = gt_video_z[:,S-1,:] if pre_video is not None else refimg_z # n c h w

        if self.use_grey:
            grey_cur_gt_video = grey_gt_video_z # n f c h w
            grey_cur_prev_img = grey_refimg_z.unsqueeze(1).repeat(1,W,1,1,1) # n c h w

            _,video_pre,_ = self.amd_model.sample(video = cur_gt_video,
                                                ref_img = cur_prev_img if not drop_prev_img else torch.zeros_like(cur_prev_img).to(cur_prev_img.device),
                                                video_grey=grey_cur_gt_video,
                                                ref_img_grey=grey_cur_prev_img,
                                                sample_step=video_sample_step,
                                                object_mask_ratio = object_mask_ratio,
                                                camera_mask_ratio = camera_mask_ratio,)
        else:
            _,video_pre,_ = self.amd_model.sample(video = cur_gt_video,
                                                ref_img = cur_prev_img if not drop_prev_img else torch.zeros_like(cur_prev_img).to(cur_prev_img.device),
                                                sample_step=video_sample_step,)
                                                # object_mask_ratio = object_mask_ratio,
                                                # camera_mask_ratio = camera_mask_ratio,
                                                # ) # n f c h w

        # decode
        result = torch.cat([refimg_z.unsqueeze(1),video_pre],dim=1) # n s*t+1 c h w
        result = vae_decode(self.vae,result)
        result = ((result / 2.0 + 0.5).clamp(0, 1) * 255).to(dtype=torch.uint8).cpu().contiguous()
        result=result[0,:]

        videos = ((videos / 2.0 + 0.5).clamp(0, 1) * 255).to(dtype=torch.uint8).cpu().contiguous()
        
        # save
        self.export_video(result,fps=fps,output_path=output_path)

        # self.export_video(videos,fps=fps,output_path=output_path_gt)
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    def export_video(self,video_tensor, fps, output_path):
        # 确认批次大小为1
        assert video_tensor.dim() == 4, "仅支持批次大小为1的视频"
        F, C, H, W = video_tensor.shape

        video_frames = video_tensor.permute(0, 2, 3, 1)  # 转换为 (F, H, W, C)
        write_video(output_path, video_frames, fps=fps, video_codec='libx264')

        print("success save video to ",output_path)

    def sample_frames_with_fps(
        self,
        total_frames,
        video_fps,
        sample_num_frames,
        sample_fps,
        start_index=None
    ):
        """sample frames proportional to the length of the frames in one second
        e.g., 1s video has 30 frames, when 'fps'=3, we sample frames with spacing of 30/3=10
        return the frame indices

        Parameters
        ----------
        total_frames : length of the video
        video_fps : original fps of the video
        sample_num_frames : number of frames to sample
        sample_fps : the fps to sample frames
        start_index : the starting frame index. If it is not None, it will be used as the starting frame index  

        Returns
        -------
        frame indices
        """
        # sample_num_frames = min(sample_num_frames, total_frames)
        interval = round(video_fps / sample_fps)
        frames_range = (sample_num_frames - 1) * interval + 1

        if start_index is not None:
            start = start_index
        elif total_frames - frames_range - 1 < 0:
            start = 0
        else:
            start = random.randint(0, total_frames - frames_range - 1)

        frame_idxs = np.linspace(
            start=start, stop=min(total_frames - 1, start + frames_range), num=sample_num_frames
        ).astype(int)

        return frame_idxs
    





class AMDPipeLine_single_cross(ModelMixin, ConfigMixin):
    @register_to_config
    def __init__(self,
                 vae,
                 amd_model,
                 window:int = 16, # inference num of frames per loop
                 device = "cuda:0",
                 use_grey = False,
                 **kwargs,
                 ):
        super().__init__()

        self.vae = vae
        self.amd_model = amd_model
        self.window = window
        self.ddevice = device
        self.use_grey = use_grey

        # if window != amd_model.target_frame:
        #     amd_model.reset_infer_num_frame(window)

        self.pixel_transforms = transforms.Compose([
            transforms.Resize(256),
            transforms.CenterCrop(256),
            transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5], inplace=True),
        ])

    @torch.no_grad()
    def sample(self,
                video_path_1: torch.Tensor,
                video_path_2: torch.Tensor,
                output_path: str,
                output_path_gt: str = None,
                video_sample_step:int = 4,
                object_mask_ratio:float = None,
                camera_mask_ratio:float = None,
                fps:int = 30,
                drop_prev_img:bool = False,
                drop_ref_img:bool = False,
                **kwargs):
        """
        Args:
            ref_img: N,F,C,H,W
            ref_audio: N,F,M,D
            audio: N,T,M,D
        Return:
            video: N,T+1,C,H,W
        """
        W = self.window
        print(f'sample window {W}')

        # load video
        video_reader_1 = VideoReader(video_path_1, ctx=cpu(0))
        video_length_1 = min(len(video_reader_1),self.window+1) # 1 for refimg
        batch_index_1 = np.arange(video_length_1)
        # print("batch_index",batch_index)
        videos_1 = torch.from_numpy(video_reader_1.get_batch(batch_index_1).asnumpy()).permute(0, 3, 1, 2).contiguous() #(N,H,W,C)->(N,C,H,W)
        videos_1 = videos_1 / 255.0 
        videos_1 = self.pixel_transforms(videos_1).to(self.ddevice)

        # load video
        video_reader_2 = VideoReader(video_path_2, ctx=cpu(0))
        video_length_2 = min(len(video_reader_2),self.window+1) # 1 for refimg
        batch_index_2 = np.arange(video_length_2)
        # print("batch_index",batch_index)
        videos_2 = torch.from_numpy(video_reader_2.get_batch(batch_index_2).asnumpy()).permute(0, 3, 1, 2).contiguous() #(N,H,W,C)->(N,C,H,W)
        videos_2 = videos_2 / 255.0 
        videos_2 = self.pixel_transforms(videos_2).to(self.ddevice)

        # vae encode
        videos_z_1 = vae_encode(self.vae,videos_1) # F,C,H,W
        refimg_z_1 = videos_z_1[:1,:] # N,C,H,W
        gt_video_z_1 = videos_z_1[1:,:].unsqueeze(0) # N,F,C,H,W

        videos_z_2 = vae_encode(self.vae,videos_2) # F,C,H,W
        refimg_z_2 = videos_z_2[:1,:] # N,C,H,W
        gt_video_z_2 = videos_z_2[1:,:].unsqueeze(0) # N,F,C,H,W

        if self.use_grey:
            frames_1 = video_reader_1.get_batch(batch_index_1).asnumpy()
            grey_frames_1 = np.zeros((frames_1.shape[0], frames_1.shape[1], frames_1.shape[2]))
            for i in range(frames_1.shape[0]):
                # 将 RGB 转换为 BGR，因为 OpenCV 默认使用 BGR 格式
                bgr_frame_1 = cv2.cvtColor(frames_1[i, ...], cv2.COLOR_RGB2BGR)
                # 将 BGR 转换为灰度
                grey_frames_1[i, ...] = cv2.cvtColor(bgr_frame_1, cv2.COLOR_BGR2GRAY)
            grey_videos_1 = torch.from_numpy(grey_frames_1).unsqueeze(1).contiguous()
            grey_videos_1 = grey_videos_1.repeat(1,3,1,1)
            grey_videos_1 = grey_videos_1 / 255.0
            grey_videos_1 = self.pixel_transforms(grey_videos_1).to(dtype=torch.float32,device=self.ddevice)
            # vae encode
            grey_videos_z_1 = vae_encode(self.vae,grey_videos_1) # F,C,H,W
            grey_refimg_z_1 = grey_videos_z_1[:1,:] # N,C,H,W
            grey_gt_video_z_1 = grey_videos_z_1[1:,:].unsqueeze(0) # N,F,C,H,W


            frames_2 = video_reader_2.get_batch(batch_index_2).asnumpy()
            grey_frames_2 = np.zeros((frames_2.shape[0], frames_2.shape[1], frames_2.shape[2]))
            for i in range(frames_2.shape[0]):
                # 将 RGB 转换为 BGR，因为 OpenCV 默认使用 BGR 格式
                bgr_frame_2 = cv2.cvtColor(frames_2[i, ...], cv2.COLOR_RGB2BGR)
                # 将 BGR 转换为灰度
                grey_frames_2[i, ...] = cv2.cvtColor(bgr_frame_2, cv2.COLOR_BGR2GRAY)
            grey_videos_2 = torch.from_numpy(grey_frames_2).unsqueeze(1).contiguous()
            grey_videos_2 = grey_videos_2.repeat(1,3,1,1)
            grey_videos_2 = grey_videos_2 / 255.0
            grey_videos_2 = self.pixel_transforms(grey_videos_2).to(dtype=torch.float32,device=self.ddevice)
            # vae encode
            grey_videos_z_2 = vae_encode(self.vae,grey_videos_2) # F,C,H,W
            grey_refimg_z_2 = grey_videos_z_2[:1,:] # N,C,H,W
            grey_gt_video_z_2 = grey_videos_z_2[1:,:].unsqueeze(0) # N,F,C,H,W


        cur_gt_video_1 = gt_video_z_1 # n f c h w
        cur_prev_img_1 = refimg_z_1.unsqueeze(1).repeat(1,W,1,1,1) # n c h w
        
        cur_gt_video_2 = gt_video_z_2 # n f c h w
        cur_prev_img_2 = refimg_z_2.unsqueeze(1).repeat(1,W,1,1,1) # n c h w
        # cur_prev_img = gt_video_z[:,S-1,:] if pre_video is not None else refimg_z # n c h w

        if self.use_grey:
            grey_cur_gt_video_1 = grey_gt_video_z_1 # n f c h w
            grey_cur_prev_img_1 = grey_refimg_z_1.unsqueeze(1).repeat(1,W,1,1,1) # n c h w
            grey_cur_gt_video_2 = grey_gt_video_z_2 # n f c h w
            grey_cur_prev_img_2 = grey_refimg_z_2.unsqueeze(1).repeat(1,W,1,1,1) # n c h w

            # _,video_pre,_ = self.amd_model.sample(video = cur_gt_video,
            #                                     ref_img = cur_prev_img if not drop_prev_img else torch.zeros_like(cur_prev_img).to(cur_prev_img.device),
            #                                     video_grey=grey_cur_gt_video,
            #                                     ref_img_grey=grey_cur_prev_img,
            #                                     sample_step=video_sample_step,
            #                                     object_mask_ratio = object_mask_ratio,
            #                                     camera_mask_ratio = camera_mask_ratio,)
            _,video_pre,_ = self.amd_model.sample_cross(video_1 = cur_gt_video_1,
                                                video_2 = cur_gt_video_2,
                                                ref_img = cur_prev_img_2,
                                                video_grey_1=grey_cur_gt_video_1,
                                                video_grey_2=grey_cur_gt_video_2,
                                                ref_img_grey=grey_cur_prev_img_2,
                                                sample_step=video_sample_step,
                                                object_mask_ratio = object_mask_ratio,
                                                camera_mask_ratio = camera_mask_ratio,)
        else:
            # _,video_pre,_ = self.amd_model.sample(video = cur_gt_video,
            #                                     ref_img = cur_prev_img if not drop_prev_img else torch.zeros_like(cur_prev_img).to(cur_prev_img.device),
            #                                     sample_step=video_sample_step,)
             _,video_pre,_ = self.amd_model.sample(video_1 = cur_gt_video_1,
                                                video_2 = cur_gt_video_2,
                                                ref_img = cur_prev_img_2,
                                                sample_step=video_sample_step,)
                                                # object_mask_ratio = object_mask_ratio,
                                                # camera_mask_ratio = camera_mask_ratio,
                                                # ) # n f c h w

        # decode
        result = torch.cat([refimg_z_2.unsqueeze(1),video_pre],dim=1) # n s*t+1 c h w
        result = vae_decode(self.vae,result)
        result = ((result / 2.0 + 0.5).clamp(0, 1) * 255).to(dtype=torch.uint8).cpu().contiguous()
        result=result[0,:]

        # videos = ((videos / 2.0 + 0.5).clamp(0, 1) * 255).to(dtype=torch.uint8).cpu().contiguous()
        
        # save
        self.export_video(result,fps=fps,output_path=output_path)

        # self.export_video(videos,fps=fps,output_path=output_path_gt)
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    def export_video(self,video_tensor, fps, output_path):
        # 确认批次大小为1
        assert video_tensor.dim() == 4, "仅支持批次大小为1的视频"
        F, C, H, W = video_tensor.shape

        video_frames = video_tensor.permute(0, 2, 3, 1)  # 转换为 (F, H, W, C)
        write_video(output_path, video_frames, fps=fps, video_codec='libx264')

        print("success save video to ",output_path)


class A2VInferencePipeLine(ModelMixin, ConfigMixin):
    @register_to_config
    def __init__(self,
                 vae,
                 amd_model,
                 a2m_model,
                 a2m_ref_num_frame:int = 8,
                 window:int = 16, # inference num of frames per loop
                 need_motion_extract_model: bool = False,
                 **kwargs,
                 ):
        super().__init__()

        self.vae = vae
        self.amd_model = amd_model
        self.a2m_model = a2m_model
        self.window = window
        self.a2m_ref_num_frame = a2m_ref_num_frame
        self.need_motion_extract_model = need_motion_extract_model
        self.a2m_randomref_num = 8

        if window != amd_model.target_frame:
            amd_model.reset_infer_num_frame(window)

        assert self.window >= self.a2m_ref_num_frame , "ref frame should be shorter than infer frame"
    
    @torch.no_grad()
    def forward(self,
                ref_img:torch.Tensor,
                ref_audio:torch.Tensor,
                audio:torch.Tensor,
                motion_sample_step:int = 4,
                video_sample_step:int = 4,
                **kwargs):
        """
        Args:
            ref_img: N,F,C,H,W
            ref_audio: N,F,M,D
            audio: N,T,M,D
        Return:
            video: N,T+1,C,H,W
        """
        W = self.window
        R = self.a2m_ref_num_frame
        print(f'sample window {W}')
        print(f'ref num frame {R}')

        # vae
        vae = self.vae

        # ref_img -> latent  | n f d h w
        pad_length = R-ref_img.shape[1]
        ref_img_pad = torch.zeros((ref_img.shape[0],pad_length, *ref_img.shape[2:]), dtype=ref_img.dtype).to(ref_img.device)
        ref_img = torch.cat([ref_img_pad,ref_img],dim=1)
        ref_img = vae_encode(vae,ref_img) # n f d h w
        
        # randomref_img | n d h w
        randomref_img = ref_img[:,-1,:]

        # cache
        pre_motion = None
        pre_video = None

        # sample loop
        sample_num = audio.shape[1] // W
        offset = audio.shape[1] % W

        for i in range(sample_num):
            S = i*W
            E = (i+1)*W -1 

            # refmotion
            if i == 0:
                ref_motion = self.amd_model.extract_motion(ref_img) # n,f,l,d
                ref_audio = self.transform_a2m_ref(ref_audio).to(ref_motion.dtype)
            else:
                if self.need_motion_extract_model:
                    ref_motion = self.amd_model.extract_motion(pre_video[:,-R:,:]) # n,f,l,d
                else:
                    ref_motion = pre_motion[:,-R:,] # n,f,l,d
                ref_audio = audio[:,S-R:S,:].to(ref_motion.dtype)
                

            # sample motion
            if i==0:
                first_motion = ref_motion[:,-1:,:] # n 1 l d
                randomref_motion = first_motion.repeat(1,self.a2m_randomref_num,1,1) # n 8 l d
            else:
                randomref_motion = torch.randint(low=0, high=pre_motion.shape[1], size=(self.a2m_randomref_num,))
                randomref_motion = pre_motion[:,randomref_motion,:,:] # n 8 l d
            motion_pre = self.a2m_model.sample(  ref_motion = ref_motion,
                                                randomref_motion = randomref_motion,
                                                audio =audio[:,S:E+1].to(ref_motion.dtype),
                                                ref_audio =ref_audio ,
                                                sample_step=motion_sample_step) # n f d h w     

            # sample video
            m2v_ref_img = randomref_img if i == 0 else pre_video[:,-1,:] # n d h w
            _,video_pre,_  = self.amd_model.sample_with_refimg_motion(ref_img = m2v_ref_img,
                                                                      motion = motion_pre,
                                                                      randomref_img = randomref_img,
                                                                      sample_step=video_sample_step) # n f d h w 

            # cache

            pre_motion = motion_pre if pre_motion is None else torch.cat([pre_motion,motion_pre],dim=1)
            pre_video = video_pre if pre_video is None else torch.cat([pre_video,video_pre],dim=1)

        if offset > 0:
            E = audio.shape[1] - 1
            S = E - W + 1
            if self.need_motion_extract_model:
                ref_motion = self.amd_model.extract_motion(pre_video[:,S-R:S,:]) # n,f,l,d
            else:
                ref_motion = pre_motion[:,S-R:S,:] # n,f,l,d
            ref_audio = audio[:,S-R:S,:].to(ref_motion.dtype)

            # sample motion
            randomref_motion = torch.randint(low=0, high=pre_motion.shape[1], size=(self.a2m_randomref_num,))
            randomref_motion = pre_motion[:,randomref_motion,:,:] # n 8 l d
            motion_pre = self.a2m_model.sample(  ref_motion = ref_motion,
                                               randomref_motion = randomref_motion,
                                                audio =audio[:,S:E+1].to(ref_motion.dtype),
                                                ref_audio =ref_audio ,
                                                sample_step=motion_sample_step) # n f d h w     

            # sample video
            m2v_ref_img = pre_video[:,S-1,:] # n d h w
            _,video_pre,_  = self.amd_model.sample_with_refimg_motion(ref_img = m2v_ref_img,
                                                                      motion = motion_pre,
                                                                      randomref_img = randomref_img,
                                                                      sample_step=video_sample_step) # n f d h w 

            pre_motion = torch.cat([pre_motion[:,:S,:],motion_pre],dim=1)
            pre_video = torch.cat([pre_video[:,:S,:],video_pre],dim=1)
            

        videos = torch.cat([ref_img[:,-1:,:],pre_video],dim=1) # n s*t+1 c h w

        
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        return videos


    def initial_blocks(self,ref_img,randomref_img,ref_audio,audio,window):
        N,T,_,_= audio.shape
        W = window
        blocks = []

        idx = 0
        for i in range(0,T-1,W):
            block = Block(idx = idx,
                          start_frame=i,
                          end_frame=i+W-1,

                          ref_img = self.transform_a2m_ref(ref_img) if i == 0 else None,
                          randomref_img= randomref_img,
                          ref_audio = self.transform_a2m_ref(ref_audio) if i == 0 else self.transform_a2m_ref(audio[:,i:i+1,:]),
                          audio = audio[:,i:i+W])

            blocks.append(block)
            idx +=1 
        return blocks

    def transform_a2m_ref(self,ref:torch.Tensor):

        R = self.a2m_ref_num_frame
        
        if ref.shape[1] >= R:
            result = ref[:,-R:,:]
        else:
            pad_length = R-ref.shape[1]
            pad = torch.zeros((ref.shape[0], pad_length,*ref.shape[2:]), dtype=ref.dtype).to(ref.device)
            result = torch.cat([pad,ref],dim=1)

        assert result.shape[1] == R, f"padding result.shape{result.shape} should be equal to {R}"
        return result


    def export_video_with_audio(self,video_tensor, audio_path, start_time, fps, output_path):
        """
        将生成的视频张体与音频文件合并，生成带音频的MP4文件。

        参数：
        video_tensor (torch.Tensor): 形状为 ( F, C, H, W) 的视频张量，值范围为0-255，uint8类型。
        audio_path (str): 输入的.wav音频文件路径。
        start_time (float): 音频开始时间（秒）。
        fps (int): 视频的帧率。
        output_path (str): 输出文件路径，应以.mp4结尾。
        """
        # 确认批次大小为1
        assert video_tensor.dim() == 4, "仅支持批次大小为1的视频"
        F, C, H, W = video_tensor.shape
        duration = F / fps  # 计算视频时长

        # 创建临时文件保存视频
        with tempfile.NamedTemporaryFile(suffix='.mp4', delete=False) as tmp_vid:
            temp_video = tmp_vid.name
            # 调整张量维度并保存视频
            video_frames = video_tensor.permute(0, 2, 3, 1)  # 转换为 (F, H, W, C)
            write_video(temp_video, video_frames, fps=fps, video_codec='libx264')

        # 创建临时文件保存截取的音频
        with tempfile.NamedTemporaryFile(suffix='.wav', delete=False) as tmp_aud:
            temp_audio = tmp_aud.name
            # 使用FFmpeg截取音频
            cmd_extract = [
                'ffmpeg',
                '-i', audio_path,        # 输入音频
                '-y',  # 覆盖输出文件
                '-ss', str(start_time),  # 开始时间
                '-t', str(duration),     # 持续时间
                '-acodec', 'copy',       # 直接复制音频流
                temp_audio
            ]
            subprocess.run(cmd_extract, check=True)

        # 合并音视频
        cmd_merge = [
            'ffmpeg',
            '-y',
            '-i', temp_video,    # 输入视频
            '-i', temp_audio,    # 输入音频
            '-c:v', 'copy',      # 复制视频流
            '-c:a', 'aac',       # 编码音频为AAC
            '-strict', 'experimental',
            output_path
        ]
        subprocess.run(cmd_merge, check=True)

        # 清理临时文件
        os.remove(temp_video)
        os.remove(temp_audio)


class ImageAudio2VideoPipeLine(ModelMixin, ConfigMixin):
    @register_to_config
    def __init__(self,
                 vae,
                 amd_model,
                 a2m_model,
                 audio_processor,
                 a2m_ref_num_frame:int = 8,
                 window:int = 16, # inference num of frames per loop
                 sample_size:int = 256,
                 need_motion_extract_model: bool = False,
                 **kwargs,
                 ):
        super().__init__()

        self.vae = vae
        self.amd_model = amd_model
        self.a2m_model = a2m_model
        self.audioprocessor = audio_processor
        self.window = window
        self.a2m_ref_num_frame = a2m_ref_num_frame
        self.need_motion_extract_model = need_motion_extract_model
        self.a2m_randomref_num = 8

        self.pixel_transforms =  transforms.Compose([
            transforms.Resize(sample_size),
            transforms.CenterCrop(sample_size),
            transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5], inplace=True),
        ])

        if window != amd_model.target_frame:
            amd_model.reset_infer_num_frame(window)

        assert self.window >= self.a2m_ref_num_frame , "ref frame should be shorter than infer frame"
    
    @torch.no_grad()
    def predict(self,
                ref_img:torch.Tensor,
                ref_audio:torch.Tensor,
                audio:torch.Tensor,
                motion_sample_step:int = 4,
                video_sample_step:int = 4,
                mask_ratio:float = 0.0,
                **kwargs):
        """
        Args:
            ref_img: N,F,C,H,W
            ref_audio: N,F,M,D
            audio: N,T,M,D
        Return:
            video: N,T+1,C,H,W
        """
        W = self.window
        R = self.a2m_ref_num_frame
        print(f'sample window {W}')
        print(f'ref num frame {R}')

        # vae
        vae = self.vae

        # ref_img -> latent  | n f d h w
        pad_length = R-ref_img.shape[1]
        ref_img_pad = torch.zeros((ref_img.shape[0],pad_length, *ref_img.shape[2:]), dtype=ref_img.dtype).to(ref_img.device)
        ref_img = torch.cat([ref_img_pad,ref_img],dim=1)
        ref_img = vae_encode(vae,ref_img) # n f d h w
        
        # randomref_img | n d h w
        randomref_img = ref_img[:,-1,:]

        # cache
        first_motion_ = None
        pre_motion = None
        pre_video = None

        # sample loop
        sample_num = audio.shape[1] // W
        offset = audio.shape[1] % W

        for i in range(sample_num):
            S = i*W
            E = (i+1)*W -1 

            # refmotion
            if i == 0:

                ref_motion = self.amd_model.extract_motion(ref_img,mask_ratio) # n,f,l,d
                # cur_ref = ref_img[:,-1:,:].repeat(1,R,1,1,1)
                # ref_motion = self.amd_model.extract_motion(cur_ref,mask_ratio)
                ref_audio = self.transform_a2m_ref(ref_audio).to(ref_motion.dtype)
                first_motion_ = ref_motion[:,-1:,:] # n 1 l d
            else:
                if self.need_motion_extract_model:
                    ref_motion = self.amd_model.extract_motion(pre_video[:,-R:,:],mask_ratio) # n,f,l,d
                else:
                    ref_motion = pre_motion[:,-R:,] # n,f,l,d
                ref_audio = audio[:,S-R:S,:].to(ref_motion.dtype)

                # ref_audio = self.transform_a2m_ref(audio[:,S-1:S,:]).to(ref_motion.dtype)
                # cur_ref_img = torch.cat([ref_img[:,:-1,:],pre_video[:,-1:,:]],dim=1)
                # ref_motion = self.amd_model.extract_motion(cur_ref_img) # n,f,l,d

            # sample motion
            # if i==0:
            #     randomref_motion = first_motion_.repeat(1,self.a2m_randomref_num,1,1) # n 8 l d
            # else:
            #     randomref_motion = torch.randint(low=0, high=pre_motion.shape[1], size=(self.a2m_randomref_num-1,))
            #     randomref_motion = pre_motion[:,randomref_motion,:,:] # n 8 l d
            #     randomref_motion = torch.cat([first_motion_,randomref_motion],dim=1)
            if i==0:
                randomref_motion = first_motion_.repeat(1,self.a2m_randomref_num,1,1) # n 8 l d
            else:
                # randomref_motion = torch.randint(low=0, high=pre_video.shape[1], size=(self.a2m_randomref_num-1,))
                # randomref_motion = pre_motion[:,randomref_motion,:,:] # n 8 l d
                # randomref_motion = torch.cat([first_motion_,randomref_motion],dim=1)
                randomref_motion = first_motion_.repeat(1,self.a2m_randomref_num,1,1) # n 8 l d

            
            motion_pre = self.a2m_model.sample(  ref_motion = ref_motion,
                                               randomref_motion = randomref_motion,
                                                audio =audio[:,S:E+1].to(ref_motion.dtype),
                                                ref_audio =ref_audio ,
                                                sample_step=motion_sample_step) # n f d h w     

            # sample video
            m2v_ref_img = randomref_img if i == 0 else pre_video[:,-1,:] # n d h w
            _,video_pre,_  = self.amd_model.sample_with_refimg_motion(ref_img = m2v_ref_img,
                                                                      motion = motion_pre,
                                                                      randomref_img = randomref_img,
                                                                      sample_step=video_sample_step) # n f d h w 

            # cache

            pre_motion = motion_pre if pre_motion is None else torch.cat([pre_motion,motion_pre],dim=1)
            pre_video = video_pre if pre_video is None else torch.cat([pre_video,video_pre],dim=1)

        if offset > 0:
            E = audio.shape[1] - 1
            S = E - W + 1
            if self.need_motion_extract_model:
                ref_motion = self.amd_model.extract_motion(pre_video[:,S-R:S,:],mask_ratio) # n,f,l,d
            else:
                ref_motion = pre_motion[:,S-R:S,:] # n,f,l,d
            ref_audio = audio[:,S-R:S,:].to(ref_motion.dtype)

            # sample motion
            randomref_motion = torch.randint(low=0, high=pre_motion.shape[1], size=(self.a2m_randomref_num,))
            randomref_motion = pre_motion[:,randomref_motion,:,:] # n 8 l d
            motion_pre = self.a2m_model.sample(  ref_motion = ref_motion,
                                               randomref_motion = randomref_motion,
                                                audio =audio[:,S:E+1].to(ref_motion.dtype),
                                                ref_audio =ref_audio ,
                                                sample_step=motion_sample_step) # n f d h w     

            # sample video
            m2v_ref_img = pre_video[:,S-1,:] # n d h w
            _,video_pre,_  = self.amd_model.sample_with_refimg_motion(ref_img = m2v_ref_img,
                                                                      motion = motion_pre,
                                                                      randomref_img = randomref_img,
                                                                      sample_step=video_sample_step) # n f d h w 

            pre_motion = torch.cat([pre_motion[:,:S,:],motion_pre],dim=1)
            pre_video = torch.cat([pre_video[:,:S,:],video_pre],dim=1)
            

        videos = torch.cat([ref_img[:,-1:,:],pre_video],dim=1) # n s*t+1 c h w

        
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        return videos

    @torch.no_grad()
    def sample(self,
               refimg_path:str,
               audio_path:str,
               motion_sample_step:int = 8,
               video_sample_step:int = 20,
               video_save_dir:str = "output",
               fps:int = 25,
               mask_ratio:float = 0.0,
               device = 'cuda:0'):

        # read img
        refimg = read_image(refimg_path) / 255.0
        refimg = self.pixel_transforms(refimg).unsqueeze(0).unsqueeze(0).to(device) # 1,1,C,H,W

        # read audio
        audio_emb, mel_feature = self.audioprocessor.preprocess(audio_path)
        audio_emb = audio_emb.unsqueeze(0).to(device) # 1,T,M,D
        audio_emb = audio_emb[:,:256,:]


        print("# AMD sampling ......")
        print("* A2M motion sample step:",motion_sample_step)
        print("* AMD video sample step:",video_sample_step)
        print("* Audio feature shape:",audio_emb.shape)

        # predict
        pre_video_latent = self.predict(ref_img=refimg,
                                        ref_audio=audio_emb[:,:1,:],
                                        audio=audio_emb[:,1:,:],
                                        motion_sample_step=motion_sample_step,
                                        video_sample_step=video_sample_step,
                                        mask_ratio=mask_ratio) 

        # vae decode       
        pre_video = vae_decode(self.vae,pre_video_latent)
        pre_video = ((pre_video / 2.0 + 0.5).clamp(0, 1) * 255).to(dtype=torch.uint8).cpu().contiguous()
        pre_video=pre_video[0,:]

        # save video
        name = f"Video-{Path(refimg_path).stem}-Audio-{Path(audio_path).stem}"
        os.makedirs(video_save_dir,exist_ok=True)
        output_path = os.path.join(video_save_dir,f"{name}.mp4")
        self.export_video_with_audio(video_tensor=pre_video,
                                    audio_path=audio_path,
                                    start_time=0.0,
                                    fps=fps,
                                    output_path=output_path)     

        print('new video saved at:',output_path)
        del pre_video,pre_video_latent

    def initial_blocks(self,ref_img,randomref_img,ref_audio,audio,window):
        N,T,_,_= audio.shape
        W = window
        blocks = []

        idx = 0
        for i in range(0,T-1,W):
            block = Block(idx = idx,
                          start_frame=i,
                          end_frame=i+W-1,

                          ref_img = self.transform_a2m_ref(ref_img) if i == 0 else None,
                          randomref_img= randomref_img,
                          ref_audio = self.transform_a2m_ref(ref_audio) if i == 0 else self.transform_a2m_ref(audio[:,i:i+1,:]),
                          audio = audio[:,i:i+W])

            blocks.append(block)
            idx +=1 
        return blocks

    def transform_a2m_ref(self,ref:torch.Tensor):

        R = self.a2m_ref_num_frame
        
        if ref.shape[1] >= R:
            result = ref[:,-R:,:]
        else:
            pad_length = R-ref.shape[1]
            pad = torch.zeros((ref.shape[0], pad_length,*ref.shape[2:]), dtype=ref.dtype).to(ref.device)
            result = torch.cat([pad,ref],dim=1)

        assert result.shape[1] == R, f"padding result.shape{result.shape} should be equal to {R}"
        return result


    def export_video_with_audio(self,video_tensor, audio_path, start_time, fps, output_path):
        """
        将生成的视频张体与音频文件合并，生成带音频的MP4文件。

        参数：
        video_tensor (torch.Tensor): 形状为 ( F, C, H, W) 的视频张量，值范围为0-255，uint8类型。
        audio_path (str): 输入的.wav音频文件路径。
        start_time (float): 音频开始时间（秒）。
        fps (int): 视频的帧率。
        output_path (str): 输出文件路径，应以.mp4结尾。
        """
        # 确认批次大小为1
        assert video_tensor.dim() == 4, "仅支持批次大小为1的视频"
        F, C, H, W = video_tensor.shape
        duration = F / fps  # 计算视频时长

        # 创建临时文件保存视频
        with tempfile.NamedTemporaryFile(suffix='.mp4', delete=False) as tmp_vid:
            temp_video = tmp_vid.name
            # 调整张量维度并保存视频
            video_frames = video_tensor.permute(0, 2, 3, 1)  # 转换为 (F, H, W, C)
            write_video(temp_video, video_frames, fps=fps, video_codec='libx264')

        # 创建临时文件保存截取的音频
        with tempfile.NamedTemporaryFile(suffix='.wav', delete=False) as tmp_aud:
            temp_audio = tmp_aud.name
            # 使用FFmpeg截取音频
            cmd_extract = [
                'ffmpeg',
                '-i', audio_path,        # 输入音频
                '-y',  # 覆盖输出文件
                '-ss', str(start_time),  # 开始时间
                '-t', str(duration),     # 持续时间
                '-acodec', 'copy',       # 直接复制音频流
                temp_audio
            ]
            subprocess.run(cmd_extract, check=True)

        # 合并音视频
        cmd_merge = [
            'ffmpeg',
            '-y',
            '-i', temp_video,    # 输入视频
            '-i', temp_audio,    # 输入音频
            '-c:v', 'copy',      # 复制视频流
            '-c:a', 'aac',       # 编码音频为AAC
            '-strict', 'experimental',
            output_path
        ]
        subprocess.run(cmd_merge, check=True)

        # 清理临时文件
        os.remove(temp_video)
        os.remove(temp_audio)
