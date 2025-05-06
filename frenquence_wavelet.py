# from diffusers import AutoencoderKL
import os
import einops
import imageio
import cv2

import numpy as np
import torchvision.transforms as transforms
import random
import torch
import pandas as pd
import shutil
import pywt

from diffusers import AutoencoderKL
from tqdm import tqdm
from decord import cpu, gpu
from decord import VideoReader


from model.wavelet import DWT, IWT
from model.utils import vae_encode, vae_decode

# import debugpy
# debugpy.connect(('localhost', 5680))

def main(video_path):

    grey_img = True
    device = "cpu"
    vae = AutoencoderKL.from_pretrained("/mnt/pfs-mc0p4k/tts/team/digital_avatar_group/sunwenzhang/qiyuan/model-checkpoints/sd-vae-ft-mse", subfolder="vae").requires_grad_(False).to(device=device)
    output_dir = "/mnt/pfs-gv8sxa/tts/dhg/liuhuaize/code/AMD2/exp_lhz/pywt_db2_level2_grey"

    #读取视频
    video_name = os.path.split(video_path)[-1]
    copy_path = os.path.join("/mnt/pfs-gv8sxa/tts/dhg/liuhuaize/code/AMD2/exp_lhz/example",video_name)
    shutil.copy(video_path, copy_path)

    video_reader = VideoReader(video_path, ctx=cpu(0))
    video_length = len(video_reader)
    video_fps = video_reader.get_avg_fps()

    sample_frames = 32
    sample_stride = 4
    clip_length = min(video_length, (sample_frames - 1) * sample_stride + 1)
    start_idx   = random.randint(0, video_length - clip_length)
    batch_index = np.linspace(start_idx, start_idx + clip_length - 1, sample_frames, dtype=int)
    if grey_img:
        frames = video_reader.get_batch(batch_index).asnumpy()
        gray_frames = np.zeros((frames.shape[0], frames.shape[1], frames.shape[2]), dtype=np.uint8)
        
        for i in range(frames.shape[0]):
            # 将 RGB 转换为 BGR，因为 OpenCV 默认使用 BGR 格式
            bgr_frame = cv2.cvtColor(frames[i, ...], cv2.COLOR_RGB2BGR)
            # 将 BGR 转换为灰度
            gray_frames[i, ...] = cv2.cvtColor(bgr_frame, cv2.COLOR_BGR2GRAY)

        
        videos = torch.from_numpy(gray_frames).unsqueeze(1).contiguous()
        videos = videos.repeat(1,3,1,1)
    else:
        videos = torch.from_numpy(video_reader.get_batch(batch_index).asnumpy()).permute(0, 3, 1, 2).contiguous()
    videos = videos / 255.0

    pixel_transforms = transforms.Compose([
            transforms.Resize(256),
            transforms.CenterCrop(256),
            transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5], inplace=True),
        ])
    
    
    videos = pixel_transforms(videos)

    #视频vae编码encode
    videos = videos.unsqueeze(0)
    video_latents = vae_encode(vae,videos)
    n,t,c,h,w = video_latents.shape

    # 小波变换视频滤波
    video_latents = einops.rearrange(video_latents, "n t c h w -> (n t) c h w")
    
    # 基于pywt实现
    # cA, (cH, cV, cD) = pywt.dwt2(video_latents, 'db2')
    cA, (cH, cV, cD), (cH1, cV1, cD1) = pywt.wavedec2(video_latents, 'db2',level=2)

    # 基于自己写的函数实现
    # dwt= DWT()
    # video_latents = 2 * video_latents - 1.0
    # cA, cH, cV, cD = dwt(video_latents)

    cA = torch.from_numpy(einops.rearrange(cA, "(n t) c h w -> n t c h w", n=n))
    cH = torch.from_numpy(einops.rearrange(cH, "(n t) c h w -> n t c h w", n=n))
    cV = torch.from_numpy(einops.rearrange(cV, "(n t) c h w -> n t c h w", n=n))
    cD = torch.from_numpy(einops.rearrange(cD, "(n t) c h w -> n t c h w", n=n))

    #视频vae编码decode
    LF_video = vae_decode(vae,cA)
    LF_video = ((LF_video / 2.0 + 0.5).clamp(0, 1) * 255).to(dtype=torch.uint8).cpu().contiguous().numpy()
    LF_video = einops.rearrange(LF_video,"n t c h w -> n t h w c").squeeze(0)

    cHF_video = vae_decode(vae,cH)
    cHF_video = ((cHF_video / 2.0 + 0.5).clamp(0, 1) * 255).to(dtype=torch.uint8).cpu().contiguous().numpy()
    cHF_video = einops.rearrange(cHF_video,"n t c h w -> n t h w c").squeeze(0)

    cVF_video = vae_decode(vae,cV)
    cVF_video = ((cVF_video / 2.0 + 0.5).clamp(0, 1) * 255).to(dtype=torch.uint8).cpu().contiguous().numpy()
    cVF_video = einops.rearrange(cVF_video,"n t c h w -> n t h w c").squeeze(0)

    cDF_video = vae_decode(vae,cD)
    cDF_video = ((cDF_video / 2.0 + 0.5).clamp(0, 1) * 255).to(dtype=torch.uint8).cpu().contiguous().numpy()
    cDF_video = einops.rearrange(cDF_video,"n t c h w -> n t h w c").squeeze(0)

    LF_path = os.path.join(output_dir,"low_frq_"+video_name)
    cHF_path = os.path.join(output_dir,"ch_high_frq_"+video_name)
    cVF_path = os.path.join(output_dir,"cv_high_frq_"+video_name)
    cDF_path = os.path.join(output_dir,"cd_high_frq_"+video_name)

    imageio.mimsave(LF_path, LF_video, fps=8)
    imageio.mimsave(cHF_path, cHF_video, fps=8)
    imageio.mimsave(cVF_path, cVF_video, fps=8)
    imageio.mimsave(cDF_path, cDF_video, fps=8)





if __name__ == "__main__":
    video_csv_path = "/mnt/pfs-gv8sxa/tts/dhg/wenzhangsun/.sun/dataset/webvid/vidtwin_csv/final.csv"
    metadata = pd.read_csv(
                video_csv_path,
                on_bad_lines="skip",
                encoding="ISO-8859-1",
                engine="python",
                sep=",",)

    for i,file_path in enumerate(tqdm(metadata['videos'])):
        if i==10:
            break
        main(file_path)