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

from diffusers import AutoencoderKL
from tqdm import tqdm
from decord import cpu, gpu
from decord import VideoReader
import glob

from model.frequency_utils import gaussian_low_pass_filter, freq_3d_filter
from model.utils import vae_encode, vae_decode

# import debugpy
# debugpy.connect(('localhost', 5680))

def main(video_path):

    grey_img = False
    device = "cpu"
    vae = AutoencoderKL.from_pretrained("/mnt/pfs-mc0p4k/tts/team/digital_avatar_group/sunwenzhang/qiyuan/model-checkpoints/sd-vae-ft-mse", subfolder="vae").requires_grad_(False).to(device=device)
    output_dir = "/mnt/pfs-gv8sxa/tts/dhg/liuhuaize/dataset/latents_role/c16t16m32t8_object"

    #读取视频
    video_name = os.path.split(video_path)[-1]
    # copy_path = os.path.join("/mnt/pfs-gv8sxa/tts/dhg/liuhuaize/code/AMD2/exp_lhz/example",video_name)
    # shutil.copy(video_path, copy_path)

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
    # n,t,c,h,w = videos.shape

    # # 快速傅里叶视频滤波
    freq_filter = gaussian_low_pass_filter([t, h, w],d_s=0.4,d_t=0.4)
    freq_filter = freq_filter.unsqueeze(0).unsqueeze(0).repeat(n, c, 1, 1, 1)
    freq_filter = freq_filter.to(device=device)

    video_latents = einops.rearrange(video_latents, "n t c h w -> n c t h w")
    # video_latents = einops.rearrange(videos, "n t c h w -> n c t h w")
    LF_video_latents, HF_video_latents = freq_3d_filter(video_latents, freq_filter)
    LF_video_latents = einops.rearrange(LF_video_latents, "n c t h w -> n t c h w")
    HF_video_latents = einops.rearrange(HF_video_latents, "n c t h w -> n t c h w")

    videos = ((videos / 2.0 + 0.5).clamp(0, 1) * 255).to(dtype=torch.uint8).cpu().contiguous().numpy()
    videos = einops.rearrange(videos,"n t c h w -> n t h w c").squeeze(0)
    # #视频vae编码decode
    LF_video = vae_decode(vae,LF_video_latents)
    LF_video = ((LF_video / 2.0 + 0.5).clamp(0, 1) * 255).to(dtype=torch.uint8).cpu().contiguous().numpy()
    LF_video = einops.rearrange(LF_video,"n t c h w -> n t h w c").squeeze(0)

    HF_video = vae_decode(vae,HF_video_latents)
    HF_video = ((HF_video / 2.0 + 0.5).clamp(0, 1) * 255).to(dtype=torch.uint8).cpu().contiguous().numpy()
    HF_video = einops.rearrange(HF_video,"n t c h w -> n t h w c").squeeze(0)

    LF_path = os.path.join(output_dir,"low_"+video_name)
    HF_path = os.path.join(output_dir,"high_"+video_name)

    # imageio.mimsave(LF_path, LF_video, fps=8)
    imageio.mimsave(HF_path, HF_video, fps=8)
    # path = os.path.join(output_dir,"Grey_"+video_name)
    # imageio.mimsave(path, videos, fps=8)





if __name__ == "__main__":
    # video_csv_path = "/mnt/pfs-gv8sxa/tts/dhg/wenzhangsun/.sun/dataset/webvid/vidtwin_csv/final.csv"
    # metadata = pd.read_csv(
    #             video_csv_path,
    #             on_bad_lines="skip",
    #             encoding="ISO-8859-1",
    #             engine="python",
    #             sep=",",)

    # for i,file_path in enumerate(tqdm(metadata['videos'])):
    #     if i==10:
    #         break
    #     main(file_path)
    # file_path = '/mnt/pfs-gv8sxa/tts/dhg/liuhuaize/code/AMD2/inference_amd_exp_new/c8t8m16t8-UCF-101-TEST-2025-4-24-21:32/v_ApplyEyeMakeup_g01_c04_gt.mp4'
    # main(file_path)
    video_dir = '/mnt/pfs-gv8sxa/tts/dhg/liuhuaize/dataset/latents_role/GT_256_8'
    video_list =  glob.glob(os.path.join(video_dir, '**', '*.mp4*'), recursive=True)
    for i,file_path in enumerate(video_list):
        main(file_path)