import torch
import os
import json
import cv2
import numpy as np
from einops import rearrange
from PIL import Image
import imageio
import cv2
import einops
from matplotlib import pyplot as plt
import re

import torch
import torchvision

from safetensors.torch import load_model, save_model,safe_open
import einops
def cat_video(amd_model,z_video:torch.Tensor,ref_img:torch.Tensor,motion_seq_len:int=15):
    '''
    Args:
        z_video (torch.Tensor): shape = (B,F,C,H,W)
        motion_seq_len (torch.Tensor): motion transformer output 
        ref_img : B,C,H,W
    '''
    n,f,_,_,_ = z_video.shape
    assert (f - 1) % motion_seq_len == 0, f"no. frames miss match"
    motion_list = []
    for i in range(1,f,motion_seq_len):
        motion_list.append(amd_model.extract_motion(z_video[:,i-1:i+motion_seq_len],None))

    # ref_motion
    ref_frame = ref_img.unsqueeze(1)
    mix_frame = ref_frame.repeat(1,2,1,1,1)
    ref_motion = amd_model.extract_motion(mix_frame,None) # 4,1,256,4,4
    ref_motion = ref_motion.squeeze(1)
    return torch.concat(motion_list,dim=1),ref_motion


def save_cfg(path, args):
    os.makedirs(path, exist_ok=True)
    # if not os.path.exists(f'{path}/args.txt'):
    with open(f'{path}/args.txt', 'w') as f:
        json.dump(args.__dict__, f, indent=2)
    # else:
    print(f'Experiment of the same name already exists. Are you trying to resume training?')
        # assert args.resume > 0, f'Experiment of the same name already exists. Are you trying to resume training?'
        
def _freeze_parameters(model):
    for param in model.parameters():
        param.requires_grad = False
    model._requires_grad = False
    return model

def save_videos_grid(videos: torch.Tensor, path: str, rescale=False, n_rows=8, fps=8):
    """
    Args:
        videos: videos of shape (b, c ,t, h, w) # Default videos in [0,1]
        rescale: rescale the videos to [0, 1] # (True if videos are in [-1, 1]) 
    """
    videos = rearrange(videos, "b c t h w -> t b c h w")
    outputs = []
    for x in videos:
        x = torchvision.utils.make_grid(x, nrow=n_rows)
        x = x.transpose(0, 1).transpose(1, 2).squeeze(-1)
        if rescale:
            x = (x + 1.0) / 2.0  # -1,1 -> 0,1
        x = (x * 255).clamp(0, 255).numpy().astype(np.uint8)
        outputs.append(x)
    os.makedirs(os.path.dirname(path), exist_ok=True)
    # export_to_video(outputs, output_video_path=path, fps=fps)

    imageio.mimsave(path, outputs, fps=fps)


def save_images_grid(images, grid_size, save_path):
    """
    将多个 PIL.Image.Image 对象组成一个网格，并保存为 .png 文件

    :param images: List of PIL.Image.Image 对象
    :param grid_size: (rows, cols) 格式的元组，表示网格的行数和列数
    :param save_path: 保存图片的路径
    """
    rows, cols = grid_size
    assert len(images) <= rows * cols, "图像数量多于网格容量"

    # 获取每个图像的尺寸（假设所有图像尺寸相同）
    img_width, img_height = images[0].size

    # 创建一个新的图像，大小为网格的总尺寸
    grid_img = Image.new('RGB', (cols * img_width, rows * img_height))

    # 将每个图像粘贴到网格中
    for idx, img in enumerate(images):
        row = idx // cols
        col = idx % cols
        grid_img.paste(img, (col * img_width, row * img_height))

    # 保存结果图像
    grid_img.save(save_path)


def print_param_num(model):
    """
    打印模型的参数数量
    """
    total_params = sum(p.numel() for p in model.parameters())

    train_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    freeze_params = sum(p.numel() for p in model.parameters() if p.requires_grad is False)

    print(f'#### #### 模型总参数数量:{total_params / 1_000_000:.2f}M')
    print(f'####  模型训练数量:{train_params / 1_000_000:.2f}M')
    print(f'####  模型冻结参数数量:{freeze_params / 1_000_000:.2f}M')


def vae_encode(vae,latents):
    # video : N,T,C,H,W
    latents_type = None
    
    if len(latents.shape) == 5:
        N,T,C,H,W = latents.shape
        latents_type = 'video'
        latents = einops.rearrange(latents,'n t c h w -> (n t) c h w')
    else:
        N,C,H,W = latents.shape
        latents_type = 'image'
        
    with torch.no_grad():
        latents = vae.encode(latents).latent_dist
        latents = latents.sample()
        latents = latents * 0.18215
    
    if latents_type == 'video':
        latents = einops.rearrange(latents,'(n t) c h w -> n t c h w',n=N,t=T)
    return latents

def vae_decode(vae,latents):
    latents_type = None
    
    if len(latents.shape) == 5:
        N,T,C,H,W = latents.shape
        latents_type = 'video'
        latents = einops.rearrange(latents,'n t c h w -> (n t) c h w')
    else:
        N,C,H,W = latents.shape
        latents_type = 'image'

    latents = 1 / 0.18215 * latents
    with torch.no_grad():
        latents = vae.decode(latents).sample # (nt)chw
        
    if latents_type == 'video':
        latents = einops.rearrange(latents,'(n t) c h w -> n t c h w',n=N,t=T)

    return latents

def latents_to_videos(latents,batch_size):
    if len(latents.shape) == 4:
        M,C,H,W = latents.shape
        T = M // batch_size
        latents = einops.rearrange(latents,'(bt) c h w -> b t c h w',b=batch_size,t=T)

    videos = ((latents / 2.0 + 0.5).clamp(0, 1) * 255).to(dtype=torch.uint8).cpu().contiguous()
    return videos


def freeze(model):
    for param in model.parameters():
        param.requires_grad = False

def frozen_model(model,frozen_name):
    for name, module in model.named_modules():
        if frozen_name in name:
            for params in module.parameters():
                params.requires_grad_(False)

def model_load_pretrain(model, path, not_load_keyword='decoder',strict=False):
    tensors = {}
    with safe_open(path, framework="pt") as f:
        for k in f.keys():
            if not_load_keyword not in k:
                tensors[k] = f.get_tensor(k)

    model.load_state_dict(tensors,strict=strict)

def display_images(images,vae=None,need_decode=False):
    if len(images.shape) == 5:
        images = einops.rearrange(images, 'b t c h w -> (b t) c h w')
        
    if need_decode:
        images = vae_decode(vae,images)

    t, c, h, w = images.shape
    fig, axs = plt.subplots(1, t, figsize=(t * 3.0, 3.0))  # 每张图像宽3英寸
    
    if t == 1:
        axs = [axs]
    # 遍历每张图像
    for i in range(t):
        image = images[i]
        image = image.permute(1, 2, 0)
        
        image_np = ((image / 2.0 + 0.5).clamp(0, 1) * 255).to(dtype=torch.uint8).cpu().contiguous().numpy()

        
        # image = (image / 2.0 + 0.5).clamp(0, 1)
        
        # image_np = image.numpy()
        
        axs[i].imshow(image_np)
        axs[i].axis('off')  # 关闭坐标轴
        axs[i].set_title(f"Image {i+1}")
    plt.tight_layout()
    plt.show()

def find_latest_checkpoint(checkpoint_dir):
    max_step = -1
    latest_path = None
    
    # 遍历目标目录
    for name in os.listdir(checkpoint_dir):
        # 用正则匹配数字部分
        match = re.match(r"checkpoint-(\d+)$", name)
        if match:
            current_step = int(match.group(1))
            # 更新最大值
            if current_step > max_step:
                max_step = current_step
                latest_path = os.path.join(checkpoint_dir, name)
    
    if latest_path is None:
        raise ValueError(f"No valid checkpoint found in {checkpoint_dir}")
    
    result = os.path.join(latest_path,'model.safetensors')
    
    return result


