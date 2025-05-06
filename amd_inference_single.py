import os
from argparse import ArgumentParser

from tqdm.auto import tqdm
from omegaconf import OmegaConf
from datetime import datetime
import numpy as np
import torch
from torchvision.io import write_video
from torch.utils.data import DataLoader
from diffusers.optimization import get_scheduler
from diffusers import AutoencoderKL
import glob
from pathlib import Path

from accelerate.utils import ProjectConfiguration, set_seed

from model.utils import save_cfg, vae_encode,cat_video,_freeze_parameters,vae_decode,save_videos_grid,model_load_pretrain
from model import AMDModel,AMD_models
from model.loss import l2
from safetensors.torch import load_model
from omegaconf import OmegaConf
import einops
from model.model_AMD import AMDModel,AMDModel_New
from model.pipeline import A2VPipeLine,AMDTestPipeLine,A2VInferencePipeLine,AMDPipeLine,AMDPipeLine_single, AMDPipeLine_single_cross
from model.utils import find_latest_checkpoint
from model import set_vis_atten_flag

# import debugpy
# debugpy.connect(('localhost', 5684))

set_vis_atten_flag(False)

now = datetime.now()
current_time = f'{now.year}-{now.month}-{now.day}-{now.hour}:{now.minute}'

def get_cfg():
    parser = ArgumentParser()
    
    def str2bool(v):
        if isinstance(v, bool):
            return v
        if v.lower() in ('yes', 'true', 't', 'y', '1'):
            return True
        elif v.lower() in ('no', 'false', 'f', 'n', '0'):
            return False
        else:
            raise argparse.ArgumentTypeError('Boolean value expected.')

    # data
    parser.add_argument('--camera_mask_ratio',type=float,default=None)
    parser.add_argument('--object_mask_ratio',type=float,default=None)
    parser.add_argument('--data_path_1', type=str)
    parser.add_argument('--data_path_2', type=str)
    parser.add_argument('--save_dir', default='/mnt/pfs-mc0p4k/cvg/team/didonglin/zqy/exp')
    parser.add_argument('--exp_name', default='default')
    parser.add_argument('--seed', type=int, default=None, help='A seed for reproducible training.')
    parser.add_argument('--sample_window', type=int, default=8, help='sample window')
    parser.add_argument('--video_sample_step',type=int,default=10)
    parser.add_argument('--num_workers',type=int,default=16)
    parser.add_argument('--pipeline_type',type=str,default='AMDPipeLine')
    parser.add_argument('--fps',type=int,default=25)
    parser.add_argument('--drop_prev_img',type=str2bool,default=False)
    parser.add_argument('--use_grey',type= str2bool,default=False, help='wheather use grey motion')
    parser.add_argument('--inference_single',type=str2bool,default=False)

    parser.add_argument('--use_cross',type= str2bool,default=True)
    # checkpoints
    parser.add_argument('--vae_version',type=str,default='/mnt/pfs-gv8sxa/tts/dhg/zqy/model/sd-vae-ft-mse')
    parser.add_argument('--amd_model_type',type=str,default='AMDModel',help='AMDModel,AMDModel_Rec')
    parser.add_argument('--amd_config',type=str, default="/mnt/pfs-gv8sxa/tts/dhg/zqy/iccvexp/amd_ablation/amd-s-t1-d512-doubleref-balance-freeze/config.json", help='amd model config path')
    parser.add_argument('--amd_ckpt',type=str,default="/mnt/pfs-gv8sxa/tts/dhg/zqy/iccvexp/amd_ablation/amd-s-t1-d512-doubleref-balance-freeze/checkpoints",help="amd model checkpoint path")

    args = parser.parse_args()

    return args


# Main Func
def get_pipe():
    
    # args
    args = get_cfg()
    
    # Seed everything
    if args.seed is not None:
        set_seed(args.seed)

    device = torch.device('cuda:0')

    # get Model 
    # VAE
    vae = AutoencoderKL.from_pretrained(args.vae_version, subfolder="vae").to(device).requires_grad_(False)
    
    # window = args.sample_window
    # if args.amd_config["args.amd_config"] != window:
    #     args.amd_config["args.amd_config"] = window
    
    # DAE
    config = eval(args.amd_model_type).load_config(args.amd_config)
    window = args.sample_window
    if config["video_frames"] != window:
        config["video_frames"] = window

    # config['use_camera'] = False


    amd_model = eval(args.amd_model_type).from_config(config).to(device).requires_grad_(False)
    amd_ckpt = find_latest_checkpoint(args.amd_ckpt) if '.safetensors' not in args.amd_ckpt else args.amd_ckpt
    load_model(amd_model,amd_ckpt)

    # if args.inference_single:
    #     return args,AMDPipeLine_single(vae=vae,
    #                                 amd_model=amd_model,
    #                                 window=window,
    #                                 use_grey = args.use_grey)
    # else:
    if args.use_cross:
        return args,AMDPipeLine_single_cross(vae=vae,
                            amd_model=amd_model,
                            window=window,
                            use_grey = args.use_grey)
    else:
        return args,AMDPipeLine_single(vae=vae,
                            amd_model=amd_model,
                            window=window,
                            use_grey = args.use_grey)
    


if __name__ == '__main__':
    args,pipeline = get_pipe()

    # save_dir = os.path.join(args.save_dir,f"{args.exp_name}-{current_time}")
    save_dir = os.path.join(args.save_dir,f"{args.exp_name}")
    os.makedirs(save_dir,exist_ok=True)

    # videos_path =  glob.glob(os.path.join(args.data_dir, '**', '*.avi'), recursive=True)
    # videos_path =  glob.glob(os.path.join(args.data_dir, '**', '*.mp4*'), recursive=True)
    # save_videos_path = glob.glob(os.path.join(save_dir, '**', '*.mp4*'), recursive=True)
    # save_videos_name = [Path(v).stem for v in save_videos_path]

    save_videos_name = Path(args.data_path_1).stem
    save_videos_path = os.path.join(save_dir, save_videos_name+'.mp4')
    if args.use_cross:
        pipeline.sample(video_path_1=args.data_path_1,
                        video_path_2=args.data_path_2,
                        output_path=save_videos_path,
                        video_sample_step = args.video_sample_step,
                        fps= args.fps,
                        drop_prev_img=args.drop_prev_img,
                        camera_mask_ratio=args.camera_mask_ratio,
                        object_mask_ratio=args.object_mask_ratio,)
    else:
        pipeline.sample(video_path=args.data_path_1,
                        output_path=save_videos_path,
                        video_sample_step = args.video_sample_step,
                        fps= args.fps,
                        drop_prev_img=args.drop_prev_img,
                        camera_mask_ratio=args.camera_mask_ratio,
                        object_mask_ratio=args.object_mask_ratio,)
    # for v in videos_path:
    #     try:
    #         name = Path(v).stem
    #         if name in save_videos_name:
    #             continue
    #         save_path = os.path.join(save_dir,name+'.mp4')
    #         save_path_gt = os.path.join("/mnt/pfs-gv8sxa/tts/dhg/liuhuaize/dataset/reconstruction_view/gt_256_video",name+'_gt.mp4')
    #         pipeline.sample(video_path=v,
    #                         output_path=save_path,
    #                         output_path_gt=save_path_gt,
    #                         video_sample_step = args.video_sample_step,
    #                         fps= args.fps,
    #                         drop_prev_img=args.drop_prev_img,
    #                         camera_mask_ratio=args.camera_mask_ratio,
    #                         object_mask_ratio=args.object_mask_ratio,)
    #     except Exception as e:
    #         print("error:",e)
    #         continue

        # pipeline.sample(video_path=v,
        #                 output_path=save_path,
        #                 video_sample_step = args.video_sample_step,
        #                 fps= args.fps,
        #                 drop_prev_img=args.drop_prev_img,
        #                 )