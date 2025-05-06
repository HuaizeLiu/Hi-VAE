
import os
from argparse import ArgumentParser
from tqdm.auto import tqdm
from omegaconf import OmegaConf
from datetime import datetime
import einops
import numpy as np
import math
import shutil
import gc
import itertools

import torch
from torch.utils.data import DataLoader

from diffusers.optimization import get_scheduler
from diffusers import AutoencoderKL

from accelerate import Accelerator
from accelerate.logging import get_logger
from accelerate.utils import ProjectConfiguration, set_seed
from accelerate import DistributedDataParallelKwargs
from safetensors.torch import load_model, save_model

from model.utils import save_cfg, vae_encode, vae_decode, freeze, print_param_num,model_load_pretrain,save_videos_grid, frozen_model
from model import AMD_models,AMDModel
from model.loss import l2,LpipsMseLoss
from dataset.dataset import AMDConsecutiveVideo,AMDRandomPair

import debugpy
debugpy.connect(('localhost', 5684))

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
            raise print('Boolean value expected.')

    # data
    parser.add_argument('--train_datapath', type=str, default='', help='path to the root directory of datasets. All datasets will be under this directory.')
    parser.add_argument('--eval_datapath', type=str, default='', help='path to the root directory of datasets. All datasets will be under this directory.')
    parser.add_argument('--sample_size', type=str, default="(256,256)", help='Sample size as a tuple, e.g., (256, 256).')
    parser.add_argument('--sample_stride', type=int, default=4, help='data sample stride')
    parser.add_argument('--sample_n_frames', type=int, default=32, help='sample_n_frames.')
    parser.add_argument('--batch_size', type=int, default=4, help='batch size used in training.')
    parser.add_argument('--ref_drop_ratio', type=float, default=0.0, help='ref_drop_ratio')
    parser.add_argument('--dataset_type', type=str,default='AMDRandomPair', help='AMDRandomPair AMDConsecutiveVideo')
    parser.add_argument('--target_fps', type=int, default=8, help='sample_frames_target_fps.')
    parser.add_argument('--mask_ratio', type=float, default=None, help='wheather mask img_hidden_state')
    parser.add_argument('--camera_mask_ratio', type=float, default=None, help='wheather mask camera motion')
    parser.add_argument('--object_mask_ratio', type=float, default=None, help='wheather mask object motion')
    parser.add_argument('--use_camera',type= str2bool,default=False)
    parser.add_argument('--use_object',type= str2bool,default=False)
    parser.add_argument('--use_mask',type= str2bool,default=True)
    parser.add_argument('--frozen_name',type=str,default=None,help='frozen modules')

    # experiment
    parser.add_argument('--exp_root', default='/mnt/pfs-mc0p4k/cvg/team/didonglin/zqy/exp', required=True, help='exp_root')
    parser.add_argument('--name', default=f'{current_time}', required=True, help='name of the experiment to load.')
    parser.add_argument('--log_with',default='tensorboard',choices=['tensorboard', 'wandb'],help='accelerator tracker.')
    parser.add_argument('--seed', type=int, default=42, help='A seed for reproducible training.')
    
    parser.add_argument('--mp', type=str, default='fp16', choices=['fp16', 'bf16', 'no'], help='use mixed precision')
    parser.add_argument('--num_workers', type=int, default=16)
    parser.add_argument('--max_train_epoch', type=int, default=200000000, help='maximum number of training steps')
    parser.add_argument('--max_train_steps', type=int, default=100000, help='max_train_steps')
    

    parser.add_argument('--weight_decay', type=float, default=1e-4, help='weight decay in optimization.')
    parser.add_argument('--gradient_accumulation_steps', type=int, default=1, help='number of steps for gradient accumulation')
    parser.add_argument('--lr_warmup_steps', type=int, default=20, help='lr_warmup_steps')
    parser.add_argument("--max_grad_norm", default=1.0, type=float, help="Max gradient norm.")
    parser.add_argument('--eval_interval_step', type=int, default=1000, help='eval_interval_step')
    parser.add_argument('--checkpoint_total_limit', type=int, default=5, help='checkpoint_total_limit')
    parser.add_argument('--save_checkpoint_interval_step', type=int, default=100, help='save_checkpoint_interval_step')
    parser.add_argument("--lr_scheduler", type=str, default="constant",help=('The scheduler type to use. Choose between ["linear", "cosine", "cosine_with_restarts", "polynomial",'' "constant", "constant_with_warmup"]'))
    parser.add_argument("--resume_training", type=str, default=None,help=('input checkpoingt path'))
    
    # validation
    parser.add_argument('--n_save_fig', default=10, help='number of batches to save as image during validation.')
    parser.add_argument('--valid_batch_size', type=int, default=4, help='batch size to use for validation.')
    parser.add_argument('--val_num_step', type=int, default=2, help='number of epochs per validation.')

    # checkpoints
    parser.add_argument('--vae_version',type=str,default='/mnt/pfs-mc0p4k/cvg/team/didonglin/zqy/model-checkpoints/Huggingface-Model/sd-vae-ft-mse')
    
    # model cfg
    # ---------- AMD
    parser.add_argument('--amd_model_type', type=str, default='AMD_S', help='AMD_S,AMD_M,AMD_L')
    parser.add_argument('--amd_block_out_channels_down', type=str, default='[64,128,256,256]', help='duoframedownsample channels')
    parser.add_argument('--amd_image_patch_size', type=int, default=2, help='image patch')
    parser.add_argument('--amd_motion_patch_size', type=int, default=1, help='motion patch ')
    parser.add_argument('--amd_num_step', type=int, default=1000, help='diffusion step')
    parser.add_argument("--amd_from_pretrained", type=str, default=None,help='input checkpoingt path')
    parser.add_argument('--lr', type=float, default=2e-4, help='learning rate in optimization')
    parser.add_argument('--amd_from_config',type=str,default=None,help='config when resume training')
    parser.add_argument('--motion_drop_ratio',type=float, default=0.0, help='motion_loss_thread')
    
    # camera motion and object motion
    parser.add_argument('--motion_token_num',type=int, default=12, help='motion_token_num')
    parser.add_argument('--motion_token_channel',type=int, default=128, help='motion_token_channel')
    parser.add_argument('--camera_motion_token_num',type=int, default=12, help='motion_token_num')
    parser.add_argument('--camera_motion_token_channel',type=int, default=128, help='motion_token_channel')
    parser.add_argument('--camera_enc_num_layers',type=int, default=8, help='motion_enc_num_layers')
    parser.add_argument('--object_motion_token_num',type=int, default=12, help='motion_token_num')
    parser.add_argument('--object_motion_token_channel',type=int, default=128, help='motion_token_channel')
    parser.add_argument('--object_enc_num_layers',type=int, default=8, help='motion_token_num')
    # AMD encode 
    parser.add_argument('--use_filter',type= str2bool,default=True, help='wheather use frequency filter')
    parser.add_argument('--filter_num',type= float,default=0.4, help='frequency number')
    parser.add_argument('--use_grey',type= str2bool,default=False, help='wheather use frequency filter')
    parser.add_argument('--use_camera_down',type= str2bool,default=False, help='wheather use frequency filter')
    parser.add_argument('--motion_need_norm_out',type= str2bool,default=False)
    parser.add_argument('--use_motiontemporal',type= str2bool,default=False)

    # loss
    parser.add_argument('--use_regularizers',type= str2bool,default=False)

    parser.add_argument('--need_motion_transformer',type= str2bool,default=False)
    parser.add_argument('--diffusion_model_type',type=str,default='default')
    parser.add_argument('--motion_type',type=str,default='plus',help='the way different motion fusion')

    args = parser.parse_args()

    return args



# Main Func
def main():
    
    # --------------- Step1 : Exp Setting --------------- #
    # args
    args = get_cfg()

    # dir
    proj_dir = os.path.join(args.exp_root, args.name)
    video_save_dir = os.path.join(proj_dir,'sample')
    
    # Seed everything
    if args.seed is not None:
        set_seed(args.seed)
    
    # --------------- Step2 : Accelerator Initialize --------------- #
    # initialize accelerator.    
    project_config = ProjectConfiguration(project_dir=proj_dir)
    accelerator = Accelerator(
        gradient_accumulation_steps = args.gradient_accumulation_steps,
        mixed_precision             = args.mp,
        log_with                    = args.log_with,
        project_config              = project_config,
    )

    # --------------- Step3 : Save Exp Config --------------- #
    # save args
    if accelerator.is_main_process:
        save_cfg(proj_dir, args)

    # --------------- Step4 : Load Model & Datasets & Optimizer--------------- #
    # AMD !!
    if args.amd_from_config is not None:
        amd_model = AMDModel.from_config(args.amd_from_config)
    else:
        amd_model_kwargs = {
                        'image_inchannel':4,
                        'image_height':eval(args.sample_size)[0] // 8,
                        'image_width':eval(args.sample_size)[1] // 8,
                        'image_patch_size':args.amd_image_patch_size,
                        'video_frames':args.sample_n_frames,
                        'scheduler_num_step':args.amd_num_step,
                        'need_motion_transformer':args.need_motion_transformer,
                        'use_filter':args.use_filter,
                        'filter_num':args.filter_num,
                        'use_grey':args.use_grey,
                        "use_camera_down":args.use_camera_down,
                        "use_mask":args.use_mask,
                        "use_camera":args.use_camera,
                        "use_object":args.use_object,
                        
                        'use_regularizers':args.use_regularizers,
                        'use_motiontemporal':args.use_motiontemporal,

                        'motion_token_num':args.motion_token_num,
                        'motion_token_channel':args.motion_token_channel,

                        'object_motion_token_num':args.object_motion_token_num,
                        'object_motion_token_channel':args.object_motion_token_channel,
                        'object_enc_num_layers':args.object_enc_num_layers,

                        'camera_motion_token_num':args.camera_motion_token_num,
                        'camera_motion_token_channel':args.camera_motion_token_channel,
                        'camera_enc_num_layers':args.camera_enc_num_layers,

                        'motion_patch_size':args.amd_motion_patch_size, 
                        'motion_need_norm_out':args.motion_need_norm_out,
                        'diffusion_model_type':args.diffusion_model_type,
    } 
        amd_model = AMD_models[args.amd_model_type](**amd_model_kwargs)

    if args.amd_from_pretrained is not None:
        model_load_pretrain(amd_model,args.amd_from_pretrained,not_load_keyword='abcabcacbd',strict=False)
        if accelerator.is_main_process:
            print(f'######### load AMD weight from {args.amd_from_pretrained} #############')

    if args.frozen_name is not None:
        frozen_model(amd_model,args.frozen_name)
        print(f"*********AMD weight from {args.frozen_name} been frozen")

    vae = AutoencoderKL.from_pretrained(args.vae_version, subfolder="vae").requires_grad_(False)
    if accelerator.is_main_process:
        amd_model.save_config(proj_dir)
        print_param_num(amd_model)

    # Dataset 
    train_dataset = eval(args.dataset_type)(video_dir=args.train_datapath,
                        ref_drop_ratio=args.ref_drop_ratio,
                        sample_size=eval(args.sample_size),
                        sample_stride=args.sample_stride,
                        sample_n_frames=args.sample_n_frames,
                        target_fps = args.target_fps,
                        use_grey=args.use_grey,
                        use_mask=args.use_mask,
                        mask_video_ratio=args.mask_ratio)
    
    eval_dataset = eval(args.dataset_type)(video_dir=args.eval_datapath,
                        ref_drop_ratio=0.0,
                        sample_size=eval(args.sample_size),
                        sample_stride=args.sample_stride,
                        sample_n_frames=args.sample_n_frames,
                        target_fps = args.target_fps,
                        use_grey=args.use_grey,
                        use_mask=args.use_mask,
                        mask_video_ratio=args.mask_ratio)

    # train_dataloader = DataLoader(train_dataset, batch_size=args.batch_size,num_workers=args.num_workers, shuffle=True, collate_fn=train_dataset.collate_fn,pin_memory=True)
    # eval_dataloader = DataLoader(eval_dataset, batch_size=args.batch_size, num_workers=args.num_workers,shuffle=True, collate_fn=eval_dataset.collate_fn,pin_memory=True)
    train_dataloader = DataLoader(train_dataset, batch_size=args.batch_size,num_workers=args.num_workers, shuffle=True, pin_memory=True)
    eval_dataloader = DataLoader(eval_dataset, batch_size=args.batch_size, num_workers=args.num_workers,shuffle=True, pin_memory=True)

    # Optimizer & Learning Schedule

    optimizer = torch.optim.AdamW(amd_model.parameters(),lr=args.lr)
    lr_scheduler = get_scheduler(          # scheduler from diffuser, auto warm-up
        name = args.lr_scheduler,
        optimizer=optimizer,
        num_warmup_steps=args.lr_warmup_steps * args.gradient_accumulation_steps,
        num_training_steps=args.max_train_steps * args.gradient_accumulation_steps,
    )

    # --------------- Step5 : Accelerator Prepare --------------- #
    # Prepare
    device = accelerator.device
    amd_model, optimizer, training_dataloader,lr_scheduler = accelerator.prepare(
        amd_model, optimizer, train_dataloader,lr_scheduler
    )
    vae.to(device)

    if accelerator.is_main_process:
        accelerator.init_trackers('tracker')


    # -----------------------------------------------  Base Component(Progress & Tracker )  ------------------------------------------------ #

    # -------------------------------------------------------  Train   --------------------------------------------------------------------- #

    # Info!!
    total_batch_size = args.batch_size * accelerator.num_processes * args.gradient_accumulation_steps
    accelerator.print(f"{accelerator.state}")
    accelerator.print("***** Running training *****")
    accelerator.print(f"  Num examples = {len(train_dataset)}")
    accelerator.print(f"  Num Epochs = {args.max_train_epoch}")
    accelerator.print(f"  Instantaneous batch size per device = {args.batch_size}")
    accelerator.print(f"  Total train batch size (w. parallel, distributed & accumulation) = {total_batch_size}")
    accelerator.print(f"  Gradient Accumulation steps = {args.gradient_accumulation_steps}")
    global_step = 0
    train_loss = 0.0
    first_epoch = 0

    # resume training
    num_update_steps_per_epoch = math.ceil(len(train_dataloader) / args.gradient_accumulation_steps)
    if args.resume_training is not None:
        model_path = args.resume_training
        accelerator.print(f"Resuming from checkpoint {model_path}")
        accelerator.load_state( model_path)
        global_step = int(os.path.basename(model_path).split("-")[1])
        first_epoch = global_step // num_update_steps_per_epoch


    # progress bar for a epoch
    progress_bar = tqdm(
    range(0, args.max_train_steps),
    initial=global_step,
    desc="Steps",
    disable=not accelerator.is_local_main_process,  # Only show the progress bar once on each machine.
    )

    # val 
    @torch.inference_mode()
    def log_validation( amd_model, vae, eval_dataloader, device,accelerator = None,global_step = 0):

        accelerator.print(f"Running validation....\n")
        if accelerator is not None:
            amd_model = accelerator.unwrap_model(amd_model)

        # data
        for data in eval_dataloader:
            x = data['videos'].to(device) # N,T,C,H,W
            ref_img = data['ref_img'].to(device) # N,T,C,H,W
            x = x[:args.valid_batch_size,:]
            ref_img = ref_img[:args.valid_batch_size,:]
            if args.use_grey:
                grey_videos = data['grey_videos'].to(device)
                grey_videos = grey_videos[:args.valid_batch_size,:]
                ref_grey_img = data['ref_grey_img'].to(device)
                ref_grey_img = ref_grey_img[:args.valid_batch_size,:]
            if args.use_mask:
                camera_mask = data['camera_mask'].to(device)
                camera_mask = camera_mask[:args.valid_batch_size,:]
            break
        
        # encode
        z = vae_encode(vae,x) # N,T,C,H,W
        ref_img_z = vae_encode(vae,ref_img) # N,C,H,W
        if args.use_grey:
            z_grey = vae_encode(vae,grey_videos)
            ref_img_z_grey = vae_encode(vae,ref_grey_img)
        assert not torch.any(torch.isnan(z)), 'Finding *Nan in data after vae.'
        N,T,C,H,W = z.shape

        # forward
        sample_step = args.val_num_step
        if args.use_grey:
            if args.use_mask:
                video_start,sample,gt = amd_model.sample(video=z,ref_img=ref_img_z,video_grey=z_grey,ref_img_grey=ref_img_z_grey,camera_mask=camera_mask,sample_step = sample_step,mask_ratio=args.mask_ratio) # (n,c,h,w)
            else:
                video_start,sample,gt = amd_model.sample(video=z,ref_img=ref_img_z,video_grey=z_grey,ref_img_grey=ref_img_z_grey,sample_step = sample_step,camera_mask_ratio=args.camera_mask_ratio,object_mask_ratio=args.object_mask_ratio) # (n,c,h,w)
        else:
            video_start,sample,gt = amd_model.sample(video=z,ref_img=ref_img_z,sample_step = sample_step,camera_mask_ratio=args.camera_mask_ratio,object_mask_ratio=args.object_mask_ratio) # (n,c,h,w)
        
        # decode
        N = x.shape[0]
        T = sample.shape[0] // N

        video_start_cache = vae_decode(vae,video_start) # n,t,c,h,w
        video_start = ((video_start_cache / 2.0 + 0.5).clamp(0, 1) * 255).to(dtype=torch.uint8).cpu().contiguous().numpy()

        sample_cache = vae_decode(vae,sample) # n,t,c,h,w
        sample = ((sample_cache / 2.0 + 0.5).clamp(0, 1) * 255).to(dtype=torch.uint8).cpu().contiguous().numpy()

        gt_cache = vae_decode(vae,gt) # n,t,c,h,w
        gt = ((gt_cache / 2.0 + 0.5).clamp(0, 1) * 255).to(dtype=torch.uint8).cpu().contiguous().numpy()

        def image_log_k_sample_per_batch(sample,k=4):
            sample = sample[:,-k:,:]
            sample = einops.rearrange(sample,'n t c h w -> (n t) h w c') # (n*k,h,w,c)
            return sample

        for tracker in accelerator.trackers:
            if tracker.name == "tensorboard":
                video_start = image_log_k_sample_per_batch(video_start)
                np_start = np.stack([np.asarray(img) for img in video_start])

                images = image_log_k_sample_per_batch(sample)
                np_images = np.stack([np.asarray(img) for img in images])
                
                images_gt = image_log_k_sample_per_batch(gt)
                np_gt = np.stack([np.asarray(img) for img in images_gt])
                
                tracker.writer.add_images(f"video_start", np_start, global_step, dataformats="NHWC")
                tracker.writer.add_images(f"video_end_pre", np_images, global_step, dataformats="NHWC")
                tracker.writer.add_images(f"video_end", np_gt, global_step, dataformats="NHWC")
                
                # videos
                np_videos = np.stack([np.asarray(vid) for vid in sample])
                tracker.writer.add_video("validation_pre", np_videos, global_step, fps=8)

                np_videos = np.stack([np.asarray(vid) for vid in gt])
                tracker.writer.add_video("validation_gt", np_videos, global_step, fps=8)
        
        # save video
        sample = einops.rearrange(sample_cache.cpu(),'n t c h w -> n c t h w')
        gt = einops.rearrange(gt_cache.cpu(),'n t c h w -> n c t h w')
        mix = torch.cat([gt,sample],dim=0)

        save_path = os.path.join(video_save_dir,f'{global_step}-step{sample_step}.mp4')
        save_videos_grid(mix,save_path,rescale=True)
        
        gc.collect()
        torch.cuda.empty_cache()

    
    if accelerator.is_main_process:
        log_validation(amd_model, vae, eval_dataloader,device,accelerator, global_step)

    for epoch in range(first_epoch,args.max_train_epoch):
        accelerator.print(f"Epoch {epoch} start!!")
        if global_step >= args.max_train_steps:
            break
        
        # train loop in 1 epoch
        for step,data in enumerate(training_dataloader):
            if global_step >= args.max_train_steps:
                break
            
            # train !!!
            amd_model.train()

            with accelerator.accumulate(amd_model):  
                # input
                x = data['videos'] # N,T,C,H,W
                ref_img = data['ref_img'] # N,TC,H,W
                z = vae_encode(vae,x) # N,T,c,h,w
                ref_img = vae_encode(vae,ref_img) # N,C,H,W
                if args.use_grey:
                    grey_videos = data['grey_videos']
                    ref_grey_img = data['ref_grey_img']
                    z_grey = vae_encode(vae,grey_videos)
                    ref_img_z_grey = vae_encode(vae,ref_grey_img)
                if args.use_mask:
                    camera_mask = data['camera_mask'].to(device)

                assert not torch.any(torch.isnan(z)), 'Finding *Nan in data after vae.'
                assert not torch.any(torch.isnan(ref_img)), 'Finding *Nan in data after vae.'

                N,T,C,H,W = z.shape

                # if args.mask_video_ratio > 0:

                #     b_z, f_z, c_z, h_z, w_z = z.shape
                #     rand_mask = torch.rand(h_z, w_z).to(device=z.device)
                #     mask = rand_mask > args.mask_video_ratio
                #     mask = mask.unsqueeze(0).unsqueeze(0).unsqueeze(0)  
                #     mask = mask.expand(b_z, f_z, c_z, h_z, w_z) 
                #     z = z * mask

                #     if args.use_grey:
                #         z_grey = z_grey * mask
 
                # AMD forward
                if args.use_grey:
                    if args.use_mask:
                        pre,gt,loss_dict = amd_model(z,ref_img,video_grey=z_grey,ref_img_grey=ref_img_z_grey,camera_mask=camera_mask,mask_ratio=args.mask_ratio) # (n,c,h,w)
                    else:
                        pre,gt,loss_dict = amd_model(z,ref_img,video_grey=z_grey,ref_img_grey=ref_img_z_grey,camera_mask_ratio=args.camera_mask_ratio,object_mask_ratio=args.object_mask_ratio) # (n,c,h,w)
                else:
                    pre,gt,loss_dict = amd_model(z,ref_img,camera_mask_ratio=args.camera_mask_ratio,object_mask_ratio=args.object_mask_ratio) # (n,c,h,w)
                loss = loss_dict['loss']
                assert not torch.any(torch.isnan(loss)), 'Finding *Nan in data after loss.'

                # progress bar 
                if accelerator.sync_gradients:
                    global_step += 1

                    loss_cache = {}
                    # AMD log , Gather the losses across all processes for logging (if we use distributed training).
                    for key in loss_dict.keys():
                        avg_loss = accelerator.gather(loss_dict[key].repeat(args.batch_size)).mean()
                        train_loss = avg_loss.item() / args.gradient_accumulation_steps
                        loss_cache[key] = train_loss
                    
                    # tqdm
                    logs = {'global_step': loss_cache['loss']}
                    progress_bar.set_postfix(**logs)
                    progress_bar.update(1)

                    # print
                    txt = ''.join([f"{key:<10} {value:<10.6f}" for key,value in loss_cache.items()])
                    txt = f'Step {global_step:<5} :' + txt
                    accelerator.print(txt)

                    # log
                    for key,val in loss_cache.items():   
                        accelerator.log({key: val}, step=global_step)

                # backpropagate
                accelerator.backward(loss)
                if accelerator.sync_gradients:  # checking sync_gradients
                    params_to_clip = amd_model.parameters()
                    accelerator.clip_grad_norm_(params_to_clip, args.max_grad_norm) 

                optimizer.step()
                lr_scheduler.step()
                optimizer.zero_grad()

                # saving
                if global_step % args.save_checkpoint_interval_step == 0:
                    checkpoint_dir = os.path.join(proj_dir, "checkpoints")
                    save_path = os.path.join(checkpoint_dir,f"checkpoint-{global_step}")
                    accelerator.save_state(save_path)
                    # checkpoint limit 
                    if accelerator.is_main_process and args.checkpoint_total_limit is not None:
                        checkpoints = os.listdir(checkpoint_dir)
                        checkpoints = [d for d in checkpoints if d.startswith("checkpoint")]
                        checkpoints = sorted(checkpoints, key=lambda x: int(x.split("-")[1]))

                        # before we save the new checkpoint, we need to have at _most_ `checkpoint_total_limit ` checkpoints
                        if len(checkpoints) > args.checkpoint_total_limit:
                            num_to_remove = len(checkpoints) - args.checkpoint_total_limit 
                            removing_checkpoints = checkpoints[0:num_to_remove]
                            accelerator.print(f"removing checkpoints: {', '.join(removing_checkpoints)}")

                            for removing_checkpoint in removing_checkpoints:
                                removing_checkpoint = os.path.join(checkpoint_dir, removing_checkpoint)
                                shutil.rmtree(removing_checkpoint)


                # 6. eval !!!
                if global_step % args.eval_interval_step == 0 and accelerator.is_main_process:
                    log_validation(amd_model, vae, eval_dataloader,device,accelerator, global_step)

    # -------------------------------------------------------  End Train   --------------------------------------------------------------------- #
                        
    # Step Final : End accelerator
    accelerator.wait_for_everyone()
    accelerator.end_training() 



if __name__ == "__main__":
    # # --------- Argparse ----------- #
    # parser = argparse.ArgumentParser()
    # parser.add_argument("--args", type=str, required=True)
    # args = parser.parse_args()
    

    # # --------- Config ----------#
    # args = OmegaConf.load(args.args)
    # print(args.log_with)

    # --------- Train --------- #
    main() # 