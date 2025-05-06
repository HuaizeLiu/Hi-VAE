import os
from argparse import ArgumentParser
from tqdm.auto import tqdm
from omegaconf import OmegaConf
from datetime import datetime
import numpy as np
import math
import shutil
import gc
from accelerate import DistributedDataParallelKwargs
import torch
from torch.utils.data import DataLoader
from diffusers.optimization import get_scheduler
from diffusers import AutoencoderKL

from accelerate import Accelerator
from accelerate.utils import ProjectConfiguration, set_seed

from model.utils import save_cfg, vae_encode,cat_video,_freeze_parameters,vae_decode,save_videos_grid,model_load_pretrain
from model import AMDModel,AMD_models
from model.loss import l2
from safetensors.torch import load_model
from dataset.dataset import A2MVideoAudio,A2MVideoAudioPose,A2MVideoAudioPoseRandomRef
from omegaconf import OmegaConf
import einops
from model.model_A2M import (A2MModel_MotionrefOnly_LearnableToken,
                             A2MModel_CrossAtten_Audio,
                             A2MModel_CrossAtten_Pose,
                             A2MModel_CrossAtten_Audio_Pose,
                             A2MModel_CrossAtten_Audio_PosePre)
from model.model_AMD import AMDModel,AMDModel_Rec


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
    parser.add_argument('--trainset', type=str, default='/mnt/pfs-mc0p4k/tts/team/digital_avatar_group/sunwenzhang/qiyuan/code/AMD_linear/dataset/path/train_video_with_audio.pkl', help='trainset index file path')
    parser.add_argument('--evalset', type=str, default='/mnt/pfs-mc0p4k/tts/team/digital_avatar_group/sunwenzhang/qiyuan/code/AMD_linear/dataset/path/eval_video_with_audio.pkl', help='evalset index file path')

    parser.add_argument('--sample_size', type=str, default="(256,256)", help='Sample size as a tuple, e.g., (256, 256).')
    parser.add_argument('--sample_stride', type=int, default=1, help='data sample stride')
    parser.add_argument('--sample_n_frames', type=int, default=31, help='sample_n_frames.')
    parser.add_argument('--batch_size', type=int, default=4, help='batch size used in training.')
    parser.add_argument('--path_type', type=str, default='file', choices=['file', 'dir'], help='path type of the dataset.')

    # experiment
    parser.add_argument('--exp_root', default='/mnt/pfs-mc0p4k/cvg/team/didonglin/zqy/exp', required=True, help='exp_root')
    parser.add_argument('--name', default=f'{current_time}', required=True, help='name of the experiment to load.')
    parser.add_argument('--log_with',default='tensorboard',choices=['tensorboard', 'wandb'],help='accelerator tracker.')
    parser.add_argument('--seed', type=int, default=None, help='A seed for reproducible training.')
    
    parser.add_argument('--mp', type=str, default='fp16', choices=['fp16', 'bf16', 'no'], help='use mixed precision')
    parser.add_argument('--num_workers', type=int, default=16)
    parser.add_argument('--max_train_epoch', type=int, default=200000000, help='maximum number of training steps')
    parser.add_argument('--max_train_steps', type=int, default=100000, help='max_train_steps')
    
    parser.add_argument('--lr', type=float, default=2e-4, help='learning rate in optimization')
    parser.add_argument('--weight_decay', type=float, default=1e-4, help='weight decay in optimization.')
    parser.add_argument('--gradient_accumulation_steps', type=int, default=1, help='number of steps for gradient accumulation')
    parser.add_argument('--lr_warmup_steps', type=int, default=20, help='lr_warmup_steps')
    parser.add_argument("--max_grad_norm", default=1.0, type=float, help="Max gradient norm.")
    parser.add_argument('--eval_interval_step', type=int, default=1000, help='eval_interval_step')
    parser.add_argument('--checkpoint_total_limit', type=int, default=3, help='checkpoint_total_limit')
    parser.add_argument('--save_checkpoint_interval_step', type=int, default=100, help='save_checkpoint_interval_step')
    parser.add_argument("--lr_scheduler", type=str, default="constant",help=('The scheduler type to use. Choose between ["linear", "cosine", "cosine_with_restarts", "polynomial",'' "constant", "constant_with_warmup"]'))
    parser.add_argument("--resume_from_checkpoint", type=str, default=None,help='input checkpoingt path') 
    parser.add_argument('--motion_sample_step', type=int, default=4, help='checkpoint_total_limit')
    parser.add_argument('--video_sample_step', type=int, default=4, help='checkpoint_total_limit')
    parser.add_argument('--a2m_from_pretrained',type=str, default=None)
    parser.add_argument('--dataset_type',type=str,default='A2MVideoAudioPose')

    # checkpoints
    parser.add_argument('--vae_version',type=str,default='/mnt/pfs-mc0p4k/cvg/team/didonglin/zqy/model-checkpoints/Huggingface-Model/sd-vae-ft-mse')
    parser.add_argument('--amd_model_type',type=str,default='AMDModel',help='AMDModel,AMDModel_Rec')
    parser.add_argument('--amd_config',type=str, default="/mnt/pfs-mc0p4k/tts/team/digital_avatar_group/sunwenzhang/qiyuan/exp/amd-m-mae-s-1026-linear-final/config.json", help='amd model config path')
    parser.add_argument('--amd_ckpt',type=str,default="/mnt/pfs-mc0p4k/tts/team/digital_avatar_group/sunwenzhang/qiyuan/code/AMD_linear/ckpt/checkpoint-157000/model_1.safetensors",help="amd model checkpoint path")
    parser.add_argument('--a2m_config',type=str, default="/mnt/pfs-mc0p4k/tts/team/digital_avatar_group/sunwenzhang/qiyuan/code/AMD_linear/config/Audio2Motion.yaml")
    parser.add_argument('--use_sample_timestep',action="store_true")
    parser.add_argument('--sample_timestep_m',type=float,default=0.5)
    parser.add_argument('--sample_timestep_s',type=float,default=1.0)

    # model
    parser.add_argument('--model_type',type=str,default='type1',help='model type : type1 or type2')


    # TODO
    # parser.add_argument('--mae_config',type=str,default="")
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
    # ddp_kwargs = DistributedDataParallelKwargs(find_unused_parameters=True)
    # accelerator = Accelerator(
    #     gradient_accumulation_steps = args.gradient_accumulation_steps,
    #     mixed_precision             = args.mp,
    #     log_with                    = args.log_with,
    #     project_config              = project_config,
    #     kwargs_handlers             =[ddp_kwargs]
    # )

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
    # Model CFG
    # get Model 
    device = accelerator.device
    amd_model = eval(args.amd_model_type).from_config(eval(args.amd_model_type).load_config(args.amd_config)).to(device).requires_grad_(False)
    load_model(amd_model,args.amd_ckpt)
    amd_model.diffusion_transformer.to(torch.device('cpu')) # save some memory
    # del amd_model.diffusion_transformer # save some memory
    _freeze_parameters(amd_model)
    vae = AutoencoderKL.from_pretrained(args.vae_version, subfolder="vae").to(device).requires_grad_(False)
    # Dataset 

    train_dataset = eval(args.dataset_type)(
        video_dir = args.trainset,
        sample_size=eval(args.sample_size),
        sample_stride=args.sample_stride,
        sample_n_frames=args.sample_n_frames,
    )
    eval_dataset = eval(args.dataset_type)(
        video_dir=args.evalset,
        sample_size=eval(args.sample_size),
        sample_stride=args.sample_stride,
        sample_n_frames=args.sample_n_frames,
    )
 
    train_dataloader = DataLoader(train_dataset, batch_size=args.batch_size,num_workers=args.num_workers, shuffle=True, collate_fn=train_dataset.collate_fn,pin_memory=True)
    eval_dataloader = DataLoader(eval_dataset, batch_size=args.batch_size,num_workers=args.num_workers, shuffle=True, collate_fn=eval_dataset.collate_fn,pin_memory=True)

    a2m_config = OmegaConf.load(args.a2m_config)
    audio_decoder = eval(a2m_config['model_type'])(**a2m_config['model'])
    if accelerator.is_main_process:
        audio_decoder.save_config(proj_dir)
    if args.a2m_from_pretrained is not None:
        model_load_pretrain(audio_decoder,args.a2m_from_pretrained,not_load_keyword='abcabcacbd',strict=False)
    if accelerator.is_main_process:
        print(f'######### load A2M weight from {args.a2m_from_pretrained} #############')

    # Optimizer & Learning Schedule
    optimizer = torch.optim.AdamW(audio_decoder.parameters(),lr=args.lr)
    lr_scheduler = get_scheduler(          # scheduler from diffuser, auto warm-up
        name = args.lr_scheduler,
        optimizer=optimizer,
        num_warmup_steps=args.lr_warmup_steps * args.gradient_accumulation_steps,
        num_training_steps=args.max_train_steps * args.gradient_accumulation_steps,
    )

    # --------------- Step5 : Accelerator Prepare --------------- #
    # Prepare
    audio_decoder, optimizer, training_dataloader, scheduler = accelerator.prepare(
        audio_decoder, optimizer, train_dataloader,lr_scheduler
    )

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
    if args.resume_from_checkpoint is not None:
        model_path = args.resume_from_checkpoint
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
    def log_validation(audio_decoder,amd_model,vae,eval_dataloader, device,accelerator = None,global_step = 0,):
        accelerator.print(f"Running validation....\n")
        if accelerator is not None:
            audio_decoder = accelerator.unwrap_model(audio_decoder)
        audio_decoder.eval()
        amd_model.diffusion_transformer.to(device)

        # data
        data = next(iter(eval_dataloader))

        ref_video = data["ref_video"].to(device)  # N,C,H,W
        gt_video = data["gt_video"].to(device) # N,F,C,H,W
        ref_audio = data["ref_audio"].to(device) # N,M,D
        gt_audio = data["gt_audio"].to(device) # N,F,M,D
        ref_pose = data["ref_pose"].to(device) # N,C,H,W
        gt_pose = data["gt_pose"].to(device) # N,F,C,H,W
        mask = data["mask"].to(device) # N,F

        # vae encode
        ref_video_z = vae_encode(vae,ref_video) # N,C,H,W
        gt_video_z = vae_encode(vae,gt_video) # N,F,C,H,W
        ref_pose_z = vae_encode(vae,ref_pose) # N,D,H,W
        gt_pose_z = vae_encode(vae,gt_pose) # N,F,D,H,W

        # get motion
        with torch.no_grad():
            mix_video_z = torch.cat([ref_video_z.unsqueeze(1),gt_video_z],dim=1) # N,F+1,C,H,W
            motion = amd_model.extract_motion(mix_video_z)
            ref_motion = motion[:,0,:] # N,L,D
            gt_motion = motion[:,1:,:] # N,F,L,D

        if args.use_sample_timestep:
            timestep = torch.from_numpy(sample_timestep(gt_motion.shape[0],args.sample_timestep_m,args.sample_timestep_s,1000)).to(device,ref_video.dtype)
        else:
            timestep = torch.ones(gt_motion.shape[0]).to(device,gt_motion.dtype) * 1000

        # pre
        gt_audio = gt_audio.to(gt_motion.dtype)
        gt_audio = torch.flip(gt_audio, dims=[0])  # !TEST!
        loss_dict = audio_decoder(motion_gt=gt_motion,
                                    ref_motion=ref_motion,
                                    audio=gt_audio,
                                    ref_audio = ref_audio,
                                    pose=gt_pose_z,
                                    ref_pose = ref_pose_z,
                                    timestep = timestep)
        
        val_loss = loss_dict['loss'].item()
        accelerator.print(f'val loss = {val_loss}')
        accelerator.log({"val_loss": val_loss}, step=global_step)

        # sample 
        motion_pre = audio_decoder.sample(  ref_motion = ref_motion,
                                            audio =gt_audio,
                                            ref_audio =ref_audio ,
                                            pose =gt_pose_z,
                                            ref_pose =ref_pose_z,
                                            sample_step=args.motion_sample_step) # n f d h w
        _,video_pre_motion_gt,_  = amd_model.sample_with_refimg_motion(ref_video_z,gt_motion,sample_step=args.video_sample_step) # n f d h w
        _,video_pre_motion_pre,_ = amd_model.sample_with_refimg_motion(ref_video_z,motion_pre,sample_step=args.video_sample_step)# n f d h w
        video_gt = gt_video_z # n f d h w

        assert video_gt.shape == video_pre_motion_gt.shape , f'video_gt shape :{video_gt.shape} , video_pre_motion_gt shape:{video_pre_motion_gt.shape}'
        assert video_gt.shape == video_pre_motion_pre.shape, f'video_gt shape :{video_gt.shape} , video_pre_motion_gt shape:{video_pre_motion_pre.shape}'

        # transform
        def transform(x:torch.Tensor):
            x = vae_decode(vae,x)
            x = ((x / 2.0 + 0.5).clamp(0, 1) * 255).to(dtype=torch.uint8).cpu().contiguous().numpy()
            return x
        video_pre_motion_gt_np = transform(video_pre_motion_gt)
        video_pre_motion_pre_np = transform(video_pre_motion_pre)
        video_gt_np = transform(video_gt)

        # log in
        def log_transform(x,log_b:int,log_f:int):
            x = x[:log_b,:log_f,:]
            x = einops.rearrange(x,'n t c h w -> (n t) h w c')
            np_x = np.stack([np.asarray(img) for img in x])
            return np_x

        log_b = 4
        log_f = 8

        for tracker in accelerator.trackers:
            if tracker.name == "tensorboard":
                video_gt_out = log_transform(video_gt_np,log_b,log_f)
                video_pre_motion_gt_out = log_transform(video_pre_motion_gt_np,log_b,log_f)
                video_pre_motion_pre_out = log_transform(video_pre_motion_pre_np,log_b,log_f)
                
                tracker.writer.add_images(f"video_gt", video_gt_out, global_step, dataformats="NHWC")
                tracker.writer.add_images(f"video_pre_motion_gt", video_pre_motion_gt_out, global_step, dataformats="NHWC")
                tracker.writer.add_images(f"video_pre_motion_pre", video_pre_motion_pre_out, global_step, dataformats="NHWC")

        # save tensorboard video
        gt_videos = np.stack([np.asarray(vid) for vid in video_gt_np])
        tracker.writer.add_video("sample_gt_videos", gt_videos, global_step, fps=8)

        videos_gt_motion = np.stack([np.asarray(vid) for vid in video_pre_motion_gt_np])
        tracker.writer.add_video("sample_videos_gt_motion", videos_gt_motion, global_step, fps=8)

        videos_pre_motion = np.stack([np.asarray(vid) for vid in video_pre_motion_pre_np])
        tracker.writer.add_video("sample_videos_pre_motion", videos_pre_motion, global_step, fps=8)

        # save video
        def save_mp4(latent,suffix='pre'):
            cur_save_path = os.path.join(video_save_dir,f'{global_step}-s{args.motion_sample_step}s{args.video_sample_step}-{suffix}.mp4')
            video = vae_decode(vae,latent) 
            video = einops.rearrange(video.cpu(),'n t c h w -> n c t h w')
            save_videos_grid(video,cur_save_path,rescale=True)

        save_mp4(video_pre_motion_pre,'motionpre')
        save_mp4(video_pre_motion_gt,'motiongt')
        save_mp4(video_gt,'gt')


        amd_model.diffusion_transformer.to(torch.device('cpu')) # save some memory
        
        gc.collect()
        torch.cuda.empty_cache()

    if accelerator.is_main_process:
        log_validation(audio_decoder,
                    amd_model,
                    vae,
                    eval_dataloader,
                    device,
                    accelerator,
                    global_step)
    for epoch in range(first_epoch,args.max_train_epoch):
        accelerator.print(f"Epoch {epoch} start!!")
        if global_step >= args.max_train_steps:
            break
        # train loop in 1 epoch
        for step,data in enumerate(training_dataloader):
            if global_step >= args.max_train_steps:
                break
            audio_decoder.train()
            with accelerator.accumulate(audio_decoder):  
                ref_video = data["ref_video"]  # N,C,H,W
                gt_video = data["gt_video"] # N,F,C,H,W
                ref_audio = data["ref_audio"] # N,M,D
                gt_audio = data["gt_audio"] # N,F,M,D
                ref_pose = data["ref_pose"] # N,C,H,W
                gt_pose = data["gt_pose"] # N,F,C,H,W
                mask = data["mask"] # N,F

                # vae encode
                ref_video_z = vae_encode(vae,ref_video) 
                gt_video_z = vae_encode(vae,gt_video)
                ref_pose_z = vae_encode(vae,ref_pose) # N,D,H,W
                gt_pose_z = vae_encode(vae,gt_pose) # N,F,D,H,W

                # get motion
                with torch.no_grad():
                    mix_video_z = torch.cat([ref_video_z.unsqueeze(1),gt_video_z],dim=1) # N,F+1,C,H,W
                    motion = amd_model.extract_motion(mix_video_z)
                    ref_motion = motion[:,0,:] # N,L,D
                    gt_motion = motion[:,1:,:] # N,F,L,D

                # timestep
                if args.use_sample_timestep:
                    timestep = torch.from_numpy(sample_timestep(ref_motion.shape[0],args.sample_timestep_m,args.sample_timestep_s,1000)).to(device,ref_motion.dtype)
                else:
                    timestep = torch.randint(0,1000+1,(ref_motion.shape[0],)).to(device,ref_motion.dtype)

                # forward
                loss_dict = audio_decoder(motion_gt=gt_motion,
                                            ref_motion=ref_motion,
                                            audio=gt_audio,
                                            ref_audio = ref_audio,
                                            pose=gt_pose_z,
                                            ref_pose = ref_pose_z,
                                            timestep = timestep)
                
                loss = loss_dict['loss']

                # log
                if accelerator.sync_gradients:
                    global_step += 1

                    loss_cache = {}
                    # AMD log , Gather the losses across all processes for logging (if we use distributed training).
                    for key in loss_dict.keys():
                        avg_loss = accelerator.gather(loss_dict[key].repeat(args.batch_size)).mean()
                        train_loss = avg_loss.item() 
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

                # update
                if accelerator.sync_gradients:  # checking sync_gradients
                    params_to_clip = audio_decoder.parameters()
                    accelerator.clip_grad_norm_(params_to_clip, args.max_grad_norm) 
                optimizer.step()
                scheduler.step()
                optimizer.zero_grad()

                # 5. saving
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
                    log_validation(audio_decoder,amd_model,vae,eval_dataloader,device,accelerator, global_step)

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
    # accelerator.print(args.log_with)

    # --------- Train --------- #
    main() # 