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
from accelerate import DistributedDataParallelKwargs
import torch
from torch.utils.data import DataLoader
from diffusers.optimization import get_scheduler
from diffusers import AutoencoderKL

from accelerate import Accelerator
from accelerate.logging import get_logger
from accelerate.utils import ProjectConfiguration, set_seed

from model.utils import save_cfg, vae_encode, vae_decode, freeze ,print_param_num,cat_video,_freeze_parameters,save_videos_grid
from model import RectifiedFlow
from model.model_AMD import AMDModel,AMDModel_New
from text2motion import LabelEncoder, Label2MotionDiffusionDecoder
from model.loss import l2
from safetensors.torch import load_model
from dataset.dataset import A2MVideoUCF
from downstream_tasks.utils import sample_timestep
import time
import pdb

# import debugpy
# debugpy.connect(('localhost', 5684))

now = datetime.now()
current_time = f'{now.year}-{now.month}-{now.day}-{now.hour}:{now.minute}'


def get_cfg():
    parser = ArgumentParser()

    # data
    parser.add_argument('--trainset', type=str, default='/mnt/pfs-mc0p4k/tts/team/digital_avatar_group/sunwenzhang/qiyuan/code/AMD_linear/dataset/path/local/mead_train.pkl', help='trainset index file path')
    parser.add_argument('--evalset', type=str, default='/mnt/pfs-mc0p4k/tts/team/digital_avatar_group/sunwenzhang/qiyuan/code/AMD_linear/dataset/path/local/mead_eval.pkl', help='evalset index file path')

    parser.add_argument('--sample_size', type=str, default="(256,256)", help='Sample size as a tuple, e.g., (256, 256).')
    parser.add_argument('--sample_stride', type=int, default=1, help='data sample stride')
    parser.add_argument('--sample_n_frames', type=int, default=31, help='sample_n_frames.')
    parser.add_argument('--batch_size', type=int, default=4, help='batch size used in training.')
    parser.add_argument('--target_fps', type=int, default=8, help='batch size used in training.')
    # amd
    parser.add_argument('--amd_val_num_step', type=int, default=2, help='number of epochs per validation.')
    parser.add_argument('--camera_mask_ratio', type=float, default=None, help='wheather mask camera motion')
    parser.add_argument('--object_mask_ratio', type=float, default=None, help='wheather mask object motion')


    # experiment
    parser.add_argument('--exp_root', default='/mnt/pfs-mc0p4k/cvg/team/didonglin/zqy/exp', required=True, help='exp_root')
    parser.add_argument('--name', default=f'{current_time}', required=True, help='name of the experiment to load.')
    parser.add_argument('--log_with',default='tensorboard',choices=['tensorboard', 'wandb'],help='accelerator tracker.')
    parser.add_argument('--seed', type=int, default=None, help='A seed for reproducible training.')
    
    parser.add_argument('--mp', type=str, default='fp16', choices=['fp16', 'bf16', 'no'], help='use mixed precision')
    parser.add_argument('--num_workers', type=int, default=8)
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

    
    # checkpoints
    parser.add_argument('--vae_version',type=str,default='/mnt/pfs-mc0p4k/cvg/team/didonglin/zqy/model-checkpoints/Huggingface-Model/sd-vae-ft-mse')
    parser.add_argument('--amd_config',type=str, default="/mnt/pfs-mc0p4k/tts/team/digital_avatar_group/sunwenzhang/qiyuan/exp/amd-m-mae-s-1026-linear-final/config.json", help='amd model config path')
    parser.add_argument('--amd_ckpt',type=str,default="/mnt/pfs-mc0p4k/tts/team/digital_avatar_group/sunwenzhang/qiyuan/code/AMD_linear/ckpt/checkpoint-157000/model_1.safetensors",help="amd model checkpoint path")
    parser.add_argument('--label_dim',type=int,default=768)
    parser.add_argument('--motion_dim',type=int,default=256)
    parser.add_argument('--motion_seq_len',type=int,default=15)
    parser.add_argument('--sample_timestep_m',type=float,default=0.5)
    parser.add_argument('--sample_timestep_s',type=float,default=1.0)
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
    
    # # Seed everything
    # if args.seed is not None:
    #     set_seed(args.seed)
    
    # --------------- Step2 : Accelerator Initialize --------------- #
    # initialize accelerator.    
    project_config = ProjectConfiguration(project_dir=proj_dir)
    ddp_kwargs = DistributedDataParallelKwargs(find_unused_parameters=True)
    accelerator = Accelerator(
        gradient_accumulation_steps = args.gradient_accumulation_steps,
        mixed_precision             = args.mp,
        log_with                    = args.log_with,
        project_config              = project_config,
        kwargs_handlers             =[ddp_kwargs]
    )

    # --------------- Step3 : Save Exp Config --------------- #
    # save args
    if accelerator.is_main_process:
        save_cfg(proj_dir, args)

    # --------------- Step4 : Load Model & Datasets & Optimizer--------------- #
    # Model CFG
    # get Model 
    device = accelerator.device
    config = AMDModel_New.load_config(args.amd_config)
    config["video_frames"] = args.sample_n_frames
    amd_model = AMDModel_New.from_config(config).to(device).requires_grad_(False)
    load_model(amd_model,args.amd_ckpt)
    # amd_model.diffusion_transformer.to(torch.device('cpu')) # save some memory

    vae = AutoencoderKL.from_pretrained(args.vae_version, subfolder="vae").to(device).requires_grad_(False)
    # Dataset 
    train_dataset = A2MVideoUCF(video_dir=args.trainset,
                        sample_size=eval(args.sample_size)[0],
                        sample_stride=args.sample_stride,
                        sample_n_frames=args.sample_n_frames,
                        target_fps = args.target_fps,)

    eval_dataset = A2MVideoUCF(video_dir=args.evalset,
                        sample_size=eval(args.sample_size)[0],
                        sample_stride=args.sample_stride,
                        sample_n_frames=args.sample_n_frames,
                        target_fps = args.target_fps,)

    train_dataloader = DataLoader(train_dataset, batch_size=args.batch_size,num_workers=args.num_workers, shuffle=True)
    eval_dataloader = DataLoader(eval_dataset, batch_size=args.batch_size,num_workers=args.num_workers, shuffle=False)

    l2m_model_config = {
        "label_dim":args.label_dim,
        "motion_dim":args.motion_dim,
        "refimg_dim": 4,
        "num_frames":args.sample_n_frames,
        "num_layers": 16,
        "attention_head_dim" : 32,
        "num_attention_heads": 8,
    }
   
    label_encoder_config = {
        "num_labels": 101,
        "out_dim": args.label_dim,
        "emb_dim": 128
    }
    label_encoder = LabelEncoder(**label_encoder_config).to(device)
    l2m_decoder = Label2MotionDiffusionDecoder(**l2m_model_config).to(device)
    # Optimizer & Learning Schedule
    optimizer = torch.optim.AdamW(itertools.chain(label_encoder.parameters(), l2m_decoder.parameters()),lr=args.lr)
    lr_scheduler = get_scheduler(          # scheduler from diffuser, auto warm-up
        name = args.lr_scheduler,
        optimizer=optimizer,
        num_warmup_steps=args.lr_warmup_steps * args.gradient_accumulation_steps,
        num_training_steps=args.max_train_steps * args.gradient_accumulation_steps,
    )

    # --------------- Step5 : Accelerator Prepare --------------- #
    # Prepare
    label_encoder,l2m_decoder, optimizer, training_dataloader, scheduler = accelerator.prepare(
        label_encoder,l2m_decoder, optimizer, train_dataloader,lr_scheduler
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
        accelerator.load_state(model_path)
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
    def log_validation(label_encoder,l2m_decoder,amd_model,vae,eval_dataloader,device,video_save_dir,accelerator=None,global_step = 0,):

        accelerator.print(f"Running validation....\n")
        if accelerator is not None:
            label_encoder = accelerator.unwrap_model(label_encoder)
            l2m_decoder = accelerator.unwrap_model(l2m_decoder)
        label_encoder.eval()
        l2m_decoder.eval()

        # data
        data = next(iter(eval_dataloader))
        video = data["videos"].to(device)
        label = data["label"].to(device)
        refimg = data["ref_img"].to(device)
        ref_grey_img = data["ref_grey_img"].to(device)
        grey_videos = data["grey_videos"].to(device)

        label_emb = label_encoder(label)
        # motion 
        z_video = vae_encode(vae,video)
        z_ref = vae_encode(vae,refimg)
        z_grey = vae_encode(vae,grey_videos)
        ref_img_z_grey = vae_encode(vae,ref_grey_img)

        camera_target_motion, object_source_motion, object_target_motion = amd_model.encode(z_video,z_ref,z_grey,ref_img_z_grey,camera_mask_ratio=args.camera_mask_ratio,object_mask_ratio=args.object_mask_ratio)
        refimg_input = z_ref
        # timestep
        timestep = torch.from_numpy(sample_timestep(refimg.shape[0],args.sample_timestep_m,args.sample_timestep_s,1000)).to(device,refimg.dtype)
        # timestep = torch.randint(1000,size=(refimg.shape[0],)).to(device) # 1000 is fixed TODO
        # res
        # object_vel,camera_val = l2m_decoder.sample(label_emb,refimg_input)
        object_vel,camera_val = l2m_decoder.sample_with_camera(label_emb,refimg_input,camera_target_motion,object_source_motion=object_source_motion)
        # object_vel,camera_val = l2m_decoder.sample_with_camera(label_emb,refimg_input,camera_target_motion)

        video_start,sample,gt = amd_model.sample_with_refimg_motion(z_ref,camera_val,object_vel,ref_img_grey=ref_img_z_grey)
            
        video_start_cache = vae_decode(vae,video_start) # n,t,c,h,w
        video_start = ((video_start_cache / 2.0 + 0.5).clamp(0, 1) * 255).to(dtype=torch.uint8).cpu().contiguous().numpy()

        sample_cache = vae_decode(vae,sample) # n,t,c,h,w
        sample = ((sample_cache / 2.0 + 0.5).clamp(0, 1) * 255).to(dtype=torch.uint8).cpu().contiguous().numpy()

        gt_cache = vae_decode(vae,z_video) # n,t,c,h,w
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
        label = label
        save_path = os.path.join(video_save_dir,f'{int(label[0])}-{int(label[1])}-{int(label[2])}-{int(label[3])}-step{global_step}.mp4')
        save_videos_grid(mix,save_path,rescale=True)

        gc.collect()
        torch.cuda.empty_cache()

    video_save_dir = os.path.join(proj_dir,"sample")
    os.makedirs(video_save_dir,exist_ok=True)
    # print("global_step",global_step)
    log_validation(label_encoder,
                   l2m_decoder,
                   amd_model,
                   vae,
                   eval_dataloader,
                   device,
                   video_save_dir,
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
            # train !!!
            label_encoder.train()
            l2m_decoder.train()
            with accelerator.accumulate(label_encoder,l2m_decoder):  
                video = data["videos"].to(device)
                label = data["label"].to(device)
                refimg = data["ref_img"].to(device)
                ref_grey_img = data["ref_grey_img"].to(device)
                grey_videos = data["grey_videos"].to(device)

                label_emb = label_encoder(label)

                # vae <- no grad
                z_video = vae_encode(vae,video)
                z_ref = vae_encode(vae,refimg)
                z_grey = vae_encode(vae,grey_videos)
                ref_img_z_grey = vae_encode(vae,ref_grey_img)


                with torch.no_grad():
                    # motion_gt = cat_video(amd_model,mae_model,z_video,args.motion_seq_len) 
                    # pdb.set_trace()
                    camera_target_motion, object_source_motion, object_target_motion = amd_model.encode(z_video,z_ref,z_grey,ref_img_z_grey,camera_mask_ratio=args.camera_mask_ratio,object_mask_ratio=args.object_mask_ratio)
                    # refimg
                    refimg_input = z_ref   
                # timestep
                timestep = torch.from_numpy(sample_timestep(refimg.shape[0],args.sample_timestep_m,args.sample_timestep_s,1000)).to(device,refimg.dtype)
                # res
                # pdb.set_trace()

                # input_example = (
                #     camera_target_motion,
                #     object_target_motion,
                #     label_emb,
                #     refimg_input,
                #     timestep
                # )

                # # from thop import profile
                # flops, params = profile(l2m_decoder,input_example)
                # print(f"FLOPs: {flops / 1e9:.2f} G, Params: {params / 1e6:.2f} M")

                # from torch.profiler import profile, record_function, ProfilerActivity
                # del vae
                # del amd_model
                # torch.cuda.empty_cache()
                # with profile(activities=[ProfilerActivity.CUDA], profile_memory=True, record_shapes=True) as prof:
                #     with record_function("model_inference"):
                #         res = l2m_decoder(camera_target_motion,object_target_motion,label_emb,refimg_input,timestep,object_source_motion=object_source_motion,return_meta_info=True)

                # print(prof.key_averages().table(sort_by="cuda_memory_usage", row_limit=10))
                # print(f"Total CUDA memory allocated: {torch.cuda.memory_allocated() / 1e9:.2f} GB")
                # print(f"Max CUDA memory allocated: {torch.cuda.max_memory_allocated() / 1e9:.2f} GB")
                
                # res = l2m_decoder(camera_target_motion,object_target_motion,label_emb,refimg_input,timestep,return_meta_info=True)
                res = l2m_decoder(camera_target_motion,object_target_motion,label_emb,refimg_input,timestep,object_source_motion=object_source_motion,return_meta_info=True)

                # loss_noise = l2(res["vel_gt"],res["vel_pred"])
                # loss_camera_nosie = l2(res["vel_gt_camera"],res["vel_pred_camera"])
                loss_object_noise = l2(res["vel_gt_object"],res["vel_pred_object"])

                # print("loss_noise",loss_noise)
                # print("loss_camera",loss_camera)
                # print("loss_object",loss_object)
                # loss = loss_camera_nosie + loss_object_noise
                loss = loss_object_noise

                # progress bar 
                if accelerator.sync_gradients:
                    global_step += 1

                    # log , Gather the losses across all processes for logging (if we use distributed training).
                    avg_loss = accelerator.gather(loss.repeat(args.batch_size)).mean()
                    train_loss = avg_loss.item() / args.gradient_accumulation_steps

                    # avg_camera_loss = accelerator.gather(loss_camera_nosie.repeat(args.batch_size)).mean()
                    # camera_train_loss = avg_camera_loss.item() / args.gradient_accumulation_steps
                    # avg_object_loss = accelerator.gather(loss_object_noise.repeat(args.batch_size)).mean()
                    # object_train_loss = avg_object_loss.item() / args.gradient_accumulation_steps
                    logs = {"step_loss": train_loss, "lr": scheduler.get_last_lr()[0]}
                    # logs = {"step_loss": train_loss, "camera_train_loss":camera_train_loss, "object_train_loss":object_train_loss, "lr": scheduler.get_last_lr()[0]}
                    progress_bar.set_postfix(**logs)
                    progress_bar.update(1)
                    accelerator.print(f'Step {global_step}: {train_loss}')
                    accelerator.log({"train_loss": train_loss}, step=global_step)
                    # accelerator.log({"camera_train_loss": camera_train_loss}, step=global_step)
                    # accelerator.log({"object_train_loss": object_train_loss}, step=global_step)

                # backpropagate
                accelerator.backward(loss)

                if accelerator.sync_gradients:  # checking sync_gradients
                    params_to_clip = itertools.chain(label_encoder.parameters(), l2m_decoder.parameters())
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
                    log_validation(label_encoder,l2m_decoder,amd_model,vae,eval_dataloader,device,video_save_dir,accelerator, global_step)

    # -------------------------------------------------------  End Train   --------------------------------------------------------------------- #
                        
    # Step Final : End accelerator
    accelerator.wait_for_everyone()
    accelerator.end_training() 



if __name__ == "__main__":
    # pdb.set_trace()
    # # --------- Argparse ----------- #
    # parser = argparse.ArgumentParser()
    # parser.add_argument("--args", type=str, required=True)
    # args = parser.parse_args()
    

    # # --------- Config ----------#
    # args = OmegaConf.load(args.args)
    # accelerator.print(args.log_with)

    # --------- Train --------- #
    main() # 