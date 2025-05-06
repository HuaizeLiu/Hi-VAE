
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

from model.utils import save_cfg, vae_encode, vae_decode, freeze, print_param_num,model_load_pretrain
from model.model_AE import AMD_models,AMDModel
from model import MAE_models
from model.loss import l2,LpipsMseLoss
from dataset.dataset import CelebvText,AMDVideo



now = datetime.now()
current_time = f'{now.year}-{now.month}-{now.day}-{now.hour}:{now.minute}'


def get_cfg():
    parser = ArgumentParser()

    # data
    parser.add_argument('--dataroot', type=str, default='/mnt/pfs-mc0p4k/tts/team/digital_avatar_group/sunwenzhang/qiyuan/code/AMD/dataset/AMD_data_dirs.pkl', help='path to the root directory of datasets. All datasets will be under this directory.')
    parser.add_argument('--sample_size', type=str, default="(256,256)", help='Sample size as a tuple, e.g., (256, 256).')
    parser.add_argument('--sample_stride', type=int, default=4, help='data sample stride')
    parser.add_argument('--sample_n_frames', type=int, default=32, help='sample_n_frames.')
    parser.add_argument('--batch_size', type=int, default=4, help='batch size used in training.')
    parser.add_argument('--ref_drop_ratio', type=float, default=0.0, help='ref_drop_ratio')

    # experiment
    parser.add_argument('--exp_root', default='/mnt/pfs-mc0p4k/cvg/team/didonglin/zqy/exp', required=True, help='exp_root')
    parser.add_argument('--name', default=f'{current_time}', required=True, help='name of the experiment to load.')
    parser.add_argument('--log_with',default='tensorboard',choices=['tensorboard', 'wandb'],help='accelerator tracker.')
    parser.add_argument('--seed', type=int, default=None, help='A seed for reproducible training.')
    
    parser.add_argument('--mp', type=str, default='fp16', choices=['fp16', 'bf16', 'no'], help='use mixed precision')
    parser.add_argument('--num_workers', type=int, default=32)
    parser.add_argument('--max_train_epoch', type=int, default=200000000, help='maximum number of training steps')
    parser.add_argument('--max_train_steps', type=int, default=100000, help='max_train_steps')
    

    parser.add_argument('--weight_decay', type=float, default=1e-4, help='weight decay in optimization.')
    parser.add_argument('--gradient_accumulation_steps', type=int, default=1, help='number of steps for gradient accumulation')
    parser.add_argument('--lr_warmup_steps', type=int, default=20, help='lr_warmup_steps')
    parser.add_argument("--max_grad_norm", default=1.0, type=float, help="Max gradient norm.")
    parser.add_argument('--eval_interval_step', type=int, default=1000, help='eval_interval_step')
    parser.add_argument('--checkpoint_total_limit', type=int, default=2, help='checkpoint_total_limit')
    parser.add_argument('--save_checkpoint_interval_step', type=int, default=100, help='save_checkpoint_interval_step')
    parser.add_argument("--lr_scheduler", type=str, default="constant",help=('The scheduler type to use. Choose between ["linear", "cosine", "cosine_with_restarts", "polynomial",'' "constant", "constant_with_warmup"]'))
    parser.add_argument("--resume_training", type=str, default=None,help=('input checkpoingt path'))
    
    # validation
    parser.add_argument('--n_save_fig', default=10, help='number of batches to save as image during validation.')
    parser.add_argument('--valid_batch_size', type=int, default=4, help='batch size to use for validation.')
    parser.add_argument('--val_num_step', type=int, default=50, help='number of epochs per validation.')

    # # experiment setting
    # parser.add_argument('--metric', default='lpips', choices=['psnr', 'ssim', 'lpips', 'dists'], help='most important metric to use in saving ckpts.')
    # parser.add_argument('--w_lpips', type=float, default=1)
    # parser.add_argument('--w_style', type=float, default=20.)
    # parser.add_argument('--loss_type', type=str, default='L1', choices=['L1', 'MSE', 'Laplacian', 'L1Census'], help='the base reconstruction loss to use.')
    # parser.add_argument('--charb_eps', type=float, default=1e-6)
    # parser.add_argument('--value_range', type=float, default=2.)

    # checkpoints
    parser.add_argument('--vae_version',type=str,default='/mnt/pfs-mc0p4k/cvg/team/didonglin/zqy/model-checkpoints/Huggingface-Model/sd-vae-ft-mse')
    
    # model cfg
    # ---------- MAE
    parser.add_argument('--mae_model_type',type=str,default='MAE_S',help='model type : MAE_S, MAE_L')
    parser.add_argument('--mae_patch_size',type=int,default=2,help='patch_size')
    parser.add_argument('--mae_freeze',action='store_true',help='freeze mae encoder')
    parser.add_argument('--mae_from_pretrained',type=str,default=None,help='input checkpoingt path')
    parser.add_argument('--mae_encoder_lr', type=float, default=2e-4, help='learning rate in optimization')
    parser.add_argument('--mae_not_load_key_word', type=str, default='decoder', help='not load keyword in')

    # ---------- AMD
    parser.add_argument('--amd_model_type', type=str, default='AMD_S', help='AMD_S,AMD_M,AMD_L')
    parser.add_argument('--amd_block_out_channels_down', type=str, default='[64,128,256,256]', help='duoframedownsample channels')
    parser.add_argument('--amd_image_patch_size', type=int, default=2, help='image patch')
    parser.add_argument('--amd_motion_patch_size', type=int, default=1, help='motion patch ')
    parser.add_argument('--amd_num_step', type=int, default=1000, help='diffusion step')
    parser.add_argument("--amd_from_pretrained", type=str, default=None,help='input checkpoingt path')
    parser.add_argument('--amd_mae_decoder_lr', type=float, default=2e-4, help='learning rate in optimization')
    parser.add_argument('--is_split_input',action='store_true',help='split input')
    parser.add_argument('--mae_output_with_img',action='store_true',help='motion from mae or mae+img')
    parser.add_argument('--amd_from_config',type=float,default=None,help='config when resume training')


    args = parser.parse_args()

    return args



# Main Func
def main():
    
    # --------------- Step1 : Exp Setting --------------- #
    # args
    args = get_cfg()

    # # check training state
    # assert not (args.amd_from_pretrained is not None and args.resume_training is not None) ,'you cant load pretrain weight when in (resume mode)'
    # assert not (args.mae_from_pretrained is not None and args.resume_training is not None) ,'you cant load pretrain weight when in (resume mode)'

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
    # MAE !!
    vae_ds_ratio = 8
    mae_model_kwargs = {'img_size':(eval(args.sample_size)[0] // vae_ds_ratio,eval(args.sample_size)[1] // vae_ds_ratio),
                    'patch_size': args.mae_patch_size,
                    'in_chans' : 4,}
    mae_model = MAE_models[args.mae_model_type](**mae_model_kwargs)

    if args.mae_from_pretrained is not None:
        model_load_pretrain(mae_model,args.mae_from_pretrained,not_load_keyword=args.mae_not_load_key_word,strict=False)
        if accelerator.is_main_process:
            print(f'######### load MAE weight from {args.mae_from_pretrained} #############')
            
    if args.mae_freeze:
        for name,parm in mae_model.named_parameters():
            parm.requires_grad = False
    accelerator.print(f"############ MAE Encoder freeze ? : {args.mae_freeze} ############")

    # AMD !!
    if args.amd_from_config is not None:
        amd_model = AMDModel.from_config(args.amd_from_config)
    else:
        amd_model_kwargs = {'mae_patch_size':args.mae_patch_size,
                        'mae_inchannel':mae_model.embed_dim,
                        'image_inchannel':4,
                        'image_height':eval(args.sample_size)[0] // vae_ds_ratio,
                        'image_width':eval(args.sample_size)[1] // vae_ds_ratio,
                        'video_frames':args.sample_n_frames,
                        'scheduler_num_step':args.amd_num_step,
                        'block_out_channels_down':eval(args.amd_block_out_channels_down),
                        'image_patch_size':args.amd_image_patch_size,
                        'motion_patch_size':args.amd_motion_patch_size,  
                        'is_split_input':args.is_split_input,
                        'mae_output_with_img':args.mae_output_with_img,                 
    } 
        amd_model = AMD_models[args.amd_model_type](**amd_model_kwargs)
    loss_fc = amd_model.forward_loss
    if args.amd_from_pretrained is not None:
        model_load_pretrain(amd_model,args.amd_from_pretrained,not_load_keyword='abcabcacbd',strict=True)
        if accelerator.is_main_process:
            print(f'######### load AMD weight from {args.amd_from_pretrained} #############')

    vae = AutoencoderKL.from_pretrained(args.vae_version, subfolder="vae").requires_grad_(False)
    if accelerator.is_main_process:
        amd_model.save_config(proj_dir)
        print_param_num(mae_model)
        print_param_num(amd_model)

    # Dataset 
    train_dataset = AMDVideo(video_dir=args.dataroot,
                         ref_drop_ratio=args.ref_drop_ratio,
                         sample_size=eval(args.sample_size),
                         sample_stride=args.sample_stride,
                         sample_n_frames=args.sample_n_frames)
    eval_dataset = AMDVideo(video_dir='/mnt/pfs-mc0p4k/tts/team/digital_avatar_group/sunwenzhang/qiyuan/data/cvpr-amd/CelebV-Text/eval',
                        ref_drop_ratio=0.0,
                        sample_size=eval(args.sample_size),
                        sample_stride=args.sample_stride,
                        sample_n_frames=args.sample_n_frames)

    train_dataloader = DataLoader(train_dataset, batch_size=args.batch_size,num_workers=args.num_workers, shuffle=True, collate_fn=train_dataset.collate_fn)
    eval_dataloader = DataLoader(eval_dataset, batch_size=args.batch_size, num_workers=args.num_workers,shuffle=True, collate_fn=eval_dataset.collate_fn)

    # Optimizer & Learning Schedule
    mae_enc_optimizer = torch.optim.AdamW(mae_model.encoder_param(),lr=args.mae_encoder_lr,)
    mae_enc_lr_scheduler = get_scheduler(          # scheduler from diffuser, auto warm-up
        name = args.lr_scheduler,
        optimizer=mae_enc_optimizer,
        num_warmup_steps=args.lr_warmup_steps * args.gradient_accumulation_steps,
        num_training_steps=args.max_train_steps * args.gradient_accumulation_steps,
    )
    amd_mae_dec_optimizer = torch.optim.AdamW(itertools.chain(mae_model.decoder_param(),amd_model.parameters()),lr=args.amd_mae_decoder_lr,)
    amd_mae_dec_lr_scheduler = get_scheduler(          # scheduler from diffuser, auto warm-up
        name = args.lr_scheduler,
        optimizer=amd_mae_dec_optimizer,
        num_warmup_steps=args.lr_warmup_steps * args.gradient_accumulation_steps,
        num_training_steps=args.max_train_steps * args.gradient_accumulation_steps,
    )

    # --------------- Step5 : Accelerator Prepare --------------- #
    # Prepare
    device = accelerator.device
    mae_model, amd_model, mae_enc_optimizer,amd_mae_dec_optimizer, training_dataloader,mae_enc_lr_scheduler,amd_mae_dec_lr_scheduler = accelerator.prepare(
        mae_model, amd_model, mae_enc_optimizer,amd_mae_dec_optimizer, train_dataloader,mae_enc_lr_scheduler,amd_mae_dec_lr_scheduler
    )
    # amd_model,amd_mae_dec_optimizer, training_dataloader,amd_mae_dec_lr_scheduler = accelerator.prepare(
    #     amd_model, amd_mae_dec_optimizer, train_dataloader,amd_mae_dec_lr_scheduler
    # )
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
    def log_validation( amd_model,mae_model, vae, eval_dataloader, device,accelerator = None,global_step = 0):

        accelerator.print(f"Running validation....\n")
        if accelerator is not None:
            amd_model = accelerator.unwrap_model(amd_model)

        amd_model.eval()
        mae_model.eval()

        # data
        for data in eval_dataloader:
            x = data['videos'].to(device) # N,T,C,H,W
            ref_img = data['ref_img'].to(device) # N,C,H,W
            x = x[:args.valid_batch_size,:]
            ref_img = ref_img[:args.valid_batch_size,:]
            break
        
        # encode
        z = vae_encode(vae,x) # N,T,c,h,w
        ref_img = vae_encode(vae,ref_img) # N,C,H,W

        assert not torch.any(torch.isnan(z)), 'Finding *Nan in data after vae.'
        N,T,C,H,W = z.shape
        z = z.flatten(0,1) # B,C,H,W

        # 2. forward
        mae_output = mae_model(z,mode='downstream') #  (B,C,H,W)
        z = einops.rearrange(z,'(n t) c h w -> n t c h w',n=N) # (N,T,C,H,W)
        sample_step = args.val_num_step
        sample,gt = amd_model.sample(video=z,mae_output=mae_output,ref_img=ref_img) # (n,c,h,w)
        
        # decode
        N = x.shape[0]
        T = sample.shape[0] // N

        sample = vae_decode(vae,sample)
        sample = ((sample / 2.0 + 0.5).clamp(0, 1) * 255).to(dtype=torch.uint8).cpu().contiguous().numpy()

        gt = vae_decode(vae,gt)
        gt = ((gt / 2.0 + 0.5).clamp(0, 1) * 255).to(dtype=torch.uint8).cpu().contiguous().numpy()

        def image_log_k_sample_per_batch(sample,k=4):
            sample = einops.rearrange(sample,'b 1 c h w -> (b 1) c h w')
            sample = einops.rearrange(sample,'(n t) c h w -> n t c h w',n=N,t=T)
            sample = sample[:,:k,:]
            sample = einops.rearrange(sample,'n t c h w -> (n t) h w c') # (n*k,h,w,c)
            return sample

        for tracker in accelerator.trackers:
            if tracker.name == "tensorboard":
                if sample.shape[1] == 1:
                    images = image_log_k_sample_per_batch(sample)
                    np_images = np.stack([np.asarray(img) for img in images])
                    
                    gt = image_log_k_sample_per_batch(gt)
                    np_gt = np.stack([np.asarray(img) for img in gt])
                    
                    tracker.writer.add_images(f"video_end_pre", np_images, global_step, dataformats="NHWC")
                    tracker.writer.add_images(f"video_end", np_gt, global_step, dataformats="NHWC")
                else:
                    np_videos = np.stack([np.asarray(vid) for vid in sample])
                    tracker.writer.add_video("validation", np_videos, global_step, fps=8)
        
        gc.collect()
        torch.cuda.empty_cache()


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
            mae_model.train() 

            with accelerator.accumulate(amd_model):  
                # 1. input
                x = data['videos'] # N,T,C,H,W
                ref_img = data['ref_img'] # N,C,H,W

                z = vae_encode(vae,x) # N,T,c,h,w
                ref_img = vae_encode(vae,ref_img) # N,C,H,W
                assert not torch.any(torch.isnan(z)), 'Finding *Nan in data after vae.'
                assert not torch.any(torch.isnan(ref_img)), 'Finding *Nan in data after vae.'

                N,T,C,H,W = z.shape

                # 2.1 mae forward
                z = z.flatten(0,1) # B,C,H,W
                mae_output = mae_model(z,mode='downstream') #  (B,C,H,W)
                assert not torch.any(torch.isnan(mae_output)), 'Finding *Nan in data after mae.'

                # 2.2 AMD forward
                z = einops.rearrange(z,'(n t) c h w -> n t c h w',n=N) # (N,T,C,H,W)
                pre,gt = amd_model(z,mae_output,ref_img) # (n,c,h,w)
                diff_loss = loss_fc(pre,gt)
                assert not torch.any(torch.isnan(diff_loss)), 'Finding *Nan in data after diff_loss.'

                # 3. progress bar 
                if accelerator.sync_gradients:
                    global_step += 1

                    # AMD log , Gather the losses across all processes for logging (if we use distributed training).
                    avg_loss = accelerator.gather(diff_loss.repeat(args.batch_size)).mean()
                    train_loss = avg_loss.item() / args.gradient_accumulation_steps
                    logs = {"step_loss": train_loss, "lr": amd_mae_dec_lr_scheduler.get_last_lr()[0]}
                    accelerator.print(f'Step {global_step}:{train_loss}')
                    
                    progress_bar.set_postfix(**logs)
                    progress_bar.update(1)

                    accelerator.log({"train_loss": train_loss}, step=global_step)

                # 4. backpropagate
                accelerator.backward(diff_loss)
                if accelerator.sync_gradients:  # checking sync_gradients
                    params_to_clip = itertools.chain(amd_model.parameters(),  mae_model.parameters())
                    accelerator.clip_grad_norm_(params_to_clip, args.max_grad_norm) 

                amd_mae_dec_optimizer.step()
                amd_mae_dec_lr_scheduler.step()
                amd_mae_dec_optimizer.zero_grad()
                if not args.mae_freeze:
                    mae_enc_optimizer.step()
                    mae_enc_lr_scheduler.step()
                    mae_enc_optimizer.zero_grad()


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
                    log_validation(amd_model,mae_model, vae, eval_dataloader,device,accelerator, global_step)

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