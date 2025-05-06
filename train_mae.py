
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

import torch
from torch.utils.data import DataLoader

from diffusers.optimization import get_scheduler
from diffusers import AutoencoderKL

from accelerate import Accelerator
from accelerate.logging import get_logger
from accelerate.utils import ProjectConfiguration, set_seed

from model.utils import save_cfg, vae_encode, vae_decode
from model import MAE_models
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

    # experiment
    parser.add_argument('--exp_root', default='/mnt/pfs-mc0p4k/cvg/team/didonglin/zqy/exp', required=True, help='exp_root')
    parser.add_argument('--name', default=f'{current_time}', required=True, help='name of the experiment to load.')
    parser.add_argument('--log_with',default='tensorboard',choices=['tensorboard', 'wandb'],help='accelerator tracker.')
    parser.add_argument('--seed', type=int, default=None, help='A seed for reproducible training.')
    
    parser.add_argument('--mp', type=str, default='fp16', choices=['fp16', 'bf16', 'no'], help='use mixed precision')
    parser.add_argument('--num_workers', type=int, default=32)
    parser.add_argument('--max_train_epoch', type=int, default=20, help='maximum number of training steps')
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

    
    # validation
    parser.add_argument('--n_save_fig', default=10, help='number of batches to save as image during validation.')
    parser.add_argument('--valid_batch_size', type=int, default=4, help='batch size to use for validation.')
    parser.add_argument('--val_num_step', type=int, default=50, help='number of epochs per validation.')

    # checkpoints
    parser.add_argument('--model_type',type=str,default='MAE_S',help='model type : MAE_S, MAE_L')
    parser.add_argument('--patch_size',type=int,default=1,help='patch_size')
    parser.add_argument('--vae_version',type=str,default='/mnt/pfs-mc0p4k/cvg/team/didonglin/zqy/model-checkpoints/Huggingface-Model/sd-vae-ft-mse')


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

    # get Model 

    vae_ds_ratio = 8
    model_kwargs = {'img_size':(eval(args.sample_size)[0] // vae_ds_ratio,eval(args.sample_size)[1] // vae_ds_ratio),
                    'patch_size': args.patch_size,
                    'in_chans' : 4,}
    
    model = MAE_models[args.model_type](**model_kwargs)
    vae = AutoencoderKL.from_pretrained(args.vae_version, subfolder="vae").requires_grad_(False)

    # Dataset 
    train_dataset = AMDVideo(video_dir=args.dataroot,
                         sample_size=eval(args.sample_size),
                         sample_stride=args.sample_stride,
                         sample_n_frames=args.sample_n_frames)
    eval_dataset = AMDVideo(video_dir='/mnt/pfs-mc0p4k/tts/team/digital_avatar_group/sunwenzhang/qiyuan/code/AMD/demo/train',
                        sample_size=eval(args.sample_size),
                        sample_stride=args.sample_stride,
                        sample_n_frames=args.sample_n_frames)
    train_dataloader = DataLoader(train_dataset, batch_size=args.batch_size,num_workers=args.num_workers, shuffle=True, collate_fn=train_dataset.collate_fn)
    eval_dataloader = DataLoader(eval_dataset, batch_size=args.batch_size,num_workers=args.num_workers, shuffle=False, collate_fn=eval_dataset.collate_fn)

    # Optimizer & Learning Schedule
    optimizer = torch.optim.AdamW(model.parameters(),lr=args.lr,)
    lr_scheduler = get_scheduler(          # scheduler from diffuser, auto warm-up
        name = args.lr_scheduler,
        optimizer=optimizer,
        num_warmup_steps=args.lr_warmup_steps * args.gradient_accumulation_steps,
        num_training_steps=args.max_train_steps * args.gradient_accumulation_steps,
    )

    # --------------- Step5 : Accelerator Prepare --------------- #
    # Prepare
    device = accelerator.device
    model, optimizer, training_dataloader, scheduler = accelerator.prepare(
        model, optimizer, train_dataloader,lr_scheduler
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
    def log_validation( model, vae, eval_dataloader, device,accelerator = None,global_step = 0):

        accelerator.print(f"Running validation....\n")
        if accelerator is not None:
            model = accelerator.unwrap_model(model)
        model.eval()
        
        # data
        for data in eval_dataloader:
            x = data['videos'].to(device) # N,T,C,H,W
            x = x[:args.valid_batch_size,:]
            break
        N,T,_,_,_ = x.shape
        
        # encode
        z = vae_encode(vae,x) # N,T,c,h,w
        z = z.flatten(0,1) # N,c,h,w
        assert not torch.any(torch.isnan(z)), 'Finding *Nan in data after vae.'

        # forward
        loss, pred, mask = model(z) # pred (N,L,D)
        rec_img = model.unpatchify(pred) # (N,C,H,W)

        # vae decode
        z = z.unsqueeze(1) # (N,1,C,H,W)
        gt = vae_decode(vae,z)
        gt = ((gt / 2.0 + 0.5).clamp(0, 1) * 255).to(dtype=torch.uint8).cpu().contiguous().numpy()

        rec_img = rec_img.unsqueeze(1) # (N,1,C,H,W)
        rec_img = vae_decode(vae,rec_img)
        rec_img = ((rec_img / 2.0 + 0.5).clamp(0, 1) * 255).to(dtype=torch.uint8).cpu().contiguous().numpy()


        for tracker in accelerator.trackers:
            if tracker.name == "tensorboard":
                if gt.shape[1] == 1:
                    gt = einops.rearrange(gt,'n t c h w -> (n t) h w c')
                    np_gt = np.stack([np.asarray(img) for img in gt])

                    rec_img = einops.rearrange(rec_img,'n t c h w -> (n t) h w c')
                    np_rec_img = np.stack([np.asarray(img) for img in rec_img])
                    
                    tracker.writer.add_images(f"gt", np_gt, global_step, dataformats="NHWC")
                    tracker.writer.add_images(f"rec_img", np_rec_img, global_step, dataformats="NHWC")
                else:
                    np_videos = np.stack([np.asarray(vid) for vid in rec_img])
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
            model.train()
            with accelerator.accumulate(model):  
                # 1. input
                x = data['videos'] # N,T,C,H,W
                z = vae_encode(vae,x) # N,T,c,h,w
                assert not torch.any(torch.isnan(z)), 'Finding *Nan in data after vae.'
                z = z.flatten(0,1) # N,c,h,w

                # 2. forward
                loss, pred, mask = model(z)

                # 3. progress bar 
                if accelerator.sync_gradients:
                    global_step += 1

                    # log , Gather the losses across all processes for logging (if we use distributed training).
                    avg_loss = accelerator.gather(loss.repeat(args.batch_size)).mean()
                    train_loss = avg_loss.item() / args.gradient_accumulation_steps
                    logs = {"step_loss": train_loss, "lr": scheduler.get_last_lr()[0]}
                    progress_bar.set_postfix(**logs)
                    progress_bar.update(1)


                    accelerator.print('-'*50)
                    accelerator.print(f'{global_step}:{train_loss}')



                    accelerator.log({"train_loss": train_loss}, step=global_step)

                # 4. backpropagate
                accelerator.backward(loss)
                if accelerator.sync_gradients:  # checking sync_gradients
                    params_to_clip = model.parameters()
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
                    log_validation(model, vae, eval_dataloader,device,accelerator, global_step)

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