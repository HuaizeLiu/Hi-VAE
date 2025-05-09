
accelerate launch \
  --config_file config/accelerate_config_1.yaml \
  train_amd.py \
  --exp_root /mnt/pfs-mc0p4k/cvg/team/didonglin/zqy/exp\
  --name amd-v52-4ds-10-6\
  --dataroot /mnt/pfs-mc0p4k/cvg/team/didonglin/zqy/data/CelebV-Text/CelebV-Text-split\
  --sample_size '[256,256]'\
  --sample_stride 4\
  --sample_n_frames 16\
  --batch_size 2\
  --vae_version /mnt/pfs-mc0p4k/cvg/team/didonglin/zqy/model-checkpoints/Huggingface-Model/sd-vae-ft-mse\
  --eval_interval_step 200\
  --max_train_epoch 100\
  --max_train_steps 100000 \
  --save_checkpoint_interval_step 100\
  --model_type AMD_S\
  --image_patch_size 2\
  --motion_patch_size 1\
  --num_step 1000\
  --block_out_channels_down '[64,128,256]'\
  --resume_from_checkpoint /mnt/pfs-mc0p4k/cvg/team/didonglin/zqy/exp/amd-v52-4ds-10-6/checkpoints/checkpoint-2500\


