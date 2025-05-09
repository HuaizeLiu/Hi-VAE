conda init bash;
cd /mnt/pfs-gv8sxa/tts/dhg/zqy/code/AMD2 # 要进入到自己的目录，否则会在lizrunv2对应机器上的/workspace
source activate /mnt/pfs-gv8sxa/tts/dhg/zqy/envs/zqy
conda activate /mnt/pfs-gv8sxa/tts/dhg/zqy/envs/zqy
which python



accelerate launch \
  --config_file config/accelerate_config_8.yaml \
  train_amd.py \
  --name $1\
  --exp_root /mnt/pfs-gv8sxa/tts/dhg/zqy/newexp/amd\
  --train_datapath /mnt/pfs-gv8sxa/tts/dhg/zqy/code/AMD2/dataset/path/amd2/train.pkl\
  --eval_datapath /mnt/pfs-gv8sxa/tts/dhg/zqy/code/AMD2/dataset/path/amd2/eval.pkl\
  --vae_version /mnt/pfs-mc0p4k/tts/team/digital_avatar_group/sunwenzhang/qiyuan/model-checkpoints/sd-vae-ft-mse\
  --num_workers 16\
  --dataset_type AMDConsecutiveVideo\
  --sample_size '[256,256]'\
  --sample_stride 1\
  --sample_n_frames 8\
  --batch_size 8\
  --lr 0.0001\
  --eval_interval_step 500\
  --val_num_step 4\
  --max_train_steps 10000000 \
  --save_checkpoint_interval_step 1000\
  --amd_model_type AMD_S\
  --amd_image_patch_size 2\
  --amd_motion_patch_size 1\
  --amd_num_step 1000\
  --motion_token_num 1\
  --motion_token_channel 512\
  --motion_need_norm_out false\
  --need_motion_transformer false\
  --mask_ratio 0.0\
  --diffusion_model_type spatial\
  --amd_from_pretrained /mnt/pfs-gv8sxa/tts/dhg/zqy/exp/amd-s-t1-d512-nonorm-spatial/checkpoints/checkpoint-131000/model.safetensors\
  # --resume_training /mnt/pfs-gv8sxa/tts/dhg/zqy/exp/amd-s-t1-d512-nonorm-spatial/checkpoints/checkpoint-57000\


  # --resume_training /mnt/pfs-gv8sxa/tts/dhg/zqy/exp/amd-s-t1-d512-nonorm-spatial/checkpoints/checkpoint-2000\

