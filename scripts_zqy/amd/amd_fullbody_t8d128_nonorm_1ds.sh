conda init bash;
cd /mnt/pfs-gv8sxa/tts/dhg/zqy/code/AMD2 # 要进入到自己的目录，否则会在lizrunv2对应机器上的/workspace
source activate /mnt/pfs-gv8sxa/tts/dhg/zqy/envs/zqy
conda activate /mnt/pfs-gv8sxa/tts/dhg/zqy/envs/zqy
which python


accelerate launch \
  --config_file config/accelerate_config_8.yaml \
  train_amd.py \
  --name $1\
  --exp_root /mnt/pfs-gv8sxa/tts/dhg/zqy/exp\
  --train_datapath /mnt/pfs-gv8sxa/tts/dhg/zqy/code/AMD2/dataset/path/amd_fullbody/train.pkl\
  --eval_datapath /mnt/pfs-gv8sxa/tts/dhg/zqy/code/AMD2/dataset/path/amd_fullbody/eval.pkl\
  --vae_version /mnt/pfs-mc0p4k/tts/team/digital_avatar_group/sunwenzhang/qiyuan/model-checkpoints/sd-vae-ft-mse\
  --sample_size '(1024,576)'\
  --sample_stride 4\
  --sample_n_frames 6\
  --batch_size 2\
  --lr 0.0002\
  --eval_interval_step 500\
  --max_train_steps 10000000 \
  --save_checkpoint_interval_step 1000\
  --amd_model_type AMD_S\
  --amd_image_patch_size 4\
  --amd_motion_patch_size 1\
  --amd_num_step 1000\
  --motion_token_num 8\
  --motion_token_channel 128\
  --resume_training /mnt/pfs-gv8sxa/tts/dhg/zqy/exp/amd-s-fullbody-t8-d128-nonorm-1ds/checkpoints/checkpoint-2000\
  # --amd_from_config /mnt/pfs-mc0p4k/tts/team/digital_avatar_group/sunwenzhang/qiyuan/exp/amd-s-t12-d128/config.json\

