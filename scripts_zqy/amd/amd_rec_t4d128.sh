conda init bash;
cd /mnt/pfs-mc0p4k/tts/team/digital_avatar_group/sunwenzhang/qiyuan/code/AMD2 # 要进入到自己的目录，否则会在lizrunv2对应机器上的/workspace
source activate /mnt/pfs-mc0p4k/tts/team/digital_avatar_group/sunwenzhang/qiyuan/envs/zqy
conda activate /mnt/pfs-mc0p4k/tts/team/digital_avatar_group/sunwenzhang/qiyuan/envs/zqy
which python
/mnt/pfs-mc0p4k/tts/team/nihao3/tools/bin/adt mountbos --group tts --token 58477ba3d08178bca9faf332867db5fc


accelerate launch \
  --config_file config/accelerate_config_8.yaml \
  train_amd.py \
  --name $1\
  --exp_root /mnt/pfs-gv8sxa/tts/dhg/zqy/exp\
  --train_datapath /mnt/pfs-mc0p4k/tts/team/digital_avatar_group/sunwenzhang/qiyuan/code/AMD2/dataset/path/amd2/train.pkl\
  --eval_datapath /mnt/pfs-mc0p4k/tts/team/digital_avatar_group/sunwenzhang/qiyuan/code/AMD2/dataset/path/amd2/eval.pkl\
  --vae_version /mnt/pfs-mc0p4k/tts/team/digital_avatar_group/sunwenzhang/qiyuan/model-checkpoints/sd-vae-ft-mse\
  --sample_size '[256,256]'\
  --sample_stride 4\
  --sample_n_frames 12\
  --batch_size 8\
  --lr 0.0002\
  --eval_interval_step 2000\
  --max_train_steps 10000000 \
  --save_checkpoint_interval_step 3000\
  --amd_model_type AMD_S_Rec\
  --amd_image_patch_size 2\
  --amd_motion_patch_size 1\
  --amd_num_step 1000\
  --motion_token_num 4\
  --motion_token_channel 128\
  # --resume_training /mnt/pfs-mc0p4k/tts/team/digital_avatar_group/sunwenzhang/qiyuan/exp/amd-s-rec-t12-d128/checkpoints/checkpoint-40000\

