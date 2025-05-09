conda init bash;
cd /mnt/pfs-gv8sxa/tts/dhg/zqy/code/AMD2 # 要进入到自己的目录，否则会在lizrunv2对应机器上的/workspace
source activate /mnt/pfs-gv8sxa/tts/dhg/zqy/envs/zqy
conda activate /mnt/pfs-gv8sxa/tts/dhg/zqy/envs/zqy
which python

accelerate launch \
  --config_file config/accelerate_config_8.yaml \
  /mnt/pfs-gv8sxa/tts/dhg/zqy/code/AMD2/train_a2m.py \
  --name $1\
  --exp_root /mnt/pfs-gv8sxa/tts/dhg/zqy/exp \
  --trainset /mnt/pfs-gv8sxa/tts/dhg/zqy/code/AMD2/dataset/path/a2m/train.pkl \
  --evalset /mnt/pfs-gv8sxa/tts/dhg/zqy/code/AMD2/dataset/path/a2m/eval.pkl \
  --path_type dir \
  --sample_size '[256,256]'\
  --sample_stride 1\
  --sample_n_frames 16\
  --batch_size 12\
  --eval_interval_step 1000\
  --max_train_epoch 10000\
  --max_train_steps 1000000 \
  --save_checkpoint_interval_step 1000\
  --num_workers 16 \
  --lr 0.0001 \
  --vae_version /mnt/pfs-mc0p4k/tts/team/digital_avatar_group/sunwenzhang/qiyuan/model-checkpoints/sd-vae-ft-mse \
  --amd_model_type AMDModel \
  --amd_config /mnt/pfs-gv8sxa/tts/dhg/zqy/exp/amd-s-t2-d256-nonorm-motion-transformer/config_sample_motion_wo_mt.json \
  --amd_ckpt /mnt/pfs-gv8sxa/tts/dhg/zqy/exp/amd-s-t2-d256-nonorm-motion-transformer-new/checkpoints/checkpoint-85000/model.safetensors \
  --a2m_config /mnt/pfs-gv8sxa/tts/dhg/zqy/code/AMD2/config/a2m/cross_audio_posepre_t2d256_l16_dim1024.yaml \
  --resume_from_checkpoint /mnt/pfs-gv8sxa/tts/dhg/zqy/exp/a2m-token2-frame16-posepre-motion-wo-mt/checkpoints/checkpoint-222000\

  # --sample_timestep_m 0.8 \
  # --sample_timestep_s 1.0 \
  # --use_sample_timestep \

