conda init bash;
cd /mnt/pfs-gv8sxa/tts/dhg/liuhuaize/code/AMD2/ # 要进入到自己的目录，否则会在lizrunv2对应机器上的/workspace
source activate /mnt/pfs-gv8sxa/tts/dhg/zqy/envs/zqy
conda activate /mnt/pfs-gv8sxa/tts/dhg/zqy/envs/zqy
which python

# 参考项
# --train_datapath /mnt/pfs-gv8sxa/tts/dhg/zqy/code/AMD2/dataset/path/amd2/train.pkl\
# --eval_datapath /mnt/pfs-gv8sxa/tts/dhg/zqy/code/AMD2/dataset/path/amd2/eval.pkl\
# --train_datapath /mnt/pfs-gv8sxa/tts/dhg/wenzhangsun/.sun/dataset/webvid/vidtwin_csv/final.csv\
# --eval_datapath /mnt/pfs-gv8sxa/tts/dhg/wenzhangsun/.sun/dataset/webvid/vidtwin_csv/final.csv\

accelerate launch \
  --config_file config/accelerate_config_8.yaml \
  train_amd.py \
  --name $1\
  --exp_root /mnt/pfs-gv8sxa/tts/dhg/liuhuaize/code/AMD2/exp\
  --train_datapath /mnt/pfs-gv8sxa/tts/dhg/wenzhangsun/.sun/dataset/webvid/vidtwin_csv/final_full.csv\
  --eval_datapath /mnt/pfs-gv8sxa/tts/dhg/wenzhangsun/.sun/dataset/webvid/vidtwin_csv/final.csv\
  --vae_version /mnt/pfs-mc0p4k/tts/team/digital_avatar_group/sunwenzhang/qiyuan/model-checkpoints/sd-vae-ft-mse\
  --num_workers 16\
  --dataset_type AMDConsecutiveVideo\
  --sample_size '[256,256]'\
  --sample_stride 2\
  --sample_n_frames 16\
  --batch_size 4\
  --lr 0.0001\
  --eval_interval_step 5000\
  --val_num_step 2\
  --max_train_steps 10000000 \
  --save_checkpoint_interval_step 10000\
  --amd_model_type AMD_S\
  --amd_image_patch_size 2\
  --amd_motion_patch_size 1\
  --amd_num_step 1000\
  --motion_token_channel 512\
  --motion_token_num 4\
  --object_motion_token_num 4\
  --object_motion_token_channel 256\
  --object_enc_num_layers 10\
  --camera_motion_token_num 4\
  --camera_motion_token_channel 512\
  --camera_enc_num_layers 8\
  --motion_need_norm_out false\
  --need_motion_transformer false\
  --diffusion_model_type spatial\
  --use_regularizers false\
  --use_motiontemporal true\
  --use_filter false  2>&1  | tee logs/$1.log
  # --resume_training /mnt/pfs-gv8sxa/tts/dhg/liuhuaize/code/AMD2/exp/full-data-original/checkpoints/checkpoint-40000\
  # 2>&1  | tee logs/$1.log
  # --amd_from_pretrained /mnt/pfs-gv8sxa/tts/dhg/zqy/exp/amd-s-t1-d512-nonorm-spatial/checkpoints/checkpoint-57000/model.safetensors\
  # --resume_training /mnt/pfs-gv8sxa/tts/dhg/zqy/exp/amd-s-t1-d512-nonorm-spatial/checkpoints/checkpoint-57000\


  # --resume_training /mnt/pfs-gv8sxa/tts/dhg/zqy/exp/amd-s-t1-d512-nonorm-spatial/checkpoints/checkpoint-2000\

  # bash scripts/amd/amd_t1d512_nonorm_spatial.sh test

