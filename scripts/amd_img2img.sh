conda init bash;
cd /mnt/pfs-gv8sxa/tts/dhg/liuhuaize/code/AMD2 # 要进入到自己的目录，否则会在lizrunv2对应机器上的/workspace
source activate /mnt/pfs-gv8sxa/tts/dhg/zqy/envs/zqy
conda activate /mnt/pfs-gv8sxa/tts/dhg/zqy/envs/zqy
which python

# name = $1
name=vidtok-img2img-fq-grey-cd-c32m64-t9-highfq-stage3
# sleep 365d


accelerate launch \
  --config_file config/accelerate_config_8.yaml \
  train_amd.py \
  --name $name\
  --exp_root /mnt/pfs-gv8sxa/tts/dhg/liuhuaize/code/AMD2/exp_video\
  --train_datapath /mnt/pfs-gv8sxa/tts/dhg/wenzhangsun/.sun/dataset/webvid/vidtwin_csv/final_full.csv\
  --eval_datapath /mnt/pfs-gv8sxa/tts/dhg/wenzhangsun/.sun/dataset/webvid/vidtwin_csv/final.csv\
  --vae_version /mnt/pfs-mc0p4k/tts/team/digital_avatar_group/sunwenzhang/qiyuan/model-checkpoints/sd-vae-ft-mse\
  --num_workers 16\
  --dataset_type AMDConsecutiveVideo\
  --sample_size '[256,256]'\
  --target_fps 8\
  --sample_n_frames 16\
  --batch_size 4\
  --lr 0.0001\
  --eval_interval_step 5000\
  --val_num_step 4\
  --max_train_steps 10000000 \
  --save_checkpoint_interval_step 1000\
  --amd_model_type AMD_S\
  --amd_image_patch_size 2\
  --amd_motion_patch_size 1\
  --amd_num_step 1000\
  --motion_token_channel 64\
  --motion_token_num 10\
  --object_motion_token_num 9\
  --object_motion_token_channel 64\
  --object_enc_num_layers 9\
  --camera_motion_token_num 9\
  --camera_motion_token_channel 32\
  --camera_enc_num_layers 6\
  --motion_need_norm_out false\
  --need_motion_transformer false\
  --diffusion_model_type spatial\
  --use_filter true\
  --use_regularizers false\
  --use_motiontemporal false\
  --use_grey true\
  --use_camera_down true\
  --use_mask true\
  --motion_type plus\
  --mask_video_ratio 0.6\
  --amd_from_pretrained /mnt/pfs-gv8sxa/tts/dhg/liuhuaize/code/AMD2/img_exp/vidtok-img2img-fq-grey-cd-c32m64-t9-highfq-stage2/checkpoints/checkpoint-108000/model.safetensors 2>&1  | tee logs/$name.log
# --resume_training /mnt/pfs-gv8sxa/tts/dhg/liuhuaize/code/AMD2/exp_video/vidtok-videorec-fq-o128c32m128-t8/checkpoints/checkpoint-10000 2>&1  | tee logs/$1.log
  
  # --main_process_port 29501\
# accelerate launch \
#   --config_file config/accelerate_config_8.yaml \
#   train_amd.py \
#   --name $name\
#   --exp_root /mnt/pfs-gv8sxa/tts/dhg/liuhuaize/code/AMD2/img_exp\
#   --train_datapath /mnt/pfs-gv8sxa/tts/dhg/wenzhangsun/.sun/dataset/webvid/vidtwin_csv/final_full.csv\
#   --eval_datapath /mnt/pfs-gv8sxa/tts/dhg/wenzhangsun/.sun/dataset/webvid/vidtwin_csv/final.csv\
#   --vae_version /mnt/pfs-mc0p4k/tts/team/digital_avatar_group/sunwenzhang/qiyuan/model-checkpoints/sd-vae-ft-mse\
#   --num_workers 16\
#   --dataset_type AMDRandomPair\
#   --sample_size '[256,256]'\
#   --sample_stride 4\
#   --sample_n_frames 10\
#   --batch_size 12\
#   --lr 0.0001\
#   --eval_interval_step 5000\
#   --val_num_step 4\
#   --max_train_steps 10000000 \
#   --save_checkpoint_interval_step 2000\
#   --amd_model_type AMD_S\
#   --amd_image_patch_size 2\
#   --amd_motion_patch_size 1\
#   --amd_num_step 1000\
#   --motion_token_channel 64\
#   --motion_token_num 9\
#   --object_motion_token_num 9\
#   --object_motion_token_channel 64\
#   --object_enc_num_layers 9\
#   --camera_motion_token_num 9\
#   --camera_motion_token_channel 32\
#   --camera_enc_num_layers 6\
#   --motion_need_norm_out false\
#   --need_motion_transformer false\
#   --diffusion_model_type default\
#   --use_filter true\
#   --use_regularizers false\
#   --use_motiontemporal false\
#   --use_grey true\
#   --use_camera_down true\
#   --mask_video_ratio 0.6\
#   --use_mask true\
#   --resume_training /mnt/pfs-gv8sxa/tts/dhg/liuhuaize/code/AMD2/img_exp/vidtok-img2img-fq-grey-cd-c32m64-t9-highfq-stage2-greyGT/checkpoints/checkpoint-60000  2>&1  | tee logs/$name.log
  # --amd_from_pretrained /mnt/pfs-gv8sxa/tts/dhg/liuhuaize/code/AMD2/img_exp/vidtok-img2img-fq-grey-cd-c32m64-t9-cameraonly/checkpoints/checkpoint-92000/model.safetensors 2>&1  | tee logs/$name.log


  # --resume_training /mnt/pfs-gv8sxa/tts/dhg/liuhuaize/code/AMD2/img_exp/vidtok-img2img-fq-grey-cd-c32m64-t9-highfq-stage2-greyGT/checkpoints/checkpoint-60000\

