conda init bash;
cd /mnt/pfs-gv8sxa/tts/dhg/liuhuaize/code/AMD2/ # 要进入到自己的目录，否则会在lizrunv2对应机器上的/workspace
source activate /mnt/pfs-gv8sxa/tts/dhg/zqy/envs/zqy
conda activate /mnt/pfs-gv8sxa/tts/dhg/zqy/envs/zqy
which python

sleep 365d
# name=vidtok-img-fq-grey-c16o64-objectonly
# name=vidtok-video-fq-grey-c16t8-o64-cameraonly-greyGT-mask0.5
# name=vidtok-video-fq-grey-c32t16-o64-cameraonly-greyGT-mask0.6
# name=vidtok-video-fq-grey-cd-c16m64-cameraonly-greyGT
# name=vidtok-video-fq-grey-cd-c16m64t16-cameraonly-greyGT-FirstFramePlus
# name=vidtok-video-fq-grey-c16m64t16-cameraonly-greyGT-nomask-FirstTwoFramePlus
# name=vidtok-video-fq-grey-cd-c16m64t16-cameraonly-greyGT-nomask-SpicalDecode
# name=vidtok-video-fq-grey-c16m64t16-cameraonly-greyGT-mask0.5-SpicalDecode
# name=vidtok-video-fq-grey-c8m32t8-cameraonly-greyGT-mask0.7-SpicalDecode
# name=vidtok-video-fq-grey-cd-c8t8-greyGT-mask0.5-SpicalDecode
# name=vidtok-video-fq-grey-c16t16m32t8-BOTH-greyGT-mask0.5-SpicalDecode
# name=vidtok-video-fq-grey-c16t16m64t8-BOTH-greyGT-mask0.5-SpicalDecode-hq0.5lq0.6
# name=vidtok-video-fq-grey-cd-c8t8m16t8-BOTH-mask0.5-SpicalDecode-hq0.6lq0.6
# name=vidtok-video-fq-grey-cd-c8t8m32t8-BOTH-mask0.5-SpicalDecode-hq0.6lq0.6
# name=vidtok-video-fq-grey-c8t8m32t8-BOTH-mask0.5-SpicalDecode-hq0.6lq0.6
name=vidtok-video-fq-grey-cd-c8t8m16t8-BOTH-mask0.5-SpicalDecode-hq0.6lq0.6-newcamera
# name=vidtok-video-fq-cd-c8t8m32t16-BOTH-mask0-SpicalDecode-hq0.6lq0.6-newcamera


accelerate launch \
  --config_file config/accelerate_config_8.yaml \
  train_amd.py \
  --name $name\
  --exp_root /mnt/pfs-gv8sxa/tts/dhg/liuhuaize/code/AMD2/exp_video\
  --train_datapath /mnt/pfs-gv8sxa/tts/dhg/wenzhangsun/.sun/dataset/webvid/vidtwin_csv/final_full.csv\
  --eval_datapath /mnt/pfs-gv8sxa/tts/dhg/wenzhangsun/.sun/dataset/webvid/vidtwin_csv/final.csv\
  --vae_version /mnt/pfs-gv8sxa/tts/dhg/zqy/model/sd-vae-ft-mse\
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
  --save_checkpoint_interval_step 5000\
  --amd_model_type AMD_N\
  --amd_image_patch_size 2\
  --amd_motion_patch_size 1\
  --amd_num_step 1000\
  --motion_token_num 40\
  --object_motion_token_num 8\
  --object_motion_token_channel 16\
  --object_enc_num_layers 8\
  --camera_motion_token_num 8\
  --camera_motion_token_channel 8\
  --camera_enc_num_layers 8\
  --motion_need_norm_out false\
  --need_motion_transformer false\
  --diffusion_model_type spatial\
  --use_filter true\
  --use_regularizers false\
  --use_grey false\
  --use_camera_down true\
  --use_mask false\
  --use_camera true\
  --use_object true\
  --frozen_name camera_motion_encoder\
  --resume_training /mnt/pfs-gv8sxa/tts/dhg/liuhuaize/code/AMD2/exp_video/vidtok-video-fq-grey-cd-c8t8m16t8-BOTH-mask0.5-SpicalDecode-hq0.6lq0.6-newcamera/checkpoints/checkpoint-125000 2>&1  | tee logs/$name.log
  # --amd_from_pretrained /mnt/pfs-gv8sxa/tts/dhg/liuhuaize/code/AMD2/exp_video/vidtok-video-fq-grey-cd-c8t8-greyGT-mask0.5-SpicalDecode/checkpoints/checkpoint-275000/model.safetensors 2>&1  | tee logs/$name.log
  # --resume_training /mnt/pfs-gv8sxa/tts/dhg/liuhuaize/code/AMD2/exp_video/vidtok-video-fq-grey-cd-c8t8-greyGT-mask0.5-SpicalDecode/checkpoints/checkpoint-165000 2>&1  | tee logs/$name.log
  # --amd_from_pretrained /mnt/pfs-gv8sxa/tts/dhg/liuhuaize/code/AMD2/exp_video/vidtok-video-fq-grey-c16m64t16-cameraonly-greyGT-mask0.5-SpicalDecode/checkpoints/checkpoint-80000/model.safetensors 2>&1  | tee logs/$name.log
  # --main_process_port 29501\
    # --resume_training /mnt/pfs-gv8sxa/tts/dhg/liuhuaize/code/AMD2/exp_video/vidtok-video-fq-grey-c8m32t8-cameraonly-greyGT-mask0.7-SpicalDecode/checkpoints/checkpoint-50000 2>&1  | tee logs/$name.log
  # --frozen_name camera\
  # --camera_mask_ratio 0.2\


# accelerate launch \
#   --config_file config/accelerate_config_8.yaml \
#   train_amd.py \
#   --name $name\
#   --exp_root /mnt/pfs-gv8sxa/tts/dhg/liuhuaize/code/AMD2/img_exp\
#   --train_datapath /mnt/pfs-gv8sxa/tts/dhg/wenzhangsun/.sun/dataset/webvid/vidtwin_csv/final_full.csv\
#   --eval_datapath /mnt/pfs-gv8sxa/tts/dhg/wenzhangsun/.sun/dataset/webvid/vidtwin_csv/final.csv\
#   --vae_version /mnt/pfs-gv8sxa/tts/dhg/zqy/model/sd-vae-ft-mse\
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
#   --save_checkpoint_interval_step 5000\
#   --amd_model_type AMD_N\
#   --amd_image_patch_size 2\
#   --amd_motion_patch_size 1\
#   --amd_num_step 1000\
#   --motion_token_num 8\
#   --object_motion_token_num 8\
#   --object_motion_token_channel 64\
#   --object_enc_num_layers 8\
#   --camera_motion_token_num 16\
#   --camera_motion_token_channel 16\
#   --camera_enc_num_layers 8\
#   --motion_need_norm_out false\
#   --need_motion_transformer false\
#   --diffusion_model_type default\
#   --use_filter true\
#   --use_regularizers false\
#   --use_motiontemporal false\
#   --use_grey true\
#   --use_camera_down false\
#   --camera_mask_ratio 0.7\
#   --object_mask_ratio 0.4\
#   --use_mask false\
#   --use_camera false\
#   --use_object true\
#   --resume_training /mnt/pfs-gv8sxa/tts/dhg/liuhuaize/code/AMD2/img_exp/vidtok-img-fq-grey-c16o64-objectonly/checkpoints/checkpoint-40000  2>&1  | tee logs/$name.log
  # --amd_from_pretrained /mnt/pfs-gv8sxa/tts/dhg/liuhuaize/code/AMD2/img_exp/vidtok-img2img-fq-grey-cd-c32m64-t9-cameraonly/checkpoints/checkpoint-92000/model.safetensors 2>&1  | tee logs/$name.log


  # --resume_training /mnt/pfs-gv8sxa/tts/dhg/liuhuaize/code/AMD2/img_exp/vidtok-img2img-fq-grey-cd-c32m64-t9-highfq-stage2-greyGT/checkpoints/checkpoint-60000\

