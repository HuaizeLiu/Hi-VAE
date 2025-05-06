conda init bash;
cd /mnt/pfs-gv8sxa/tts/dhg/liuhuaize/code/AMD2 # 要进入到自己的目录，否则会在lizrunv2对应机器上的/workspace
source activate /mnt/pfs-gv8sxa/tts/dhg/zqy/envs/zqy
conda activate /mnt/pfs-gv8sxa/tts/dhg/zqy/envs/zqy
which python

# name=am64lm64-t2m-ucf-threeloss-c8t8m32t16
# name=labm64-t2m-ucf-Sperateloss-c8t8m32t16-nocontent-speratenorm
# name=labm64-t2m-ucf-objectloss-c8t8m32t16-CondCamera
# name=labm64-t2m-ucf-objectloss-c16t16m32t8-CondCamera
# name=labm64-t2m-ucf-objectloss-c16t16m32t8-CondCamera-objectsource
# name=labm128-hiddenstatet16*64-ucf-objectloss-c8t8m32t16-CondCamera-objectsource-plusImg
# name=labm128-hiddenstatet16*64-ucf-objectloss-c16t16m32t8-CondCamera-plusImg
# name=labm128-hiddenstatet16*128-ucf-objectloss-c8t8m32t16-CondCamera-plusImg
name=labm128-hiddenstatet16*128-layer20-ucf-objectloss-c8t8m32t16-CondCamera-objectsource-plusImg
# name=labm128-hiddenstatet16*128-layer20-ucf-objectloss-c16t16m32t8-CondCamera-objectsource-plusImg

# accelerate launch \
#   --config_file config/accelerate_config_8.yaml \
#   train_t2m.py\
#   --name $name\
#   --exp_root /mnt/pfs-gv8sxa/tts/dhg/liuhuaize/code/AMD2/exp_t2m\
#   --trainset /mnt/pfs-gv8sxa/tts/dhg/liuhuaize/dataset/UCF-train.csv\
#   --evalset /mnt/pfs-gv8sxa/tts/dhg/liuhuaize/dataset/UCF-test.csv\
#   --vae_version /mnt/pfs-gv8sxa/tts/dhg/zqy/model/sd-vae-ft-mse\
#   --sample_size '[256,256]'\
#   --num_workers 16\
#   --target_fps 16\
#   --sample_n_frames 16\
#   --batch_size 16\
#   --lr 0.0001\
#   --eval_interval_step 2000\
#   --max_train_steps 30000 \
#   --save_checkpoint_interval_step 2000\
#   --motion_dim 32\
#   --label_dim 256\
#   --amd_config /mnt/pfs-gv8sxa/tts/dhg/liuhuaize/code/AMD2/exp_video/vidtok-video-fq-cd-c8t8m32t16-BOTH-mask0-SpicalDecode-hq0.6lq0.6-newcamera/config.json\
#   --amd_ckpt /mnt/pfs-gv8sxa/tts/dhg/liuhuaize/code/AMD2/exp_video/vidtok-video-fq-cd-c8t8m32t16-BOTH-mask0-SpicalDecode-hq0.6lq0.6-newcamera/checkpoints/checkpoint-315000/model.safetensors\
  # --camera_mask_ratio 0.5\
  # --resume_from_checkpoint /mnt/pfs-gv8sxa/tts/dhg/liuhuaize/code/AMD2/exp_t2m/am64lm64-t2m-ucf/checkpoints/checkpoint-28000\

accelerate launch \
  --config_file config/accelerate_config_8.yaml \
  train_t2m.py\
  --name $name\
  --exp_root /mnt/pfs-gv8sxa/tts/dhg/liuhuaize/code/AMD2/exp_t2m\
  --trainset /mnt/pfs-gv8sxa/tts/dhg/liuhuaize/dataset/UCF-train.csv\
  --evalset /mnt/pfs-gv8sxa/tts/dhg/liuhuaize/dataset/UCF-test.csv\
  --vae_version /mnt/pfs-gv8sxa/tts/dhg/zqy/model/sd-vae-ft-mse\
  --sample_size '[256,256]'\
  --num_workers 16\
  --target_fps 16\
  --sample_n_frames 16\
  --batch_size 16\
  --lr 0.0001\
  --eval_interval_step 2000\
  --max_train_steps 60000 \
  --save_checkpoint_interval_step 2000\
  --motion_dim 32\
  --label_dim 128\
  --amd_config /mnt/pfs-gv8sxa/tts/dhg/liuhuaize/code/AMD2/exp_video/vidtok-video-fq-cd-c8t8m32t16-BOTH-mask0-SpicalDecode-hq0.6lq0.6-newcamera/config.json\
  --amd_ckpt /mnt/pfs-gv8sxa/tts/dhg/liuhuaize/code/AMD2/exp_video/vidtok-video-fq-cd-c8t8m32t16-BOTH-mask0-SpicalDecode-hq0.6lq0.6-newcamera/checkpoints/checkpoint-315000/model.safetensors\
  --resume_from_checkpoint /mnt/pfs-gv8sxa/tts/dhg/liuhuaize/code/AMD2/exp_t2m/labm128-hiddenstatet16*128-layer20-ucf-objectloss-c8t8m32t16-CondCamera-objectsource-plusImg/checkpoints/checkpoint-12000


# 如果参考vidtwin，label_dim为256， latents_dim为1024 = 128*8