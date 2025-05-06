conda init bash;
cd /mnt/pfs-gv8sxa/tts/dhg/liuhuaize/code/AMD2 # 要进入到自己的目录，否则会在lizrunv2对应机器上的/workspace
source activate /mnt/pfs-gv8sxa/tts/dhg/zqy/envs/zqy
conda activate /mnt/pfs-gv8sxa/tts/dhg/zqy/envs/zqy
which python

# python /mnt/pfs-gv8sxa/tts/dhg/liuhuaize/code/AMD2/amd_inference.py \
# --exp_name amd_realistate_cd_c16m64_cameraonly_grey_m0.5 \
# --data_dir /mnt/pfs-gv8sxa/tts/dhg/liuhuaize/dataset/realistim_data/videos \
# --save_dir /mnt/pfs-gv8sxa/tts/dhg/liuhuaize/code/AMD2/inference_amd_exp_new \
# --video_sample_step 20 \
# --sample_window 16 \
# --fps 8 \
# --amd_model_type AMDModel_New \
# --drop_prev_img false \
# --use_grey true \
# --inference_single true \
# --camera_mask_ratio 0.5 \
# --object_mask_ratio 0.5 \
# --amd_config /mnt/pfs-gv8sxa/tts/dhg/liuhuaize/code/AMD2/exp_video/vidtok-video-fq-grey-cd-c16m64-cameraonly-greyGT/config.json \
# --amd_ckpt /mnt/pfs-gv8sxa/tts/dhg/liuhuaize/code/AMD2/exp_video/vidtok-video-fq-grey-cd-c16m64-cameraonly-greyGT/checkpoints 2>&1  | tee logs/amd_inference.log

# /mnt/pfs-gv8sxa/tts/dhg/liuhuaize/dataset/test_data/videos_256
# /mnt/pfs-gv8sxa/tts/dhg/liuhuaize/dataset/realistim_data/videos

python /mnt/pfs-gv8sxa/tts/dhg/liuhuaize/code/AMD2/amd_inference.py \
--exp_name c16t16m32t8 \
--data_dir /mnt/pfs-gv8sxa/tts/dhg/liuhuaize/dataset/latents_role/GT_256_8 \
--save_dir /mnt/pfs-gv8sxa/tts/dhg/liuhuaize/dataset/latents_role \
--video_sample_step 20 \
--sample_window 16 \
--fps 8 \
--amd_model_type AMDModel_New \
--drop_prev_img false \
--use_grey true \
--inference_single true \
--amd_config /mnt/pfs-gv8sxa/tts/dhg/liuhuaize/code/AMD2/exp_video/vidtok-video-fq-grey-c16t16m32t8-BOTH-greyGT-mask0.5-SpicalDecode/config.json \
--amd_ckpt /mnt/pfs-gv8sxa/tts/dhg/liuhuaize/code/AMD2/exp_video/vidtok-video-fq-grey-c16t16m32t8-BOTH-greyGT-mask0.5-SpicalDecode/checkpoints 2>&1  | tee logs/amd_inference.log
# # --object_mask_ratio 0 \
# # --camera_mask_ratio 0.5 \

# cross-inference
# python /mnt/pfs-gv8sxa/tts/dhg/liuhuaize/code/AMD2/amd_inference_single.py \
# --exp_name c8t8m32t16_cross_reconstruction_result \
# --data_path_1 /mnt/pfs-gv8sxa/tts/dhg/liuhuaize/dataset/test_data/videos/0_A_basketball_moves_across_a_clean_table,_and_there_is_only_t_0_seed1234_000000.mp4 \
# --data_path_2 /mnt/pfs-gv8sxa/tts/dhg/liuhuaize/dataset/test_data/videos/0_A_square_box_moves_across_a_clean_table,_and_there_is_only_t_0_seed1234_000000.mp4 \
# --save_dir /mnt/pfs-gv8sxa/tts/dhg/liuhuaize/dataset \
# --video_sample_step 20 \
# --sample_window 16 \
# --fps 8 \
# --amd_model_type AMDModel_New \
# --drop_prev_img false \
# --use_grey true \
# --inference_single true \
# --camera_mask_ratio 0.8 \
# --amd_config /mnt/pfs-gv8sxa/tts/dhg/liuhuaize/code/AMD2/exp_video/vidtok-video-fq-cd-c8t8m32t16-BOTH-mask0-SpicalDecode-hq0.6lq0.6-newcamera/config.json \
# --amd_ckpt /mnt/pfs-gv8sxa/tts/dhg/liuhuaize/code/AMD2/exp_video/vidtok-video-fq-cd-c8t8m32t16-BOTH-mask0-SpicalDecode-hq0.6lq0.6-newcamera/checkpoints 2>&1  | tee logs/amd_inference.log
# --object_mask_ratio 0 \
# --camera_mask_ratio 0.5 \
# /mnt/pfs-gv8sxa/tts/dhg/liuhuaize/code/AMD2/exp_video/vidtok-video-fq-cd-c8t8m32t16-BOTH-mask0-SpicalDecode-hq0.6lq0.6-newcamera