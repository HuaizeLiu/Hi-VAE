#!/bin/bash
lizrun start -j $2 \
-n 1 \
-g 8 \
-i reg-ai.chehejia.com/ssai/lizr/cu118/py310/pytorch:2.0.1-multinode-lizr-nccl \
-c "bash /mnt/pfs-gv8sxa/tts/dhg/zqy/code/AMD2/scripts/amd/$1 $2" \
-p dhg-a800

# gpu数量 默认=8
# $1: jobname, $2: GPU id, $3: xxx_std.yaml

# -c "your script absolute path" -n number_of_nodes -j job_name -i image_name -p resource_pool_name


# 查看GPU池；   (经常用)
# lizrun pool get -p dhg-a800 -d
# lizrun pool get -p wq-dhg -d
# lizrun pool get -p a800 -d
# lizrun pool get -p dhg -d

# 启动任务
# bash lizrun_amd_mae.sh amd-v54-ae-4ds
# bash lizrun.sh amd-b-mae-s-dif-lr

# 查看job日志； (看log)
# lizrun logs amd-v54-ae-4ds-didonglin-master-0 -f

# 登录job镜像； (任务行)
# lizrun login amd-v54-ae-4ds-didonglin-master-0    # 10/3

# 查看job状态； (没啥用)
# lizrun status -j zqy-amd-s-didonglin

# 查看job列表；
# lizrun list

# 销毁job；
# lizrun stop amd-v54-ae-4ds-didonglin



# ---------------------------------------------

# bash lizrun.sh amd_t2d256_nonorm.sh amd-s-t2-d256-nonorm
# lizrun logs amd-s-t2-d256-nonorm-didonglin-master-0 -f
# lizrun login amd-s-t2-d256-nonorm-didonglin-master-0    
# lizrun stop amd-s-t2-d256-nonorm-didonglin 

# bash lizrun.sh amd_t1d512_nonorm.sh amd-s-t1-d512-nonorm
# lizrun logs amd-s-t1-d512-nonorm-didonglin-master-0 -f
# lizrun login amd-s-t1-d512-nonorm-didonglin-master-0    
# lizrun stop amd-s-t1-d512-nonorm-didonglin 


# bash lizrun.sh amd_t1d1024_nonorm_512.sh amd-s-t1-d1024-nonorm-512px
# lizrun logs amd-s-t1-d1024-nonorm-512px-didonglin-master-0 -f
# lizrun login amd-s-t1-d1024-nonorm-512px-didonglin-master-0    
# lizrun stop amd-s-t1-d1024-nonorm-512px-didonglin 


# bash lizrun.sh amd_fullbody_t8d128_nonorm.sh amd-s-fullbody-t8-d128-nonorm-2ds
# lizrun logs amd-s-fullbody-t8-d128-nonorm-2ds-didonglin-master-0 -f
# lizrun login amd-s-fullbody-t8-d128-nonorm-2ds-didonglin-master-0    
# lizrun stop amd-s-fullbody-t8-d128-nonorm-2ds-didonglin 

# bash lizrun.sh amd_fullbody_t8d128_nonorm_1ds.sh amd-s-fullbody-t8-d128-nonorm-1ds
# lizrun logs amd-s-fullbody-t8-d128-nonorm-1ds-didonglin-master-0 -f
# lizrun login amd-s-fullbody-t8-d128-nonorm-1ds-didonglin-master-0    
# lizrun stop amd-s-fullbody-t8-d128-nonorm-1ds-didonglin 

# ----------------------------- motion transformer ------------------------

# bash lizrun.sh amd_t4d128_nonorm_mt.sh amd-s-t4-d128-nonorm-motion-transformer
# lizrun logs amd-s-t4-d128-nonorm-motion-transformer-didonglin-master-0 -f
# lizrun login amd-s-t4-d128-nonorm-motion-transformer-didonglin-master-0    
# lizrun stop amd-s-t4-d128-nonorm-motion-transformer-didonglin 

# bash lizrun.sh amd_t2d256_nonorm_mt.sh amd-s-t2-d256-nonorm-motion-transformer-new
# lizrun logs amd-s-t2-d256-nonorm-motion-transformer-new-didonglin-master-0 -f
# lizrun login amd-s-t2-d256-nonorm-motion-transformer-new-didonglin-master-0    
# lizrun stop amd-s-t2-d256-nonorm-motion-transformer-new-didonglin 


# bash lizrun.sh amd_t1d512_nonorm_mt.sh amd-s-t1-d512-nonorm-motion-transformer-new
# lizrun logs amd-s-t1-d512-nonorm-motion-transformer-new-didonglin-master-0 -f
# lizrun login amd-s-t1-d512-nonorm-motion-transformer-new-didonglin-master-0    
# lizrun stop amd-s-t1-d512-nonorm-motion-transforme-newr-didonglin 


# ----------------------------- rec ------------------------

# bash lizrun.sh amd_rec_t2d256_nonorm.sh amd-rec-t2d256-nonorm
# lizrun logs amd-rec-t2d256-nonorm-didonglin-master-0 -f
# lizrun login amd-rec-t2d256-nonorm-didonglin-master-0    
# lizrun stop amd-rec-t2d256-nonorm-didonglin 

# bash lizrun.sh amd_rec_t1d512_nonorm.sh amd-rec-t1d512-nonorm
# lizrun logs amd-rec-t1d512-nonorm-didonglin-master-0 -f
# lizrun login amd-rec-t1d512-nonorm-didonglin-master-0    
# lizrun stop amd-rec-t1d512-nonorm-didonglin 

# bash lizrun.sh amd_rec_t1d1024_nonorm.sh amd-rec-t1d1024-nonorm
# lizrun logs amd-rec-t1d1024-nonorm-didonglin-master-0 -f
# lizrun login amd-rec-t1d1024-nonorm-didonglin-master-0    
# lizrun stop amd-rec-t1d1024-nonorm-didonglin 



# ----------------------------- sample  ------------------------

# bash lizrun.sh amd_t2d256_nonorm_mt_sample.sh amd-s-t2-d256-nonorm-mt-sample
# lizrun logs amd-s-t2-d256-nonorm-mt-sample-didonglin-master-0 -f
# lizrun login amd-s-t2-d256-nonorm-mt-sample-didonglin-master-0    
# lizrun stop amd-s-t2-d256-nonorm-mt-sample-didonglin 

# ----------------------------- dual  ------------------------
# bash lizrun.sh amd_t1d512_nonorm_dual.sh amd-s-t1-d512-nonorm-dual
# lizrun logs amd-s-t1-d512-nonorm-dual-didonglin-master-0 -f
# lizrun login amd-s-t1-d512-nonorm-dual-didonglin-master-0    
# lizrun stop amd-s-t1-d512-nonorm-dual-didonglin 

# bash lizrun.sh amd_t2d256_nonorm_dual.sh amd-s-t2-d256-nonorm-dual
# lizrun logs amd-s-t2-d256-nonorm-dual-didonglin-master-0 -f
# lizrun login amd-s-t2-d256-nonorm-dual-didonglin-master-0    
# lizrun stop amd-s-t2-d256-nonorm-dual-didonglin 

# ----------------------------- spatial  ------------------------
# bash lizrun.sh amd_t1d512_nonorm_spatial.sh amd-s-t1-d512-nonorm-spatial
# lizrun logs amd-s-t1-d512-nonorm-spatial-didonglin-master-0 -f
# lizrun login amd-s-t1-d512-nonorm-spatial-didonglin-master-0    
# lizrun stop amd-s-t1-d512-nonorm-spatial-didonglin 

# bash lizrun.sh amd_t2d256_nonorm_spatial.sh amd-s-t2-d256-nonorm-spatial
# lizrun logs amd-s-t2-d256-nonorm-spatial-didonglin-master-0 -f
# lizrun login amd-s-t2-d256-nonorm-spatial-didonglin-master-0    
# lizrun stop amd-s-t2-d256-nonorm-spatial-didonglin 

# bash lizrun.sh amd_t1d512_nonorm_spatial.sh amd-s-t1-d512-nonorm-spatial-f30
# lizrun logs amd-s-t1-d512-nonorm-spatial-f30-didonglin-master-0 -f
# lizrun login amd-s-t1-d512-nonorm-spatial-f30-didonglin-master-0    
# lizrun stop amd-s-t1-d512-nonorm-spatial-f30-didonglin 

# bash lizrun.sh amd_t2d256_nonorm_spatial.sh amd-s-t2-d256-nonorm-spatial-f30
# lizrun logs amd-s-t2-d256-nonorm-spatial-f30-didonglin-master-0 -f
# lizrun login amd-s-t2-d256-nonorm-spatial-f30-didonglin-master-0    
# lizrun stop amd-s-t2-d256-nonorm-spatial-f30-didonglin 