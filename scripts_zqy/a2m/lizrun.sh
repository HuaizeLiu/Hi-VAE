#!/bin/bash
lizrun start -j $2 \
-n 1 \
-g 8 \
-i reg-ai.chehejia.com/ssai/lizr/cu118/py310/pytorch:2.0.1-multinode-lizr-nccl \
-c "bash /mnt/pfs-gv8sxa/tts/dhg/zqy/code/AMD2/scripts/a2m/$1 $2" \
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


# bash lizrun.sh amd-s-t4d128-frame16.sh a2m-amd-s-t4d128-frame16
# lizrun logs a2m-amd-s-t4d128-frame16-didonglin-master-0 -f
# lizrun login a2m-amd-s-t4d128-frame16-didonglin-master-0    
# lizrun stop a2m-amd-s-t4d128-frame16-didonglin 

# bash lizrun.sh amd-s-t4d128-frame64.sh a2m-amd-s-t4d128-frame64
# lizrun logs a2m-amd-s-t4d128-frame64-didonglin-master-0 -f
# lizrun login a2m-amd-s-t4d128-frame64-didonglin-master-0    
# lizrun stop a2m-amd-s-t4d128-frame64-didonglin 



# bash lizrun.sh amd-s-t4d128-frame32.sh a2m-amd-s-t4d128-frame32
# lizrun logs a2m-amd-s-t4d128-frame32-didonglin-master-0 -f
# lizrun login a2m-amd-s-t4d128-frame32-didonglin-master-0    
# lizrun stop a2m-amd-s-t4d128-frame32-didonglin 



# ------------------------- t4d128 ----------------------------

# bash lizrun.sh amd-s-t4d128-cross-audio-frame32.sh amd-s-t4d128-cross-audio-frame32
# lizrun logs amd-s-t4d128-cross-audio-frame32-didonglin-master-0 -f
# lizrun login amd-s-t4d128-cross-audio-frame32-didonglin-master-0    
# lizrun stop amd-s-t4d128-cross-audio-frame32-didonglin 

# bash lizrun.sh amd-s-t4d128-cross-audio-pose-frame32.sh amd-s-t4d128-cross-audio-pose-frame32
# lizrun logs amd-s-t4d128-cross-audio-pose-frame32-didonglin-master-0 -f
# lizrun login amd-s-t4d128-cross-audio-pose-frame32-didonglin-master-0    
# lizrun stop amd-s-t4d128-cross-audio-pose-frame32-didonglin 


# bash lizrun.sh amd-s-t4d128-cross-audio-frame16.sh amd-s-t4d128-cross-audio-frame16
# lizrun logs amd-s-t4d128-cross-audio-frame16-didonglin-master-0 -f
# lizrun login amd-s-t4d128-cross-audio-frame16-didonglin-master-0    
# lizrun stop amd-s-t4d128-cross-audio-frame16-didonglin 


# bash lizrun.sh amd-s-t4d128-cross-audio-pose-frame16.sh amd-s-t4d128-cross-audio-pose-frame16
# lizrun logs amd-s-t4d128-cross-audio-pose-frame16-didonglin-master-0 -f
# lizrun login amd-s-t4d128-cross-audio-pose-frame16-didonglin-master-0    
# lizrun stop amd-s-t4d128-cross-audio-pose-frame16-didonglin 


# ------------------------- t2d256 ----------------------------

# bash lizrun.sh amd-s-t2d256-cross-audio-frame16.sh a2m-token2-frame16
# lizrun logs a2m-token2-frame16-didonglin-master-0 -f
# lizrun login a2m-token2-frame16-didonglin-master-0    
# lizrun stop a2m-token2-frame16-didonglin 

# bash lizrun.sh amd-s-t2d256-cross-audio-pose-frame16.sh a2m-dwpose-token2-frame16
# lizrun logs a2m-dwpose-token2-frame16-didonglin-master-0 -f
# lizrun login a2m-dwpose-token2-frame16-didonglin-master-0    
# lizrun stop a2m-dwpose-token2-frame16-didonglin 

# bash lizrun.sh amd-s-t2d256-cross-audio-posepre-frame16.sh a2m-dwpose-pre-token2-frame16
# lizrun logs a2m-dwpose-pre-token2-frame16-didonglin-master-0 -f
# lizrun login a2m-dwpose-pre-token2-frame16-didonglin-master-0    
# lizrun stop a2m-dwpose-pre-token2-frame16-didonglin 


# bash lizrun.sh amd-s-t2d256-cross-audio-frame16-wo-mt.sh a2m-token2-frame16-motion-wo-mt
# lizrun logs a2m-token2-frame16-motion-wo-mt-didonglin-master-0 -f
# lizrun login a2m-token2-frame16-motion-wo-mt-didonglin-master-0    
# lizrun stop a2m-token2-frame16-motion-wo-mt-didonglin 

# bash lizrun.sh amd-s-t2d256-cross-audio-posepre-frame16-wo-mt.sh a2m-token2-frame16-posepre-motion-wo-mt
# lizrun logs a2m-token2-frame16-posepre-motion-wo-mt-didonglin-master-0 -f
# lizrun login a2m-token2-frame16-posepre-motion-wo-mt-didonglin-master-0    
# lizrun stop a2m-token2-frame16-posepre-motion-wo-mt-didonglin 


# ------------------------- t1d512 ----------------------------

# bash lizrun.sh amd-s-t1d512-cross-audio-frame16.sh amd-s-t1d512-cross-audio-frame16
# lizrun logs amd-s-t1d512-cross-audio-frame16-didonglin-master-0 -f
# lizrun login amd-s-t1d512-cross-audio-frame16-didonglin-master-0    
# lizrun stop amd-s-t1d512-cross-audio-frame16-didonglin 

# bash lizrun.sh amd-s-t1d512-cross-audio-posepre-frame16.sh amd-s-t1d512-cross-audio-posepre-frame16
# lizrun logs amd-s-t1d512-cross-audio-posepre-frame16-didonglin-master-0 -f
# lizrun login amd-s-t1d512-cross-audio-posepre-frame16-didonglin-master-0    
# lizrun stop amd-s-t1d512-cross-audio-posepre-frame16-didonglin 

# bash lizrun.sh amd-s-t1d512-cross-audio-posepre-frame32.sh amd-s-t1d512-cross-audio-posepre-frame32
# lizrun logs amd-s-t1d512-cross-audio-posepre-frame32-didonglin-master-0 -f
# lizrun login amd-s-t1d512-cross-audio-posepre-frame32-didonglin-master-0    
# lizrun stop amd-s-t1d512-cross-audio-posepre-frame32-didonglin 

# bash lizrun.sh amd-s-t1d512-cross-audio-frame16-wo-mt.sh a2m-token1-frame16-motion-wo-mt
# lizrun logs a2m-token1-frame16-motion-wo-mt-didonglin-master-0 -f
# lizrun login a2m-token1-frame16-motion-wo-mt-didonglin-master-0    
# lizrun stop a2m-token1-frame16-motion-wo-mt-didonglin 

# bash lizrun.sh amd-s-t1d512-cross-audio-posepre-frame16-wo-mt.sh a2m-token1-frame16-posepre-motion-wo-mt
# lizrun logs a2m-token1-frame16-posepre-motion-wo-mt-didonglin-master-0 -f
# lizrun login a2m-token1-frame16-posepre-motion-wo-mt-didonglin-master-0    
# lizrun stop a2m-token1-frame16-posepre-motion-wo-mt-didonglin

# bash lizrun.sh amd-s-t1d512-cross-audio-pose-frame16-wo-mt.sh a2m-token1-frame16-pose-motion-wo-mt
# lizrun logs a2m-token1-frame16-pose-motion-wo-mt-didonglin-master-0 -f
# lizrun login a2m-token1-frame16-pose-motion-wo-mt-didonglin-master-0    
# lizrun stop a2m-token1-frame16-pose-motion-wo-mt-didonglin


# ------------------------- spatial ----------------------------
# bash lizrun.sh a2m-t1d512-frame16-spatial.sh a2m-t1d512-f16-spatial
# lizrun logs a2m-t1d512-f16-spatial-didonglin-master-0 -f
# lizrun login a2m-t1d512-f16-spatial-didonglin-master-0    
# lizrun stop a2m-t1d512-f16-spatial-didonglin 

# bash lizrun.sh a2m-t1d512-frame16-posepre-spatial.sh a2m-t1d512-f16-posepre-spatial
# lizrun logs a2m-t1d512-f16-posepre-spatial-didonglin-master-0 -f
# lizrun login a2m-t1d512-f16-posepre-spatial-didonglin-master-0    
# lizrun stop a2m-t1d512-f16-posepre-spatial-didonglin 

# bash lizrun.sh a2m-t2d256-frame16-spatial.sh a2m-t2d256-f16-spatial
# lizrun logs a2m-t2d256-f16-spatial-didonglin-master-0 -f
# lizrun login a2m-t2d256-f16-spatial-didonglin-master-0    
# lizrun stop a2m-t2d256-f16-spatial-didonglin 

# bash lizrun.sh a2m-t2d256-frame16-posepre-spatial.sh a2m-t2d256-f16-posepre-spatial
# lizrun logs a2m-t2d256-f16-posepre-spatial-didonglin-master-0 -f
# lizrun login a2m-t2d256-f16-posepre-spatial-didonglin-master-0    
# lizrun stop a2m-t2d256-f16-posepre-spatial-didonglin 