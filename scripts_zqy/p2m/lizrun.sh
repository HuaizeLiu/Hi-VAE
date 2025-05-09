#!/bin/bash
lizrun start -j $2 \
-n 1 \
-g 8 \
-i reg-ai.chehejia.com/ssai/lizr/cu118/py310/pytorch:2.0.1-multinode-lizr-nccl \
-c "bash /mnt/pfs-gv8sxa/tts/dhg/zqy/code/AMD2/scripts/p2m/$1 $2" \
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


# ------------------------- p2m ----------------------------

# bash lizrun.sh p2m-t1d512.sh p2m-t1d512-f16
# lizrun logs p2m-t1d512-f16-didonglin-master-0 -f
# lizrun login p2m-t1d512-f16-didonglin-master-0    
# lizrun stop p2m-t1d512-f16-didonglin 
