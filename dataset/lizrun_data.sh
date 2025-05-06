lizrunv2 start -j $1 \
-n 1 \
-g 1 \
-i reg-ai.chehejia.com/ssai/lizr/cu118/py310/pytorch:2.0.1-multinode-lizr-nccl \
-c "bash /mnt/pfs-mc0p4k/tts/team/digital_avatar_group/sunwenzhang/qiyuan/code/AMD_linear/dataset/dataprocess.sh $1" \
-p dhg-a800


# data-celebv

# bash lizrun_data.sh data-celebv
# lizrunv2 logs data-celebv-didonglin-master-0 -f
# lizrunv2 login data-celebv-didonglin-master-0    
# lizrunv2 stop data-celebv-didonglin 