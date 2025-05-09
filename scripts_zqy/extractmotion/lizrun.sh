#!/bin/bash
lizrun start -j $1 \
-n 1 \
-g 8 \
-i reg-ai.chehejia.com/ssai/lizr/cu118/py310/pytorch:2.0.1-multinode-lizr-nccl \
-c "bash /mnt/pfs-gv8sxa/tts/dhg/zqy/code/AMD2/scripts/extractmotion/get_motion.sh $1" \
-p dhg-a800


# bash lizrun.sh extract-motion
# lizrun logs extract-motion-didonglin-master-0 -f
# lizrun login extract-motion-didonglin-master-0    
# lizrun stop extract-motion-didonglin 