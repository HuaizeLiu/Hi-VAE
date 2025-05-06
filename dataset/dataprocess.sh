conda init bash;
cd /mnt/pfs-mc0p4k/tts/team/digital_avatar_group/liuhuaize/hallo2 # 要进入到自己的目录，否则会在lizrunv2对应机器上的/workspace
source activate /mnt/pfs-mc0p4k/tts/team/digital_avatar_group/liuhuaize/envs/hallo2
conda activate /mnt/pfs-mc0p4k/tts/team/digital_avatar_group/liuhuaize/envs/hallo2
which python

python -m scripts.data_preprocess --input_dir /mnt/pfs-mc0p4k/tts/team/digital_avatar_group/sunwenzhang/qiyuan/data/celebv-hq/videos --step 2 -p 16