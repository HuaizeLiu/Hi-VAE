conda init bash;
cd /mnt/pfs-gv8sxa/tts/dhg/zqy/code/AMD2 # 要进入到自己的目录，否则会在lizrunv2对应机器上的/workspace
source activate /mnt/pfs-gv8sxa/tts/dhg/zqy/envs/zqy
conda activate /mnt/pfs-gv8sxa/tts/dhg/zqy/envs/zqy
which python

 commands=(
    " python extract_motion.py \
    --data_pkl /mnt/pfs-gv8sxa/tts/dhg/zqy/code/AMD2/dataset/path/a2m/videos_pkl/0.pkl"
    " python extract_motion.py \
    --data_pkl /mnt/pfs-gv8sxa/tts/dhg/zqy/code/AMD2/dataset/path/a2m/videos_pkl/1.pkl"
    " python extract_motion.py \
    --data_pkl /mnt/pfs-gv8sxa/tts/dhg/zqy/code/AMD2/dataset/path/a2m/videos_pkl/2.pkl"
    " python extract_motion.py \
    --data_pkl /mnt/pfs-gv8sxa/tts/dhg/zqy/code/AMD2/dataset/path/a2m/videos_pkl/3.pkl"
    " python extract_motion.py \
    --data_pkl /mnt/pfs-gv8sxa/tts/dhg/zqy/code/AMD2/dataset/path/a2m/videos_pkl/4.pkl"
    " python extract_motion.py \
    --data_pkl /mnt/pfs-gv8sxa/tts/dhg/zqy/code/AMD2/dataset/path/a2m/videos_pkl/5.pkl"
    " python extract_motion.py \
    --data_pkl /mnt/pfs-gv8sxa/tts/dhg/zqy/code/AMD2/dataset/path/a2m/videos_pkl/6.pkl"
    " python extract_motion.py \
    --data_pkl /mnt/pfs-gv8sxa/tts/dhg/zqy/code/AMD2/dataset/path/a2m/videos_pkl/7.pkl"
)
for i in "${!commands[@]}"; do
    cmd="env CUDA_VISIBLE_DEVICES=$i  ${commands[$i]}"
    echo "Executing: $cmd"
    $cmd &
done
wait
echo "所有命令执行完毕"