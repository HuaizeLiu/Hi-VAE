# cd /mnt/pfs-gv8sxa/tts/dhg/zqy/code/AMD2
# env CUDA_VISIBLE_DEVICES=1 python extract_motion.py --data_pkl /mnt/pfs-gv8sxa/tts/dhg/zqy/code/AMD2/dataset/path/a2m_split/mead_test.pkl

from model import AMD_models,AMDModel
from model.utils import save_cfg, vae_encode, vae_decode, freeze, print_param_num,model_load_pretrain
import torchvision.transforms as transforms
from diffusers import AutoencoderKL
from decord import VideoReader
from decord import cpu, gpu
import os
import sys
import torch
import argparse
import pickle
import time
import gc

# args
parser = argparse.ArgumentParser(description='Process audio files.')
parser.add_argument('--data_pkl', type=str, required=True, help='data pkl')
args = parser.parse_args()

# device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# transform
sample_size = (256,256)
pixel_transforms = transforms.Compose([
            transforms.Resize(min(sample_size)),
            transforms.CenterCrop(sample_size),
            transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5], inplace=True),
        ])

# dir
amd_config = '/mnt/pfs-gv8sxa/tts/dhg/zqy/exp/amd-s-t4-d128-nonorm/config.json'
amd_ckpt = '/mnt/pfs-gv8sxa/tts/dhg/zqy/exp/amd-s-t4-d128-nonorm/checkpoints/checkpoint-188000/model.safetensors'
vae_version = '/mnt/pfs-mc0p4k/tts/team/digital_avatar_group/sunwenzhang/qiyuan/model-checkpoints/sd-vae-ft-mse'


# vae
vae = AutoencoderKL.from_pretrained(vae_version, subfolder="vae").requires_grad_(False)
vae.to(device)

# amd_model
amd_model = AMDModel.from_config(AMDModel.load_config(amd_config))
model_load_pretrain(amd_model,amd_ckpt,not_load_keyword='abcabcacbd',strict=True)
amd_model.to(device)
print(f'######### load AMD weight from {amd_ckpt} #############')

# load pkl
with open(args.data_pkl, 'rb') as f:
    datas = pickle.load(f)


# log
log = []
log_path = args.data_pkl.split('.')[0]+'_log.pkl'

total_num = len(datas)
for i,data in enumerate(datas):
    try:
        video_path = data['video_path']
        motion_path = data['motion_path']

        if os.path.exists(motion_path):
            continue

        # read
        video_reader = VideoReader(video_path, ctx=cpu(0))
        idx = [j for j in range(len(video_reader))]
        videos = torch.from_numpy(video_reader.get_batch(idx).asnumpy()).permute(0, 3, 1, 2).contiguous() #(T,H,W,C)->(T,C,H,W)
        videos = videos / 255.0
        videos = pixel_transforms(videos)
        videos = videos.to(device)
        videos = videos.unsqueeze(0) #(N,T,C,H,W)

        with torch.no_grad():
            z = vae_encode(vae,videos).to(device) # N,T,c,h,w

            # # test loss
            # test_video = z[:,:4]
            # test_ref_img = z[:,-4:]
            # _,_,loss_dict = amd_model(test_video,test_ref_img)

            # get motion
            motion = amd_model.extract_motion(z)
            motion = motion.squeeze(0)

            # log
            log.append({
                'video_path':video_path,
                'motion_path':motion_path,
                'num_frames':len(video_reader),
            })

        # save
        torch.save(motion,motion_path)

        gc.collect()
        torch.cuda.empty_cache()
        del video_reader,z,motion,videos

        # print
        print(f'{i} has been processed, total:{total_num}')


    except Exception as e:
        # file_name = self.metadata_list[idx]['name']
        # print(file_name)
        print('error',e)
        gc.collect()
        torch.cuda.empty_cache()
        continue

with open(log_path, 'wb') as file:
    # 使用pickle模块的dump方法保存数据
    pickle.dump(log, file)

print('******** All Finished **********')

