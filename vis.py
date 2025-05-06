#%%
from omegaconf import OmegaConf
from model.model_A2M import A2MModel_CrossAtten_Audio_PosePre 
from safetensors.torch import load_model
import numpy as np
import torch
import os
from torchvision.io import read_video, write_video
from torchvision import transforms
from diffusers.models import AutoencoderKL  
from model.utils import vae_encode, vae_decode
from einops import rearrange
device = torch.device("cuda:1")
dtype = torch.float32
vae_path = "/mnt/pfs-mc0p4k/tts/team/digital_avatar_group/sunwenzhang/qiyuan/model-checkpoints/sd-vae-ft-mse"
vae = AutoencoderKL.from_pretrained(vae_path, subfolder="vae").to(device).requires_grad_(False)
config = OmegaConf.load("/mnt/pfs-gv8sxa/tts/dhg/zqy/code/AMD2/config/a2m/a2m_t1d512_posepre.yaml")
audio_decoder = A2MModel_CrossAtten_Audio_PosePre(**config['model']).to(device, dtype)
load_model(audio_decoder, "/mnt/pfs-gv8sxa/tts/dhg/zqy/newexp/a2m-t1d512-f16-posepre-spatial/checkpoints/checkpoint-75000/model.safetensors")
#%%
pixel_transforms = transforms.Compose([
            transforms.Resize(256),
            transforms.CenterCrop(256),
            transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5], inplace=True),
])
bs = 4
audio_emb_path = "/mnt/pfs-gv8sxa/tts/dhg/zqy/data/FaceVid_240h/whisper_embs"
pv_path = "/mnt/pfs-gv8sxa/tts/dhg/zqy/data/FaceVid_240h/videos_dwpose"
afps = os.listdir(audio_emb_path)[:bs]
rpps = list(map(lambda x: x.replace(".pt", ".mp4"), afps))
afs = []
pvs = []
sample_frames = 17
for i in range(bs):
    af_path = os.path.join(audio_emb_path, afps[i])
    rp_path = os.path.join(pv_path, rpps[i])
    af = torch.load(af_path)
    pv,_,_ = read_video(rp_path, pts_unit="sec", output_format="TCHW")
    assert len(af) >= sample_frames, "too short"
    s = np.random.randint(0, len(af) - sample_frames)
    index = torch.linspace(s, min(s + sample_frames, len(af)),sample_frames, dtype=torch.int32)
    af = af[index]
    pv = pv[index] / 255.0
    pv = pixel_transforms(pv)
    afs.append(af)
    pvs.append(pv)
afs = torch.stack(afs).to(device,dtype)
pvs = torch.stack(pvs).to(device,dtype)
rps = pvs[:,0]
rps = vae_encode(vae,rps)
#%%
with torch.no_grad():
    audio_feature = audio_decoder.audio_encoder(afs) # (N,F+1,W,D)
    mix_pose_pre = audio_decoder.pose_predictor(rps,audio_feature) # N,F+1,C,H,W
pose_pred = vae_decode(vae, mix_pose_pre)
pose_pred = (pose_pred / 2 + 0.5) * 255
pose_pred_vis = torch.clamp(pose_pred, 0, 255).to(torch.uint8).detach().cpu()
pose_pred_vis = rearrange(pose_pred_vis, "b f c h w -> f h (b w) c")
write_video("./test_pose_vis.mp4",pose_pred_vis, fps=8)