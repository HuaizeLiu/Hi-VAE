import torch
from torch import nn
from torch.nn import functional as F
from torch.utils.data import DataLoader

from model.utils import save_videos_grid
from model.modules import DuoFrameDownEncoder,Upsampler
from model.model_AE import AMDModel1
from model.utils import save_cfg, vae_encode, vae_decode
from omegaconf import OmegaConf
from diffusers import AutoencoderKL
from model.utils import print_param_num
import einops


# 测试一下在原始的数据集，相邻帧的loss
def test_origin_loss():
    data_path = 'demo/train'
    dataset = CelebvText(video_dir=data_path,sample_size=(256,256),sample_stride=4,sample_n_frames=32)
    dataloader = DataLoader(dataset, batch_size=2, shuffle=False, collate_fn=dataset.collate_fn)

    vae_version = '/mnt/lpai-dione/ssai/cvg/team/didonglin/zqy/Huggingface-Model/sd-vae-ft-mse'
    vae = AutoencoderKL.from_pretrained(vae_version, subfolder="vae").requires_grad_(False)

    for i, batch in enumerate(dataloader):
        x = batch['videos'] # (b,t,c,h,w)
        z = vae_encode(vae,x)
        loss = F.mse_loss(z[:,1:,:], z[:,1:,:],reduction='mean')
        print(f'相邻帧loss={loss}')

        return

def test_vae():
    vae_version = '/mnt/lpai-dione/ssai/cvg/team/didonglin/zqy/Huggingface-Model/sd-vae-ft-mse'
    vae = AutoencoderKL.from_pretrained(vae_version, subfolder="vae").requires_grad_(False)
    num_params = sum(p.numel() for p in vae.parameters())
    print(f'#### #### model 模型参数数量:{num_params / 1_000_000:.2f}M')


# Test 把video读取到dataloader，再保存回视频
def test_dataload_datasave():
    
    data_path = 'demo/video'
    dataset = CelebvText(video_dir=data_path,sample_size=(256,256),sample_stride=4,sample_n_frames=16)
    dataloader = DataLoader(dataset, batch_size=2, shuffle=True, collate_fn=dataset.collate_fn)
    
    for i, batch in enumerate(dataloader):
        video_data = batch['videos'] # (b,t,c,h,w)
        print(batch['name'], batch['videos'].shape)
        break
    
    # transform video data shape
    video_data = video_data.permute(0,2,1,3,4).contiguous()
    save_videos_grid(video_data, 'demo/test/test_save.mp4',rescale=True)

# Test DuoFrameEncoder
def TestDownSample():

    block_out_channels_down = (64,128,256,256)
    block_out_channels_up = list(reversed(block_out_channels_down))
    encoder = DuoFrameDownEncoder(in_channel=4,block_out_channels=block_out_channels_down)
    decoder = Upsampler(in_channel=block_out_channels_up[0],out_channel=4,block_out_channels=block_out_channels_up)
    
    num_params = sum(p.numel() for p in encoder.parameters())
    print(f'#### #### encoder 模型参数数量:{num_params / 1_000_000:.2f}M')
    num_params = sum(p.numel() for p in decoder.parameters())
    print(f'#### #### decoder 模型参数数量:{num_params / 1_000_000:.2f}M')

    x = torch.randn(2,4,64,64)
    x = encoder(x)
    print('After Encoder: X shape',x.shape)

    x = decoder(x)
    print('After Decoder: X shape',x.shape)


# Test DuoFrameEncoder
def TestModel1():
    inchannel = 4
    upsampler_outchannel = 4
    block_out_channels_down  = [128,256,512,512]

    model = AMDModel1(inchannel,upsampler_outchannel,block_out_channels_down)
    num_params = sum(p.numel() for p in model.parameters())
    print(f'#### #### model 模型参数数量:{num_params / 1_000_000:.2f}M')

    # x = torch.randn(2,32,4,32,32)
    # pre = model(x)
    # loss = model.forward_loss(pre,x)

    # print(loss)
    
def select_pairs(K):
    import numpy as np
    # 获取序列长度
    n = 16
    
    if K > n - 1:
        raise ValueError("K is too large for the given sequence length.")
    
    # 随机选择K个不同的索引作为左侧点i
    left_indices = torch.randint(0, n-2, (K,), dtype=torch.long)
    print('left_indices:',left_indices)
    
    # 对于每个左侧点i，随机选择一个右侧点j
    max_indices = torch.full((K,), n, dtype=torch.long)
    print('max_indices',max_indices)
    
    right_indices_np = np.random.randint(left_indices.numpy() + 1, max_indices.numpy(), size=(K,))
    right_indices = torch.from_numpy(right_indices_np)
    # right_indices = torch.randint(left_indices + 1, max_indices, (K,), dtype=torch.long)
    print('right_indices',right_indices)
    # 根据索引对提取值，并将它们转换成形状为[K, 2]的张量
    pairs_tensor = torch.stack([left_indices,right_indices], dim=1)
    print(pairs_tensor)
    return pairs_tensor

def test_AMD_model():
    from model import AMD_models
    model = AMD_models['AMD_S']()
    
    encoder = model.dfd_encoder
    motion_transformer = model.motion_transformer
    diffusion_transformer = model.diffusion_transformer

    N,F,C,H,W = 2,16,4,64,64
    video = torch.randn((N,F,C,H,W))
    print(f'N = {N};F = {F};C = {C};H = {H};W = {W}')
    
    print('************** ENCODER **************')
    encoder_in = model.prepare_encoder_input(video)
    encoder_out = encoder(encoder_in)
    print('video shape',video.shape)
    print('encoder_in_video shape:',encoder_in.shape)
    print('-'*50)
    print('encoder_out_motion shape:',encoder_out.shape)

    print('************** MOTION TRANSFORMER **************')
    import einops
    motion_transformer_in = einops.rearrange(encoder_out,'(b t) c h w -> b t c h w',b=N)
    print('motion_transformer_in_motion shape:',motion_transformer_in.shape)
    motion = motion_transformer(motion_transformer_in)
    print('-'*50)
    print('motion shape:',motion.shape)

    print('************** DIFFUSION TRANSFORMER **************')
    # video_start,video_end,motion = self.prepare_diffusion_transformer_data_input(video,motion,self.groups_per_batch) # (n*k,C,H,W),(n*k,C,H,W),(n*k,l-1,c,h,w)
    # time_step = self.prepare_timestep(video_start.shape[0],device) # (b,)
    # z_t,gt = self.prepare_train_target(video_start,video_end,time_step) # (b,C,H,W),(b,C,H,W)
    # pre = self.diffusion_transformer(motion,z_t,time_step) # (b,C,H,W)
    video_start,video_end,dit_motion_in = model.prepare_diffusion_transformer_data_input(video,motion,2) # (n*k,C,H,W),(n*k,C,H,W),(n*k,l-1,c,h,w)
    print('video_start shape',video_start.shape)
    print('video_end shape',video_end.shape)
    print('dit_motion_in shape',dit_motion_in.shape)
    time_step = model.prepare_timestep(video_start.shape[0],video_start.device) # (b,)
    print('time_step shape',time_step.shape)

    from model import RectifiedFlow
    scheduler = RectifiedFlow(num_steps=1000)
    time_step = torch.randint(0,1000,(video_start.shape[0],),device=video_start.device) 
    z_t,gt = model.prepare_train_target(scheduler,video_start,video_end,time_step) # (b,C,H,W),(b,C,H,W)
    print('zt shape',z_t.shape)
    print('gt shape',gt.shape)

    pre = model.diffusion_transformer(dit_motion_in,z_t,time_step) # (b,C,H,W)
    print('-'*50)
    print('pre shape',pre.shape)

def test_MAE():
    from model import mae_vit_base_patch8,mae_vit_large_patch8
    from model.utils import print_param_num
    model = mae_vit_large_patch8()
    
    img = torch.randn((2,4,256,256))
    loss, pred, mask = model(img)
    print('loss:',loss)
    print('pred:',pred.shape)  
    print('mask:',mask.shape)

    print_param_num(model)



if __name__ == '__main__':
    
    # test_dataload_datasave()

    # TestDownSample()

    # test_origin_loss()

    # test_vae()

    # TestModel1()


    # test_AMD_model()

    # test_MAE()
    from dataset.dataset import A2MVideoAudio
    from torch.utils.data import DataLoader
    datadir = '/mnt/pfs-gv8sxa/tts/dhg/zqy/code/AMD2/dataset/path/a2m/eval.pkl'
    dataset = A2MVideoAudio(datadir,sample_stride=60,sample_n_frames=8)

    # dataloader = DataLoader(dataset, batch_size=4, num_workers=4,shuffle=True, collate_fn=dataset.collate_fn,pin_memory=True)

    # for data in dataloader:
    #     print(data['meta'])
    #     print(data['ref_video'].shape)
    #     print(data['gt_video'].shape)
    #     print(data['ref_audio'].shape)
    #     print(data['gt_audio'].shape)
    #     print(data['mask'].shape)
    #     print(data['mask'])
    #     break

    data = dataset[1]
    print(data['meta'])
    print(data['ref_video'].shape)
    print(data['gt_video'].shape)
    print(data['ref_audio'].shape)
    print(data['gt_audio'].shape)
    print(data['mask'].shape)
    print(data['mask'])

    



