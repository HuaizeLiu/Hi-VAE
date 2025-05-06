from model.transformer import AMDDiffusionTransformerModel,MotionEncoderLearnTokenTransformer,MotionTransformer
from model.model_AMD import AMDModel,AMD_models
from model.model_A2M import A2MModel_CrossAtten_Audio_PosePre
from model.utils import print_param_num
import torch
import pickle
from model.modules import A2MCrossAttnBlock,A2MMotionSelfAttnBlock

# device = torch.device('cuda:3')
# model = A2MModel_CrossAtten_Audio_Pose(motion_frames=16,motion_num_token=4,diffusion_num_layers=8).to(device)
# print_param_num(model)



device = torch.device('cuda:3')
model = A2MModel_CrossAtten_Audio_PosePre(motion_num_token=4,motion_frames=16)

model.to(device)

print_param_num(model)

i = 0
for name,parm in model.named_parameters():
    print(i,name,parm.shape)
    i +=1

# ref_motion = torch.randn((4,4,128)).to(device)
# gt_motion = torch.randn((4,16,4,128)).to(device)
# audio =  torch.randn((4,16,50,384)).to(device)
# ref_audio = torch.randn((4,50,384)).to(device)
# pose = torch.randn((4,16,4,32,32)).to(device)
# ref_pose = torch.randn((4,4,32,32)).to(device)
# loss_dict = model(gt_motion,ref_motion,audio,ref_audio,pose,ref_pose)
# print(loss_dict)

# sample = model.sample(ref_motion,audio,ref_audio,pose,ref_pose)
# print(sample.shape)