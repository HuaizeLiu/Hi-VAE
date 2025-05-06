import torch
from torch import nn
from .base_model import BaseDiffusionModel,BaseDecoder
from .modules import AudioProjModel
from diffusers.configuration_utils import ConfigMixin, register_to_config
from diffusers.models.modeling_utils import ModelMixin

class Audio2MotionDiffusionDecoder(BaseDiffusionModel):

    def __init__(
        self,
        **config 
    ):
        super().__init__(**config)

        # TODO
        self.extra_encoder = 1
        self.model = 1

        # temporal_embedding = get_1d_sincos_pos_embed_from_grid(audio_dim,torch.arange(num_frames))
        # audio_pos_embedding = torch.zeros(1,*temporal_embedding.shape,requires_grad=False)
        # audio_pos_embedding.data.copy_(torch.from_numpy(temporal_embedding))
        # self.register_buffer("audio_pos_embedding",audio_pos_embedding,persistent=False)
        # hidden_dim = attention_head_dim * num_attention_heads
        # self.audio_proj_in = Mlp(audio_dim,hidden_dim,hidden_dim)
        self.audio_encoder = AudioProjModel(**config)
        self.cfg = config
    def cond_injection(self,
                       extra:torch.Tensor,
                       refimg:torch.Tensor,
                       timestep:torch.Tensor
                       ):
        """
        ref_img (torch.Tensor): (N,C,H,W)
        extra (torch.Tensor): (N,F,M,D)
        timestep (torch.Tensor): (N,) <= num_steps
        """

        return self.audio_encoder(extra),refimg,timestep

class Audio2VideoPipe(nn.Module):
    def __init__(self,
                  *args, 
                  **kwargs):
        super().__init__()
    def generate(self,):
        pass

if __name__ == "__main__":
    def print_param_num(model):
        total_params = sum(p.numel() for p in model.parameters())
        train_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
        freeze_params = sum(p.numel() for p in model.parameters() if p.requires_grad is False)
        print(f'####  total :{total_params / 1_000_000:.2f}M')
        print(f'####  train:{train_params / 1_000_000:.2f}M')
        print(f'####  freeze:{freeze_params / 1_000_000:.2f}M') 
    