import torch
from tqdm import tqdm
from torch import nn
from diffusers.utils import is_torch_version
from .modules import TransformerBlock,PatchEmbed,Wav2Vec2ModelLerp
from diffusers.models.embeddings import TimestepEmbedding, Timesteps, get_3d_sincos_pos_embed,get_2d_sincos_pos_embed,get_1d_sincos_pos_embed_from_grid
from diffusers.configuration_utils import ConfigMixin, register_to_config
from diffusers.models.modeling_utils import ModelMixin
from typing import Optional,Union,Dict,Any
from einops import rearrange
from transformers import Wav2Vec2Config
from timm.layers import Mlp
# This block only do wav2vec model forwarding and reversed for future development.
class AudioEncoder(nn.Module):
    def __init__(
        self,
        model_path:str,
        # from_pretrained:bool=False,
        only_last_features:bool=True
    ):  
        super().__init__()

        self._only_last_features = only_last_features

        # self.audio_encoder_config = Wav2Vec2Config.from_pretrained(model_path, local_files_only=True)
        # if from_pretrained:
        self.audio_encoder = Wav2Vec2ModelLerp.from_pretrained(model_path, local_files_only=True)
        # else:
        # self.audio_encoder = Wav2Vec2ModelLerp(self.audio_encoder_config)
        self.audio_encoder.feature_extractor._freeze_parameters()


        # hidden_size = self.audio_encoder_config.hidden_size

        # self.in_fn = nn.Linear(hidden_size, latent_dim)
       
        # self.out_fn = nn.Linear(latent_dim, out_dim)
        # nn.init.constant_(self.out_fn.weight, 0)
        # nn.init.constant_(self.out_fn.bias, 0)


    def forward(self, input_value, seq_len):
        embeddings = self.audio_encoder(input_value, seq_len=seq_len, output_hidden_states=True)

        if self._only_last_features:
            hidden_states = embeddings.last_hidden_state
        else:
            hidden_states = sum(embeddings.hidden_states) / len(embeddings.hidden_states)

        # layer_in = self.in_fn(hidden_states)
        # out = self.out_fn(layer_in)

        return hidden_states
class Audio2MotionDiffusionDecoder(ModelMixin, ConfigMixin):

    @register_to_config
    def __init__(
        self,
        
        audio_dim:int = 512,
        motion_dim:int = 512,
        motion_width:int = 4,
        motion_height:int = 4,
        refimg_width: int = 32,
        refimg_height: int = 32,
        refimg_patch_size: int = 4,
        refimg_dim:int = 4,
        num_frames: int = 16,

        num_steps:int = 1000,
        flip_sin_to_cos: bool = True,
        freq_shift: int = 0,
        time_embed_dim: int = 768,
        timestep_activation_fn: str = "silu",

        attention_head_dim : int = 64,
        num_attention_heads: int = 8,
        num_layers: int = 8,
        dropout: float = 0.0,
        norm_elementwise_affine: bool = True,
        norm_eps: float = 1e-5,
        spatial_interpolation_scale: float = 1.0,
        temporal_interpolation_scale: float = 1.0,
    ):
        super().__init__()
        
        # 1. Setting
        hidden_dim = num_attention_heads * attention_head_dim
        post_patch_height = refimg_height // refimg_patch_size
        post_patch_width = refimg_width // refimg_patch_size
        self.img_num_patches = post_patch_height * post_patch_width 
        self.num_steps= num_steps 
        self.patch_size = refimg_patch_size
        self.out_channels = motion_dim
        self.motion_height = motion_height 
        self.motion_width = motion_width
        self.num_frames = num_frames
        self.motion_seq_len = motion_height * motion_width * (num_frames - 1)
        #Patch embedding
        self.patch_embed = PatchEmbed(patch_size=refimg_patch_size,
                                      in_channels=refimg_dim,
                                      embed_dim=hidden_dim,
                                      bias=True)
        self.embedding_dropout = nn.Dropout(dropout)

        # embeddings
        spatial_pos_embedding = get_2d_sincos_pos_embed(  
            hidden_dim,
            (post_patch_width, post_patch_height),
            cls_token=False,
            interpolation_scale=spatial_interpolation_scale,
        ) 
        spatial_pos_embedding = torch.from_numpy(spatial_pos_embedding) # [H/p * W/p, D]
        # pos_embedding = spatial_pos_embedding.unsqueeze(0)# [1,T*H*W, D]
        
        img_pos_embedding = torch.zeros(1,*spatial_pos_embedding.shape, requires_grad=False)
        img_pos_embedding.data.copy_(spatial_pos_embedding)
        self.register_buffer("img_pos_embedding", img_pos_embedding, persistent=False)
        
        spatial_temporal_embedding = get_3d_sincos_pos_embed(
            motion_dim,
            (motion_height,motion_width),num_frames-1,
            spatial_interpolation_scale=spatial_interpolation_scale,
            temporal_interpolation_scale=temporal_interpolation_scale
        )

        motion_pos_embedding = torch.zeros(1,self.num_frames-1,motion_dim,motion_height,motion_width,requires_grad=False) 
        spatial_temporal_embedding = torch.from_numpy(spatial_temporal_embedding)
        spatial_temporal_embedding = rearrange(spatial_temporal_embedding.unsqueeze(0),"1 t (h w) d -> 1 t d h w", h = motion_height)
        motion_pos_embedding.data.copy_(spatial_temporal_embedding)
        self.register_buffer("motion_pos_embedding", motion_pos_embedding,persistent=False) #[h * w * f, D]

        temporal_embedding = get_1d_sincos_pos_embed_from_grid(
            audio_dim,torch.arange(num_frames))
        audio_pos_embedding = torch.zeros(1,*temporal_embedding.shape,requires_grad=False)
        audio_pos_embedding.data.copy_(torch.from_numpy(temporal_embedding))
        self.register_buffer("audio_pos_embedding",audio_pos_embedding,persistent=False)
        
        self.time_proj = Timesteps(time_embed_dim, flip_sin_to_cos, freq_shift)
        self.time_embedding = TimestepEmbedding(time_embed_dim, hidden_dim, timestep_activation_fn)


        # 3. transformers blocks
        self.motion_proj_in = Mlp(motion_dim,hidden_dim,hidden_dim)
        self.audio_proj_in = Mlp(audio_dim,hidden_dim,hidden_dim)
        self.transformer_blocks = nn.ModuleList(
            [
                TransformerBlock(
                    head_dim=attention_head_dim,
                    num_heads=num_attention_heads,
                )
                for _ in range(num_layers)
            ]
        )

        self.norm_final = nn.LayerNorm(hidden_dim, eps=norm_eps, elementwise_affine=norm_elementwise_affine)

        self.proj_out = nn.Linear(hidden_dim,motion_dim)

        self.gradient_checkpointing = False

    def _set_gradient_checkpointing(self, module, value=False):
        self.gradient_checkpointing = value

    def forward(self,
                motion_gt:torch.Tensor,
                audio:torch.Tensor,
                ref_img:torch.Tensor,
                timestep: Union[int, float, torch.LongTensor], # Timesteps should be a 1d-array
                timestep_cond: Optional[torch.Tensor] = None,
                return_meta_info=False):
        """
        Args:
            motion_gt (torch.Tensor): (B,F,C,h,w)
            audio (torch.Tensor): (B,F,D)
            ref_img (torch.Tensor): (B,C,H,W)
            timestep (torch.Tensor): (B) <= num_steps
        """
        # embeddings include: 
        # reference image embedding
        # audio embedding
        # timestep embedding
        device = ref_img.device
        dtype = ref_img.dtype
        image_hidden_state = self.patch_embed(ref_img)
        image_hidden_state = self.embedding_dropout(image_hidden_state + self.img_pos_embedding)
        audio = self.audio_proj_in(audio + self.audio_pos_embedding)
        t_emb = self.time_proj(timestep)
        t_emb = t_emb.to(dtype=dtype)
        emb = self.time_embedding(t_emb, timestep_cond)
        # prepare motion
        t = (1 - timestep / self.num_steps)[:,None,None,None,None]
        noise = torch.randn_like(motion_gt)
        vel_gt = motion_gt - noise
        motion_with_noise = t * motion_gt +  (1 - t) * noise
        motion_hidden_state = motion_with_noise + self.motion_pos_embedding
        motion_hidden_state = rearrange(motion_hidden_state,"b f c h w -> b (f h w) c")
        motion_hidden_state = self.motion_proj_in(motion_hidden_state)
        # concat all tokens
        hidden_state = torch.cat([motion_hidden_state,audio,image_hidden_state],dim=1)

        for i, block in enumerate(self.transformer_blocks):
            if self.training and self.gradient_checkpointing:

                def create_custom_forward(module):
                    def custom_forward(*inputs):
                        return module(*inputs)

                    return custom_forward

                ckpt_kwargs: Dict[str, Any] = {"use_reentrant": False} if is_torch_version(">=", "1.11.0") else {}
                hidden_state = torch.utils.checkpoint.checkpoint(
                    create_custom_forward(block),
                    hidden_state,
                    emb,
                    **ckpt_kwargs,
                )
            else:
                hidden_state = block(
                    hidden_state,
                    emb,
                )
        hidden_state = self.norm_final(hidden_state)
        hidden_state = self.proj_out(hidden_state)
        vel_pred = hidden_state[:,:self.motion_seq_len]
        vel_pred = rearrange(vel_pred, "b (f h w) c -> b f c h w",f = self.num_frames-1,h=self.motion_height,w=self.motion_width) 

        motion_pred = motion_with_noise + (1 - t) * vel_pred
        if return_meta_info:
            return {
                    'motion_with_noise' : motion_with_noise,  # (b f c h w) 
                    'motion_pred' : motion_pred,              # (b f c h w)
                    'vel_pred' : vel_pred,                    # (b (f h w) c)
                    'vel_gt' : vel_gt,                        # (b (f h w) c)
                    }
        else:
            return motion_with_noise,motion_pred,vel_pred,vel_gt 
    @torch.no_grad()
    def sample(self,
               audio:torch.Tensor,
               ref_img:torch.Tensor,
               sample_steps:int=10,
               timestep_cond=None):
        n = audio.shape[0]
        device = ref_img.device
        dtype = ref_img.dtype
        image_hidden_state = self.patch_embed(ref_img)
        image_hidden_state = self.embedding_dropout(image_hidden_state + self.img_pos_embedding)
        audio = self.audio_proj_in(audio + self.audio_pos_embedding)
        timestep = torch.ones(n).to(device) * self.num_steps
        motion_with_noise = torch.randn(n,self.num_frames-1,self.out_channels,self.motion_height,self.motion_width).to(device)
        dt = 1. / sample_steps
        for i in tqdm(range(sample_steps)):
            t_emb = self.time_proj(timestep)
            t_emb = t_emb.to(dtype=dtype)
            emb = self.time_embedding(t_emb, timestep_cond)
            # prepare motion
            motion_hidden_state = motion_with_noise + self.motion_pos_embedding
            motion_hidden_state = rearrange(motion_hidden_state,"b f c h w -> b (f h w) c")
            motion_hidden_state = self.motion_proj_in(motion_hidden_state)
            # concat all tokens
            hidden_state = torch.cat([motion_hidden_state,audio,image_hidden_state],dim=1)

            for j, block in enumerate(self.transformer_blocks):
                hidden_state = block( hidden_state, emb)
            hidden_state = self.norm_final(hidden_state)
            hidden_state = self.proj_out(hidden_state)
            vel_pred = hidden_state[:,:self.motion_seq_len]
            vel_pred = rearrange(vel_pred, "b (f h w) c -> b f c h w",f = self.num_frames-1,h=self.motion_height,w=self.motion_width) 
            motion_with_noise = motion_with_noise + dt * vel_pred
            timestep = timestep - dt * self.num_steps
        return motion_with_noise
class Audio2VideoPipe(nn.Module):
    def __init__(self,
                  *args, 
                  **kwargs):
        super().__init__()
    def generate(self,):
        pass