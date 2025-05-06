import torch
from torch import nn
from tqdm import tqdm
from diffusers.utils import is_torch_version
from .modules import TransformerBlock,PatchEmbed
from diffusers.models.embeddings import TimestepEmbedding, Timesteps, get_3d_sincos_pos_embed,get_2d_sincos_pos_embed,get_1d_sincos_pos_embed_from_grid
from diffusers.configuration_utils import ConfigMixin, register_to_config
from diffusers.models.modeling_utils import ModelMixin
from typing import Optional,Union,Dict,Any
from einops import rearrange
from transformers import Wav2Vec2Config
from timm.models.layers import Mlp
import einops
# This block only do wav2vec model forwarding and reversed for future development.
import pdb

class Label2MotionDiffusionDecoder(ModelMixin, ConfigMixin):

    @register_to_config
    def __init__(
        self,
        
        label_dim:int = 512,
        motion_dim:int = 512,
        refimg_width: int = 32,
        refimg_height: int = 32,
        refimg_patch_size: int = 2,
        refimg_dim:int = 4,
        num_frames: int = 16,

        num_steps:int = 1000,
        flip_sin_to_cos: bool = True,
        freq_shift: int = 0,
        time_embed_dim: int = 768,
        timestep_activation_fn: str = "silu",

        attention_head_dim : int = 128,
        num_attention_heads: int = 16,
        num_layers: int = 20,
        dropout: float = 0.0,
        norm_elementwise_affine: bool = True,
        norm_eps: float = 1e-5,
        spatial_interpolation_scale: float = 1.0,
        temporal_interpolation_scale: float = 1.0,

        camera_token_num = 8,
        object_token_num = 16,
        camera_channel = 8,
        object_channel = 32,
    ):
        super().__init__()
        
        # camera_token_num = 128,
        # object_token_num = 8,
        # camera_channel = 16,
        # object_channel = 32,

        # camera_token_num = 8,
        # object_token_num = 16,
        # camera_channel = 8,
        # object_channel = 32,

        # 1. Setting
        hidden_dim = num_attention_heads * attention_head_dim
        post_patch_height = refimg_height // refimg_patch_size
        post_patch_width = refimg_width // refimg_patch_size
        self.img_num_patches = post_patch_height * post_patch_width 
        self.num_steps= num_steps 
        self.patch_size = refimg_patch_size
        self.out_channels = motion_dim
        self.num_frames = num_frames

        self.camera_channel = camera_channel
        self.camera_token_num = camera_token_num
        self.object_channel = object_channel
        self.object_token_num = object_token_num
        self.motion_seq_len = camera_token_num + object_token_num

        self.camera_motion_norm = nn.LayerNorm(self.camera_channel)
        self.object_motion_norm = nn.LayerNorm(self.object_channel)

        self.motion_ailgn_c = nn.Parameter(torch.zeros(1, self.object_channel),requires_grad=True)
        self.motion_ailgn_o = nn.Parameter(torch.zeros(1, self.object_channel),requires_grad=True)

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
        ) 
        spatial_pos_embedding = torch.from_numpy(spatial_pos_embedding) # [H/p * W/p, D]
        # pos_embedding = spatial_pos_embedding.unsqueeze(0)# [1,T*H*W, D]
        
        img_pos_embedding = torch.zeros(1,*spatial_pos_embedding.shape, requires_grad=False)
        img_pos_embedding.data.copy_(spatial_pos_embedding)
        self.register_buffer("img_pos_embedding", img_pos_embedding, persistent=False)
        
        # spatial_temporal_embedding = get_3d_sincos_pos_embed(
        #     motion_dim,
        #     (motion_height,motion_width),num_frames-1,
        #     spatial_interpolation_scale=spatial_interpolation_scale,
        #     temporal_interpolation_scale=temporal_interpolation_scale
        # )

        # motion_pos_embedding = torch.zeros(1,self.num_frames-1,motion_dim,motion_height,motion_width,requires_grad=False) 
        # spatial_temporal_embedding = torch.from_numpy(spatial_temporal_embedding)
        # spatial_temporal_embedding = rearrange(spatial_temporal_embedding.unsqueeze(0),"1 t (h w) d -> 1 t d h w", h = motion_height)
        # motion_pos_embedding.data.copy_(spatial_temporal_embedding)
        # self.register_buffer("motion_pos_embedding", motion_pos_embedding,persistent=False) #[h * w * f, D]

        self.time_proj = Timesteps(time_embed_dim, flip_sin_to_cos, freq_shift)
        self.time_embedding = TimestepEmbedding(time_embed_dim, hidden_dim, timestep_activation_fn)


        # 3. transformers blocks
        self.motion_proj_in = Mlp(motion_dim,hidden_dim,hidden_dim)
        self.label_proj_in = Mlp(label_dim,hidden_dim,hidden_dim)
        

        self.motion_transformer_blocks = nn.ModuleList(
            [
                TransformerBlock(
                    head_dim=attention_head_dim,
                    num_heads=num_attention_heads,
                )
                for _ in range(num_layers)
            ]
        )
        self.image_transformer_blocks = nn.ModuleList(
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
        self.camera_proj_out = nn.Linear(self.object_channel,self.camera_channel)
        self.camera_proj_in = nn.Linear(self.camera_channel,self.object_channel)
        # self.object_proj_out = nn.Linear(self.object_channel,self.camera_channel)
        self.object_proj_in = nn.Linear(self.object_channel,self.object_channel)

        self.gradient_checkpointing = False

    def _set_gradient_checkpointing(self, module, value=False):
        self.gradient_checkpointing = value

    def forward(self,
                camera_target_motion:torch.Tensor,
                object_target_motion:torch.Tensor,
                label:torch.Tensor,
                ref_img:torch.Tensor,
                timestep: Union[int, float, torch.LongTensor], # Timesteps should be a 1d-array
                timestep_cond: Optional[torch.Tensor] = None,
                object_source_motion:torch.Tensor = None,
                return_meta_info=True
                ):
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
        n,t,c,h,w = ref_img.shape
        
        ref_img = einops.rearrange(ref_img, 'n t c h w -> (n t) c h w')
        image_hidden_state = self.patch_embed(ref_img)
        image_hidden_state = self.embedding_dropout(image_hidden_state + self.img_pos_embedding)
        n_i,n_t,n_c = image_hidden_state.shape
        
        label_emb = self.label_proj_in(label)
        t_emb = self.time_proj(timestep)
        t_emb = t_emb.to(dtype=dtype)
        emb = self.time_embedding(t_emb, timestep_cond) + label_emb
        emb = emb.repeat(t,1)

        # get gt motion
        camera_target_motion = camera_target_motion.flatten(0,1)

        bc,_,cc = camera_target_motion.shape
        bo,_,oc = object_target_motion.shape
        # pdb.set_trace()
        # camera_motion = self.camera_proj_in(camera_target_motion)
        xc = oc // cc
        # camera_motion = camera_target_motion.repeat(1,1,xc)
        # print("camera_motion.shape",camera_motion.shape)
        # print("camera_target_motion.shape",camera_target_motion.shape)
        # motion_gt = torch.cat([object_target_motion,camera_motion],dim=1)
        # _,motion_seq_len,_ = motion_gt.shape
        # prepare motion
        step = (1 - timestep / self.num_steps)[:,None,None].repeat(t,1,1)
        
        # 是否归一化和映射
        # camera_target_motion = self.camera_motion_norm(camera_target_motion)
        # camera_target_motion = self.camera_proj_in(camera_target_motion)
        # object_target_motion = self.object_motion_norm(object_target_motion)
        # object_target_motion = self.object_proj_in(object_target_motion)
        
        # print("camera_target_motion.mean(dim=-1)",camera_target_motion.mean(dim=-1))
        # print("camera_target_motion.mean()",camera_target_motion.mean())
        # print("camera_target_motion.var(dim=-1)",camera_target_motion.var(dim=-1))
        # print("camera_target_motion.var()",camera_target_motion.var())

        # print("object_target_motion.mean(dim=-1)",object_target_motion.mean(dim=-1))
        # print("object_target_motion.mean()",object_target_motion.mean())
        # print("object_target_motion.var(dim=-1)",object_target_motion.var(dim=-1))
        # print("object_target_motion.var()",object_target_motion.var())

        # vel_gt = motion_gt - noise
        # motion_with_noise = step * motion_gt +  (1 - step) * noise
        # motion_hidden_state = motion_with_noise
        # noise_camera = torch.randn_like(camera_target_motion)
        noise_object = torch.randn_like(object_target_motion)
        # vel_gt_camera = camera_target_motion - noise_camera
        # camera_motion_with_noise = step * camera_target_motion +  (1 - step) * noise_camera
        vel_gt_object = object_target_motion - noise_object
        object_motion_with_noise = step * object_target_motion +  (1 - step) * noise_object

        
        # concat all tokens
        # print("motion_hidden_state.shape",motion_hidden_state.shape)
        # print("image_hidden_state.shape",image_hidden_state.shape)
        # print("motion_gt.shape",motion_gt.shape)
        # camera_motion_noise = self.camera_proj_in(camera_motion_with_noise)
        camera_target_motion = self.camera_proj_in(camera_target_motion)
        
        motion_ailgn_c = self.motion_ailgn_c.unsqueeze(0).repeat(bc,1,1)
        motion_ailgn_o = self.motion_ailgn_o.unsqueeze(0).repeat(bo,1,1)
        # motion_hidden_state = torch.cat([object_motion_with_noise,camera_motion_noise],dim=1)
        # pdb.set_trace()
        if object_source_motion != None:
            motion_hidden_state = torch.cat([object_motion_with_noise,motion_ailgn_o,object_source_motion,motion_ailgn_c,camera_target_motion],dim=1)
        else:
            motion_hidden_state = torch.cat([object_motion_with_noise,motion_ailgn_c,camera_target_motion],dim=1)
        hidden_state = self.motion_proj_in(motion_hidden_state)

        # hidden_state = torch.cat([motion_hidden_state,image_hidden_state],dim=1)

        # for i, m_block, in enumerate(self.motion_transformer_blocks):

        #     hidden_state = m_block(
        #         hidden_state,
        #         emb,
        #     )
        for i, (m_block,i_block) in enumerate(zip(self.motion_transformer_blocks,self.image_transformer_blocks)):

            hidden_state = m_block(
                hidden_state,
                emb,
            )

            i_hidden_state = torch.cat([hidden_state,image_hidden_state],dim=1)
            i_hidden_state = i_block(
                i_hidden_state,
                emb,
            )
            hidden_state = i_hidden_state[:,:n_t]

        hidden_state = self.norm_final(hidden_state)
        hidden_state = self.proj_out(hidden_state)
        # vel_pred = hidden_state[:,:self.motion_seq_len]
        if object_source_motion != None:
            vel_pred_object = hidden_state[:,:self.object_token_num]
            vel_pred_camera = hidden_state[:,2*self.object_token_num+2:]
            # pdb.set_trace()
            # print("vel_pred_camera.shape",vel_pred_camera.shape)
        else:
            vel_pred_object = hidden_state[:,:self.object_token_num]
            vel_pred_camera = hidden_state[:,self.object_token_num+1:]
        vel_pred_camera = self.camera_proj_out(vel_pred_camera)
        # vel_pred = rearrange(vel_pred, "b (f h w) c -> b f c h w",f = self.num_frames-1,h=self.motion_height,w=self.motion_width) 
        # print("vel_pred.shape",vel_pred.shape)
        # object_vel = object_motion_with_noise + (1 - step) * vel_pred_object
        # camera_val = camera_motion_with_noise + (1 - step) * vel_pred_camera

        # object_vel = motion_pred[:,:self.object_token_num]
        # camera_val = motion_pred[:,self.object_token_num:]
        # camera_val = self.camera_proj_out(camera_val)
        # camera_val = camera_val[:,:,:self.camera_channel]

        # camera_val = einops.rearrange(camera_val, "(n t) s c -> n t s c",n=n)
        # print("camera_val.shape",camera_val.shape)
        # if return_meta_info:
        # return {
        #         'camera_motion_with_noise' : camera_motion_with_noise,  # (b f c h w) 
        #         'vel_pred_camera' : vel_pred_camera,                    # (b (f h w) c)
        #         'vel_gt_camera' : vel_gt_camera,                        # (b (f h w) c)
        #         'object_motion_with_noise' : object_motion_with_noise,  # (b f c h w) 
        #         'vel_pred_object' : vel_pred_object,                    # (b (f h w) c)
        #         'vel_gt_object' : vel_gt_object,                        # (b (f h w) c)
        #         }
        return {
                'vel_pred_camera' : vel_pred_camera,                    # (b (f h w) c)
                'object_motion_with_noise' : object_motion_with_noise,  # (b f c h w) 
                'vel_pred_object' : vel_pred_object,                    # (b (f h w) c)
                'vel_gt_object' : vel_gt_object,                        # (b (f h w) c)
                }
        # else:
        #     return motion_with_noise,motion_pred,vel_pred,vel_gt
        
    @torch.no_grad()
    def sample(self,
               label:torch.Tensor,
               ref_img:torch.Tensor,
               sample_steps:int=10,
               timestep_cond=None):
        n = label.shape[0]
        device = ref_img.device
        dtype = ref_img.dtype
        n,t,c,h,w = ref_img.shape
        ref_img = einops.rearrange(ref_img, 'n t c h w -> (n t) c h w')
        image_hidden_state = self.patch_embed(ref_img)
        image_hidden_state = self.embedding_dropout(image_hidden_state + self.img_pos_embedding)
        label = self.label_proj_in(label)
        timestep = torch.ones(n).to(device) * self.num_steps
        
        # 初始化纯噪声形式的motion
        # motion_with_noise = torch.randn([n*self.num_frames,self.camera_token_num+self.object_token_num,self.out_channels]).to(device)
        camera_motion_with_noise = torch.randn([n*self.num_frames,self.camera_token_num,self.camera_channel]).to(device)
        object_motion_with_noise = torch.randn([n*self.num_frames,self.object_token_num,self.object_channel]).to(device)

        dt = 1. / sample_steps
        for i in tqdm(range(sample_steps)):
            t_emb = self.time_proj(timestep)
            t_emb = t_emb.to(dtype=dtype)
            emb = self.time_embedding(t_emb, timestep_cond) + label
            emb = emb.repeat(t,1)
            # prepare motion
            camera_motion_noise = self.camera_proj_in(camera_motion_with_noise)
            motion_hidden_state = torch.cat([object_motion_with_noise,camera_motion_noise],dim=1)
            hidden_state = self.motion_proj_in(motion_hidden_state)
            # concat all tokens
            # print("motion_hidden_state.shape",motion_hidden_state.shape)
            # print("image_hidden_state.shape",image_hidden_state.shape)
            # hidden_state = torch.cat([motion_hidden_state,image_hidden_state],dim=1)
            # print("emb.shape",emb.shape)
            # print("timestep.shape",timestep.shape)
            # print("label",label.shape)
            for j, block in enumerate(self.transformer_blocks):
                hidden_state = block(
                    hidden_state,
                    emb,
                )
            hidden_state = self.norm_final(hidden_state)
            hidden_state = self.proj_out(hidden_state)
            # vel_pred = hidden_state[:,:self.motion_seq_len]
            vel_pred_object = hidden_state[:,:self.object_token_num]
            vel_pred_camera = hidden_state[:,self.object_token_num:]
            vel_pred_camera = self.camera_proj_out(vel_pred_camera)
            # vel_pred = rearrange(vel_pred, "b (f h w) c -> b f c h w",f = self.num_frames-1,h=self.motion_height,w=self.motion_width) 
            camera_motion_with_noise = camera_motion_with_noise + dt * vel_pred_camera
            object_motion_with_noise = object_motion_with_noise + dt * vel_pred_object
            timestep = timestep - dt * self.num_steps

        assert torch.allclose(timestep,torch.zeros_like(timestep)), "denosing error"


        # object_vel = motion_with_noise[:,:self.object_token_num]
        # camera_val = motion_with_noise[:,self.object_token_num:]
        # camera_val = self.camera_proj_out(camera_val)
        object_vel = object_motion_with_noise
        camera_val = camera_motion_with_noise
        camera_val = einops.rearrange(camera_val, "(n t) s c -> n t s c",n=n)
        # print("object_vel.shape",object_vel.shape)
        # print("camera_val.shape",camera_val.shape)

        return object_vel,camera_val
    

    @torch.no_grad()
    def sample_with_camera(self,
               label:torch.Tensor,
               ref_img:torch.Tensor,
               camera_target_motion:torch.Tensor,
               object_source_motion:torch.Tensor = None,
               sample_steps:int=10,
               timestep_cond=None):
        n = label.shape[0]
        device = ref_img.device
        dtype = ref_img.dtype
        n,t,c,h,w = ref_img.shape
        ref_img = einops.rearrange(ref_img, 'n t c h w -> (n t) c h w')
        image_hidden_state = self.patch_embed(ref_img)
        image_hidden_state = self.embedding_dropout(image_hidden_state + self.img_pos_embedding)
        n_i,n_t,n_c = image_hidden_state.shape

        label = self.label_proj_in(label)
        timestep = torch.ones(n).to(device) * self.num_steps
        
        # 初始化纯噪声形式的motion
        # motion_with_noise = torch.randn([n*self.num_frames,self.camera_token_num+self.object_token_num,self.out_channels]).to(device)
        # camera_motion_with_noise = torch.randn([n*self.num_frames,self.camera_token_num,self.camera_channel]).to(device)
        camera_target_motion = camera_target_motion.flatten(0,1)
        # camera_target_motion = self.camera_motion_norm(camera_target_motion)
        object_motion_with_noise = torch.randn([n*self.num_frames,self.object_token_num,self.object_channel]).to(device)
        camera_motion = self.camera_proj_in(camera_target_motion)
        dt = 1. / sample_steps
        for i in tqdm(range(sample_steps)):
            t_emb = self.time_proj(timestep)
            t_emb = t_emb.to(dtype=dtype)
            emb = self.time_embedding(t_emb, timestep_cond) + label
            emb = emb.repeat(t,1)
            # prepare motion
            # camera_motion_noise = self.camera_proj_in(camera_motion_with_noise)
            motion_ailgn_c = self.motion_ailgn_c.unsqueeze(0).repeat(n*self.num_frames,1,1)
            motion_ailgn_o = self.motion_ailgn_o.unsqueeze(0).repeat(n*self.num_frames,1,1)
            # pdb.set_trace()
            if object_source_motion != None:
                motion_hidden_state = torch.cat([object_motion_with_noise,motion_ailgn_o,object_source_motion,motion_ailgn_c,camera_motion],dim=1)
            else:
                motion_hidden_state = torch.cat([object_motion_with_noise,motion_ailgn_c,camera_motion],dim=1)
            hidden_state = self.motion_proj_in(motion_hidden_state)
            # concat all tokens
            # print("motion_hidden_state.shape",motion_hidden_state.shape)
            # print("image_hidden_state.shape",image_hidden_state.shape)
            # hidden_state = torch.cat([motion_hidden_state,image_hidden_state],dim=1)
            # print("emb.shape",emb.shape)
            # print("timestep.shape",timestep.shape)
            # print("label",label.shape)

            # for i, m_block in enumerate(self.motion_transformer_blocks):
            #     hidden_state = m_block(
            #         hidden_state,
            #         emb,
            #     )

            for i, (m_block,i_block) in enumerate(zip(self.motion_transformer_blocks,self.image_transformer_blocks)):
                hidden_state = m_block(
                    hidden_state,
                    emb,
                )
                # print(motion_hidden_state.shape)
                i_hidden_state = torch.cat([hidden_state,image_hidden_state],dim=1)
                i_hidden_state = i_block(
                    i_hidden_state,
                    emb,
                )
                hidden_state = i_hidden_state[:,:n_t]

            hidden_state = self.norm_final(hidden_state)
            hidden_state = self.proj_out(hidden_state)
            # vel_pred = hidden_state[:,:self.motion_seq_len]
            # vel_pred_object = hidden_state[:,:self.object_token_num]
            # camera_motion = hidden_state[:,self.object_token_num:]
            if object_source_motion != None:
                vel_pred_object = hidden_state[:,:self.object_token_num]
                camera_motion = hidden_state[:,2*self.object_token_num+2:]
                # print("camera_motion.shape",camera_motion.shape)
            else:
                vel_pred_object = hidden_state[:,:self.object_token_num]
                camera_motion = hidden_state[:,self.object_token_num+1:]
            # vel_pred_camera = hidden_state[:,self.object_token_num:]
            # vel_pred_camera = self.camera_proj_out(vel_pred_camera)
            # vel_pred = rearrange(vel_pred, "b (f h w) c -> b f c h w",f = self.num_frames-1,h=self.motion_height,w=self.motion_width) 
            # camera_motion_with_noise = camera_motion_with_noise + dt * vel_pred_camera
            object_motion_with_noise = object_motion_with_noise + dt * vel_pred_object
            timestep = timestep - dt * self.num_steps

        assert torch.allclose(timestep,torch.zeros_like(timestep)), "denosing error"


        # object_vel = motion_with_noise[:,:self.object_token_num]
        # camera_val = motion_with_noise[:,self.object_token_num:]
        # camera_val = self.camera_proj_out(camera_val)
        object_vel = object_motion_with_noise
        # camera_val = camera_motion_with_noise
        camera_val = camera_target_motion
        camera_val = einops.rearrange(camera_val, "(n t) s c -> n t s c",n=n)
        # print("object_vel.shape",object_vel.shape)
        # print("camera_val.shape",camera_val.shape)

        return object_vel,camera_val