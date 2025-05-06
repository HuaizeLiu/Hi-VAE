import torch
from torch import nn
import torch.nn.functional as F
from .lpips import LPIPS
from einops import rearrange,repeat
from typing import Optional
import einops


def l1(pre, gt):
    return F.l1_loss(pre, gt)

def l2(pre, gt):
    return F.mse_loss(pre, gt,reduction='mean')

def vae_decode(vae,latents):
    latents_type = None
    
    if len(latents.shape) == 5:
        N,T,C,H,W = latents.shape
        latents_type = 'video'
        latents = einops.rearrange(latents,'n t c h w -> (n t) c h w')
    else:
        N,C,H,W = latents.shape
        latents_type = 'image'

    latents = 1 / 0.18215 * latents
    latents = vae.decode(latents).sample # (nt)chw
        
    if latents_type == 'video':
        latents = einops.rearrange(latents,'(n t) c h w -> n t c h w',n=N,t=T)

    return latents


# class LPIPSWithDiscriminator3D(nn.Module):
#     def __init__(self,
#         # --- Discriminator Loss ---
#         disc_start,
#         latent_width: int = 32,
#         latent_height: int = 32,
#         latten_chs:int = 4,
#         disc_factor=1.0,
#         disc_weight=0.2,
#         disc_loss="hinge",
#         loss_type: str = "l2",
#         num_frames: int = 15,
#         # --- Perception Loss ---
#         perceptual_weight=0.5,
#         # logvar
#         logvar_init=0.0,
#         use_sigmoid=False,
#         learn_logvar=False,
#         disc_rec=False
#     ):

#         super().__init__()
#         assert disc_loss in ["hinge", "vanilla","adv"]
#         self.logvar = nn.Parameter(torch.ones(size=()) * logvar_init,requires_grad=learn_logvar)
#         self.perceptual_loss = LPIPS().eval()
#         self.num_frames = num_frames
#         self.perceptual_weight = perceptual_weight
#         self.discriminator = NLayerDiscriminator3D(
#             input_nc = latten_chs,
#         ).apply(weights_init)
#         self.discriminator_weight = disc_weight
#         self.disc_factor = disc_factor
#         self.discriminator_iter_start = disc_start
#         self.disc_loss = adv_loss if disc_loss == "adv" else hinge_d_loss if disc_loss == "hinge" else vanilla_d_loss
#         self.loss_func = l1 if loss_type == "l1" else l2
#         self.disc_rec = disc_rec
        
#     def calculate_adaptive_weight(self, nll_loss, g_loss, last_layer=None):
#         layer = last_layer if last_layer is not None else self.last_layer[0]
#         nll_grads = torch.autograd.grad(nll_loss, layer, retain_graph=True)[0]
#         g_grads = torch.autograd.grad(g_loss, layer, retain_graph=True)[0]
#         d_weight = torch.norm(nll_grads) / (torch.norm(g_grads) + 1e-4)
#         d_weight = torch.clamp(d_weight, 0.0, 1e4).detach()
#         d_weight = d_weight * self.discriminator_weight
#         return d_weight
#     def loss(self,
#              reconstructed:torch.Tensor,
#              inputs:torch.Tensor,
#              zj_pred:torch.Tensor,
#              zj_gt:torch.Tensor,
#              v_pred:torch.Tensor,
#              v_gt:torch.Tensor,
#              opt:int,
#              global_step:int,
#              last_layer:Optional[torch.Tensor] = None,
#              weights:Optional[float] = None,
#              adaptive_g_loss:bool=False,
#              zi:Optional[torch.Tensor]=None
#              ):
#         """ calculate loss(reconstruction loss, lpips loss, adversarial loss), 
#             perception loss = LPIPS(reconstructied, inputs)
#             adv_loss = disc(concat(zj,zt),t) = disc(concat(zj_pred,zt),t) or disc(concat(zj_gt,zt),t)
#         Args:
#             reconstructed (torch.Tensor): xj_hat = vae.decode(zj), reconstructed video
#             inputs (torch.Tensor): xj, shape = (N,C,H,W),video ground truth
#             zj_pred (torch.Tensor): zj, shape = (N,c,h,w)
#             zj_gt (torch.Tensor): ground truth of zj
#             v_pred (torch.Tensor): predicted velocity
#             v_gt (torch.Tensor): ground truth velocity
#             opt (int): 1 -> training discriminator, 0 for training generator
#             global_step (int): currrent optimizer step
#             last_layer (torch.Tensor): last layer's weight of generator, only used when training generator
#         Returns:
#             loss (torch.Tensor): loss for backward
#             log (Dict): logging
#         """        
#         if opt == 1:
#             # train discriminator
#             # zj_pred_exp = rearrange(zj_pred,"(b f) c h w -> b c f h w",f = self.frames_each)
#             # zj_gt_exp = rearrange(zj_gt,"(b f) c h w -> b c f h w",f = self.frames_each)
#             # zt_exp = rearrange(zt,"(b f) c h w -> b c f h w",f = self.frames_each)
#             # t = t.reshape(-1,self.frames_each)
#             if self.disc_rec:
#                 d_loss,logits_real,logits_fake = self.discriminator_loss(reconstructed,inputs,global_step)
#             else:
#                 d_loss,logits_real,logits_fake = self.discriminator_loss(zj_pred,zj_gt,zi,global_step)
#             log = {
#                 "discriminator_loss" : d_loss.clone().detach(),
#                 "logits_real": logits_real.detach().mean(),
#                 "logits_fake": logits_fake.detach().mean()
#             }
#             return d_loss, log
#         else: 
#             # zj_pred_exp = rearrange(zj_pred,"(b f) c h w -> b c f h w",f = self.frames_each)
#             # zt_exp = rearrange(zt,"(b f) c h w -> b c f h w",f = self.frames_each)
#             # t = t.reshape(-1,self.frames_each)
#             rec_loss = self.reconstruction_loss(v_pred,v_gt)
#             if self.perceptual_weight > 0:
#                 p_loss = self.lpips_loss(reconstructed,inputs)
#             else:
#                 p_loss = torch.zeros_like(rec_loss)
#             # rec_p_loss = (rec_loss + self.perceptual_weight * p_loss).mean()
#             weighted_nll_loss,nll_loss = self.nll_loss(rec_loss,p_loss,weights)
#             # computation graph too large
#             # considering replace nll_loss with rec_loss, thus skip vae decode, while limiting perception weight
#             nll_loss_ = torch.sum(rec_loss) / rec_loss.shape[0]
#             if self.disc_rec:
#                 g_loss,score = self.generator_loss(reconstructed,global_step,last_layer,nll_loss_,adaptive_g_loss)
#             else:
#                 g_loss,score = self.generator_loss(zj_pred,zi,global_step,last_layer,nll_loss_,adaptive_g_loss)
#             loss = weighted_nll_loss + g_loss
            
#             log =  {
#                     "generator_loss" : loss.clone().detach(),
#                     "g_loss": g_loss.detach(),
#                     "p_loss": p_loss.detach().mean(),
#                     "rec_loss": rec_loss.detach().mean(),
#                     "nll_loss": nll_loss.detach().mean(),
#                     "score": score.detach().mean(),
#                 }
#             return loss, log
        
#     def reconstruction_loss(self,pred:torch.Tensor,gt:torch.Tensor):
#         return self.loss_func(pred,gt)
#     def lpips_loss(self,pred:torch.Tensor,gt:torch.Tensor):
#         return self.perceptual_loss(pred,gt)
#     def discriminator_loss(self,
#                            zj_pred:torch.Tensor,
#                            zj_gt:torch.Tensor,
#                            zi:torch.Tensor,
#                            global_step:int
#                            ):

#         zi_input = rearrange(zi,"(n t) c h w -> n c t h w",t=self.num_frames)
#         real_input = rearrange(zj_gt,"(n t) c h w -> n c t h w",t = self.num_frames)
#         fake_input = rearrange(zj_pred,"(n t) c h w -> n c t h w",t = self.num_frames)
#         logits_real = self.discriminator(real_input.contiguous().detach(),zi_input[:,:,:1,:,:].contiguous()) # (N,) \in (0,1)
#         logits_fake = self.discriminator(fake_input.contiguous().detach(),zi_input[:,:,:1,:,:].contiguous()) # (N,)
#         disc_factor = adopt_weight(
#             self.disc_factor, global_step, threshold=self.discriminator_iter_start
#         )
#         d_loss = disc_factor * self.disc_loss(logits_real, logits_fake)
#         return d_loss,logits_real,logits_fake
#     def generator_loss(self,
#                        zj_pred:torch.Tensor,
#                        zi:torch.Tensor,
#                        global_step:int,last_layer:torch.Tensor,nll_loss:torch.Tensor,adaptive:bool=False):
#         if self.discriminator_weight > 0:
#             fake_input = rearrange(zj_pred,"(n t) c h w -> n c t h w",t = self.num_frames)
#             zi_input = rearrange(zi,"(n t) c h w -> n c t h w",t=self.num_frames)
#             logits_fake = self.discriminator(fake_input.contiguous(),zi_input[:,:,:1,:,:].contiguous())
#             g_loss = - logits_fake.mean()
#             if global_step >= self.discriminator_iter_start:
#                 if self.disc_factor > 0.0:
#                     if adaptive:
#                         d_weight = self.calculate_adaptive_weight(
#                             nll_loss, g_loss, last_layer=last_layer
#                         )
#                     else:
#                         d_weight = self.discriminator_weight
#                 else:
#                     d_weight = torch.tensor(1.0)
#             else:
#                 d_weight = torch.tensor(0.0)
#                 g_loss = torch.tensor(0.0, requires_grad=True)

#             disc_factor = adopt_weight(
#                 self.disc_factor, global_step, threshold=self.discriminator_iter_start
#             )
#             return d_weight * disc_factor * g_loss,logits_fake
#         else:
#             return torch.tensor(0.0, requires_grad=True),torch.tensor(0.0, requires_grad=True)
#     def nll_loss(self,rec_loss:torch.Tensor,p_loss:torch.Tensor,weights:float):
#         rec_p_loss = rec_loss + self.perceptual_weight * p_loss
#         nll_loss = rec_p_loss / torch.exp(self.logvar) + self.logvar
#         weighted_nll_loss = nll_loss
#         if weights is not None:
#             weighted_nll_loss = weights * nll_loss
#         weighted_nll_loss = torch.sum(weighted_nll_loss) / weighted_nll_loss.shape[0]
#         nll_loss = torch.sum(nll_loss) / nll_loss.shape[0]
#         return weighted_nll_loss,nll_loss
#     def forward(self,
#                 reconstructed:torch.Tensor,
#                 inputs:torch.Tensor,
#                 zj_pred:torch.Tensor,
#                 zj_gt:torch.Tensor,
#                 v_pred:torch.Tensor,
#                 v_gt:torch.Tensor,
#                 opt:int,
#                 global_step:int,
#                 weights:float=None,
#                 last_layer:torch.Tensor=None,
#                 adaptive_g_loss:bool=False,
#                 zi:Optional[torch.Tensor]=None
#             ):
#         return self.loss(
#                 reconstructed=reconstructed,
#                 inputs=inputs,
#                 zj_pred=zj_pred,
#                 zj_gt=zj_gt,
#                 v_pred=v_pred,
#                 v_gt=v_gt,
#                 opt=opt,
#                 global_step=global_step,
#                 last_layer=last_layer,
#                 weights=weights,
#                 adaptive_g_loss=adaptive_g_loss,
#                 zi=zi
#         )



class LpipsMseLoss(nn.Module):
    def __init__(self,
        # --- Discriminator Loss ---
        latten_chs:int = 4,

        loss_type: str = "l2",
        num_frames: int = 15,
        # --- Perception Loss ---
        perceptual_weight=0.5,

    ):

        super().__init__()

        self.perceptual_loss = LPIPS().eval()
        self.num_frames = num_frames
        self.perceptual_weight = perceptual_weight
        self.loss_func = l1 if loss_type == "l1" else l2
        
    def loss(self,
             vae,
             video_gt:torch.Tensor, # (N,C,H,W),video ground truth
             zj_pred:torch.Tensor, # (N,c,h,w)
             v_pred:torch.Tensor,
             v_gt:torch.Tensor,
             ):
        """ calculate loss(reconstruction loss, lpips loss, adversarial loss), 
            perception loss = LPIPS(reconstructied, inputs)
        Args:
            inputs (torch.Tensor): xj, shape = (N,C,H,W),video ground truth
            zj_pred (torch.Tensor): zj, shape = (N,c,h,w)
            zj_gt (torch.Tensor): ground truth of zj
            v_pred (torch.Tensor): predicted velocity
            v_gt (torch.Tensor): ground truth velocity
        """        

        # vae decode
        video_pre = vae_decode(vae,zj_pred)


        # rec loss
        rec_loss = self.reconstruction_loss(v_pred,v_gt)

        # transform
        if len(video_gt.shape) == 5:
            video_gt = einops.rearrange(video_gt,'n t c h w -> (n t) c h w')
        if len(video_pre.shape) == 5:  
            video_pre = einops.rearrange(video_pre,'n t c h w -> (n t) c h w')

        # lpips loss
        if self.perceptual_weight > 0:
            p_loss = self.lpips_loss(video_pre,video_gt)
        else:
            p_loss = torch.zeros_like(rec_loss)


        loss = rec_loss + self.perceptual_weight * p_loss
        log =  {
                "loss" : loss.detach(),
                "rec_loss": rec_loss.detach(),
                "lpips_loss": p_loss.detach(),
            }
        return loss, log
        
    def reconstruction_loss(self,pred:torch.Tensor,gt:torch.Tensor):
        return self.loss_func(pred,gt)
    def lpips_loss(self,pred:torch.Tensor,gt:torch.Tensor):
        return self.perceptual_loss(pred,gt)

    def forward(self,
                vae,
                video_gt:torch.Tensor, # (N,C,H,W),video ground truth
                zj_pred:torch.Tensor, # (N,c,h,w)
                v_pred:torch.Tensor,
                v_gt:torch.Tensor,
            ):
        return self.loss(
                vae,
                video_gt, # (N,C,H,W),video ground truth
                zj_pred, # (N,c,h,w)
                v_pred,
                v_gt,
        )
