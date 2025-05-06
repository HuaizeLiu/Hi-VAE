import torch
from .loss import (l1,l2)
import numpy as np
from tqdm import tqdm

class RectifiedFlow:
  def __init__(self, num_steps=1000 ,theta=1e-5):
    self.num_step = num_steps
    self.theta = theta
  
  def train_loss(self,model,z0,z1,loss_type='l2'):
    zt,t,gt = self.get_train_tuple(z0,z1)
    pre = model(zt,t)
    
    loss_fn = eval(loss_type)
    return loss_fn(pre,gt)
  
  def get_train_tuple(self, z0=None, z1=None,time_step=None):
    """
    Args:
        z0 (torch.Tensor) : start distribution d0 , can be Gaussian noise. default shape : (n,c,h,w)
        z1 (torch.Tensor) : target distribution d1 , default shape : (n,c,h,w)
        time_step (torch.Tensor) : (batch_size,)
    Returns:
        z_t (torch.Tensor) : intermediate distribution z_t, default shape : (n,c,h,w)
        t (torch.Tensor) : interpolation factor t, default shape : (n,1)
        target (torch.Tensor) : target distribution, default shape : (n,c,h,w)
    """
    if time_step is None:
      t = torch.rand((z1.shape[0], 1))
    else:
      t = self.timestep_to_time(time_step) # (batch_size,1)
    
    if z0 is None:
      z0 = torch.randn_like(z1) # gaussian noise
    
    z_t =  t * z1 + (1.- t) * z0
    target = z1 - z0 
        
    return z_t, target
  
  def get_target_with_zt_vel(self, zt,vel,time_step):
    t = self.timestep_to_time(time_step) # (batch_size,1)
    target = zt + (1.-t)*vel
    return target



  @torch.no_grad()
  def sample_ode_process(self,model=None, z0=None, N=None):
    ### NOTE: Use Euler method to sample from the learned flow
    if N is None:
      N = self.num_step    
    dt = 1./N
    traj = [] # to store the trajectory
    z = z0.detach().clone()
    batchsize = z.shape[0]
    
    traj.append(z.detach().clone())
    for i in range(N):
      t = torch.ones((batchsize,1)) * i / N
      pred = model(z, t)
      z = z.detach().clone() + pred * dt
      
      traj.append(z.detach().clone())

    return traj
  
  @torch.no_grad()
  def timestep_to_time(self,time_step):
    # use for transform time_step(int) to time(float)
    t = (self.num_step - time_step) / self.num_step
    if len(t.shape) == 1:
      t = t.view(-1,1,1,1)
    return t
  
  @torch.no_grad()
  def sample_loop(self,model,video_start,sample_step,start_step=None,motion = None,motion_available_length=None,reinforce_condition=False): 
    """predict z1 from z0

    Args:
        video_start : `torch.Tensor` (n,c,h,w)
        motion : `torch.Tensor` (n,t,c,h,w)
        start_step : `int` denoise start step
        sample_step : `int` <= self.num.step
    Returns:
        z1 (n,c,h,w) : predict z1 from z0
    """
    if start_step is None:
      start_step = self.num_step

    step_seq = np.linspace(0, start_step, num=sample_step, endpoint=True,dtype=int) # [0,5,10,15,....,start_step]
    
    dt = 1./sample_step
    
    sample = video_start
    
    for i in tqdm(list(reversed(step_seq))):
      # time_step
      time_step = torch.ones((sample.shape[0],)).to(sample.device)  
      time_step = time_step * i
      
      # input
      if reinforce_condition:
        zt = torch.cat((video_start,sample),dim=1)
      else:
        zt = sample
      
      # forward
      if motion is None:
        pre = model(zt,time_step)
      else:
        pre = model(motion,zt,time_step,motion_available_length) # (n,c,h,w)
      sample = sample + pre * dt
    
    return sample # (n,c,h,w)
  
  @torch.no_grad()
  def sample_step(self,model,zt,sample_step,start_step=None):
    """
    one-step denoise
    Args:
        model : torch.nn.Module
        start_step : `int`
    Returns:
        z : torch.Tensor , z[t-1]
    """
    if start_step is None:
      start_step = self.num_step
    
    dt = 1./sample_step
    

    time_step = torch.ones((zt.shape[0],1)) * start_step
    pre = model(zt,time_step)
    z = zt + pre * dt
    
    return z
    
      