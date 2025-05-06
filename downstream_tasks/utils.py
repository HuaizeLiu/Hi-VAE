import torch
import numpy as np
from typing import Dict,List
import math
def logit(x):
    return 1 / (1 + np.exp(- x))

def sample_t(num_samples,m,s):
    mu = m + s * np.random.randn((num_samples)) 
    return logit(mu)
def sample_timestep(num_samples,m,s,num_steps):
    mu = m + s * np.random.randn((num_samples)) 
    return np.array(logit(mu) * num_steps,dtype=np.int32)
def get_sample_t_schedule(t_schedule:Dict,sample_steps) -> List[float]:
    """ construct sample t schedule

    Args:
        m (int): intrinsic param
        n (int): intrinsic param
    Returns:
        t (torch.Tensor): sample t schedulem shape = (N,K)
    """    
    m = t_schedule.get("m",1)
    n = t_schedule.get("n",100)
    logm = math.log(m)
    logn = math.log(n)
    progress = torch.linspace(0,1,sample_steps + 1)
    logmn = torch.log(progress * (m - n) + n)
    t = 1 - (logm - logmn) / (logm - logn)
    return torch.diff(t)
if __name__ == "__main__":
    steps = get_sample_t_schedule({},sample_steps=10)
    print(torch.sum(steps) - 1)