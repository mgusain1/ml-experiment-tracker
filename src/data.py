import torch

def make_dataset(n_samples,noise):
    x = torch.randn(n_samples,1)
    y = 3*x+2+noise*torch.randn(n_samples,1)
    return x,y