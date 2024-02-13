import torch

def bfloat16_numpy(self:torch.Tensor):
    if self.dtype != torch.bfloat16:
        return self.numpy()
    return self.to(torch.float32).numpy()
