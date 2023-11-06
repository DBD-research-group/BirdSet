import hydra
import torch
from omegaconf import DictConfig

def load_loss(loss_cfg: DictConfig):

    weight_params = {}

    for key, value in loss_cfg.items():
        if "weight" in key:
            weight_params[key] = torch.tensor(value).float()
    
    loss = hydra.utils.instantiate(loss_cfg, **weight_params)
    return loss