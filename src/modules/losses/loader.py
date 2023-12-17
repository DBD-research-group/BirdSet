import hydra
import torch
from omegaconf import DictConfig

def load_loss(loss_cfg: DictConfig, label_counts):

    weight_params = {}

    for key, value in loss_cfg.items():
        if "weight" in key:
            weight_params[key] = torch.tensor(value).float()
    
    if label_counts and "focal" in loss_cfg._target_:
        label_counts = torch.tensor(label_counts)

        #calculate sample weights
        sample_weights = (label_counts / label_counts.sum())**(-0.5)

        # if we do not have the 0 label --> it is inf. so replace with small value
        sample_weights = torch.where(
            condition=torch.isinf(sample_weights), 
            input=torch.tensor(1e-4), 
            other=sample_weights)
        
        weight_params["alpha"] = sample_weights
    
    loss = hydra.utils.instantiate(loss_cfg, **weight_params)
    return loss