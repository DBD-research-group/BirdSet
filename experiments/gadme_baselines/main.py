#%%
import os 
import time 
import json 
import logging

import torch 
import torch.nn as nn
import hydra
import math
import transformers

import lightning as L 
#from lightning.pytorch import seed_everything
import wandb

from omegaconf import OmegaConf
from torch.utils.data import DataLoader
from utils import initialize_wandb

from gadme import build_dataset

#%%


@hydra.main(version_base=None, config_path="./configs", config_name="baselines")
def main(args):
    logging.info('Using config: \n%s', OmegaConf.to_yaml(args))
    args.output_dir = os.path.expanduser(args.output_dir)
    L.seed_everything(args.random_seed)
    os.makedirs(args.output_dir, exist_ok=True)

    # logging
    results = {}
    initialize_wandb(args)

    # Setup data
    logging.info('Building dataset %s', args.dataset.name)
    data= build_dataset(args)


#%%
main()