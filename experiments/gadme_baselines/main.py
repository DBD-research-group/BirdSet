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

from gadme import datasets

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
    data_module = datasets.BaseGADME(
        dataset_name=args.dataset.name_hf,
        dataset_path=args.dataset_path,
        seed=args.random_seed,
        train_batch_size=args.train_batch_size,
        eval_batch_size=args.eval_batch_size,
        val_split=args.val_split
    )

    data_module.prepare_data()
    data_module.setup()


    # Setup model 
    logging.info("Building model: %s", args.model_name)
    model = 

    print(next(iter(data_module.train_dataloader())))


#%%
if __name__ == "__main__":
    main()
# %%
