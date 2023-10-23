import hydra
from lightning import Callback
from omegaconf import DictConfig
from omegaconf import OmegaConf
from lightning.pytorch.loggers import Logger
import logging
import wandb

def instantiate_callbacks(callbacks_args):
    callbacks = []

    if not callbacks_args:
        logging.warning("No callbacks found")
        return callbacks
    
    for _, cb_conf in callbacks_args.items():
        if isinstance(cb_conf, DictConfig) and "_target_" in cb_conf:
            logging.info(f"Instantiate callbacks <{cb_conf._target_}>")
            callbacks.append(hydra.utils.instantiate(cb_conf))
    
    return callbacks
    

def instantiate_wandb(args):
    logger_args = args.loggers
    logger = None

    if not logger_args:
        logging.warning("No callbacks found")
        return logger
    
    #logging.info(f"Instantiating logger <{logger_args._target_}>")

    logger = hydra.utils.instantiate(
        logger_args
    )
    wandb.config.update(OmegaConf.to_container(
            args,
            resolve=True,
            throw_on_missing=True
        ))
    return logger
    
