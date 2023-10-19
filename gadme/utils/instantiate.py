import hydra
from lightning import Callback
from omegaconf import DictConfig
from lightning.pytorch.loggers import Logger
import logging


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
    
