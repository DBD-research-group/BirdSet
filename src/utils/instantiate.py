import hydra
from lightning import Callback
from omegaconf import DictConfig
from omegaconf import OmegaConf
from src.utils import pylogger
import wandb

from pytorch_lightning.loggers import WandbLogger
log = pylogger.get_pylogger(__name__)

def instantiate_callbacks(callbacks_args):
    callbacks = []

    if not callbacks_args:
        log.warning("No callbacks found")
        return callbacks
    
    for _, cb_conf in callbacks_args.items():
        if isinstance(cb_conf, DictConfig) and "_target_" in cb_conf:
            log.info(f"Instantiate callbacks <{cb_conf._target_}>")
            callbacks.append(hydra.utils.instantiate(cb_conf))
    
    return callbacks
    

def instantiate_wandb(args):
    logger_args = args.logger

    if not logger_args:
        log.warning("No callbacks found")
        return None
    
    #logging.info(f"Instantiating logger <{logger_args._target_}>")
    log.info(f"Instantiate logger <{logger_args.wandb._target_}>")
    logger = hydra.utils.instantiate(
        logger_args
    )
    logger = logger.wandb
    # wandb.config.update(OmegaConf.to_container(
    #     args,
    #     resolve=True,
    #     throw_on_missing=True
    # ))
    return logger


def initialize_wandb_logger(args):
    wandb_logger = WandbLogger(
        name=args.module.model_name+'_'+args.datamodule.dataset.dataset_name+'#'+str(args.seed),
        save_dir=args.paths.log_dir,
        project=args.logger.wandb.project,
        mode=args.logger.wandb.mode,
        entity=args.logger.wandb.entity,
        config=OmegaConf.to_container(
            args,
            resolve=True,
            throw_on_missing=True
        )
    )

    return wandb_logger
