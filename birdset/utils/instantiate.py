import hydra
from lightning import Callback
from omegaconf import DictConfig
from omegaconf import OmegaConf
from birdset.utils import pylogger
from pytorch_lightning.loggers import Logger
from typing import Any, Callable, List, Optional

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


def instantiate_wandb(cfg):
    logger_cfg = cfg.logger

    if not logger_cfg:
        log.warning("No callbacks found")
        return None

    # logging.info(f"Instantiating logger <{logger_cfg._target_}>")
    log.info(f"Instantiate logger <{logger_cfg.wandb._target_}>")
    logger = hydra.utils.instantiate(logger_cfg)
    logger = logger.wandb
    # wandb.config.update(OmegaConf.to_container(
    #     args,
    #     resolve=True,
    #     throw_on_missing=True
    # ))
    return logger


def instantiate_loggers(logger_cfg):
    logger: List[Logger] = []

    if not logger_cfg:
        log.warning("No loger configs found, skipping...")
        return logger

    if not isinstance(logger_cfg, DictConfig):
        raise TypeError("Logger config must be a DictConfig!")

    for _, lg_conf in logger_cfg.items():
        if isinstance(lg_conf, DictConfig) and "_target_" in lg_conf:
            log.info(f"Instantiate logger <{lg_conf._target_}>")
            logger.append(hydra.utils.instantiate(lg_conf))

    try:
        log.info(f"Wandb version to resume training: {logger_cfg.wandb.version}")

    except:
        log.info("Wandb not used")

    return logger
