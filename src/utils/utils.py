
from importlib.util import find_spec
from src.utils import pylogger
from pytorch_lightning.utilities import rank_zero_only
from omegaconf import OmegaConf

log = pylogger.get_pylogger(__name__)

def close_loggers() -> None:
    """Makes sure all loggers closed properly (prevents logging failure during
    multirun)."""

    log.info("Closing loggers...")

    if find_spec("wandb"):  # if wandb is installed
        import wandb

        if wandb.run:
            log.info("Closing wandb!")
            wandb.finish()

@rank_zero_only
def log_hyperparameters(object_dict):
    hparams = {}

    cfg = object_dict["cfg"]
    trainer = object_dict["trainer"]
  
    hparams["datamodule"] = cfg["datamodule"]
    hparams["trainer"] = cfg["trainer"]

    hparams["callbacks"] = cfg.get("callbacks")
    hparams["extras"] = cfg.get("extras")

    hparams["task_name"] = cfg.get("task_name")
    hparams["tags"] = cfg.get("tags")
    hparams["ckpt_path"] = cfg.get("ckpt_path")
    hparams["seed"] = cfg.get("seed")

    #trainer.logger.experiment.config.update(hparams, allow_val_change=True)
    trainer.logger.log_hyperparams(OmegaConf.to_container(cfg))
    #trainer.logger.log_hyperparams(hparams)
    