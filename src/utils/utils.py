
from importlib.util import find_spec
from src.utils import pylogger

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