from importlib.util import find_spec
from typing import Callable, Any
from hydra import compose, initialize_config_dir
from hydra.core.global_hydra import GlobalHydra
from omegaconf import OmegaConf
from pytorch_lightning.utilities import rank_zero_only
from functools import wraps
from birdset.modules.losses import load_loss
from birdset.modules.metrics import load_metrics
from birdset.utils import pylogger
import argparse
import sys

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

    # trainer.logger.experiment.config.update(hparams, allow_val_change=True)
    trainer.logger.log_hyperparams(OmegaConf.to_container(cfg))
    # trainer.logger.log_hyperparams(hparams)


def get_args_parser() -> argparse.ArgumentParser:
    """Get parser for additional Hydra's command line flags."""
    parser = argparse.ArgumentParser(
        description="Additional Hydra's command line flags parser."
    )

    parser.add_argument(
        "--config-path",
        "-cp",
        nargs="?",
        default=None,
        help="""Overrides the config_path specified in hydra.main().
                    The config_path is absolute or relative to the Python file declaring @hydra.main()""",
    )

    parser.add_argument(
        "--config-name",
        "-cn",
        nargs="?",
        default=None,
        help="Overrides the config_name specified in hydra.main()",
    )

    parser.add_argument(
        "--config-dir",
        "-cd",
        nargs="?",
        default=None,
        help="Adds an additional config dir to the config search path",
    )
    return parser


# flexible metric and loss names for callbacks
def register_custom_resolvers(
    version_base: str, config_path: str, config_name: str
) -> Callable:
    """Optional decorator to register custom OmegaConf resolvers. It is
    excepted to call before `hydra.main` decorator call.

    Replace resolver: To avoiding copying of loss and metric names in configs,
    there is custom resolver during hydra initialization which replaces
    `__loss__` to `loss.__class__.__name__` and `__metric__` to
    `main_metric.__class__.__name__` For example: ${replace:"__metric__/valid"}
    Use quotes for defining internal value in ${replace:"..."} to avoid grammar
    problems with hydra config parser.

    Args:
        version_base (str): Hydra version base.
        config_path (str): Hydra config path.
        config_name (str): Hydra config name.

    Returns:
        Callable: Decorator that registers custom resolvers before running
            main function.
    """

    # parse additional Hydra's command line flags
    parser = get_args_parser()
    args, _ = parser.parse_known_args()
    if args.config_path:
        config_path = args.config_path
    if args.config_dir:
        config_path = args.config_dir
    if args.config_name:
        config_name = args.config_name

    # register of replace resolver
    if not OmegaConf.has_resolver("replace"):
        with initialize_config_dir(version_base=version_base, config_dir=config_path):
            overrides = sys.argv[1:]  # get arguments, except filename

            cfg = compose(
                config_name=config_name, return_hydra_config=True, overrides=overrides
            )
        cfg_tmp = cfg.copy()
        loss = load_loss(cfg_tmp.module.loss, None, None)
        metric = load_metrics(cfg_tmp.module.metrics)
        metric = metric["main_metric"]
        GlobalHydra.instance().clear()

        OmegaConf.register_new_resolver(
            "replace",
            lambda item: item.replace("__loss__", loss.__class__.__name__).replace(
                "__metric__", metric.__class__.__name__
            ),
        )

    def decorator(function: Callable) -> Callable:
        @wraps(function)
        def wrapper(*args: Any, **kwargs: Any) -> Any:
            return function(*args, **kwargs)

        return wrapper

    return decorator
