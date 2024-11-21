import os
import hydra
import json
import pyrootutils
from omegaconf import open_dict
import lightning as L
from pathlib import Path

from birdset import utils

root = pyrootutils.setup_root(
    search_from=__file__,
    indicator=[".git"],
    pythonpath=True,
    dotenv=True,
)

_HYDRA_PARAMS = {
    "version_base": None,
    "config_path": str(root / "configs"),
    "config_name": "train.yaml",
}

log = utils.get_pylogger(__name__)


@hydra.main(**_HYDRA_PARAMS)
def eval(cfg):
    # log.info('Using config: \n%s', OmegaConf.to_yaml(cfg))
    log.info("Starting Evaluation")
    log.info(f"Dataset path: <{os.path.abspath(cfg.paths.dataset_path)}>")
    os.makedirs(cfg.paths.dataset_path, exist_ok=True)

    log.info(f"Log path: <{os.path.abspath(cfg.paths.log_dir)}>")
    os.makedirs(cfg.paths.log_dir, exist_ok=True)

    log.info(f"Root Dir:<{os.path.abspath(cfg.paths.log_dir)}>")
    log.info(f"Work Dir:<{os.path.abspath(cfg.paths.work_dir)}>")
    log.info(f"Output Dir:<{os.path.abspath(cfg.paths.output_dir)}>")

    if cfg.get("seed"):
        log.info(f"Seed everything with <{cfg.seed}>")
        L.seed_everything(cfg.seed)

    log.info(f"Instantiate datamodule <{cfg.datamodule._target_}>")
    datamodule = hydra.utils.instantiate(cfg.datamodule)
    datamodule.prepare_data()

    log.info(f"Instantiate model <{cfg.module.network.model._target_}>")
    with open_dict(cfg):
        cfg.module.metrics["num_labels"] = datamodule.num_classes
        cfg.module.network.model["num_classes"] = datamodule.num_classes
    model = hydra.utils.instantiate(
        cfg.module,
        num_epochs=cfg.trainer.max_epochs,
        len_trainset=datamodule.len_trainset,
        batch_size=datamodule.loaders_config.train.batch_size,
        pretrain_info=cfg.module.network.model.pretrain_info,
    )

    log.info(f"Instantiate logger")
    logger = utils.instantiate_loggers(cfg.get("logger"))
    # override standard TF logger to handle rare logger error
    logger.append(utils.TBLogger(Path(cfg.paths.log_dir)))

    trainer = hydra.utils.instantiate(cfg.trainer, logger=logger)

    object_dict = {
        "cfg": cfg,
        "datamodule": datamodule,
        "model": model,
        "logger": logger,
        "trainer": trainer,
    }

    log.info("Logging Hyperparams")
    utils.log_hyperparameters(object_dict)

    if cfg.get("test"):
        log.info("Starting Testing")
        trainer.test(model=model, datamodule=datamodule, ckpt_path=None)
        test_metrics = trainer.callback_metrics
    else:
        log.info("Predict not yet implemented")

    metric_dict = {**test_metrics}
    metric_dict = [
        {"name": k, "value": v.item() if hasattr(v, "item") else v}
        for k, v in metric_dict.items()
    ]

    file_path = os.path.join(cfg.paths.output_dir, "finalmetrics.json")
    with open(file_path, "w") as json_file:
        json.dump(metric_dict, json_file)

    utils.close_loggers()


if __name__ == "__main__":
    eval()
