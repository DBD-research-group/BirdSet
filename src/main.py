import os 
import rootutils
import hydra
import lightning as L 
from omegaconf import OmegaConf
from src.utils.instantiate import instantiate_callbacks, instantiate_wandb
from src.utils.pylogger import get_pylogger
from src.utils import utils 

log = get_pylogger(__name__)
rootutils.setup_root(__file__, indicator=".project-root", pythonpath=True)

@hydra.main(version_base=None, config_path="../configs", config_name="main")
def main(cfg):
    log.info('Using config: \n%s', OmegaConf.to_yaml(cfg))

    log.info(f"Dataset path: <{os.path.abspath(cfg.paths.dataset_path)}>")
    os.makedirs(cfg.paths.dataset_path, exist_ok=True)

    log.info(f"Log path: <{os.path.abspath(cfg.paths.log_dir)}>")
    os.makedirs(cfg.paths.log_dir, exist_ok=True)

    log.info(f"Seed everything with <{cfg.seed}>")
    L.seed_everything(cfg.seed)

    # Setup data
    log.info(f"Instantiate datamodule <{cfg.datamodule._target_}>")
    datamodule = hydra.utils.instantiate(cfg.datamodule)
    datamodule.prepare_data()

    # Setup model 
    log.info(f"Instantiate model <{cfg.module.model._target_}>")     
    model = hydra.utils.instantiate(
        cfg.module,
        num_epochs=cfg.trainer.max_epochs,
        len_trainset=datamodule.len_trainset,
        _recursive_=False # manually instantiate!
    )

    # Setup logger
    log.info(f"Instantiate logger")
    logger = instantiate_wandb(cfg) 

    # Setup callbacks
    log.info(f"Instantiate callbacks")
    callbacks = instantiate_callbacks(cfg["callbacks"])

    # Training
    log.info(f"Instantiate trainer <{cfg.trainer._target_}>")
    trainer = hydra.utils.instantiate(
        cfg.trainer, callbacks= callbacks, logger=logger
    )

    object_dict = {
        "cfg": cfg, 
        "datamodule": datamodule,
        "model": model,
        "callbacks": callbacks,
        "logger": logger,
        "trainer": trainer
    }

    log.info("Logging Hyperparams")
    utils.log_hyperparameters(object_dict)

    if cfg.get("train"):
        log.info(f"Starting training")
        trainer.fit(
            model=model, 
            datamodule=datamodule,
            ckpt_path=cfg.get("ckpt_path"))
    
    train_metrics = trainer.callback_metrics

    if cfg.get("test"):
        log.info(f"Starting testing")
        ckpt_path = trainer.checkpoint_callback.best_model_path
        if ckpt_path == "":
            log.warning(
                "No ckpt saved or found. Using current weights for testing"
            )
            ckpt_path = None
        trainer.test(model=model, datamodule=datamodule, ckpt_path=ckpt_path)
    test_metrics = trainer.callback_metrics

    metric_dict = {**train_metrics, **test_metrics}
    
    utils.close_loggers()

if __name__ == "__main__":    
    main()
