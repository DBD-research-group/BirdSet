import os 
import hydra
import lightning as L 
from omegaconf import OmegaConf
import json
from gadme import utils
import pyrootutils 

log = utils.get_pylogger(__name__)
#rootutils.setup_root(__file__, indicator=".project-root", pythonpath=True)
root = pyrootutils.setup_root(
    search_from=__file__,
    indicator=[".git"],
    pythonpath=True,
    dotenv=True,
)

_HYDRA_PARAMS = {
    "version_base":None,
    #"config_path": "../configs",
    "config_path": str(root / "configs"),
    "config_name": "main.yaml"
}

@utils.register_custom_resolvers(**_HYDRA_PARAMS)
@hydra.main(**_HYDRA_PARAMS)
def main(cfg):
    log.info('Using config: \n%s', OmegaConf.to_yaml(cfg))

    log.info(f"Dataset path: <{os.path.abspath(cfg.paths.dataset_path)}>")
    os.makedirs(cfg.paths.dataset_path, exist_ok=True)

    log.info(f"Log path: <{os.path.abspath(cfg.paths.log_dir)}>")
    os.makedirs(cfg.paths.log_dir, exist_ok=True)

    log.info(f"Root Dir:<{os.path.abspath(cfg.paths.log_dir)}>")
    log.info(f"Work Dir:<{os.path.abspath(cfg.paths.work_dir)}>")
    log.info(f"Output Dir:<{os.path.abspath(cfg.paths.output_dir)}>")
    log.info(f"Background Dir:<{os.path.abspath(cfg.paths.background_path)}>")

    log.info(f"Seed everything with <{cfg.seed}>")
    L.seed_everything(cfg.seed)
    #log.info(f"Instantiate logger {[loggers for loggers in cfg['logger']]}")

    # Setup data
    log.info(f"Instantiate datamodule <{cfg.datamodule._target_}>")
    datamodule = hydra.utils.instantiate(cfg.datamodule)
    datamodule.prepare_data() # has to be called before model for len_traindataset!

    # Setup logger
    log.info(f"Instantiate logger")
    logger = utils.instantiate_loggers(cfg.get("logger")) 

    # Setup callbacks
    log.info(f"Instantiate callbacks")
    callbacks = utils.instantiate_callbacks(cfg["callbacks"])

    # Training
    log.info(f"Instantiate trainer <{cfg.trainer._target_}>")
    trainer = hydra.utils.instantiate(
        cfg.trainer, callbacks= callbacks, logger=logger
    )

    # Setup model 
    log.info(f"Instantiate model <{cfg.module.network.model._target_}>")     
    model = hydra.utils.instantiate(
        cfg.module,
        num_epochs=cfg.trainer.max_epochs, #?
        len_trainset=datamodule.len_trainset,
        batch_size=datamodule.loaders_config.train.batch_size,
        label_counts=datamodule.num_train_labels,
        _recursive_=False # manually instantiate!
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
        ckpt = cfg.get("ckpt_path")
        if ckpt:
            log.info(f"Resume training from checkpoint {ckpt}")
        else:
            log.info("No checkpoint found. Training from scratch!")
                     
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
        else:
            log.info(
                f"The best checkpoint for {cfg.callbacks.model_checkpoint.monitor}"
                f" is {trainer.checkpoint_callback.best_model_score}"
                f" and saved in {ckpt_path}"   
            )
        trainer.test(model=model, datamodule=datamodule, ckpt_path=ckpt_path)

    test_metrics = trainer.callback_metrics 

    if cfg.get("save_state_dict"):
        log.info(f"Saving state dicts")
        utils.save_state_dicts(
            trainer=trainer,
            model=model, 
            dirname=cfg.paths.output_dir,
            **cfg.extras.state_dict_saving_params  
        )

    if cfg.get("dump_metrics"):
        log.info(f"Dumping final metrics locally to {cfg.paths.output_dir}")
        metric_dict = {**train_metrics, **test_metrics}

        metric_dict = [{'name': k, 'value': v.item() if hasattr(v, 'item') else v} for k, v in metric_dict.items()]

        file_path = os.path.join(cfg.paths.output_dir, "finalmetrics.json")
        with open(file_path, 'w') as json_file:
            json.dump(metric_dict, json_file)
    
    utils.close_loggers()

if __name__ == "__main__":    
    main()
