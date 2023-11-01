import os 
import rootutils
import hydra
import lightning as L 
from omegaconf import OmegaConf
from src.utils.instantiate import instantiate_callbacks, instantiate_wandb
from src.utils.pylogger import get_pylogger
from src.utils.utils import close_loggers

log = get_pylogger(__name__)
rootutils.setup_root(__file__, indicator=".project-root", pythonpath=True)

@hydra.main(version_base=None, config_path="../configs", config_name="main")
def main(args):
    log.info('Using config: \n%s', OmegaConf.to_yaml(args))

    L.seed_everything(args.seed)
    os.makedirs(args.paths.dataset_path, exist_ok=True)
    os.makedirs(args.paths.log_dir, exist_ok=True)

    log.info(f"Instantiate logger {[loggers for loggers in args['logger']]}")

    # TODO: This line throws an error in .fit which has to be fixed
    #logger = instantiate_wandb(args)

    # Setup data
    log.info(f"Instantiate data module , <{args.datamodule._target_}>")
    data_module = hydra.utils.instantiate(args.datamodule)
    data_module.prepare_data()

    # Setup model 
    log.info("Building model: %s", args.module.model_name)
    model = hydra.utils.instantiate(
        args.module,
        num_epochs=args.trainer.max_epochs,
        len_trainset=data_module.len_trainset
    )

    # Training
    log.info('Instantiate callbacks %s', [callbacks for callbacks in args["callbacks"]])
    callbacks = instantiate_callbacks(args["callbacks"])

    log.info(f"Instantiate trainer")
    trainer = hydra.utils.instantiate(
        args.trainer, callbacks= callbacks, logger=logger
    )

    trainer.fit(
        model=model, 
        datamodule=data_module,
        ckpt_path=args.get("ckpt_path"))

    # Evaluation
    trainer.test(ckpt_path="last", datamodule=data_module)
    close_loggers()

if __name__ == "__main__":    
    main()
