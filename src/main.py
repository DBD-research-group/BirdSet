import os 
import logging
import rootutils
import hydra
import lightning as L 
from omegaconf import OmegaConf
#from utils import initialize_wandb_logger
from src.utils.instantiate import instantiate_callbacks, instantiate_wandb, initialize_wandb_logger

rootutils.setup_root(__file__, indicator=".project-root", pythonpath=True)

@hydra.main(version_base=None, config_path="../configs", config_name="main")
def main(args):
    logging.info('Using config: \n%s', OmegaConf.to_yaml(args))

    L.seed_everything(args.seed)
    os.makedirs(args.paths.dataset_path, exist_ok=True)
    os.makedirs(args.paths.log_dir, exist_ok=True)

    # logging
    results = {}
    logging.info(f"Instantiate logger {[loggers for loggers in args['loggers']]}")
    wandb_logger = initialize_wandb_logger(args)
    #wandb_logger = instantiate_wandb(args) # throws an error in .fit

    # Setup data
    logging.info("Instantiate data module %s", args.dataset)
    data_module = hydra.utils.instantiate(args.dataset)
    data_module.prepare_data()

    # Setup model 
    logging.info("Building model: %s", args.model.model_name)
    model = hydra.utils.instantiate(
        args.model,
        num_epochs=args.trainer.max_epochs,
        len_trainset=data_module.len_trainset
    )

    # Training
    logging.info('Instantiate callbacks %s', [callbacks for callbacks in args["callbacks"]])
    callbacks = instantiate_callbacks(args["callbacks"])

    logging.info(f"Instantiate trainer")
    trainer = hydra.utils.instantiate(
        args.trainer, callbacks= callbacks, logger=wandb_logger
    )

    trainer.fit(model, data_module)

    # Evaluation
    trainer.test(ckpt_path="last", datamodule=data_module)
    #predictions = trainer.predict(model, test_dataloader)
    print("finished")

if __name__ == "__main__":    
    main()
