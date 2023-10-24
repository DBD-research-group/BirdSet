import os 
import logging
import rootutils
import hydra
import lightning as L 
from omegaconf import OmegaConf
from utils import build_model, initialize_wandb_logger
from gadme.utils.instantiate import instantiate_callbacks, instantiate_wandb

rootutils.setup_root(__file__, indicator=".project-root", pythonpath=True)

@hydra.main(version_base=None, config_path="../../configs", config_name="default")
def main(args):
    logging.info('Using config: \n%s', OmegaConf.to_yaml(args))

    L.seed_everything(args.random_seed)
    os.makedirs(args.paths.data_dir, exist_ok=True)
    os.makedirs(args.paths.log_dir, exist_ok=True)

    # logging
    results = {}
    logging.info(f"Instantiate logger {[loggers for loggers in args['loggers']]}")
    wandb_logger = initialize_wandb_logger(args)
    #wandb_logger = instantiate_wandb(args) # throws an error in .fit

    # Setup data
    # instantiate and mapping are a problem, turn the omegaconf into dict!
    logging.info("Instantiate data module %s", args.dataset.instantiate)
    #data_module = build_dataset(args)
    data_module = hydra.utils.instantiate(args.dataset.instantiate)
    data_module.prepare_data()
    #data_module.setup()

    # Setup model 
    logging.info("Building model: %s", args.model.extras.name)
    model = hydra.utils.instantiate(
        args.model.instantiate,
        num_epochs=5,
        len_trainset=data_module.len_trainset
    )

    # Training
    logging.info('Instantiate callbacks %s', [callbacks for callbacks in args["callbacks"]])
    callbacks = instantiate_callbacks(args["callbacks"])

    trainer = L.Trainer(
        max_epochs=5,
        default_root_dir=args.paths.output_dir,
        callbacks=callbacks,
        enable_checkpointing=False,
        fast_dev_run=False,
        enable_progress_bar=True,
        devices=1,
        accelerator="gpu",
        strategy="auto",
        logger=wandb_logger
    )
    trainer.fit(model, data_module)

    # Evaluation
    trainer.test(ckpt_path="last", datamodule=data_module)
    #predictions = trainer.predict(model, test_dataloader)
    print("finished")

if __name__ == "__main__":    
    main()
