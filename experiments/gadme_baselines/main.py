import os 
import logging
import hydra
import lightning as L 
from lightning.pytorch.callbacks.early_stopping import EarlyStopping
from lightning.pytorch.callbacks.lr_monitor import LearningRateMonitor
from omegaconf import OmegaConf
from utils import build_dataset, build_model, initialize_wandb_logger
from gadme.callbacks import TimeCallback



@hydra.main(version_base=None, config_path="./configs", config_name="baselines")
def main(args):
    logging.info('Using config: \n%s', OmegaConf.to_yaml(args))
    args.output_dir = os.path.expanduser(args.output_dir)
    L.seed_everything(args.random_seed)
    os.makedirs(args.output_dir, exist_ok=True)

    # logging
    results = {}
    wandb_logger = initialize_wandb_logger(args)


    # Setup data
    # instantiate and mapping are a problem, turn the omegaconf into dict!
    logging.info("Instantiate data module %s", args.dataset.name)
    data_module = build_dataset(args)
    data_module.prepare_data()
    #data_module.setup()

    # Setup model 
    logging.info("Building model: %s", args.model.name)
    model = build_model(args, len_trainset=500)
    #print(next(iter(data_module.train_dataloader()))
    # Training
    trainer = L.Trainer(
        max_epochs=args.model.trainer.n_epochs,
        default_root_dir=args.output_dir,
        callbacks=[
            EarlyStopping(monitor="val_loss", patience=2),
            TimeCallback(),
            LearningRateMonitor("step")
            ],
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

if __name__ == "__main__":    main()
