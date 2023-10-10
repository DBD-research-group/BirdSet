#%%
import os 
import time 
import json 
import logging
import torch 
import torch.nn as nn
import hydra
import math
import transformers
import lightning as L 
#from lightning.pytorch import seed_everything
import wandb
import torchmetrics

from omegaconf import OmegaConf
from torch.utils.data import DataLoader
from utils import initialize_wandb

from gadme import datasets
from gadme import models


@hydra.main(version_base=None, config_path="./configs", config_name="baselines")
def main(args):
    logging.info('Using config: \n%s', OmegaConf.to_yaml(args))
    args.output_dir = os.path.expanduser(args.output_dir)
    L.seed_everything(args.random_seed)
    os.makedirs(args.output_dir, exist_ok=True)

    # logging
    results = {}
    initialize_wandb(args)

    # Setup data
    logging.info('Building dataset %s', args.dataset.name)
    data_module = datasets.BaseGADME(
        dataset_name=args.dataset.name_hf,
        feature_extractor=args.model.name_hf,
        dataset_path=args.dataset_path,
        seed=args.random_seed,
        train_batch_size=args.train_batch_size,
        eval_batch_size=args.eval_batch_size,
        val_split=args.val_split
    )
    #data_module.prepare_data()´
    #data_module.setup()

    # Setup model 
    logging.info("Building model: %s", args.model.name)
    model = build_model(args, len_trainset=500)
    #print(next(iter(data_module.train_dataloader())))

    # Training
    trainer = L.Trainer(
        max_epochs=args.model.trainer.n_epochs,
        default_root_dir=args.output_dir,
        enable_checkpointing=False,
        fast_dev_run=False,
        enable_progress_bar=True,
        callbacks=None,
        devices=1,
        accelerator="gpu",
        strategy="auto"
    )

    trainer.fit(model, data_module)
    print("hallo")

def build_model(args, **kwargs):
    len_trainset = kwargs["len_trainset"]

    if args.model.name == "wav2vec2":
        base_model = models.w2v2.Wav2vec2SequenceClassifier(
            checkpoint=args.model.name_hf,
            num_classes=args.dataset.n_classes
        )
    
    elif args.model.name == "hubert":
        base_model = models.hubert.HubertSequenceClassifier(
            checkpoint=args.model.name_hf,
            num_classes=args.dataset.n_classes
        )
    
    else:
        raise NotImplementedError (f'Model {args.model.name} not implemented')
    
    model = torch.compile(base_model)

    # instantiate optimizer via hydra
    opt_params = args.model.optimizer
    slr_params = args.model.scheduler

    optimizer = hydra.utils.instantiate(
        opt_params,
        params=base_model.parameters()
    )
    scheduler = hydra.utils.instantiate(
        slr_params["scheduler"],
        optimizer=optimizer,
        num_warmup_steps=math.ceil(
            args.model.trainer.n_epochs * len_trainset *args.model.scheduler.extras.warmup_ratio
        ),
        num_training_steps=args.model.trainer.n_epochs * len_trainset,
        _convert_="partial"
    )

    # optimizer = torch.optim.AdamW(
    #     base_model.parameters(),
    #     lr=args.model.optimizer_lr,%
    #     weight_decay=args.model.optimizer.weight_decay 
    # )

    # scheduler = transformers.get_linear_schedule_with_warmup(
    #     optimizer=optimizer,
    #     num_warmup_steps=math.ceil(args.model.n_epochs * len_trainset * args.model.optimizer.warmup_ratio),
    #     num_training_steps=args.model.n_epochs * len_trainset
    # )

    model = models.BaseModuleTransformer(
        model=base_model,
        loss_fn=nn.CrossEntropyLoss(),
        optimizer=optimizer,
        lr_scheduler=scheduler,
        scheduler_interval="step",
        train_metrics={"train_acc": torchmetrics.classification.Accuracy(
            task="multiclass", 
            num_classes=args.dataset.n_classes)},
        val_metrics=None
    )



    return model


#%%
if __name__ == "__main__":
    main()
# %%
