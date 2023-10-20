import wandb 
import torch
import torch.nn as nn
import math
import torchmetrics
import hydra 
from omegaconf import OmegaConf
from gadme import datasets
from gadme.modules import models,base_module
from pytorch_lightning.loggers import WandbLogger

def initialize_wandb(args):
    wandb.init(
        project=args.wandb.project,
        entity=args.wandb.entity,
        group=args.wandb.group,
        reinit=args.wandb.reinit,
        mode = args.wandb.mode,
        name=args.model.name+'_'+args.dataset.name+'#'+str(args.random_seed),
        config = OmegaConf.to_container(
            args, 
            resolve=True, 
            throw_on_missing=True
        ),
        dir=args.paths.log_dir
    )

def initialize_wandb_logger(args):
    wandb_logger = WandbLogger(
        name=args.model.name+'_'+args.dataset.dataset_name+'#'+str(args.random_seed),
        save_dir=args.paths.log_dir,
        project=args.loggers.wandb.project,
        mode=args.loggers.wandb.mode,
        entity=args.loggers.wandb.entity,
        config=OmegaConf.to_container(
            args,
            resolve=True,
            throw_on_missing=True
        )
    )

    return wandb_logger


# def build_dataset (args, **kwargs):
#     if args.dataset.name == "sapsucker":
#         datamodule = datasets.SapsuckerWoods(
#             data_dir=args.paths.dataset_path,
#             dataset_name=args.dataset.name,
#             feature_extractor_name=args.model.name_hf,
#             dataset_loading=dict(args.dataset.loading),
#             seed=args.random_seed,
#             train_batch_size=args.train_batch_size,
#             eval_batch_size=args.eval_batch_size,
#             val_split=args.val_split,
#             column_list=["input_values", "ebird_code"]
#         )
    
#     elif args.dataset.name == "hsn":
#          datamodule = datasets.SapsuckerWoods(
#             data_dir=args.dataset_path,
#             dataset_name=args.dataset.name,
#             feature_extractor_name=args.model.name_hf,
#             dataset_loading=dict(args.dataset.loading),
#             seed=args.random_seed,
#             train_batch_size=args.train_batch_size,
#             eval_batch_size=args.eval_batch_size,
#             val_split=args.val_split,
#             column_list=["input_values", "ebird_code"]
#         )       
    
#     elif args.dataset.name == "esc50":
#         datamodule = datasets.ESC50(
#             data_dir=args.dataset_path,
#             dataset_name=args.dataset.name,
#             feature_extractor_name=args.model.name_hf,
#             dataset_loading=dict(args.dataset.loading),
#             seed=args.random_seed,
#             train_batch_size=args.train_batch_size,
#             eval_batch_size=args.eval_batch_size,
#             val_split=args.val_split,
#             column_list=["input_values", "target"]
#         )
        
#     else:
#         raise NotImplementedError(
#             f'Dataset {args.dataset.name} not implemented.'
#         )

#     return datamodule

def build_model(args, **kwargs):
    len_trainset = kwargs["len_trainset"]
    num_classes = kwargs["num_classes"]

    if args.model.name == "wav2vec2":
        
        base_model = models.w2v2.Wav2vec2SequenceClassifier(
            checkpoint=args.model.name_hf,
            num_classes=num_classes
        )
    
    elif args.model.name == "hubert":
        base_model = models.hubert.HubertSequenceClassifier(
            checkpoint=args.model.name_hf,
            num_classes=num_classes
        )
    
    elif args.model.name == "ast":
        base_model = models.ast.ASTSequenceClassifier(
            checkpoint=args.model.name_hf,
            num_classes=num_classes
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

    model = base_module.BaseModule(
        model=base_model,
        loss_fn=nn.CrossEntropyLoss(),
        optimizer=optimizer,
        lr_scheduler=scheduler,
        scheduler_interval="step",
        train_metrics={"train_acc": torchmetrics.classification.Accuracy(
            task="multiclass", 
            num_classes=num_classes)},
        eval_metrics={"acc": torchmetrics.classification.Accuracy(
            task="multiclass", 
            num_classes=num_classes)},
    )

    return model