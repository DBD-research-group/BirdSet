from typing import Any, List, Optional
import lightning.pytorch as pl
import torch
import torch.nn as nn
import torch.distributed as dist
import functools
import logging
import torch.optim.scheduler as scheduler

from pytorch_lightning import Callback, LightningModule, Trainer

#logger = logging.getLogger(__name__)
import lightning as L

class BaseModule(L.LightningModule):

    class BaseModule:
        def __init__(
            self,
            model: Any,
            loss: Any,
            optimizer: Any,
            scheduler: scheduler._LRScheduler,
            train_metrics: List[Any],
            eval_metrics: List[Any],
            scheduler_interval: int,
            compile: bool
        ) -> None:
            super(BaseModule, self).__init__()
            self.model = model
            self.loss = loss
            self.optimizer = optimizer
            self.scheduler = scheduler
            self.scheduler_interval = scheduler_interval
            self.train_metrics = nn.ModuleDict(train_metrics)
            self.eval_metrics = nn.ModuleDict(eval_metrics)
            self.compile = compile
             # this line allows to access init params with 'self.hparams' attribute
            # also ensures init params will be stored in ckpt
            self.save_hyperparameters(logger=False)
    
    def forward(self, *args, **kwargs):
        return self.model.forward(*args, **kwargs)
    
    def training_step(self, batch):
        logits = self(**batch)
        loss = self.loss(logits, batch["labels"])
        self.log("train_loss", loss, prog_bar=True)
        #self.log_train_metrics
        return {"loss": loss}

    def validation_step(self, batch, batch_idx):
        logits = self(**batch)
        loss = self.loss(logits, batch["labels"])
        self.log("val_loss", loss, prog_bar=True)
        return loss
    
    def test_step(self, batch, batch_idx):
        logits = self(**batch)
        targets = batch['labels']
        self.log_eval_metrics(logits, targets)
        return logits, targets 
    
    def test_step(self, batch, batch_idx):
        logits = self(**batch)
        targets = batch['labels']
        self.log_eval_metrics(logits, targets)
        return logits, targets 

    def log_train_metrics(self, logits, targets):
        metrics = {metric_name: metric(logits, targets) for metric_name, metric in self.train_metrics.items()}
        self.log_dict(metrics, prog_bar=True)

    def log_eval_metrics(self, logits, targets):
        metrics = {metric_name: metric(logits, targets) for metric_name, metric in self.eval_metrics.items()}
        self.log_dict(metrics, prog_bar=True, on_epoch=True)

    def configure_optimizers(self):
        if self.optimizer is None:
            self.optimizer = torch.optim.AdamW(self.parameters(), lr=5e-5)
        if isinstance(self.optimizer, functools.partial):
            self.optimizer = self.optimizer(self.parameters())

        if self.scheduler is None: 
            return {
                "optimizer": self.optimizer,
                "scheduler": None
            }
        if isinstance(self.scheduler, functools.partial):
            self.scheduler = self.scheduler(self.optimizer)

        return {
            "optimizer": self.optimizer, 
            "scheduler": {
                "scheduler": self.scheduler,
                "interval" : self.scheduler_interval
            }
        }
    def on_train_batch_start(self, batch: Any, batch_idx: int):
        pass

    def on_test_epoch_end(self):
        pass








