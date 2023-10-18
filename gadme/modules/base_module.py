from typing import Any, Optional
import lightning.pytorch as pl
import torch
import torch.nn as nn
import torch.distributed as dist
import functools
import logging

from pytorch_lightning import Callback, LightningModule, Trainer

#logger = logging.getLogger(__name__)
import lightning as L

class BaseModule(L.LightningModule):
    def __init__(
        self,
        model,
        loss_fn,
        optimizer,
        lr_scheduler,
        train_metrics,
        eval_metrics,
        scheduler_interval):

        super(BaseModule, self).__init__()
        self.model = model
        self.loss_fn = loss_fn
        self.optimizer = optimizer
        self.lr_scheduler = lr_scheduler
        self.scheduler_interval = scheduler_interval
        self.train_metrics = nn.ModuleDict(train_metrics)
        self.eval_metrics = nn.ModuleDict(eval_metrics)
    
    def forward(self, *args, **kwargs):
        return self.model.forward(*args, **kwargs)
    
    def training_step(self, batch):
        logits = self(**batch)
        loss = self.loss_fn(logits, batch["labels"])
        self.log("train_loss", loss, prog_bar=True)
        #self.log_train_metrics
        return {"loss": loss}

    def validation_step(self, batch, batch_idx):
        logits = self(**batch)
        loss = self.loss_fn(logits, batch["labels"])
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

        if self.lr_scheduler is None: 
            return {
                "optimizer": self.optimizer,
                "lr_scheduler": None
            }
        if isinstance(self.lr_scheduler, functools.partial):
            self.lr_scheduler = self.lr_scheduler(self.optimizer)

        return {
            "optimizer": self.optimizer, 
            "lr_scheduler": {
                "scheduler": self.lr_scheduler,
                "interval" : self.scheduler_interval
            }
        }
    def on_train_batch_start(self, batch: Any, batch_idx: int):
        pass

    def on_test_epoch_end(self):
        pass








