import torch
import torch.nn as nn
import torch.distributed as dist


import lightning as L

class BaseModule(L.LightningModule):
    def __init__(
        self,
        model,
        loss_fn,
        optimizer,
        lr_scheduler,
        train_metrics,
        val_metrics,
        scheduler_interval):

        super(BaseModule).__init__()
        self.model = model
        self.loss_fn = loss_fn
        self.optimizer = optimizer
        self.lr_scheduler = lr_scheduler
        self.scheduler_interval = scheduler_interval
        self.train_metrics = nn.ModuleDict(train_metrics)
        self.val_metrics = nn.ModuleDict(val_metrics)

    def forward(self, *args, **kwargs):
        return self.model.forward(*args, **kwargs)
    
    def training_step(self, batch):
        logits = self(batch["input_ids"], batch["attention_mask"])
        loss = self.loss_fn(logits, batch["labels"])
        self.log("train_loss")
    
