import torch
import torch.nn as nn
import torch.distributed as dist
import functools

import lightning as L

class BaseModuleTransformer(L.LightningModule):
    def __init__(
        self,
        model,
        loss_fn,
        optimizer,
        lr_scheduler,
        train_metrics,
        val_metrics,
        scheduler_interval):

        super(BaseModuleTransformer, self).__init__()
        self.model = model
        self.loss_fn = loss_fn
        self.optimizer = optimizer
        self.lr_scheduler = lr_scheduler
        self.scheduler_interval = scheduler_interval
        self.train_metrics = nn.ModuleDict(train_metrics)
        self.val_metrics = nn.ModuleDict(val_metrics)

    
    # def model_step(self, batch, *args, **kwargs):
    #     logits = 

    def forward(self, *args, **kwargs):
        return self.model.forward(*args, **kwargs)
    
    def training_step(self, batch):
        logits = self(batch["input_ids"], batch["attention_mask"])
        loss = self.loss_fn(logits, batch["labels"])
        self.log("train_loss", loss, prog_bar=True)
        #self.log_train_metrics
        return {"loss": loss}

    def validation_step(self, batch, batch_idx):
        logits = self(batch["input_ids"], batch["attention_mask"])
        loss = self.loss_fn(logits, batch["labels"])
        self.log("val_loss", loss, prog_bar=True)
        return loss
    
    def predict_step(self, batch, batch_idx, dataloader_idx=0):
        logits = self(batch['input_ids'], batch['attention_mask'])
        logits = self._gather(logits)
        targets = self._gather(batch['labels'])
        #cc_fn = metrics.Accuracy().to('cuda')
        #self.log('test_accuracy', acc_fn(logits, targets), prog_bar=True)
        return logits, targets 

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



        




