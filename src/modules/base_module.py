from typing import Any, Optional
import lightning.pytorch as pl
import torch
import torch.nn as nn
import torch.distributed as dist
import functools
import logging
import math 
import hydra
from pytorch_lightning import Callback, LightningModule, Trainer

from src.modules.losses import load_loss
from src.modules.metrics import load_metrics


#logger = logging.getLogger(__name__)
import lightning as L

class BaseModule(L.LightningModule):
    def __init__(
        self,
        network,
        output_activation,
        loss,
        optimizer,
        lr_scheduler,
        metrics,
        logging_params,
        num_epochs,
        len_trainset,
        task):

        super(BaseModule, self).__init__()

        self.model = hydra.utils.instantiate(network.model)
        self.opt_params = optimizer
        self.lrs_params = lr_scheduler

        self.loss = load_loss(loss)
        self.output_activation = hydra.utils.instantiate(
            output_activation,
            _partial_=True
        )
        self.logging_params = logging_params
        
        self.metrics = load_metrics(metrics)
        self.train_metric = self.metrics["main_metric"].clone()
        self.train_add_metrics = self.metrics["add_metrics"].clone(postfix="/train")

        self.valid_metric = self.metrics["main_metric"].clone()
        self.valid_metric_best = self.metrics["val_metric_best"].clone()
        self.valid_add_metrics = self.metrics["add_metrics"].clone(postfix="/valid")

        self.test_metric = self.metrics["main_metric"].clone()
        self.test_add_metrics = self.metrics["add_metrics"].clone(postfix="/test")


        # self.train_metrics = nn.ModuleDict(train_metrics)
        # self.eval_metrics = nn.ModuleDict(eval_metrics)
        self.torch_compile = network.torch_compile
        self.model_name = network.model_name

        # partial
        self.num_epochs = num_epochs
        self.len_trainset = len_trainset
        self.task = task 

        self.save_hyperparameters()
        
    def forward(self, *args, **kwargs):
        return self.model.forward(*args, **kwargs)

    def configure_optimizers(self):
        optimizer = hydra.utils.instantiate(
            self.opt_params, 
            params=self.parameters(),
            _convert_='partial'
        )

        if self.lrs_params.get("scheduler"):
            if self.lrs_params.scheduler._target_ == "transformers.get_linear_schedule_with_warmup":
                scheduler = hydra.utils.instantiate(
                    self.lrs_params.scheduler,
                    optimizer=optimizer,
                    num_warmup_steps=math.ceil(
                        self.num_epochs * self.len_trainset * self.lrs_params.extras.warmup_ratio
                    ),
                    num_training_steps=self.num_epochs * self.len_trainset,
                    _convert_="partial"
                )
            else:
                scheduler = hydra.utils.instantiate(
                    self.lrs_params.scheduler,
                    optimizer=self.optimizer,
                    _convert_="partial"
                )
            
            lr_scheduler_dict = {"scheduler": scheduler}

            if self.lrs_params.get("extras"):
                for key, value in self.lrs_params.get("extras").items():
                    lr_scheduler_dict[key] = value
            
            return {"optimizer": optimizer, "lr_scheduler": lr_scheduler_dict}
        
        return {"optimizer": optimizer}
    
    def model_step(self, batch, batch_idx):
        logits = self.forward(**batch)
        loss = self.loss(logits, batch["labels"])
        preds = self.output_activation(logits)
        return loss, preds, batch["labels"]
    
    def on_train_start(self):
        self.valid_metric_best.reset()       

    def training_step(self, batch, batch_idx):
        train_loss, preds, targets = self.model_step(batch, batch_idx)
        self.log(
            f"train_loss",
            train_loss,
            on_step=True,
            on_epoch=True,
            prog_bar=True
        )

        self.train_metric(preds, targets)
        self.log(
            f"train_{self.train_metric.__class__.__name__}",
            self.train_metric,
            **self.logging_params
        )

        #self.train_add_metrics(preds, targets)
        #self.log_dict(self.train_add_metrics, **self.logging_params)

        return {"loss": train_loss}
    
    def validation_step(self, batch, batch_idx):
        val_loss, preds, targets = self.model_step(batch, batch_idx)

        self.log(
            f"val_loss",
            val_loss,
            on_step=True,
            on_epoch=True,
            prog_bar=True
        )

        self.valid_metric(preds, targets)
        self.log(
            f"val_{self.valid_metric.__class__.__name__}",
            self.valid_metric,
            **self.logging_params,
        )

        self.valid_add_metrics(preds, targets)
        self.log_dict(self.valid_add_metrics, **self.logging_params)
        return {"loss": val_loss, "preds": preds, "targets": targets}
    
    def on_validation_epoch_end(self):
        valid_metric = self.valid_metric.compute()  # get current valid metric
        self.valid_metric_best(valid_metric)  # update best so far valid metric

        self.log(
            f"val_best_{self.valid_metric.__class__.__name__}",
            self.valid_metric_best.compute(),
        )
    
    def test_step(self, batch, batch_idx):
        test_loss, preds, targets = self.model_step(batch, batch_idx)
        self.log(
            f"test_loss",
            test_loss, 
            on_step=False,
            on_epoch=True,
            prog_bar=True
        )

        self.test_metric(preds, targets)
        self.log(
            f"test_{self.test_metric.__class__.__name__}",
            self.test_metric,
            **self.logging_params,
        )

        self.test_add_metrics(preds, targets)
        self.log_dict(self.test_add_metrics, **self.logging_params)
        return {"loss": test_loss, "preds": preds, "targets": targets}
        

    # def log_train_metrics(self, logits, targets):
    #     metrics = {metric_name: metric(logits, targets) for metric_name, metric in self.train_metrics.items()}
    #     self.log_dict(metrics, prog_bar=True)

    # def log_eval_metrics(self, logits, targets):
    #     metrics = {metric_name: metric(logits, targets) for metric_name, metric in self.eval_metrics.items()}
    #     self.log_dict(metrics, prog_bar=True, on_epoch=True)

    def setup(self, stage):
        if self.torch_compile and stage=="fit":
            self.model = torch.compile(self.model)


    def on_train_batch_start(self, batch: Any, batch_idx: int):
        pass

    def on_test_epoch_end(self):
        pass






