from birdset.modules.base_module import BaseModule
from functools import partial
from typing import Callable, Literal, Type, Optional, Union
from torch.nn import CrossEntropyLoss
from torch.nn.modules.loss import _Loss
from torch.optim import AdamW, Optimizer
from birdset.configs import NetworkConfig, LoggingParamsConfig, LRSchedulerConfig, MulticlassMetricsConfig, MultilabelMetricsConfig, MultilabelMetricsConfig as MetricsConfig
from birdset.datamodule.embedding_datamodule import EmbeddingModuleConfig
from birdset.utils import pylogger
from birdset.modules.finetune_module import FinetuneModule

import torch
import math

log = pylogger.get_pylogger(__name__)
class HybridModule(FinetuneModule):
    def __init__(
            self,
            network: NetworkConfig = NetworkConfig(),
            output_activation: Callable[[torch.Tensor], torch.Tensor] = torch.sigmoid,
            loss: _Loss = CrossEntropyLoss(),
            optimizer: partial[Type[Optimizer]] = partial(
                AdamW,
                lr=1e-2,
                weight_decay=5e-4,
            ),
            lr_scheduler: Optional[LRSchedulerConfig] = LRSchedulerConfig(),
            metrics: MulticlassMetricsConfig | MultilabelMetricsConfig = MulticlassMetricsConfig(),
            logging_params: LoggingParamsConfig = LoggingParamsConfig(),
            num_epochs: int = 50,
            len_trainset: int = 13878, # set to property from datamodule
            batch_size: int = 32,
            task: Literal['multiclass', 'multilabel'] = "multiclass",
            num_gpus: int = 1,
            pretrain_info = None,
            embedding_model: EmbeddingModuleConfig = EmbeddingModuleConfig(), # Model for extracting the embeddings
            ft_optimizer: partial[Type[Optimizer]] = partial( # Optimizer to use for fine-tuning
                AdamW,
                lr=1e-4,
                weight_decay=0.01,
            ),
            ft_max_epochs: int = 5,
            ):
        super().__init__(
            network = network,
            output_activation=output_activation,
            loss=loss,
            optimizer=optimizer,
            lr_scheduler=lr_scheduler,
            metrics=metrics,
            logging_params=logging_params,
            num_epochs=num_epochs,
            len_trainset=len_trainset,
            batch_size=batch_size,
            task=task,
            num_gpus=num_gpus,
            pretrain_info = pretrain_info,
            embedding_model=embedding_model
        )
        self.linear_probing = True
        self.ft_optimizer = ft_optimizer
        self.ft_max_epochs = ft_max_epochs
        
    def configure_optimizers(self):
        if self.linear_probing: 
            #print("PROBING")
            self.embedding_model.eval()
            for param in self.embedding_model.parameters(): # Just to be sure
                param.requires_grad = False
            self.optimizer = self.optimizer(list(self.model.parameters()))   
        else:
            #print("TUNING")
            self.embedding_model.train()  
            for param in self.embedding_model.parameters():
                param.requires_grad = True 
            self.optimizer = self.ft_optimizer(list(self.model.parameters())+list(self.embedding_model.parameters()))
        
        #! REMOVE OR CHECK THIS SCHEDULER THING
        '''    if self.lr_scheduler is not None:
                print("HALLO")
                # TODO: Handle the case when we do not want warmup
                num_training_steps = math.ceil((self.num_epochs * self.len_trainset) / self.batch_size * self.num_gpus)
                num_warmup_steps = math.ceil(
                        num_training_steps * self.lr_scheduler.warmup_ratio
                    )
                # TODO: Handle the case when drop_last=True more explicitly   

                self.scheduler = self.lr_scheduler.scheduler(
                    optimizer=self.optimizer,
                    num_training_steps=num_training_steps,
                    num_warmup_steps=num_warmup_steps,
                )

                scheduler_dict = {
                    "scheduler": self.scheduler,
                    "interval": self.lr_scheduler.interval,
                    "warmup_ratio":self.lr_scheduler.warmup_ratio}                      

                return {"optimizer": self.optimizer, "lr_scheduler": scheduler_dict}'''

        return {"optimizer": self.optimizer}


    def on_train_end(self):
        if self.linear_probing:
            #print("STARTING fine-tuning...")
            self.linear_probing = False
            self.trainer.fit_loop.max_epochs += self.ft_max_epochs
            for param_group in self.optimizer.param_groups:
                print(f"Learning Rate: {param_group['lr']}")
            #print(self.trainer.checkpoint_callback.best_model_path)
            self.trainer.fit(self,  datamodule=self.trainer.datamodule)
                
    def model_step(self, batch, batch_idx):
        logits = self.forward(**batch)
        if self.class_mask and (not self.pretrain_info.valid_test_only or not self.trainer.training):
            if batch["labels"].shape == logits.shape:
                batch["labels"] = batch["labels"][:, self.class_mask]
            logits = logits[:, self.class_mask]
        loss = self.loss(logits, batch["labels"])
        preds = self.output_activation(logits)
        return loss, preds, batch["labels"]            