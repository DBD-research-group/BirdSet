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
            output_activation: Callable[[torch.Tensor], torch.Tensor] = partial(
                torch.softmax,
                dim=1
            ),
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
            #ft_lr: int = 5,
            #ft_max_epochs: int = 5,
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
        #self.ft_lr = ft_lr
        #self.ft_max_epochs = ft_max_epochs
        
    """def configure_optimizers(self):
        if not self.linear_probing:
            return self.optimizer
        
        #self.embedding_model.eval()
        for param in self.embedding_model.parameters(): # Just to be sure
            param.requires_grad = False  
        self.optimizer = self.optimizer(list(self.model.parameters())+list(self.embedding_model.parameters()))
        
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

        return {"optimizer": self.optimizer}"""

    def configure_optimizers(self):
        if not self.linear_probing:
            self.embedding_model.train()  
            #self.optimizer.add_param_group({'params': self.embedding_model.parameters()})

            for param in self.embedding_model.parameters():
                param.requires_grad = True 
                
            for param_group in self.trainer.optimizers[0].param_groups:
                param_group['lr'] = 1e-4
            
            return self.optimizer
        
        self.optimizer = self.optimizer(list(self.model.parameters())+list(self.embedding_model.parameters())) #! Changed this
        for param in self.embedding_model.parameters():
                param.requires_grad = False
        if self.lr_scheduler is not None:
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

            return {"optimizer": self.optimizer, "lr_scheduler": scheduler_dict}

        return {"optimizer": self.optimizer}


    def on_train_end(self):
        if self.linear_probing:
            print("STARTING fine-tuning...")
            self.linear_probing = False
            #print(self.trainer.checkpoint_callback.best_model_path)
            #self.trainer.fit(self,  datamodule=self.trainer.datamodule)
            
            #self.embedding_model.train()  
            #for param in self.embedding_model.parameters():
                #param.requires_grad = True 
                
            #for param_group in self.trainer.optimizers[0].param_groups:
                #param_group['lr'] = 1e-4 # Set your new learning rate here 
            #print("Erhöht")
            
            self.trainer.fit_loop.max_epochs += 5
            
            
            self.trainer.fit(self,  datamodule=self.trainer.datamodule, ckpt_path=self.trainer.checkpoint_callback.best_model_path)         
            #self.trainer.fit_loop.min_epochs = self.trainer.current_epoch + 5  
            #self.trainer.fit(self,  datamodule=self.trainer.datamodule, ckpt_path=self.trainer.checkpoint_callback.best_model_path)         
                        