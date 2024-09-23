from birdset.modules.base_module import BaseModule
from functools import partial
from typing import Callable, Literal, Type, Optional, Union
from torch.nn import CrossEntropyLoss
from torch.nn.modules.loss import _Loss
from torch.optim import AdamW, Optimizer, Adam
from birdset.configs import NetworkConfig, LoggingParamsConfig, LRSchedulerConfig, MulticlassMetricsConfig, MultilabelMetricsConfig, MultilabelMetricsConfig as MetricsConfig
from birdset.datamodule.embedding_datamodule import EmbeddingModuleConfig
from birdset.utils import pylogger
from birdset.modules.finetune_module import FinetuneModule
from torch.optim.lr_scheduler import LambdaLR

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
            ft_lr: int = 1e-2,
            ft_max_epochs: int = 10,
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
        self.embedding_model.train()
        #self.linear_probing = True
        self.ft_lr = ft_lr
        self.ft_max_epochs = ft_max_epochs
        # Freeze the embedding model at the start
        self.freeze_embedding_model()
        
    def configure_optimizers(self):
        # Set the AdamW optimizer with the initial learning rate
        self.optimizer = self.optimizer(self.model.parameters())
        #self.optimizer = Adam(self.model.parameters(), lr=1e-2) 
        # Attach the learning rate scheduler
        return {"optimizer": self.optimizer}
        #optimizer = self.optimizer([{'name':'classifier','params': list(self.model.parameters())+list(self.embedding_model.parameters()), 'lr': 1e-2}])#, {'name':'embedding','params': self.embedding_model.parameters(), 'lr': 0.0}])

        # Custom scheduler: starts with warmup, then reduces to 0, and gradually increases after unfreezing
        '''def lr_lambda(epoch):
            if epoch < self.ft_max_epochs:
                # Linearly increase the learning rate during warmup
                return 1.0
            elif epoch == self.ft_max_epochs:
                # Reduce LR to zero after warmup
                return 0.0
            else:
                # Gradually increase LR from 0 to max_lr after unfreezing
                total_epochs = self.trainer.max_epochs
                progress = (epoch - self.ft_max_epochs+1) / (total_epochs - self.ft_max_epochs+1)
                return progress * (self.ft_lr / 1e-2)
        
        scheduler = LambdaLR(optimizer, lr_lambda=lr_lambda)
        return [optimizer], [scheduler]'''

    def on_train_epoch_start(self):
        #for param_group in self.optimizer.param_groups:
            #print(self.current_epoch,param_group['lr'])
        # Change the learning rate after epoch 15
        if self.current_epoch == self.ft_max_epochs:
            print(f"Changing learning rate to {self.ft_lr}")
            self.unfreeze_embedding_model()    
            #optimizer = self.optimizers()
            #optimizer.add_param_group({'name':'embedding','params': self.embedding_model.parameters(), 'lr': 0.0})
            #for param_group in optimizer.param_groups:
                #param_group['lr'] = self.ft_lr
            
            
            
            optimizer = AdamW([
                {'params': self.model.parameters(), 'lr': self.ft_lr, 'weight_decay': 5e-4},  # Lower LR for feature extractor
                {'params': self.embedding_model.parameters(), 'lr': self.ft_lr, 'weight_decay': 5e-4}    # Higher LR for classifier head
            ])
            self.trainer.optimizers = [optimizer]
        #print(self.lr_schedulers())
        print(self.optimizers())  

    def freeze_embedding_model(self):
        """Freeze the embedding model's parameters."""
        for param in self.embedding_model.parameters():
            param.requires_grad = False
            
    def unfreeze_embedding_model(self):
        """Unfreeze the embedding model's parameters."""
        for param in self.embedding_model.parameters():
            param.requires_grad = True    