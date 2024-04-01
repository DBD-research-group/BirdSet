import torch
from .base_module import BaseModule, NetworkConfig, LRSchedulerConfig, MetricsConfig, LoggingParamsConfig
from birdset.modules.metrics.multiclass import MulticlassMetricsConfig
import wandb
from typing import Callable, Literal, Type, Optional
from torch.nn import BCEWithLogitsLoss
from torch.nn.modules.loss import _Loss
from torch.optim import AdamW, Optimizer
from functools import partial

class MultilabelModule(BaseModule):
    """
    MulticlassModule is a PyTorch Lightning module for multiclass classification tasks.

    Attributes:
        prediction_table (bool): Whether to create a prediction table. Defaults to False.
    """
    def __init__(
            self,
            network: NetworkConfig = NetworkConfig(),
            output_activation: Callable[[torch.Tensor], torch.Tensor] = torch.sigmoid,
            loss: _Loss = BCEWithLogitsLoss(),
            optimizer: partial[Type[Optimizer]] = partial(
                AdamW,
                lr=1e-5,
                weight_decay=0.01,
            ),
            lr_scheduler: Optional[LRSchedulerConfig] = LRSchedulerConfig(),
            metrics: MetricsConfig = MulticlassMetricsConfig(),
            logging_params: LoggingParamsConfig = LoggingParamsConfig(),
            num_epochs: int = 50,
            len_trainset: int = 13878, # set to property from datamodule
            batch_size: int = 32,
            task: Literal['multiclass', 'multilabel'] = "multiclass",
            class_weights_loss: Optional[bool] = None,
            label_counts: int = 21,
            num_gpus: int = 1,
            prediction_table: bool = False
            ):
    
        self.prediction_table = prediction_table

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
            task=task,
            class_weights_loss=class_weights_loss,
            label_counts=label_counts,
            batch_size=batch_size,
            num_gpus=num_gpus
        )

  