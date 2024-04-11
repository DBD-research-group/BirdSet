import torch
from .base_module import BaseModule, NetworkConfig, LRSchedulerConfig, LoggingParamsConfig
from birdset.modules.metrics.multiclass import MulticlassMetricsConfig
from typing import Callable, Literal, Type, Optional
from torch.nn import CrossEntropyLoss
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
            output_activation: Callable[[torch.Tensor], torch.Tensor] = partial(
                torch.softmax,
                dim=1
            ),
            loss: _Loss = CrossEntropyLoss(),
            optimizer: partial[Type[Optimizer]] = partial(
                AdamW,
                lr=1e-5,
                weight_decay=0.01,
            ),
            lr_scheduler: Optional[LRSchedulerConfig] = LRSchedulerConfig(),
            metrics: MulticlassMetricsConfig = MulticlassMetricsConfig(),
            logging_params: LoggingParamsConfig = LoggingParamsConfig(),
            num_epochs: int = 50,
            len_trainset: int = 13878, # set to property from datamodule
            batch_size: int = 32,
            task: Literal['multiclass', 'multilabel'] = "multiclass",
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
            batch_size=batch_size,
            num_gpus=num_gpus
        )

  