import torch
from dataclasses import dataclass
from .base_module import BaseModule, NetworkConfig, LRSchedulerConfig, MetricsConfig, LoggingParamsConfig
from typing import Callable, Literal, Type, Optional, Union
from torch.nn import BCEWithLogitsLoss, Module
from torch.nn.modules.loss import _Loss
from torch.optim import AdamW, Optimizer
from functools import partial
from .models.embedding_abstract import EmbeddingModel

@dataclass
class EmbeddingModuleConfig(NetworkConfig):
    """
    A dataclass that makes sure the model inherits from EmbeddingClassifier.

    """
    model: Union[EmbeddingModel, Module] = None # Model for extracting the embeddings


class EmbeddingModule(BaseModule):
    """
    MultilabelModule is a PyTorch Lightning module for multilabel classification tasks.

    Attributes:
        model (Network): Model for extracting the embeddings
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
            metrics: MetricsConfig = MetricsConfig(),
            logging_params: LoggingParamsConfig = LoggingParamsConfig(),
            num_epochs: int = 50,
            len_trainset: int = 13878, # set to property from datamodule
            batch_size: int = 32,
            task: Literal['multiclass', 'multilabel'] = "multiclass",
            class_weights_loss: Optional[bool] = None,
            label_counts: int = 21,
            num_gpus: int = 1,
            embedding_model: Union[EmbeddingModel, Module] = None # Model for extracting the embeddings
            ):
        self.embedding_model = embedding_model

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

    # Use the embedding model to get the embeddings and pass them to the classifier model
    def forward(self, *args, **kwargs):
        embeddings = self.embedding_model.get_embeddings(*args, **kwargs)
        embeddings = embeddings.view(embeddings.size(0), -1) # Transform for the classifier
        return self.model.forward(embeddings)

