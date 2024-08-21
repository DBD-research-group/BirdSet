import torch
import torch.nn.functional as F
from dataclasses import dataclass
from birdset.modules.base_module import BaseModule, NetworkConfig, LRSchedulerConfig, LoggingParamsConfig
from birdset.configs.module_configs import MulticlassMetricsConfig, MultilabelMetricsConfig
from typing import Callable, Literal, Type, Optional, Union
from torch.nn import Module, CrossEntropyLoss
from torch.nn.modules.loss import _Loss
from torch.optim import AdamW, Optimizer
from functools import partial
from birdset.modules.models.embedding_abstract import EmbeddingModel
from torch import nn

@dataclass
class EmbeddingModuleConfig(NetworkConfig):
    """
    A dataclass that makes sure the model inherits from EmbeddingClassifier.

    """
    model: Union[EmbeddingModel, Module] = None # Model for extracting the embeddings

class EmbeddingModule(BaseModule):
    """
    EmbeddingModule is a PyTorch Lightning module for multilabel classification tasks.
    The Module expects a embedding_model that will be used to extract embeddings and than trains a network with these embeddings as input features.

    Attributes:
        model (Network): Model for extracting the embeddings
    """
    def __init__(
            self,
            network: NetworkConfig = NetworkConfig(),
            output_activation: Callable[[torch.Tensor], torch.Tensor] = torch.sigmoid,
            loss: _Loss = CrossEntropyLoss(),
            optimizer: partial[Type[Optimizer]] = partial(
                AdamW,
                lr=1e-5,
                weight_decay=0.01,
            ),
            lr_scheduler: Optional[LRSchedulerConfig] = LRSchedulerConfig(),
            metrics: MulticlassMetricsConfig | MultilabelMetricsConfig = MulticlassMetricsConfig(),
            logging_params: LoggingParamsConfig = LoggingParamsConfig(),
            num_epochs: int = 50,
            len_trainset: int = 13878, # set to property from datamodule
            batch_size: int = 32,
            task: Literal['multiclass', 'multilabel'] = "multiclass",
            num_gpus: int = 1,
            pretrain_info = None
            #embedding_model: EmbeddingModuleConfig = EmbeddingModuleConfig() # Model for extracting the embeddings
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
        )
        #print(f"Using "+ embedding_model.model_name+" as the embedding model")
        #self.embedding_model = embedding_model.model

    # Use the embedding model to get the embeddings and pass them to the classifier model
    def forward(self, *args, **kwargs):
        # Get embeddings
        input_values = kwargs['input_values'] # Extract input tensor
        #embeddings, _ = self.embedding_model.get_embeddings(input_values)

        # Pass embeddings through the classifier to get the final output
        #embeddings = embeddings.view(embeddings.size(0), -1) # Transform for the classifier
        return self.model.forward(input_values)