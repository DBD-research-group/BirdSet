from birdset.modules.base_module import BaseModule
from functools import partial
from typing import Callable, Literal, Type, Optional, Union
from torch.nn import CrossEntropyLoss
from torch.nn.modules.loss import _Loss
from torch.optim import AdamW, Optimizer
from birdset.configs import (
    NetworkConfig,
    LoggingParamsConfig,
    LRSchedulerConfig,
    MulticlassMetricsConfig,
    MultilabelMetricsConfig,
    MultilabelMetricsConfig as MetricsConfig,
)
from birdset.datamodule.embedding_datamodule import EmbeddingModuleConfig
from birdset.utils import pylogger

import torch
import math

log = pylogger.get_pylogger(__name__)


class FinetuneModule(BaseModule):
    """
    FinetuneModule is an extension of the BaseModule that enables finetuning of compatible models (Check README). Embeddings are extracted from the model and passed to a classifier. The loss is then calculated to update the feature extractor.
    The default parameters are used for the task of 'multiclass' classification.

    Attributes:
        network (NetworkConfig): Configuration for the network.
        output_activation (Callable): The output activation function.
        loss (_Loss): The loss function.
        optimizer (partial): The optimizer function to be initalized in configure_optimizers.
        lr_scheduler (LRSchedulerConfig, optional): The learning rate scheduler configuration.
        metrics (MetricsConfig): The metrics configuration.
        logging_params (LoggingParamsConfig): The logging parameters configuration.
        num_epochs (int): The number of epochs for training.
        len_trainset (int): The length of the training set.
        batch_size (int): The batch size for training.
        task (str): The task type, can be either 'multiclass' or 'multilabel'.
        num_gpus (int): The number of GPUs to use for training.
        pretrain_info: Information about the pretraining of the model.
        embedding_model: Model for extracting the embeddings.
    """

    def __init__(
        self,
        network: NetworkConfig = NetworkConfig(),
        output_activation: Callable[[torch.Tensor], torch.Tensor] = partial(
            torch.softmax, dim=1
        ),
        loss: _Loss = CrossEntropyLoss(),
        optimizer: partial[Type[Optimizer]] = partial(
            AdamW,
            lr=1e-5,
            weight_decay=0.01,
        ),
        lr_scheduler: Optional[LRSchedulerConfig] = LRSchedulerConfig(),
        metrics: (
            MulticlassMetricsConfig | MultilabelMetricsConfig
        ) = MulticlassMetricsConfig(),
        logging_params: LoggingParamsConfig = LoggingParamsConfig(),
        num_epochs: int = 50,
        len_trainset: int = 13878,  # set to property from datamodule
        batch_size: int = 32,
        task: Literal["multiclass", "multilabel"] = "multiclass",
        num_gpus: int = 1,
        pretrain_info=None,
        embedding_model: EmbeddingModuleConfig = EmbeddingModuleConfig(),  # Model for extracting the embeddings
    ):
        super().__init__(
            network=network,
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
            pretrain_info=pretrain_info,
        )
        self.embedding_model = embedding_model.model
        self.sampling_rate = embedding_model.sampling_rate
        self.max_length = embedding_model.length
        log.info(
            f"FINETUNING using embedding model:{embedding_model.model_name} (Sampling Rate:{self.sampling_rate}, Window Size:{self.max_length})"
        )

    def configure_optimizers(self):
        self.optimizer = self.optimizer(
            list(self.model.parameters()) + list(self.embedding_model.parameters())
        )  #! Changed this
        if self.lr_scheduler is not None:
            # TODO: Handle the case when we do not want warmup
            num_training_steps = math.ceil(
                (self.num_epochs * self.len_trainset) / self.batch_size * self.num_gpus
            )
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
                "warmup_ratio": self.lr_scheduler.warmup_ratio,
            }

            return {"optimizer": self.optimizer, "lr_scheduler": scheduler_dict}

        return {"optimizer": self.optimizer}

    # Use the embedding model to get the embeddings and pass them to the classifier model
    def forward(self, *args, **kwargs):
        # Get embeddings
        input_values = kwargs["input_values"]  # Extract input tensor
        embeddings, _ = self.embedding_model.get_embeddings(input_values)

        # Pass embeddings through the classifier to get the final output
        embeddings = embeddings.unsqueeze(1)  # Transform for the classifier
        logits = self.model.forward(embeddings)
        return logits
