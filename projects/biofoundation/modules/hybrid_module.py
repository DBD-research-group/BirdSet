from functools import partial
from typing import Callable, Literal, Type, Optional
from torch.nn import CrossEntropyLoss
from torch.nn.modules.loss import _Loss
from torch.optim import AdamW, Optimizer, Adam
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
from birdset.modules.finetune_module import FinetuneModule

import torch

log = pylogger.get_pylogger(__name__)


class HybridModule(FinetuneModule):
    """
    HybridModule is an extension of the FinetuneModule that enables first only training the classifier and then after a specified amount of epochs doing complete finetuning.
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
        ft_lr: Learning rate for the finetuning part as it can differ from linear probing LR.
        ft_at_epoch: Epoch at which to start finetuning the feature extractor.
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
            lr=1e-2,
            weight_decay=5e-4,
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
        ft_lr: int = 1e-2,
        ft_at_epoch: int = 10,
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
            embedding_model=embedding_model,
        )
        self.embedding_model.train()
        self.ft_lr = ft_lr
        self.ft_at_epoch = ft_at_epoch
        # Freeze the embedding model at the start
        self.freeze_embedding_model()

    def configure_optimizers(self):
        self.optimizer = self.optimizer(
            self.model.parameters()
        )  # Model is the classifier
        return {"optimizer": self.optimizer}

    def on_train_epoch_start(self):
        # Change the learning rate after specified epoch
        if self.current_epoch == self.ft_at_epoch:
            log.info(f"Changing learning rate to {self.ft_lr}")
            self.unfreeze_embedding_model()

            # Define new optimizer with different learning rate
            optimizer = AdamW(
                [
                    {
                        "params": self.model.parameters(),
                        "lr": self.ft_lr,
                        "weight_decay": 5e-4,
                    },
                    {
                        "params": self.embedding_model.parameters(),
                        "lr": self.ft_lr,
                        "weight_decay": 5e-4,
                    },
                ]
            )
            self.trainer.optimizers = [optimizer]

    def freeze_embedding_model(self):
        """Freeze the embedding model's parameters."""
        for param in self.embedding_model.parameters():
            param.requires_grad = False

    def unfreeze_embedding_model(self):
        """Unfreeze the embedding model's parameters."""
        for param in self.embedding_model.parameters():
            param.requires_grad = True
