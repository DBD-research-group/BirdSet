from dataclasses import dataclass, field, asdict
from functools import partial
from typing import Callable, Dict, List, Literal, Type, Optional, Union

from birdset.modules.metrics.multilabel import TopKAccuracy, cmAP, cmAP5, mAP, pcmAP
from birdset.modules.models.efficientnet import EfficientNetClassifier
import torch
import math
import hydra

from birdset.modules.losses import load_loss
import datasets

import lightning as L
import torch.nn as nn
from torch.nn import BCEWithLogitsLoss
from torch.nn.modules.loss import _Loss
from torch.optim import AdamW, Optimizer, lr_scheduler 
from transformers import get_scheduler, SchedulerType
from torchmetrics import AUROC, Metric, MaxMetric, MetricCollection

def get_num_gpu(num_gpus: Union[int|str|List[int]]) -> int:
    """
    Returns the number of GPU`s infered from lightnings trainer devices argument.
    https://lightning.ai/docs/pytorch/stable/api/lightning.pytorch.trainer.trainer.Trainer.html#lightning.pytorch.trainer.trainer.Trainer
    """
     # check if num_gpus is a list
    if not isinstance(num_gpus, int) and not isinstance(num_gpus, str):
        return len(num_gpus)
    elif isinstance(num_gpus, str):
        if num_gpus == "auto" or num_gpus == "-1":
            return torch.cuda.device_count()
        elif len(num_gpus.split(",")) > 1:
            return len(num_gpus.split(",")) 
        else:
            return int(num_gpus)
    else:    
        return num_gpus

@dataclass
class NetworkConfig:
    """
    A dataclass for configuring a neural network model for training.

    Attributes:
        model (nn.Module): The model to be used for training. Defaults to an instance of `EfficientNetClassifier`.
        model_name (str): The name of the model. Defaults to "efficientnet".
        model_type (Literal['vision', 'waveform']): The type of the model, can be either 'vision' or 'waveform'. Defaults to "vision".
        torch_compile (bool): Whether to compile the model using TorchScript. Defaults to False.
        sample_rate (int): The sample rate for audio data. Defaults to 32000.
        normalize_waveform (bool): Whether to normalize the waveform data. Defaults to False.
        normalize_spectrogram (bool): Whether to normalize the spectrogram data. Defaults to True.
    """
    model: nn.Module = EfficientNetClassifier(
        architecture="efficientnet_b1",
        num_classes=21,
        num_channels=1,
        checkpoint=None
    )
    model_name: str = "efficientnet"
    model_type: Literal['vision', 'waveform'] = "vision"
    torch_compile: bool = False
    sample_rate: int = 32000
    normalize_waveform: bool = False
    normalize_spectrogram: bool = True


# @dataclass
# class LRSchedulerExtrasConfig:
#     """
#     A dataclass for configuring the extras of the learning rate scheduler.

#     Attributes:
#         interval (str): The interval at which the scheduler performs its step. Defaults to "step".
#         warmup_ratio (float): The ratio of warmup steps to total steps. Defaults to 0.5.
#     """
#     interval: str = "step"
#     warmup_ratio: float = 0.05


@dataclass
class LRSchedulerConfig:
    """
    A dataclass for configuring the learning rate scheduler.

    Attributes:
        scheduler (partial): The scheduler function. Defaults to a cosine scheduler with `num_cycles` set to 0.5 and `last_epoch` set to -1.
        extras (LRSchedulerExtrasConfig): The extras configuration for the scheduler. Defaults to an instance of `LRSchedulerExtrasConfig`.
    """
    scheduler = partial(
        get_scheduler,
        name = "cosine",
        scheduler_specific_kwargs = {
            'num_cycles': 0.5,
            'last_epoch': -1,
        }
    )

    interval: str = "step"
    warmup_ratio: float = 0.05

    #extras: LRSchedulerExtrasConfig = LRSchedulerExtrasConfig()

class MultiClassMetricsConfig:
    """
    A class for configuring the metrics used during model training and evaluation.

    Attributes:
        main_metric (Metric): The main metric used for model training.
        val_metric_best (Metric): The metric used for model validation.
        add_metrics (MetricCollection): A collection of additional metrics used during model training.
        eval_complete (MetricCollection): A collection of metrics used during model evaluation.
    """

    def __init__(
        self,
        num_labels: int = 21,
    ):
        """
        Initializes the MetricsConfig class.

        Args:
            num_labels (int): The number of labels in the dataset. Defaults to 21 as in the HSN dataset.
        """
        self.main_metric: Metric = cmAP(
            num_labels=num_labels,
            thresholds=None
        )
        self.val_metric_best: Metric = MaxMetric()
        self.add_metrics: MetricCollection = MetricCollection({})
        self.eval_complete: MetricCollection = MetricCollection({
            'cmAP': cmAP(
                num_labels=num_labels,
                thresholds=None
            ),
        })

class MetricsConfig:
    """
    A class for configuring the metrics used during model training and evaluation.

    Attributes:
        main_metric (Metric): The main metric used for model training.
        val_metric_best (Metric): The metric used for model validation.
        add_metrics (MetricCollection): A collection of additional metrics used during model training.
        eval_complete (MetricCollection): A collection of metrics used during model evaluation.
    """

    def __init__(
        self,
        num_labels: int = 21,
    ):
        """
        Initializes the MetricsConfig class.

        Args:
            num_labels (int): The number of labels in the dataset. Defaults to 21 as in the HSN dataset.
        """
        self.main_metric: Metric = cmAP(
            num_labels=num_labels,
            thresholds=None
        )
        self.val_metric_best: Metric = MaxMetric()
        self.add_metrics: MetricCollection = MetricCollection({
            'MultilabelAUROC': AUROC(
                task="multilabel",
                num_labels=num_labels,
                average='macro',
                thresholds=None
            ),
            'T1Accuracy': TopKAccuracy(topk= 1),
            'T3Accuracy': TopKAccuracy(topk= 3),
            'mAP': mAP(
                num_labels= 21,
                thresholds=None
            )  
        })
        self.eval_complete: MetricCollection = MetricCollection({
            'cmAP5': cmAP5(
                num_labels=num_labels,
                sample_threshold=5,
                thresholds=None
            ),
            'pcmAP': pcmAP(
                num_labels=num_labels,
                padding_factor=5,
                average="macro",
                thresholds=None
            )
        })

@dataclass
class LoggingParamsConfig:
    """
    A dataclass for configuring the logging parameters during model training.

    Attributes:
        on_step (bool): Whether to log metrics after each training step. Defaults to False.
        on_epoch (bool): Whether to log metrics after each epoch. Defaults to True.
        sync_dist (bool): Whether to synchronize the logging in a distributed setting. Defaults to False.
        prog_bar (bool): Whether to display a progress bar during training. Defaults to True.
    """
    on_step: bool = False
    on_epoch: bool = True
    sync_dist: bool = False
    prog_bar: bool = True         


class BaseModule(L.LightningModule):
    """
    BaseModule is a PyTorch Lightning module that serves as a base for all models.

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
        class_weights_loss (bool, optional): Whether to use class weights for the loss function.
        label_counts (int): The number of labels.
        num_gpus (int): The number of GPUs to use for training.
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
            task: Literal['multiclass', 'multilabel'] = "multilabel",
            class_weights_loss: Optional[bool] = None,
            label_counts: int = 21,
            num_gpus: int = 1
            ):

        super(BaseModule, self).__init__()
        self.network = network
        self.output_activation = output_activation
        # TODO: refactor load_loss
        # self.loss = load_loss(loss, class_weights_loss, label_counts)
        self.loss = loss
        self.optimizer = optimizer
        self.lr_scheduler = lr_scheduler
        self.warmup_ratio = 0.05
        self.metrics = metrics
        self.logging_params = logging_params

        # partial
        self.num_epochs = num_epochs
        self.batch_size = batch_size
        self.len_trainset = len_trainset
        self.task = task
        self.num_gpus = get_num_gpu(num_gpus)

        self.model = self.network.model

        # configure main metric
        self.train_metric = self.metrics.main_metric.clone()
        self.valid_metric = self.metrics.main_metric.clone()
        self.test_metric = self.metrics.main_metric.clone()
        # configure val_best metric
        self.valid_metric_best = self.metrics.val_metric_best.clone()
        # configure additional metrics
        self.valid_add_metrics = self.metrics.add_metrics.clone(prefix="val/")
        self.test_add_metrics = self.metrics.add_metrics.clone(prefix="test/")
        # configure eval_complete metrics
        self.test_complete_metrics = self.metrics.eval_complete.clone(prefix="test/")

        self.torch_compile = network.torch_compile
        self.model_name = network.model_name

        self.save_hyperparameters()

        self.test_targets = []
        self.test_preds = []
        self.class_mask = None

        # TODO: reimplement this
        if hasattr(network.model, 'pretran_info') and network.model.pretrain_info is not None:
            self.pretrain_dataset = network.model.pretrain_info["hf_pretrain_name"]
            self.hf_path = network.model.pretrain_info["hf_path"]
            self.hf_name = network.model.pretrain_info["hf_name"]
            pretrain_info = datasets.load_dataset_builder(self.hf_path, self.pretrain_dataset).info.features["ebird_code"]
            dataset_info = datasets.load_dataset_builder(self.hf_path, self.hf_name).info.features["ebird_code"]
            self.class_mask = [pretrain_info.names.index(i) for i in dataset_info.names]

    def forward(self, *args, **kwargs):
        return self.model.forward(*args, **kwargs)

    def configure_optimizers(self):
        self.optimizer = self.optimizer(self.model.parameters())
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

    def model_step(self, batch, batch_idx):
        logits = self.forward(**batch)
        if self.class_mask:
            logits = logits[:, self.class_mask]
        loss = self.loss(logits, batch["labels"])
        preds = self.output_activation(logits)
        return loss, preds, batch["labels"]

    def on_train_start(self):
        self.valid_metric_best.reset()

    def training_step(self, batch, batch_idx):
        train_loss, preds, targets = self.model_step(batch, batch_idx)
        self.log(
            f"train/{self.loss.__class__.__name__}",
            train_loss,
            on_step=True,
            on_epoch=True,
            prog_bar=True
        )

        # remove metrics from train to significantly improve training time for many classes
        # self.train_metric(preds, targets.int())
        # self.log(
        #     f"train/{self.train_metric.__class__.__name__}",
        #     self.train_metric,
        #     **asdict(self.logging_params)
        # )

        # self.train_add_metrics(preds, targets)
        # self.log_dict(self.train_add_metrics, **self.logging_params)

        return {"loss": train_loss}

    def validation_step(self, batch, batch_idx):
        val_loss, preds, targets = self.model_step(batch, batch_idx)
       
        self.log(
            f"val/{self.loss.__class__.__name__}",
            val_loss,
            on_step=True,
            on_epoch=True,
            prog_bar=True
        )

        self.valid_metric(preds, targets.int())
        self.log(
            f"val/{self.valid_metric.__class__.__name__}",
            self.valid_metric,
            **asdict(self.logging_params),
        )

        # self.valid_add_metrics(preds, targets.int())
        # self.log_dict(self.valid_add_metrics, **asdict(self.logging_params))
        return {"loss": val_loss, "preds": preds, "targets": targets}

    def on_validation_epoch_end(self):
        valid_metric = self.valid_metric.compute()  # get current valid metric
        self.valid_metric_best(valid_metric)  # update best so far valid metric

        self.log(
            f"val/{self.valid_metric.__class__.__name__}_best",
            self.valid_metric_best.compute(),
        )

    def test_step(self, batch, batch_idx):
        test_loss, preds, targets = self.model_step(batch, batch_idx)

        self.log(
            f"test/{self.loss.__class__.__name__}",
            test_loss,
            on_step=False,
            on_epoch=True,
            prog_bar=True
        )

        self.test_metric(preds, targets.int())
        self.log(
            f"test/{self.test_metric.__class__.__name__}",
            self.test_metric,
            **asdict(self.logging_params),
        )

        self.test_add_metrics(preds, targets.int())
        self.log_dict(self.test_add_metrics, **asdict(self.logging_params))

        return {"loss": test_loss, "preds": preds, "targets": targets}

    def setup(self, stage):
        if self.torch_compile and stage == "fit":
            print("COMPILE")
            self.model = torch.compile(self.model)

    def on_test_epoch_end(self):
        pass
