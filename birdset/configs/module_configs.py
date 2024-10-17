from dataclasses import dataclass
import torch.nn as nn
from typing import Literal
from transformers import get_scheduler
from functools import partial
from torchmetrics import AUROC, Accuracy, F1Score, Metric, MaxMetric, MetricCollection

from birdset.modules.metrics.multilabel import TopKAccuracy, cmAP, cmAP5, mAP, pcmAP
from birdset.modules.models.efficientnet import EfficientNetClassifier


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
        num_classes=21,
        num_channels=1,
        checkpoint=None,
        local_checkpoint=None,
        cache_dir=None,
        pretrain_info=None,
    )
    model_name: str = "efficientnet"
    model_type: Literal['vision', 'waveform'] = "vision"
    torch_compile: bool = False
    sample_rate: int = 32000
    input_length_in_s: int = 5 
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
        name="cosine",
        scheduler_specific_kwargs={
            'num_cycles': 0.5,
            'last_epoch': -1,
        }
    )

    interval: str = "step"
    warmup_ratio: float = 0.05

    # extras: LRSchedulerExtrasConfig = LRSchedulerExtrasConfig()

class MulticlassMetricsConfig:
    """
    A class for configuring multiclass metrics used during model training and evaluation.

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
        self.main_metric: Metric = Accuracy(
            task="multiclass",
            num_classes=num_labels,
        )
        self.val_metric_best: Metric = MaxMetric()
     
        self.add_metrics: MetricCollection = MetricCollection({
            'F1': F1Score(
                task="multiclass",
                num_classes=num_labels,
            ),
        })
        self.eval_complete: MetricCollection = MetricCollection({
            'acc': Accuracy(
            task="multiclass",
            num_classes=num_labels,
        )
        })

class MultilabelMetricsConfig:
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
                num_labels= num_labels,
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
