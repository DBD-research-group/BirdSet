from dataclasses import dataclass, field, asdict
from functools import partial
from typing import Callable, Dict, Literal, Type, Optional

from gadme.modules.metrics.multilabel import cmAP, cmAP5, pcmAP
from gadme.modules.models.efficientnet import EfficientNetClassifier
import torch
import math
import hydra

from gadme.modules.losses import load_loss
import datasets

import lightning as L
import torch.nn as nn
from torch.nn import BCEWithLogitsLoss
from torch.nn.modules.loss import _Loss
from torch.optim import AdamW, Optimizer, lr_scheduler 
from transformers import get_cosine_schedule_with_warmup
from torchmetrics import AUROC, Metric, MaxMetric, MetricCollection

@dataclass
class NetworkConfig:
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
#     interval: str = "step"
#     warmup_ratio: float = 0.5


@dataclass
class LRSchedulerConfig:
    scheduler: partial[Type[lr_scheduler.LRScheduler]] = partial(
        lr_scheduler.LambdaLR,
        lr_lambda=lambda epoch: epoch // 30
    )
    # extras = LRSchedulerExtrasConfig()


@dataclass
class MetricsConfig:
    main_metric: Metric = cmAP(
        num_labels = 21,
        thresholds = None
    )
    val_metric_best: Metric = MaxMetric()
    add_metrics: MetricCollection  = MetricCollection(
        metrics ={
        'MultilabelAUROC': AUROC(
            task="multilabel",
            num_labels=21,
            average='macro',
            thresholds=None
        )
        # TODO: more default metrics
    })
    eval_complete: MetricCollection = MetricCollection({
        'cmAP5': cmAP5(
            num_labels=21,
            sample_threshold=5,
            thresholds=None
        ),
        'pcmAP': pcmAP(
            num_labels=21,
            padding_factor=5,
            average="macro",
            thresholds=None
        )
    })

@dataclass
class LoggingParamsConfig:
    on_step: bool = False
    on_epoch: bool = True
    sync_dist: bool = False
    prog_bar: bool = True         




class BaseModule(L.LightningModule):
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
        self.metrics = metrics
        self.logging_params = logging_params

        # partial
        self.num_epochs = num_epochs
        self.batch_size = batch_size
        self.len_trainset = len_trainset
        self.task = task
        self.num_gpus = num_gpus



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
        # if "pretrain_info" in network.model and network.model.pretrain_info is not None:
        #     self.pretrain_dataset = network.model.pretrain_info["hf_pretrain_name"]
        #     self.hf_path = network.model.pretrain_info["hf_path"]
        #     self.hf_name = network.model.pretrain_info["hf_name"]
        #     pretrain_info = datasets.load_dataset_builder(self.hf_path, self.pretrain_dataset).info.features["ebird_code"]
        #     dataset_info = datasets.load_dataset_builder(self.hf_path, self.hf_name).info.features["ebird_code"]
        #     self.class_mask = [pretrain_info.names.index(i) for i in dataset_info.names]

    def forward(self, *args, **kwargs):
        return self.model.forward(*args, **kwargs)

    def configure_optimizers(self):
        self.optimizer = self.optimizer(self.model.parameters())
        if self.lr_scheduler is not None:
            num_training_steps = math.ceil((self.num_epochs * self.len_trainset) / self.batch_size * self.num_gpus)
            # TODO: Handle the case when drop_last=True more explicitly   

            # TODO: check if lr_scheduler can be called like this
            scheduler = self.lr_scheduler.scheduler(
                optimizer=self.optimizer,
                # num_warmup_steps=math.ceil(num_training_steps * self.lr_scheduler.extras.warmup_ratio),
                # num_training_steps=num_training_steps
            )
            # is_linear_warmup = scheduler_target == "transformers.get_linear_schedule_with_warmup"
            # is_cosine_warmup = scheduler_target == "transformers.get_cosine_schedule_with_warmup"

            # if is_linear_warmup or is_cosine_warmup:

            #     num_warmup_steps = math.ceil(
            #         num_training_steps * self.lrs_params.extras.warmup_ratio
            #     )

            #     scheduler_args = {
            #         "optimizer": self.optimizer,
            #         "num_warmup_steps": num_warmup_steps,
            #         "num_training_steps": num_training_steps,
            #         "_convert_": "partial"
            #     }
            # else:
            #     scheduler_args = {
            #         "optimizer": self.optimizer,
            #         "_convert_": "partial"
            #     }

            # instantiate hydra
            # scheduler = hydra.utils.instantiate(self.lrs_params.scheduler, **scheduler_args)
            lr_scheduler_dict = {"scheduler": scheduler}

            # if self.lr_scheduler.extras is not None and len(self.lr_scheduler.extras) > 0:
            #     for key, value in self.lr_scheduler.extras.items():
            #         lr_scheduler_dict[key] = value

            return {"optimizer": self.optimizer, "lr_scheduler": lr_scheduler_dict}

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

        self.train_metric(preds, targets.int())
        self.log(
            f"train/{self.train_metric.__class__.__name__}",
            self.train_metric,
            **asdict(self.logging_params)
        )

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
