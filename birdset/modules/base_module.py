import torch
import math
import lightning as L
import datasets
from dataclasses import asdict
from functools import partial
from typing import Callable, List, Literal, Type, Optional, Union
from torch.nn import CrossEntropyLoss
from torch.nn.modules.loss import _Loss
from torch.optim import AdamW, Optimizer

from birdset.configs import NetworkConfig, LoggingParamsConfig, LRSchedulerConfig, MulticlassMetricsConfig, MultilabelMetricsConfig, MultilabelMetricsConfig as MetricsConfig


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

class BaseModule(L.LightningModule):
    """
    BaseModule is a PyTorch Lightning module that serves as a base for all models. The default parameters are used for the task of 'multiclass' classification. See MultiLabelModule for 'multilabel' classification.

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
            metrics: MulticlassMetricsConfig | MultilabelMetricsConfig = MulticlassMetricsConfig(),
            logging_params: LoggingParamsConfig = LoggingParamsConfig(),
            num_epochs: int = 50,
            len_trainset: int = 13878, # set to property from datamodule
            batch_size: int = 32,
            task: Literal['multiclass', 'multilabel'] = "multiclass",
            num_gpus: int = 1,
            pretrain_info = None,
            ):

        super().__init__()
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
        self.pretrain_info = pretrain_info

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
        if self.pretrain_info and self.pretrain_info.get("hf_pretrain_name"):
            print("Masking Logits")
            self.pretrain_dataset = self.pretrain_info["hf_pretrain_name"]
            self.hf_path = self.pretrain_info["hf_path"]
            self.hf_name = self.pretrain_info["hf_name"]
            pretrain_classlabels = datasets.load_dataset_builder(self.hf_path, self.pretrain_dataset).info.features["ebird_code"]
            dataset_classlabels = datasets.load_dataset_builder(self.hf_path, self.hf_name).info.features["ebird_code"]
            self.class_mask = [pretrain_classlabels.names.index(i) for i in dataset_classlabels.names]

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
        if self.class_mask and (not self.pretrain_info.valid_test_only or not self.trainer.training):
            if batch["labels"].shape == logits.shape:
                batch["labels"] = batch["labels"][:, self.class_mask]
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

        #! Comment out for performance increase
        self.valid_metric(preds, targets.int())
        self.log(
            f"val/{self.valid_metric.__class__.__name__}",
            self.valid_metric,
            **asdict(self.logging_params),
        )

        # self.valid_add_metrics(preds, targets.int())
        # self.log_dict(self.valid_add_metrics, **asdict(self.logging_params))
        return {"loss": val_loss, "preds": preds, "targets": targets}

    # def on_validation_epoch_end(self):
    #     valid_metric = self.valid_metric.compute()  # get current valid metric
    #     self.valid_metric_best(valid_metric)  # update best so far valid metric
    #
    #     self.log(
    #         f"val/{self.valid_metric.__class__.__name__}_best",
    #         self.valid_metric_best.compute(),
    #     )

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
        if self.task == "multiclass":
            self.test_add_metrics(preds, targets)
        else:    
            self.test_add_metrics(preds, targets.int())
            
        self.log_dict(self.test_add_metrics, **asdict(self.logging_params))
        return {"loss": test_loss, "preds": preds, "targets": targets}

    def setup(self, stage):
        if self.torch_compile and stage == "fit":
            print("COMPILE")
            self.model = torch.compile(self.model)

    def on_test_epoch_end(self):
        pass
