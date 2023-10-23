from typing import List, Literal, Tuple

import pytorch_lightning as pl
import torch
import torchmetrics
from torch import nn
import torchvision

from src.models.base_module import BaseModule

ResNetVersion = Literal["resnet18", "resnet34", "resnet50", "resnet101", "resnet152"]

class LightningResNet(BaseModule):
    def __init__(
        self,
        baseline_architecture: ResNetVersion,
        num_classes: int,
        num_channels: int = 1,
        learning_rate: float = 1e-3,
        pretrained: bool = False,
    ):
        """Initialize the ResNet benchmark model.

        Args:
            baseline_architecture (ResNetVersion): Defines the architecture of the ResNet model.
            num_classes (int): Number of classes in the classification task.
            num_channels (int): Number of input channels in the first convolutional layer.
            learning_rate (float): Learning rate for the optimizer. Defaults to 1e-3.
            pretrained (bool): Specifies whether to initialize the model with or without pretrained weights.
        """
        super().__init__()

        self.baseline_architecture = baseline_architecture
        self.num_classes = num_classes
        self.num_channels = num_channels
        self.learning_rate = learning_rate

        # Instantiate loss criterion
        self.criterion = nn.CrossEntropyLoss()

        # Use accuracy as another evaluation metric
        self.acc = torchmetrics.Accuracy(task="multiclass", num_classes=num_classes)

        # Available resnet versions
        resnet_versions = {
            "resnet18": torchvision.models.resnet18,
            "resnet34": torchvision.models.resnet34,
            "resnet50": torchvision.models.resnet50,
            "resnet101": torchvision.models.resnet101,
            "resnet152": torchvision.models.resnet152,
        }

        # Using the resnet backbone
        # TODO: Customize this to read pre-trained weights from a specified directory
        if pretrained:
            resnet_weights = {
                "resnet18": torchvision.models.ResNet18_Weights.DEFAULT,
                "resnet34": torchvision.models.ResNet34_Weights.DEFAULT,
                "resnet50": torchvision.models.ResNet50_Weights.DEFAULT,
                "resnet101": torchvision.models.ResNet101_Weights.DEFAULT,
                "resnet152": torchvision.models.ResNet152_Weights.DEFAULT,
            }
            weights = resnet_weights[baseline_architecture]
        else:
            weights = None

        resnet_model = resnet_versions[baseline_architecture](weights=weights)

        # Replace the old FC layer with Identity, so we can train our own
        linear_size = list(resnet_model.children())[-1].in_features
        # Replace the final layer for fine-tuning (classification into num_classes classes)
        resnet_model.fc = nn.Linear(linear_size, num_classes)

        # Manually set the number of channels in the first Conv layer according to the shape of our input
        resnet_model.conv1 = nn.Conv2d(
            num_channels, 64, kernel_size=7, stride=2, padding=3, bias=False
        )

        self.model = resnet_model

    def forward(self, batch: torch.Tensor) -> torch.Tensor:
        """Defines the forward pass, i.e., the computation performed on each call.

        Args:
            batch (torch.Tensor): A batch of input data.

        Returns:
            torch.Tensor: The output tensor produced by the ResNet model.
        """
        return self.model(batch)

    def configure_optimizers(
        self,
    ) -> Tuple[
        List[torch.optim.Optimizer], List[torch.optim.scheduler._LRScheduler]
    ]:
        """Configure the optimizer and learning rate scheduler.

        Returns:
            Tuple[List[torch.optim.Optimizer], List[torch.optim.scheduler._LRScheduler]]: The optimizer and
            scheduler.
        """
        optimizer = torch.optim.Adam(self.parameters(), lr=self.learning_rate)
        scheduler = {
            "scheduler": torch.optim.scheduler.ReduceLROnPlateau(optimizer),
            "monitor": "val_loss",
        }
        return [optimizer], [scheduler]

    def training_step(
        self, batch: Tuple[torch.Tensor, torch.Tensor], batch_idx: int
    ) -> torch.Tensor:
        """Defines a single training step.

        Args:
            batch (Tuple[torch.Tensor, torch.Tensor]): A tuple containing input data and corresponding labels.
            batch_idx (int): The index of the current batch.

        Returns:
            torch.Tensor: The loss value calculated during the training step.
        """
        x, y = batch
        preds = self(x)

        loss = self.criterion(preds, y)
        acc = self.acc(preds, y)

        # Perform logging
        self.log("train_loss", loss, on_epoch=True, prog_bar=True, logger=True)
        self.log("train_acc", acc, on_epoch=True, prog_bar=True, logger=True)
        return loss

    def validation_step(self, batch: Tuple[torch.Tensor, torch.Tensor], batch_idx: int):
        """Defines a single validation step.

        Args:
            batch (Tuple[torch.Tensor, torch.Tensor]): A tuple containing input data and corresponding labels.
            batch_idx (int): The index of the current batch.
        """
        x, y = batch
        preds = self(x)

        loss = self.criterion(preds, y)
        acc = self.acc(preds, y)

        # Perform logging
        self.log("val_loss", loss, on_epoch=True, prog_bar=True, logger=True)
        self.log("val_acc", acc, on_epoch=True, prog_bar=True, logger=True)

    def test_step(self, batch: Tuple[torch.Tensor, torch.Tensor], batch_idx: int):
        """Defines a single test step.

        Args:
            batch (Tuple[torch.Tensor, torch.Tensor]): A tuple containing input data and corresponding labels.
            batch_idx (int): The index of the current batch.
        """
        x, y = batch
        preds = self(x)

        loss = self.criterion(preds, y)
        acc = self.acc(preds, y)

        # Perform logging
        self.log("test_loss", loss, prog_bar=True, logger=True)
        self.log("test_acc", acc, prog_bar=True, logger=True)
