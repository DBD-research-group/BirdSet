from typing import Literal, Optional

import torch
from torch import nn
from torchvision.models import (
    mobilenet_v2,
    mobilenet_v3_small,
    mobilenet_v3_large,
)

from src.modules.models.efficientnet import load_state_dict, update_first_cnn_layer


MobileNetVersion = Literal[
    "mobilenet_v2",
    "mobilenet_v3_small",
    "mobilenet_v3_large",
]


class MobileNetClassifier(nn.Module):
    """
    MobileNet model for audio classification.

    Attributes:
        architecture (MobileNetVersion): The version of MobileNet to use.
        num_classes (int): The number of classes for the output layer.
        num_channels (int): The number of input channels.
        checkpoint (Optional[str]): Path to a checkpoint for loading pre-trained weights.
    """

    def __init__(
        self,
        architecture: MobileNetVersion,
        num_classes: int,
        num_channels: int = 1,
        checkpoint: Optional[str] = None,
    ):
        """
        Initialize the MobileNet model.

        Args:
        architecture (MobileNetVersion): The version of the MobileNet architecture.
        num_classes (int): The number of classes for classification.
        num_channels (int): The number of input channels. Default is 1.
        checkpoint (Optional[str]): Path to a checkpoint for loading pre-trained weights. Default is None.
        """
        super().__init__()

        self.architecture = architecture
        self.num_classes = num_classes
        self.num_channels = num_channels
        self.checkpoint = checkpoint

        self.model = None

        self._initialize_model()

    def _initialize_model(self) -> nn.Module:
        """Initializes the MobileNet model based on specified attributes.

        Returns:
            nn.Module: The initialized MobileNet model.
        """
        # Initialize model based on the backbone architecture
        if self.architecture == "mobilenet_v2":
            mobilenet_model = mobilenet_v2(
                pretrained=False, num_classes=self.num_classes
            )
        elif self.architecture == "mobilenet_v3_small":
            mobilenet_model = mobilenet_v3_small(
                pretrained=False, num_classes=self.num_classes
            )
        elif self.architecture == "mobilenet_v3_large":
            mobilenet_model = mobilenet_v3_large(
                pretrained=False, num_classes=self.num_classes
            )
        else:
            raise ValueError(f"Unsupported MobileNet version: {self.architecture}")

        # Update the first layer to match num_channels if needed
        update_first_cnn_layer(model=mobilenet_model, num_channels=self.num_channels)

        # Load checkpoint if provided
        if self.checkpoint:
            state_dict = load_state_dict(self.checkpoint)
            mobilenet_model.load_state_dict(state_dict, strict=False)

        self.model = mobilenet_model

    def forward(
        self, input_values: torch.Tensor, labels: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        """
        Defines the forward pass of the MobileNet model.

        Args:
            input_values (torch.Tensor): An input batch.
            labels (Optional[torch.Tensor]): The corresponding labels. Default is None.

        Returns:
            torch.Tensor: The output of the MobileNet model.
        """
        output = self.model(input_values)

        return output

    @torch.inference_mode()
    def get_logits(self, dataloader, device):
        pass

    @torch.inference_mode()
    def get_probas(self, dataloader, device):
        pass

    @torch.inference_mode()
    def get_representations(self, dataloader, device):
        pass
