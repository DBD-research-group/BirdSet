from typing import Literal, Optional

import torch
from torch import nn
from torchvision.models import (
    convnext_tiny,
    convnext_small,
    convnext_base,
    convnext_large,
)

from gadme.modules.models.efficientnet import generate_state_dict, update_first_cnn_layer


ConvNextVersion = Literal[
    "convnext_tiny",
    "convnext_small",
    "convnext_base",
    "convnext_large",
]


class ConvNextClassifier(nn.Module):
    """
    ConvNext model for audio classification.

    Attributes:
        architecture (ConvNextVersion): The version of ConvNext to use.
        num_classes (int): The number of classes for the output layer.
        num_channels (int): The number of input channels.
        checkpoint (Optional[str]): Path to a checkpoint for loading pre-trained weights.
    """

    def __init__(
        self,
        architecture: ConvNextVersion,
        num_classes: int,
        num_channels: int = 1,
        checkpoint: Optional[str] = None,
    ):
        """
        Initialize the ConvNext model.

        Args:
        architecture (ConvNextVersion): The version of the ConvNext architecture.
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
        """Initializes the ConvNext model based on specified attributes.

        Returns:
            nn.Module: The initialized ConvNext model.
        """
        # Initialize model based on the backbone architecture
        if self.architecture == "convnext_tiny":
            convnext_model = convnext_tiny(
                pretrained=False, num_classes=self.num_classes
            )
        elif self.architecture == "convnext_small":
            convnext_model = convnext_small(
                pretrained=False, num_classes=self.num_classes
            )
        elif self.architecture == "convnext_base":
            convnext_model = convnext_base(
                pretrained=False, num_classes=self.num_classes
            )
        elif self.architecture == "convnext_large":
            convnext_model = convnext_large(
                pretrained=False, num_classes=self.num_classes
            )
        else:
            raise ValueError(f"Unsupported ConvNext version: {self.architecture}")

        # Update the first layer to match num_channels if needed
        update_first_cnn_layer(model=convnext_model, num_channels=self.num_channels)

        # Load checkpoint if provided
        if self.checkpoint:
            state_dict = load_state_dict(self.checkpoint)
            convnext_model.load_state_dict(state_dict, strict=False)

        self.model = convnext_model

    def forward(
        self, input_values: torch.Tensor, labels: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        """
        Defines the forward pass of the ConvNext model.

        Args:
            input_values (torch.Tensor): An input batch.
            labels (Optional[torch.Tensor]): The corresponding labels. Default is None.

        Returns:
            torch.Tensor: The output of the ConvNext model.
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
