from typing import Dict, Literal, Optional, Union

import torch
from torch import nn
from torchvision.models import (
    efficientnet_b0,
    efficientnet_b1,
    efficientnet_b2,
    efficientnet_b3,
    efficientnet_b4,
    efficientnet_b5,
    efficientnet_b6,
    efficientnet_b7,
    efficientnet_v2_s,
    efficientnet_v2_m,
    efficientnet_v2_l,
)


EfficientNetVersion = Literal[
    "efficientnet_b0",
    "efficientnet_b1",
    "efficientnet_b2",
    "efficientnet_b3",
    "efficientnet_b4",
    "efficientnet_b5",
    "efficientnet_b6",
    "efficientnet_b7",
    "efficientnet_v2_s",
    "efficientnet_v2_m",
    "efficientnet_v2_l",
]


def load_state_dict(
    checkpoint_path: str, map_location: Optional[Union[str, torch.device]] = None
) -> Dict[str, torch.Tensor]:
    """
    Load a model checkpoint and return the state dictionary. The function is compatible with
    checkpoints from both plain PyTorch and PyTorch Lightning.

    Args:
        checkpoint_path (str): Path to the checkpoint file.
        map_location (Optional[Union[str, torch.device]]): Specifies how to remap storage locations
        (for CPU/GPU compatibility). It can be a string ('cpu', 'cuda'), or a torch.device object.

    Returns:
        Dict[str, torch.Tensor]: The state dictionary extracted from the checkpoint. The keys are layer names,
        and values are parameter tensors of the model.

    Raises:
        FileNotFoundError: If the checkpoint file does not exist at the given path.
        IOError: If the checkpoint file cannot be loaded for reasons other than non-existence.
    """
    # Load the checkpoint file. torch.load() automatically loads the tensor to the specified device
    # if map_location is provided.
    checkpoint = torch.load(checkpoint_path, map_location=map_location)

    # Extract the state dictionary from the checkpoint. The structure of the checkpoint can vary
    # depending on whether it's from plain PyTorch or PyTorch Lightning.
    if "state_dict" in checkpoint:
        # PyTorch Lightning checkpoints usually nest the model state dictionary under the 'state_dict' key.
        state_dict = checkpoint["state_dict"]
    else:
        # For plain PyTorch checkpoints, the file directly contains the state dictionary.
        state_dict = checkpoint

    return state_dict


def update_first_cnn_layer(model, num_channels):
    """
    Updates the first layer of the given CNN model to match the specified number of input channels.

    Args:
        model (nn.Module): The CNN model whose first layer is to be updated.
        num_channels (int): The number of input channels for the first layer.
    """
    # Access the first layer
    first_layer = list(model.features.children())[0][0]
    if first_layer.in_channels != num_channels:
        # Create a new first layer with the specified number of input channels
        new_first_layer = nn.Conv2d(
            in_channels=num_channels,
            out_channels=first_layer.out_channels,
            kernel_size=first_layer.kernel_size,
            stride=first_layer.stride,
            padding=first_layer.padding,
            bias=first_layer.bias is not None,
        )
        # Replace the old first layer with the new one
        model.features[0][0] = new_first_layer


class EfficientNetClassifier(nn.Module):
    """
    EfficientNet model for audio classification.

    Attributes:
        architecture (EfficientNetVersion): The version of EfficientNet to use.
        num_classes (int): The number of classes for the output layer.
        num_channels (int): The number of input channels.
        checkpoint (Optional[str]): Path to a checkpoint for loading pre-trained weights.
    """

    def __init__(
        self,
        architecture: EfficientNetVersion,
        num_classes: int,
        num_channels: int = 1,
        checkpoint: Optional[str] = None,
    ):
        """
        Initialize the EfficientNet model.

        Args:
        architecture (EfficientNetVersion): The version of the EfficientNet architecture.
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
        """Initializes the EfficientNet model based on specified attributes.

        Returns:
            nn.Module: The initialized EfficientNet model.
        """
        # Initialize model based on the backbone architecture
        if self.architecture == "efficientnet_b0":
            efficientnet_model = efficientnet_b0(
                pretrained=False, num_classes=self.num_classes
            )
        elif self.architecture == "efficientnet_b1":
            efficientnet_model = efficientnet_b1(
                pretrained=False, num_classes=self.num_classes
            )
        elif self.architecture == "efficientnet_b2":
            efficientnet_model = efficientnet_b2(
                pretrained=False, num_classes=self.num_classes
            )
        elif self.architecture == "efficientnet_b3":
            efficientnet_model = efficientnet_b3(
                pretrained=False, num_classes=self.num_classes
            )
        elif self.architecture == "efficientnet_b4":
            efficientnet_model = efficientnet_b4(
                pretrained=False, num_classes=self.num_classes
            )
        elif self.architecture == "efficientnet_b5":
            efficientnet_model = efficientnet_b5(
                pretrained=False, num_classes=self.num_classes
            )
        elif self.architecture == "efficientnet_b6":
            efficientnet_model = efficientnet_b6(
                pretrained=False, num_classes=self.num_classes
            )
        elif self.architecture == "efficientnet_b7":
            efficientnet_model = efficientnet_b7(
                pretrained=False, num_classes=self.num_classes
            )
        elif self.architecture == "efficientnet_v2_s":
            efficientnet_model = efficientnet_v2_s(
                pretrained=False, num_classes=self.num_classes
            )
        elif self.architecture == "efficientnet_v2_m":
            efficientnet_model = efficientnet_v2_m(
                pretrained=False, num_classes=self.num_classes
            )
        elif self.architecture == "efficientnet_v2_l":
            efficientnet_model = efficientnet_v2_l(
                pretrained=False, num_classes=self.num_classes
            )
        else:
            raise ValueError(f"Unsupported EfficientNet version: {self.architecture}")

        # Update the first layer to match num_channels if needed
        update_first_cnn_layer(model=efficientnet_model, num_channels=self.num_channels)

        # Load checkpoint if provided
        if self.checkpoint:
            state_dict = load_state_dict(self.checkpoint)
            efficientnet_model.load_state_dict(state_dict, strict=False)

        self.model = efficientnet_model

    def forward(
        self, input_values: torch.Tensor, labels: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        """
        Defines the forward pass of the EfficientNet model.

        Args:
            input_values (torch.Tensor): An input batch.
            labels (Optional[torch.Tensor]): The corresponding labels. Default is None.

        Returns:
            torch.Tensor: The output of the EfficientNet model.
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
