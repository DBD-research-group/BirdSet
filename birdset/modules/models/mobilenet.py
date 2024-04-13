from typing import Dict, Optional

import datasets
import torch
from torch import nn
from transformers import AutoConfig, MobileNetV2Model


class MobileNetClassifier(nn.Module):
    """
    MobileNet model for audio classification.

    Attributes:
        num_classes (int): The number of classes for the output layer.
        num_channels (int): The number of input channels.
        checkpoint (Optional[str]): Path to a checkpoint for loading pre-trained weights.
    """

    def __init__(
        self,
        num_classes: int,
        num_channels: int = 1,
        checkpoint: Optional[str] = None,
        local_checkpoint: Optional[str] = None,
        cache_dir: Optional[str] = None,
        pretrain_info: Optional[Dict] = None,
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

        if pretrain_info:
            self.hf_path = pretrain_info.hf_path
            self.hf_name = (
                pretrain_info.hf_name
                if not pretrain_info.hf_pretrain_name
                else pretrain_info.hf_pretrain_name
            )
            self.num_classes = len(
                datasets.load_dataset_builder(self.hf_path, self.hf_name)
                .info.features["ebird_code"]
                .names
            )
        else:
            self.hf_path = None
            self.hf_name = None
            self.num_classes = num_classes

        self.num_channels = num_channels
        self.checkpoint = checkpoint
        self.local_checkpoint = local_checkpoint
        self.cache_dir = cache_dir

        self.model = None

        self._initialize_model()

    def _initialize_model(self) -> nn.Module:
        """Initializes the MobileNet model based on specified attributes.

        Returns:
            nn.Module: The initialized ConvNext model.
        """

        state_dict = None

        if self.checkpoint:
            if self.local_checkpoint:
                state_dict = torch.load(self.local_checkpoint)["state_dict"]
                state_dict = {
                    key.replace("model.model.", ""): weight
                    for key, weight in state_dict.items()
                }

            self.model = MobileNetV2Model.from_pretrained(
                self.checkpoint,
                num_labels=self.num_classes,
                num_channels=self.num_channels,
                cache_dir=self.cache_dir,
                state_dict=state_dict,
                ignore_mismatched_sizes=True,
            )
        else:
            config = AutoConfig.from_pretrained(
                "facebook/convnext-base-224-22k",
                num_labels=self.num_classes,
                num_channels=self.num_channels,
            )
            self.model = MobileNetV2Model(config)

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
