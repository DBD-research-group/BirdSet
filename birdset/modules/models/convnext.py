from typing import Dict, Optional

import datasets
import torch
from torch import nn
from transformers import AutoConfig, ConvNextForImageClassification


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
        num_classes: int,
        num_channels: int = 1,
        checkpoint: Optional[str] = None,
        local_checkpoint: Optional[str] = None,
        cache_dir: Optional[str] = None,
        pretrain_info: Optional[Dict] = None,
    ):
        """
        Initialize the ConvNext model.

        Args:
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
        """Initializes the ConvNext model based on specified attributes.

        Returns:
            nn.Module: The initialized ConvNext model.
        """

        adjusted_state_dict = None

        if self.checkpoint:
            if self.local_checkpoint:
                state_dict = torch.load(self.local_checkpoint)["state_dict"]

                # Update this part to handle the necessary key replacements
                adjusted_state_dict = {}
                for key, value in state_dict.items():
                    # Handle 'model.model.' prefix
                    new_key = key.replace("model.model.", "")

                    # Handle 'model._orig_mod.model.' prefix
                    new_key = new_key.replace("model._orig_mod.model.", "")

                    # Assign the adjusted key
                    adjusted_state_dict[new_key] = value

            self.model = ConvNextForImageClassification.from_pretrained(
                self.checkpoint,
                num_labels=self.num_classes,
                num_channels=self.num_channels,
                cache_dir=self.cache_dir,
                state_dict=adjusted_state_dict,
                ignore_mismatched_sizes=True,
            )
        else:
            config = AutoConfig.from_pretrained(
                "facebook/convnext-base-224-22k",
                num_labels=self.num_classes,
                num_channels=self.num_channels,
            )
            self.model = ConvNextForImageClassification(config)

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
        logits = output.logits

        return logits

    @torch.inference_mode()
    def get_logits(self, dataloader, device):
        pass

    @torch.inference_mode()
    def get_probas(self, dataloader, device):
        pass

    @torch.inference_mode()
    def get_representations(self, dataloader, device):
        pass
