from typing import Dict, Optional

import datasets
import torch
from torch import nn
from transformers import AutoConfig, ConvNextForImageClassification
from transformers.models.convnext.modeling_convnext import ConvNextModel
from birdset.configs import PretrainInfoConfig
from typing import Tuple
from torchaudio.compliance import kaldi
import torch.nn.functional as F


class ConvNextClassifier(nn.Module):
    """
    ConvNext model for audio classification.
    """

    def __init__(
        self,
        num_channels: int = 1,
        num_classes: Optional[int] = None,
        checkpoint: Optional[str] = None,
        local_checkpoint: Optional[str] = None,
        cache_dir: Optional[str] = None,
        pretrain_info: PretrainInfoConfig = None,
    ):
        """
        Note: Either num_classes or pretrain_info must be given
        Args:
            num_channels: Number of input channels.
            checkpoint: huggingface checkpoint path of any model of correct type
            num_classes: number of classification heads to be used in the model
            local_checkpoint: local path to checkpoint file
            cache_dir: specified cache dir to save model files at
            pretrain_info: hf_path and hf_name of info will be used to infer if num_classes is None
        """
        super().__init__()

        if pretrain_info:
            self.hf_path = pretrain_info.hf_path
            self.hf_name = (
                pretrain_info.hf_name
                if not pretrain_info.hf_pretrain_name
                else pretrain_info.hf_pretrain_name
            )
            if self.hf_path == "DBD-research-group/BirdSet":
                self.num_classes = len(
                    datasets.load_dataset_builder(self.hf_path, self.hf_name)
                    .info.features["ebird_code"]
                    .names
                )
            else:
                self.num_classes = num_classes
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

    def _initialize_model(self):
        """Initializes the ConvNext model based on specified attributes."""

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


class ConvNextEmbedding(nn.Module):
    """
    ConvNext model for audio classification.
    """

    MEAN = -4.2677393
    STD = 4.5689974

    def __init__(
        self,
        num_channels: int = 1,
        num_classes: Optional[int] = None,
        checkpoint: Optional[str] = None,
        local_checkpoint: Optional[str] = None,
        cache_dir: Optional[str] = None,
        pretrain_info: PretrainInfoConfig = None,
    ):
        """
        Note: Either num_classes or pretrain_info must be given
        Args:
            num_channels: Number of input channels.
            checkpoint: huggingface checkpoint path of any model of correct type
            local_checkpoint: local path to checkpoint file
            cache_dir: specified cache dir to save model files at
            pretrain_info: hf_path and hf_name of info will be used to infer if num_classes is None
        """
        super().__init__()

        if pretrain_info:
            self.hf_path = pretrain_info.hf_path
            self.hf_name = (
                pretrain_info.hf_name
                if not pretrain_info.hf_pretrain_name
                else pretrain_info.hf_pretrain_name
            )
        else:
            self.hf_path = None
            self.hf_name = None

        self.num_channels = num_channels
        self.checkpoint = checkpoint
        self.local_checkpoint = local_checkpoint
        self.cache_dir = cache_dir

        self.model = None

        self._initialize_model()

    def _initialize_model(self):
        """Initializes the ConvNext model based on specified attributes."""

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

            self.model = ConvNextModel.from_pretrained(
                self.checkpoint,
                num_channels=self.num_channels,
                cache_dir=self.cache_dir,
                state_dict=adjusted_state_dict,
                ignore_mismatched_sizes=True,
            )

        else:
            print("Using pretrained convnext")
            config = AutoConfig.from_pretrained(
                "facebook/convnext-base-224-22k",
                num_channels=self.num_channels,
            )
            self.model = ConvNextModel(config)

    def preprocess(
        self, input_values: torch.Tensor, input_tdim=500, sampling_rate=32000
    ) -> torch.Tensor:
        """
        Preprocesses the input values by applying mel-filterbank transformation.
        Args:
            input_values (torch.Tensor): Input tensor of shape (batch_size, num_samples).
            input_tdim (int): The number of frames to keep. Defaults to 500.
            sampling_rate (int): The sampling rate of the input tensor. Defaults to 16000.
        Returns:
            torch.Tensor: Preprocessed tensor of shape (batch_size, 1, num_mel_bins, num_frames).
        """
        device = input_values.device
        melspecs = []
        for waveform in input_values:
            melspec = kaldi.fbank(
                waveform,
                htk_compat=True,
                window_type="hanning",
                num_mel_bins=128,
                use_energy=False,
                sample_frequency=sampling_rate,
                frame_shift=10,
            )  # shape (n_frames, 128)
            # print(melspec.shape)
            if melspec.shape[0] < input_tdim:
                melspec = F.pad(melspec, (0, 0, 0, input_tdim - melspec.shape[0]))
            else:
                melspec = melspec[:input_tdim]
            melspecs.append(melspec)
        melspecs = torch.stack(melspecs).to(device)
        melspecs = melspecs.unsqueeze(1)  # shape (batch_size, 1, 128, 1024)
        melspecs = (melspecs - self.MEAN) / (self.STD * 2)
        return melspecs

    def get_embeddings(self, input_tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        input_tensor = self.preprocess(input_tensor)
        input_tensor = input_tensor.transpose(2, 3)
        output = self.model(input_tensor, output_hidden_states=True, return_dict=True)
        return output.pooler_output, None
