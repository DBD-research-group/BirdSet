from torch import nn
import torch

class BirdSetModel(nn.Module):
    def __init__(
        self,
        num_classes: int | None,
        embedding_size: int,
        classifier: nn.Module | None = None,
        local_checkpoint: str = None,
        freeze_backbone: bool = False,
        preprocess_in_model: bool = False,
    ) -> None:
        super().__init__()
        self.num_classes = num_classes
        self.local_checkpoint = local_checkpoint
        self.freeze_backbone = freeze_backbone
        self.preprocess_in_model = preprocess_in_model

    def _preprocess(self, input_values: torch.Tensor) -> torch.Tensor:
        """
        Preprocessing for the input values is done in BETAs.py
        The waveform gets resampled to 16kHz, transformed into a fbank and then normalized.
        """
        return input_values