import datasets
from torch import nn
import torch
import torchaudio
from torchvision import transforms
from birdset.modules.models.efficientnet import EfficientNetClassifier
from birdset.datamodule.components.augmentations import PowerToDB


class EfficientNetBirdSet(nn.Module):
    """
    BirdSet EfficientNet model trained on BirdSet XCL dataset.
    The model expects a raw 1 channel 5s waveform with sample rate of 32kHz as an input.
    Its preprocess function will:
        - convert the waveform to a spectrogram: n_fft: 2048, hop_length: 256, power: 2.0
        - melscale the spectrogram: n_mels: 256, n_stft: 1025
        - dbscale with top_db: 80
        - normalize the spectrogram mean: -4.268, std: 4.569 (from AudioSet)
    """

    def __init__(
        self,
        num_classes=9736,
    ):
        super().__init__()
        self.model = EfficientNetClassifier(
            checkpoint="DBD-research-group/EfficientNet-B1-BirdSet-XCL",
            num_classes=num_classes,
        )
        self.powerToDB = PowerToDB(top_db=80)
        self.config = self.model.model.config

    def preprocess(self, waveform: torch.Tensor):
        # convert waveform to spectrogram
        spectrogram = torchaudio.transforms.Spectrogram(
            n_fft=2048, hop_length=256, power=2.0
        )(waveform)
        melspec = torchaudio.transforms.MelScale(
            n_mels=256, n_stft=1025, sample_rate=32_000
        )(spectrogram)
        dbscale = self.powerToDB(melspec)
        normalized_dbscale = transforms.Normalize((-4.268,), (4.569,))(dbscale)
        # add batch dimension if needed
        if normalized_dbscale.dim() == 3:
            normalized_dbscale = normalized_dbscale.unsqueeze(0)
        return normalized_dbscale

    def forward(self, input: torch.Tensor):
        # spectrogram = self.preprocess(waveform)
        return self.model(input)
