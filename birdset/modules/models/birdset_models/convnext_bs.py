import datasets
from torch import nn
import torch
import torchaudio
from torchvision import transforms
from birdset.modules.models.convnext import ConvNextClassifier
from birdset.datamodule.components.augmentations import PowerToDB


class ConvNextBirdSet(nn.Module):
    """
    BirdSet ConvNext model trained on BirdSet XCL dataset.
    The model expects a raw 1 channel 5s waveform with sample rate of 32kHz as an input.
    Its preprocess function will:
        - convert the waveform to a spectrogram: n_fft: 1024, hop_length: 320, power: 2.0
        - melscale the spectrogram: n_mels: 128, n_stft: 513
        - dbscale with top_db: 80
        - normalize the spectrogram mean: -4.268, std: 4.569 (from esc-50)
    """

    def __init__(
        self,
        num_classes=9736,
    ):
        super().__init__()
        self.model = ConvNextClassifier(
            checkpoint="DBD-research-group/ConvNeXT-Base-BirdSet-XCL",
            num_classes=num_classes,
        )
        self.spectrogram_converter = torchaudio.transforms.Spectrogram(
            n_fft=1024, hop_length=320, power=2.0
        )
        self.mel_converter = torchaudio.transforms.MelScale(
            n_mels=128, n_stft=513, sample_rate=32_000
        )
        self.normalizer = transforms.Normalize((-4.268,), (4.569,))
        self.powerToDB = PowerToDB(top_db=80)
        self.config = self.model.model.config

    def preprocess(self, waveform: torch.Tensor):
        # convert waveform to spectrogram
        spectrogram = self.spectrogram_converter(waveform)
        spectrogram = spectrogram.to(torch.float32)
        melspec = self.mel_converter(spectrogram)
        dbscale = self.powerToDB(melspec)
        normalized_dbscale = self.normalizer(dbscale)
        # add dimension 3 from left
        normalized_dbscale = normalized_dbscale.unsqueeze(-3)

        return normalized_dbscale

    def forward(self, input: torch.Tensor):
        # spectrogram = self.preprocess(waveform)
        return self.model(input)
