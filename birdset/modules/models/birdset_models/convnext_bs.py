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
        self.powerToDB = PowerToDB(top_db=80)
        dataset_meta = datasets.load_dataset_builder(
            "dbd-research-group/BirdSet", "XCL"
        )
        self.class_names = dataset_meta.info.features["ebird_code"]

    def preprocess(self, waveform: torch.Tensor):
        # convert waveform to spectrogram
        spectrogram = torchaudio.transforms.Spectrogram(
            n_fft=1024, hop_length=320, power=2.0
        )(waveform)
        melspec = torchaudio.transforms.MelScale(
            n_mels=128, n_stft=513, sample_rate=32_000
        )(spectrogram)
        dbscale = self.powerToDB(melspec)
        normalized_dbscale = transforms.Normalize((-4.268,), (4.569,))(dbscale)
        # add batch dimension if needed
        if normalized_dbscale.dim() == 3:
            normalized_dbscale = normalized_dbscale.unsqueeze(0)
        return normalized_dbscale

    def forward(self, waveform: torch.Tensor):
        spectrogram = self.preprocess(waveform)
        return self.model(spectrogram)

    def get_ebird_code(self, logits: torch.Tensor):
        top5 = torch.topk(logits, 5, dim=1)
        print("Top 5 logits:", top5.values)
        print("Top 5 predicted classes:")
        print([self.class_names.int2str(i) for i in top5.indices.squeeze().tolist()])
        return self.class_names.int2str(torch.argmax(logits).item())
