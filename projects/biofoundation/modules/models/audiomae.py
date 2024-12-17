from typing import Optional, Tuple
import timm
import torch
from torch import nn
import torch.nn.functional as F
from torchaudio.compliance import kaldi
from birdset.modules.models.birdset_model import BirdSetModel
from birdset.configs import PretrainInfoConfig


class AudioMAEModel(BirdSetModel):
    """
    Pretrained model for audio classification using the AUDIOMAE model.
    Masked Autoencoders that Listen: https://arxiv.org/abs/2207.06405
    Pretrained weights from Huggingface: gaunernst/vit_base_patch16_1024_128.audiomae_as2m

    The model expect a 1D audio signale sampled with 16kHz and a length of 10s.
    """

    EMBEDDING_SIZE = 768
    MEAN = -4.2677393
    STD = 4.5689974

    def __init__(
        self,
        num_classes: int,
        embedding_size: int = EMBEDDING_SIZE,
        local_checkpoint: str = None,
        freeze_backbone: bool = False,
        preprocess_in_model: bool = True,
        classifier: nn.Module = None,
    ) -> None:
        super().__init__(
            num_classes=num_classes,
            embedding_size=embedding_size,
            local_checkpoint=local_checkpoint,
            freeze_backbone=freeze_backbone,
            preprocess_in_model=preprocess_in_model,
        )
        self.model = None  # Placeholder for the loaded model
        self.load_model()
        self.num_classes = num_classes

        if classifier is None:

            self.classifier = nn.Linear(
                in_features=self.EMBEDDING_SIZE, out_features=num_classes
            )
        else:
            self.classifier = classifier
        if local_checkpoint:
            state_dict = torch.load(local_checkpoint)["state_dict"]
            state_dict = {
                key.replace("model.model.", ""): weight
                for key, weight in state_dict.items()
            }
            self.model.load_state_dict(state_dict)

        if freeze_backbone:
            for param in self.model.parameters():
                param.requires_grad = False

    def load_model(self) -> None:
        """
        Load the model from Huggingface.
        """
        self.model = timm.create_model(
            "hf_hub:gaunernst/vit_base_patch16_1024_128.audiomae_as2m", pretrained=True
        )

        self.model.eval()

    def preprocess(self, input_values: torch.Tensor) -> torch.Tensor:

        device = input_values.device
        melspecs = []
        for waveform in input_values:
            melspec = kaldi.fbank(
                waveform, htk_compat=True, window_type="hanning", num_mel_bins=128
            )  # shape (n_frames, 128)
            if melspec.shape[0] < 1024:
                melspec = F.pad(melspec, (0, 0, 0, 1024 - melspec.shape[0]))
            else:
                melspec = melspec[:1024]
            melspecs.append(melspec)
        melspecs = torch.stack(melspecs).to(device)
        melspecs = melspecs.unsqueeze(1)  # shape (batch_size, 1, 128, 1024)
        melspecs = (melspecs - self.MEAN) / (self.STD * 2)
        return melspecs

    def forward(
        self, input_values: torch.Tensor, labels: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        """
        Forward pass through the model.

        Args:
            input_values (torch.Tensor): The input tensor for the classifier.
            labels (Optional[torch.Tensor]): The true labels for the input values. Default is None.

        Returns:
            torch.Tensor: The output of the classifier.
        """

        embeddings = self.get_embeddings(input_values)

        return self.classifier(embeddings)

    def get_embeddings(self, input_tensor: torch.Tensor) -> torch.Tensor:
        """
        Get the embeddings and logits from the AUDIOMAE model.

        Args:
            input_tensor (torch.Tensor): The input tensor for the model.

        Returns:
            torch.Tensor: The embeddings from the model.
        """
        if self.preprocess_in_model:
            input_values = self.preprocess(input_tensor)
        embeddings = self.model(input_values)

        return embeddings
