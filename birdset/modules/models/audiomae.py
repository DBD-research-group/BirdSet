from typing import Optional, Tuple
import timm
import torch
from torch import nn
import torch.nn.functional as F
from torchaudio.compliance import kaldi


class AudioMAEModel(nn.Module):
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
        train_classifier: bool = False,
    ) -> None:
        super().__init__()
        self.model = None  # Placeholder for the loaded model
        self.load_model()
        self.num_classes = num_classes
        self.train_classifier = train_classifier
        # Define a linear classifier to use on top of the embeddings
        self.classifier = nn.Linear(
            in_features=self.EMBEDDING_SIZE, out_features=num_classes
        )
        # self.classifier = nn.Sequential(
        #     nn.Linear(self.EMBEDDING_SIZE, 128),
        #     nn.ReLU(),
        #     nn.Dropout(0.5),
        #     nn.Linear(128, 64),
        #     nn.ReLU(),
        #     nn.Linear(64, self.num_classes),
        # )
        # freeze the model
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
        melspec = self.preprocess(input_values)
        embeddings = self.model(melspec)

        if self.train_classifier:
            # Pass embeddings through the classifier to get the final output
            output = self.classifier(embeddings)
        else:
            output = embeddings

        return output

    def get_embeddings(self, input_tensor: torch.Tensor) -> Tuple[torch.Tensor, None]:
        """
        Get the embeddings and logits from the AUDIOMAE model.

        Args:
            input_tensor (torch.Tensor): The input tensor for the model.

        Returns:
            torch.Tensor: The embeddings from the model.
        """
        melspecs = self.preprocess(input_tensor)
        embeddings = self.model(melspecs)
        return embeddings, None
