from typing import Optional, Tuple
import timm
import torch
from torch import nn
from torchaudio.compliance import kaldi



class AudioMAEModel(nn.Module)
    """
    Pretrained model for audio classification using the AUDIOMAE model.
    Masked Autoencoders that Listen: https://arxiv.org/abs/2207.06405
    Pretrained weights from Huggingface: gaunernst/vit_base_patch16_1024_128.audiomae_as2m

    The model expect a 1D audio signale sampled with 16kHz and a length of 10s.
    """
    EMBEDDING_SIZE = 767
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


    def load_model(self) -> None:
        """
        Load the model from Huggingface.
        """
        self.model = timm.create_model("hf_hub:gaunernst/vit_base_patch16_1024_128.audiomae_as2m_ft", pretrained=True)

        self.model.eval()
    
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
        # If there's an extra channel dimension, remove it
        if input_values.dim() > 2:
            input_values = input_values.squeeze(1)
        
        # Convert input to mel spectrogram
        melspec = kaldi.fbank(input_values, htk_compat=True, window_type="hanning", num_mel_bins=128)  # shape (n_frames, 128)

        # AudioMAE only accepts 1024-frame input
        if melspec.shape[0] < 1024:
            melspec = F.pad(melspec, (0, 0, 0, 1024 - melspec.shape[0]))
        else:
            melspec = melspec[:1024]
        melspec = (melspec - self.MEAN) / (self.STD * 2)

        melspec = melspec.view(1, 1, 1024, 128)  # add batch dim and channel dim
        embeddings = self.model(melspec)

        if self.train_classifier:
            # Pass embeddings through the classifier to get the final output
            output = self.classifier(embeddings)
        else:
            output = embeddings

        return output


