from typing import Optional, Tuple
import timm
import torch
from torch import nn
import torch.nn.functional as F
import torchaudio.transforms as T


class AudioMAEProcessor:
    def __init__(self, mean, std, device='cuda:0'):
        self.MEAN = mean
        self.STD = std
        self.device = device  # Store the device information
        self.mel_spectrogram = T.MelSpectrogram(
            sample_rate=16000,  # Assuming a sample rate of 16000 Hz
            n_fft=400,
            win_length=400,
            hop_length=160,
            center=False,
            pad=0,
            window_fn=torch.hann_window,  # Move window to the specified device
            n_mels=128,
            power=2.0,
            normalized=False,
        )  

    def process_batch(self, input_values):
        # get device of input tensor
        device = input_values.device
        self.mel_spectrogram.to(device)

        # Compute mel spectrogram for the batch
        melspec = self.mel_spectrogram(input_values)  # shape (batch_size, 128, n_frames)

        # Pad or truncate to 1024 frames
        max_frames = 1024
        current_frames = melspec.shape[-1]
        if current_frames < max_frames:
            padding = max_frames - current_frames
            melspec = torch.nn.functional.pad(melspec, (0, padding))
        else:
            melspec = melspec[:, :, :max_frames]
        
        # transform from (batch_size, 1, 128, 1024) to (batchsize, 1, 1024, 128) (swap two last columns)
        melspec = melspec.transpose(2, 3)


        # Normalize
        melspec = (melspec - self.MEAN) / (self.STD * 2)

        return melspec




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
        self.preprocessor = AudioMAEProcessor(mean=self.MEAN, std=self.STD, device='cuda:0')
         # Define a linear classifier to use on top of the embeddings
        # self.classifier = nn.Linear(
        #     in_features=self.EMBEDDING_SIZE, out_features=num_classes
        # )
        self.classifier = nn.Sequential(
            nn.Linear(self.EMBEDDING_SIZE, 128),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Linear(64, self.num_classes),
        )


    def load_model(self) -> None:
        """
        Load the model from Huggingface.
        """
        self.model = timm.create_model("hf_hub:gaunernst/vit_base_patch16_1024_128.audiomae_as2m", pretrained=True)

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
        melspec = self.preprocessor.process_batch(input_values)
        embeddings = self.model(melspec)

        if self.train_classifier:
            # Pass embeddings through the classifier to get the final output
            output = self.classifier(embeddings)
        else:
            output = embeddings

        return output

    def get_embeddings(
        self, input_tensor: torch.Tensor
    ) -> torch.Tensor:
        """
        Get the embeddings and logits from the AUDIOMAE model.

        Args:
            input_tensor (torch.Tensor): The input tensor for the model.

        Returns:
            torch.Tensor: The embeddings from the model.
        """
        melspecs = self.preprocessor.process_batch(input_tensor)
        embeddings = self.model(melspecs)
        return embeddings

