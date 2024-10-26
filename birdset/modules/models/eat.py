from typing import Optional, Tuple
import torch
from torch import nn
import torch.nn.functional as F
from torchaudio.compliance import kaldi
from birdset.modules.models.EAT.data2vecmultimodel import Data2VecMultiModel

class EATModel(nn.Module):
    """
    Pretrained model for audio classification using EAT model.
    
    """
    EMBEDDING_SIZE = 768
    MEAN = 0
    STD = 0.5

    def __init__(
            self,
            multimodel,
            modality,
            num_classes: int,
            train_classifier: bool = False,
        ) -> None:
        super().__init__()
        self.multimodel = multimodel
        self.modality = modality
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
        if self.train_classifier:
            for param in self.model.parameters():
                param.requires_grad = False


    def load_model(self) -> None:
        backbone = Data2VecMultiModel(multimodel=self.multimodel, modality=self.modality, skip_ema=True)

        checkpoint = torch.load('/workspace/models/eat_ssl/EAT-base_epoch30_ft.pt')['model']
        checkpoint = {k.replace('model.', ''): v for k, v in checkpoint.items()}
        checkpoint = {k.replace('modality_encoders.IMAGE', 'modality_encoder'): v for k, v in checkpoint.items()}

        missing_keys, unexpected_keys = backbone.load_state_dict(checkpoint, strict=False)
        print(f"Missing keys: {missing_keys}")
        print(f"Unexpected keys: {unexpected_keys}")
        # We don't need the decoder so it is fine that the keys are missing
        backbone.remove_pretrain_components()
        self.model = backbone 


    def preprocess(self, input_values: torch.Tensor) -> torch.Tensor:
        device = input_values.device
        melspecs = []
        for waveform in input_values:
            melspec = kaldi.fbank(waveform, htk_compat=True, window_type="hanning", num_mel_bins=128)  # shape (n_frames, 128)
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
        embeddings = self.get_embeddings(melspec)

        if self.train_classifier:
            flattend_embeddings = embeddings.reshape(embeddings.size(0), -1)
            # Pass embeddings through the classifier to get the final output
            output = self.classifier(flattend_embeddings)
        else:
            output = embeddings

        return output


    def get_embeddings(
        self, input_tensor: torch.Tensor
    ) -> Tuple[torch.Tensor, None]:
        """
        Get the embeddings and logits from the AUDIOMAE model.

        Args:
            input_tensor (torch.Tensor): The input tensor for the model.

        Returns:
            torch.Tensor: The embeddings from the model.
        """
        melspecs = self.preprocess(input_tensor)
        # Utterance level here
        result = self.model(melspecs, features_only=True, padding_mask=None,mask=False, remove_extra_tokens=False)
        embeddings = result['x']
        cls_state = embeddings[:, 0]
        return cls_state, None

