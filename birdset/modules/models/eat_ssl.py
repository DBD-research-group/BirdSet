from typing import Optional, Tuple
import torch
from torch import nn
import torch.nn.functional as F
from torchaudio.compliance import kaldi
from birdset.modules.models.EAT.data2vecmultimodel import Data2VecMultiModel
from birdset.modules.models.birdset_model import BirdSetModel

class EATSSL(BirdSetModel):
    """
    Pretrained model for audio classification using the Efficient Audio Transformer (EAT) model.
    
    This file and the EAT folder includes code that is based on EAT by Wenxi Chen, licensed under the MIT License
    Copyright (c) 2024 Wenxi Chen
    Github-Repository: https://github.com/cwx-worst-one/EAT 
    Paper: https://arxiv.org/abs/2401.03497

    We use a modified version of the EAT implementation that only relies on small local fairseq files and is compatible with Pytorch Lightning.
    This adaptation is by Paul Hahn and is also licensed under the MIT License.
    Github-Repository: https://github.com/nhaH-luaP/PyEat

    Important Parameters:
    ---------------------
    checkpoint: The path to the checkpoint to be loaded.
    multimodel: The settings for the Data2vec multimodel to be used in the model. This should best be defined in a hydra yaml.
    modality: The settings for the Image Encoder to be used in the model. This should best be defined in a hydra yaml.
    num_classes: Number of classification heads to be used in the model.
    train_classifier: If True, the model will output the embeddings and freeze the feature extractor. Default is False. 
    """
    EMBEDDING_SIZE = 768
    MEAN = torch.tensor(-4.268)
    STD = torch.tensor(4.569)



    def __init__(
        self,
        checkpoint,
        multimodel,
        modality, 
        num_classes: int | None,
        embedding_size: int = EMBEDDING_SIZE,
        local_checkpoint: str = None,
        freeze_backbone: bool = False,
        preprocess_in_model: bool = True,
        classifier: nn.Module | None = None,
    ) -> None:
        super().__init__(
            num_classes=num_classes,
            embedding_size=embedding_size,
            local_checkpoint=local_checkpoint,
            freeze_backbone=freeze_backbone,
            preprocess_in_model=preprocess_in_model,
        )
        self.checkpoint = checkpoint
        self.multimodel = multimodel
        self.modality = modality
        self.model = None  # Placeholder for the loaded model
        self.load_model()
        
         # Define a linear classifier to use on top of the embeddings
        if classifier is None:
            self.classifier = nn.Linear(embedding_size, num_classes)
        else:
            self.classifier = classifier
        
        if local_checkpoint:
            self._load_local_checkpoint()
            
        # freeze the model
        if freeze_backbone:
            for param in self.model.parameters():
                param.requires_grad = False


    def load_model(self) -> None:
        """
        Load the model by using the Data2VecMultiModel and loading a local checkpoint. The decoder is not needed to extract features so we remove it and ignore its weights from the checkpoint.
        """
        backbone = Data2VecMultiModel(multimodel=self.multimodel, modality=self.modality, skip_ema=True)

        checkpoint = torch.load(self.checkpoint)['model']
        checkpoint = {k.replace('model.', ''): v for k, v in checkpoint.items()}
        checkpoint = {k.replace('modality_encoders.IMAGE', 'modality_encoder'): v for k, v in checkpoint.items()}

        missing_keys, unexpected_keys = backbone.load_state_dict(checkpoint, strict=False)
        print(f"Missing keys: {missing_keys}")
        print(f"Unexpected keys: {unexpected_keys}")
        # We don't need the decoder so it is fine that the keys are missing
        backbone.remove_pretrain_components()
        self.model = backbone 


    def preprocess(self, input_values: torch.Tensor) -> torch.Tensor:
        """
        Preprocesses the input values by applying mel-filterbank transformation. Similar as function for AudioMae, ConvNeXt and SSAST.
        Args:
            input_values (torch.Tensor): Input tensor of shape (batch_size, num_samples).
        Returns:
            torch.Tensor: Preprocessed tensor of shape (batch_size, 1, num_mel_bins, num_frames).
        """
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
        embeddings = self.get_embeddings(input_values)

        return self.classifier(embeddings)


    def get_embeddings(
        self, input_tensor: torch.Tensor) -> torch.Tensor:
        """
        Get the embeddings and logits from the AUDIOMAE model.

        Args:
            input_tensor (torch.Tensor): The input tensor for the model.

        Returns:
            torch.Tensor: The embeddings from the model.
        """
        if self.preprocess_in_model:
            input_values = self.preprocess(input_tensor)

        # Utterance level here
        result = self.model(input_values, features_only=True, padding_mask=None,mask=False, remove_extra_tokens=False)
        embeddings = result['x']
        cls_state = embeddings[:, 0,:]
        return cls_state

