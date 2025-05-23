from typing import Optional, Literal
from biofoundation.modules.models.vit import ViT

import torch
from torch import nn


from biofoundation.modules.models.BEATs import BEATs, BEATsConfig

from birdset.utils import pylogger

log = pylogger.get_pylogger(__name__)


class BEATsModel(ViT):
    """
    Pretrained model for audio classification using the BEATs model.
    Expects a 1-channel 10s waveform input, all preprocessing is done in the network.
    """

    EMBEDDING_SIZE = 768
    MEAN = torch.tensor(-4.268)
    STD = torch.tensor(4.569)

    def __init__(
        self,
        num_classes: int | None,
        embedding_size: int = EMBEDDING_SIZE,
        local_checkpoint: str = None,
        checkpoint_path: str = '/workspace/models/beats/BEATs_iter3_plus_AS2M.pt',
        load_classifier_checkpoint: bool = True,
        freeze_backbone: bool = False,
        preprocess_in_model: bool = True,
        classifier: nn.Module | None = None,
        pretrain_info = None,
        pooling: Literal['just_cls', 'attentive', 'attentive_old', 'average', 'mean'] = "just_cls",
    ) -> None:
        super().__init__(
            num_classes=num_classes,
            embedding_size=embedding_size,
            classifier=classifier,
            local_checkpoint=local_checkpoint,
            load_classifier_checkpoint=load_classifier_checkpoint,
            freeze_backbone=freeze_backbone,
            preprocess_in_model=preprocess_in_model,
            pretrain_info=pretrain_info,
            pooling=pooling,
        )
        self.model = None  # Placeholder for the loaded model
        self.checkpoint_path = checkpoint_path
    
        self.load_model()
        if classifier is None:
            self.classifier = nn.Linear(embedding_size, num_classes)
        else:
            self.classifier = classifier

        if local_checkpoint:
            self._load_local_checkpoint()

        if freeze_backbone:
            for param in self.model.parameters():
                param.requires_grad = False

    def load_model(self) -> None:
        """
        Load the model from shared storage.
        """
        log.info(f">> Loading model from {self.checkpoint_path}")
        # load the pre-trained checkpoints
        checkpoint = torch.load(self.checkpoint_path)

        cfg = BEATsConfig(checkpoint["cfg"])
        self.model = BEATs(cfg)
        self.model.load_state_dict(checkpoint["model"])
        self.model.predictor = None  # remove the predictor head
        self.model.eval()

    def _preprocess(self, input_values: torch.Tensor) -> torch.Tensor:
        """
        Preprocessing for the input values is done in BETAs.py
        The waveform gets resampled to 16kHz, transformed into a fbank and then normalized.
        """
        return input_values

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
        embeddings = self.get_embeddings(input_values, self.pooling_type)
        # flattend_embeddings = embeddings.reshape(embeddings.size(0), -1)
        return self.classifier(embeddings)

    def get_embeddings(self, input_values: torch.Tensor, pooling_type) -> torch.Tensor:
        """
        Get the embeddings and logits from the BEATs model.

        Args:
            input_tensor (torch.Tensor): The input tensor for the model.

        Returns:
            torch.Tensor: The embeddings from the model.
        """
        if self.preprocess_in_model:
            input_values = self._preprocess(input_values)
        embeddings = self.model.extract_features(input_values)[
            0
        ]  # outputs a tensor of size 496x768
        return self.pool(embeddings, pooling_type)

    def get_num_layers(self) -> int:
        """
        Get the number of layers in the model.

        Returns:
            int: The number of layers in the model.
        """
        return len(self.model.encoder.layers)