from typing import Optional


from birdset.configs.model_configs import PretrainInfoConfig
from birdset.modules.models.BEATs import BEATs, BEATsConfig
from birdset.modules.models.birdset_model import BirdSetModel
import torch
from torch import nn


class BEATsModel(BirdSetModel):
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
        freeze_backbone: bool = False,
        preprocess_in_model: bool = True,
        classifier: nn.Module | None = None,
        pretrain_info: Optional[PretrainInfoConfig] = None,
    ) -> None:
        super().__init__(
            num_classes=num_classes,
            embedding_size=embedding_size,
            local_checkpoint=local_checkpoint,
            freeze_backbone=freeze_backbone,
            preprocess_in_model=preprocess_in_model,
            pretrain_info=pretrain_info,
        )
        self.model = None  # Placeholder for the loaded model
        self.load_model()
        if classifier is None:
            self.classifier = nn.Linear(embedding_size, num_classes)
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
        Load the model from shared storage.
        """
        # load the pre-trained checkpoints
        checkpoint = torch.load("/workspace/models/beats/BEATs_iter3_plus_AS2M.pt")

        cfg = BEATsConfig(checkpoint["cfg"])
        self.model = BEATs(cfg)
        self.model.load_state_dict(checkpoint["model"])
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
        embeddings = self.get_embeddings(input_values)
        # flattend_embeddings = embeddings.reshape(embeddings.size(0), -1)
        return self.classifier(embeddings)

    def get_embeddings(self, input_values: torch.Tensor) -> torch.Tensor:
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
        cls_state = embeddings[:, 0, :]

        return cls_state
