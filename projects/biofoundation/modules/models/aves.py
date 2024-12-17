import torch
import datasets
import torch.nn as nn
from typing import Tuple
from birdset.configs import PretrainInfoConfig
from torchaudio.models import wav2vec2_model
from birdset.modules.models.birdset_model import BirdSetModel
import json
from typing import Optional


class AvesClassifier(BirdSetModel):
    """
    Pretrained model for audio classification using the AVES model.

    This file includes code from AVES by Masato Hagiwara, licensed under the MIT License
    Copyright (c) 2022 Earth Species Project
    Github-Repository: https://github.com/earthspecies/aves
    Paper: https://arxiv.org/abs/2210.14493
    """

    EMBEDDING_SIZE = 768

    def __init__(
        self,
        num_classes: int = None,
        embedding_size: int = EMBEDDING_SIZE,
        local_checkpoint: str = None,
        freeze_backbone: bool = False,
        preprocess_in_model: bool = True,
        classifier: nn.Module | None = None,
        pretrain_info: PretrainInfoConfig = None,
    ):

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
        self.config = self.load_config("/workspace/models/aves/aves-base-bio.torchaudio.model_config.json")
        self.model = wav2vec2_model(**self.config, aux_num_out=None)
        self.model.load_state_dict(torch.load("/workspace//models/aves/aves-base-bio.torchaudio.pt"))
        self.model.feature_extractor.requires_grad_(True)

    def load_config(self, config_path):
        with open(config_path, "r") as ff:
            obj = json.load(ff)

        return obj

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
        
        input_values = input_values.squeeze(1)
        embeddings = self.model.extract_features(input_values)[0][-1]
        cls_state = embeddings[:, 0, :]

        return cls_state
