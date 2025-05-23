from typing import Literal
from biofoundation.modules.models.Pooling import AttentivePooling, AttentivePooling_old, AveragePooling
from biofoundation.modules.models.birdset_model import BirdSetModel
import torch
from torch import nn

from birdset.configs.model_configs import PretrainInfoConfig


class ViT(BirdSetModel):
    def __init__(
        self,
        num_classes: int | None,
        embedding_size: int,
        classifier: nn.Module | None = None,
        local_checkpoint: str = None,
        load_classifier_checkpoint: bool = True,
        freeze_backbone: bool = False,
        preprocess_in_model: bool = False,
        pretrain_info: PretrainInfoConfig = None,
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
            pretrain_info=pretrain_info
        )
        self.pooling_type = pooling
        if self.pooling_type == "attentive":
            attentive_heads = embedding_size // 64 # embedding_size // 64 should be 12 for 768
            self.attentive_pooling = AttentivePooling(
                dim=embedding_size, num_heads=attentive_heads
            )
        elif self.pooling_type == "attentive_old":
            attentive_heads = embedding_size // 8 # beats uses 8 heads
            self.attentive_pooling = AttentivePooling_old(
                embed_dim=embedding_size, num_heads=attentive_heads
            )
        elif self.pooling_type == "average":
            self.average_pooling = AveragePooling()
    
    def pool(self, embeddings: torch.Tensor, pooling) -> torch.Tensor:
        if pooling == "just_cls":
            # Use only the CLS token for classification
            # The CLS token is the first token in the sequence
            return embeddings[:, 0, :]
        elif pooling == "attentive":
            return self.attentive_pooling(embeddings)
        elif pooling == "attentive_old":
            return self.attentive_pooling(embeddings)
        elif pooling == "average":
            return self.average_pooling(embeddings)
        elif pooling == "mean":
            return torch.mean(embeddings, dim=1)
        else:
            raise ValueError(
                f"Pooling method '{pooling}' is not supported. Choose from 'just_cls', 'attentive', 'attentive_old', or 'average'."
            )
    
    def get_num_layers(self) -> int:
        """
        Returns the number of layers in the model.
        """
        return len(self.model.encoder.layers)