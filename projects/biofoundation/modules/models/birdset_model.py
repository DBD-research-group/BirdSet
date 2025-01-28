import datasets
from torch import nn
import torch
from birdset.configs import PretrainInfoConfig


class BirdSetModel(nn.Module):
    def __init__(
        self,
        num_classes: int | None,
        embedding_size: int,
        classifier: nn.Module | None = None,
        local_checkpoint: str = None,
        freeze_backbone: bool = False,
        preprocess_in_model: bool = False,
        pretrain_info: PretrainInfoConfig = None,
    ) -> None:
        super().__init__()
        self.num_classes = num_classes
        self.local_checkpoint = local_checkpoint
        self.freeze_backbone = freeze_backbone
        self.preprocess_in_model = preprocess_in_model
        self.classifier = classifier
        self.embedding_size = embedding_size

        if pretrain_info:
            self.hf_path = pretrain_info.hf_path
            self.hf_name = (
                pretrain_info.hf_name
                if not pretrain_info.hf_pretrain_name
                else pretrain_info.hf_pretrain_name
            )
            if self.hf_path == "DBD-research-group/BirdSet":
                self.num_classes = len(
                    datasets.load_dataset_builder(self.hf_path, self.hf_name)
                    .info.features["ebird_code"]
                    .names
                )
            else:
                self.num_classes = num_classes
        else:
            self.hf_path = None
            self.hf_name = None
            self.num_classes = num_classes

    def _preprocess(self, input_values: torch.Tensor) -> torch.Tensor:
        return input_values
