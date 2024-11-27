from torch import nn


class BirdSetModel(nn.Module):
    def __init__(
        self,
        num_classes: int | None,
        embedding_size: int,
        classifier: nn.Module | None = None,
        local_checkpoint: str = None,
        freeze_backbone: bool = False,
        preprocess_in_model: bool = False,
    ) -> None:
        super().__init__()
        self.num_classes = num_classes
        self.local_checkpoint = local_checkpoint
        self.freeze_backbone = freeze_backbone
        self.preprocess_in_model = preprocess_in_model
