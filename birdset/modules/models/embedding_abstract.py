from abc import ABC, abstractmethod
from typing import Tuple
import torch

class EmbeddingModel(ABC):

    @abstractmethod
    def get_embeddings(
        self, input_tensor
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        pass
    