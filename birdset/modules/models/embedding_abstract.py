from abc import ABC, abstractmethod

class EmbeddingModel(ABC):

    @abstractmethod
    def get_embeddings(self, *args, **kwargs):
        pass
    