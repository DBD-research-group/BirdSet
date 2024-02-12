from typing import Any
import enums as e
import logging
from datasets import Dataset, DatasetDict

from dataset.embeddings.ds_saver import DsSaver
from dataset.embeddings.embedding_backend_model import EmbedModel

class EmbeddingCreator():
    def __init__(self, embedding_model) -> None:
        self.embedding_model = embedding_model
    
    def embed(self, x):
        x = x["input_values"]
        embeddings = self.embedding_model(x)
        return {"embeddings": embeddings}
    
    def __call__(self, x, *args: Any, **kwds: Any) -> Any:
        return self.embed(x)


class BaseEmbeddingModule():
    def __init__(self) -> None:
        pass
    
    def run(self, datamodule):
        return None



class EmbeddingModule(BaseEmbeddingModule):
    def __init__(self, backend: e.BACKEND, device: e.DEVICE, save_config:DsSaver, embedding:EmbedModel) -> None:
        self.backend = backend
        self.device = device
        self.embedding = embedding
        self.save_config = save_config
    
    def prepare_splits(self, datamodule):
        train = datamodule.train_dataset
        test = datamodule.test_dataset
        test5s = datamodule.test_5s_dataset
        return train, test, test5s
    
    def run(self, datamodule):
        if datamodule is None:
            logging.error("Datamodule not initialized!")
            return
        
        logging.info(f"Creating embedding dataset for {self.embedding.model_name}")
        
        train, test, test5s = self.prepare_splits(datamodule)
        
        mapper = EmbeddingCreator(self.embedding)
        
        logging.info("Starting mapping")
        train = train.map(mapper, batched=True, batch_size=100)
        train = train.rename_column("labels", "ebird_code")
        test = test.map(mapper, batched=True, batch_size=100)
        test = test.rename_column("labels", "ebird_code")
        test5s = test5s.map(mapper, batched=True, batch_size=100)
        test5s = test5s.rename_column("labels", "ebird_code_multilabel")
        
        dataset = DatasetDict({"train": train, "test": test, "test_5s": test5s})
        # dataset.rename_column("embeddings", "audio")
        dataset.reset_format()
        
        self.save_config.save(dataset)
        
        logging.info(f"Cleaning up: {dataset.cleanup_cache_files()}")
        return dataset
        


    