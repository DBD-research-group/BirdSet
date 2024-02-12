import os
import logging

from datasets import DatasetDict

class DsSaver():
    def __init__(self) -> None:
        pass
    
    def save(self, dataset):
        logging.info("Nothing to do.. not saving")

class Local(DsSaver):
    def __init__(self, data_dir, embedding_dir, dataset_name, model_name, hf_path, hf_name, override) -> None:
        super().__init__()
        self.dataset_name = dataset_name
        self.model_name = model_name
        
        logging.info(f"Override was set to {override}. Ignoring anyways!")        
        
        
        self.embeddings_path = os.path.join(data_dir, embedding_dir)
        logging.info(f"Creating paths for embedding dataset at {self.embeddings_path}")
        os.makedirs(self.embeddings_path, exist_ok=True)
        
    def save(self, dataset: DatasetDict):
        logging.info(f"Saving dataset in embeddings path {self.embeddings_path}")
        ds_path = os.path.join(self.embeddings_path, self.dataset_name, self.model_name)
        logging.info(f"Saving to: <{os.path.abspath(ds_path)}>")
        os.makedirs(ds_path, exist_ok=True)
        
        dataset.save_to_disk(ds_path)
        
        logging.info("all done saving dataset")
        
        return
