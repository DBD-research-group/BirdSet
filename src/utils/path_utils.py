import os
import logging
from typing import Literal
from datasets import Dataset, DatasetDict, load_from_disk, load_dataset

# this enables saving and loading of datasets at specified locations
# its intended use is in the datamodule and in creating the emebddings

def get_ds_manager(mode:Literal["local", "hf"], data_dir, dataset_name, hf_path, hf_name, num_proc):
    if mode=="hf":
        return HfDsManager(data_dir, dataset_name, hf_path, hf_name, num_proc)
    elif mode == "local":
        return LocalDsManager(data_dir, dataset_name, hf_path, hf_name, num_proc)

class DsManager():
    def __init__(self, data_dir, dataset_name, hf_path, hf_name, num_proc) -> None:
        self.data_dir = data_dir
        self.dataset_name = dataset_name
        self.ensure_data_dir_exists()
    
    def save(self, dataset: Dataset | DatasetDict):
        logging.error("Unsupported Operation!")
        assert False
    
    def load(self) -> Dataset | DatasetDict:
        logging.error("Unsupported Operation")
        assert False
    
    def ensure_data_dir_exists(self, exist_ok=True):
        self.ensure_path_exists(self.data_dir, exist_ok=exist_ok)

    def ensure_path_exists(self, path, exist_ok=True):
        logging.info(f"Making sure the dir {self.data_dir} exists. Creating otherwise")
        os.makedirs(path, exist_ok=exist_ok)
    
    def _return_ds(self, dataset: Dataset | DatasetDict) -> Dataset | DatasetDict:
        if isinstance(dataset, Dataset | DatasetDict):
            logging.info(f"Successfully loaded Dataset.")
            return dataset
        
        logging.error(f"Something went wrong, trying to get the dataset. Failing now...")
        assert False
        

class LocalDsManager(DsManager):
    def __init__(self, data_dir, dataset_name, hf_path, hf_name, num_proc) -> None:
        super().__init__(data_dir, dataset_name, hf_name, hf_path, num_proc)
        self.ds_dir = os.path.join(data_dir, dataset_name)

    def ensure_ds_dir_exists(self, exist_ok=True):
        self.ensure_path_exists(self.ds_dir, exist_ok=exist_ok)
            
    def save(self, dataset: Dataset | DatasetDict):
        logging.info(f"Datasets will be saved in {self.data_dir}")
        self.ensure_ds_dir_exists()
        ds_path = self.ds_dir
        logging.info(f"Saving dataset to {ds_path}")
        dataset.save_to_disk(ds_path)
        logging.info (f"Successfully saved Dataset!")
    
    def load(self) -> Dataset | DatasetDict:
        ds_path = self.ds_dir
        logging.info(f"Attempting to load dataset from {ds_path}")
        if not os.path.exists(ds_path):
            logging.error(f"Path to dataset {ds_path} does not exist! Cannot load. Is the path correct?")
            assert False
        
        dataset = load_from_disk(ds_path)
        
        return self._return_ds(dataset)



class HfDsManager(DsManager):
    def __init__(self, data_dir, dataset_name, hf_path, hf_name, num_proc) -> None:
        super().__init__(data_dir, dataset_name, hf_path, hf_name, num_proc)
        self.hf_path = hf_path
        self.hf_name = hf_name
        self.num_proc =  num_proc
    
    def save(self, dataset: Dataset | DatasetDict):
        return super().save(dataset)
    
    def load(self) -> Dataset | DatasetDict:
        logging.info(f"Loading dataset {self.hf_name} from huggingface path {self.hf_path} (checking at local path {self.data_dir}) ")
        self.ensure_data_dir_exists()
        dataset = load_dataset(
            name = self.hf_name,
            path = self.hf_path,
            cache_dir=self.data_dir,
            num_proc=self.num_proc
        )
        
        return self._return_ds(dataset)
        
        
        