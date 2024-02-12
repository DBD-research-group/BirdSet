import os
from typing import Any, Literal
from datasets import Dataset, DatasetDict, disable_caching
import rootutils
import hydra
import lightning as L 
from omegaconf import OmegaConf
from src import utils
import pyrootutils 

from chirp.inference import models

from src.datamodule.base_datamodule import DatasetConfig, LoadersConfig
from src.datamodule.components.event_mapping import XCEventMapping
from src.datamodule.gadme_datamodule import GADMEDataModule

from dataset.embeddings.embedding_transforms import EmbeddingTransforms

log = utils.get_pylogger(__name__)


file_path = os.path.abspath(os.path.dirname(__file__))
config_path = os.path.join(file_path, "configs")

_HYDRA_PARAMS = {
    "version_base":None,
    # "config_path": "../configs",
    "config_path": config_path,
    "config_name": "main.yaml"
}

print("Hallo Welt")
# models.model_class_map()["tfhub_model"]

print(models.model_class_map()["tfhub_model"])

# print(os.path.abspath(os.path.dirname(__file__)))

# set the datamodule and load the data
# set the embedding model and embed the data

@utils.register_custom_resolvers(**_HYDRA_PARAMS)
@hydra.main(**_HYDRA_PARAMS)
def main(cfg):
    print(cfg)
    


if __name__ == "__main__":   
    disable_caching() 
    main()