import os
from typing import Any, Literal
from datasets import Dataset, DatasetDict, disable_caching
import rootutils
import hydra
import lightning as L 
from omegaconf import OmegaConf
from src import utils
import pyrootutils 

from src.datamodule.base_datamodule import DatasetConfig, LoadersConfig
from src.datamodule.components.event_mapping import XCEventMapping
from src.datamodule.gadme_datamodule import GADMEDataModule

from dataset.embeddings.embedding_transforms import EmbeddingTransforms
from dataset.embeddings.embedding_module import EmbeddingModule


log = utils.get_pylogger(__name__)

root = pyrootutils.setup_root(
    search_from=__file__,
    indicator=[".git"],
    pythonpath=True,
    dotenv=True,
)

_HYDRA_PARAMS = {
    "version_base":None,
    #"config_path": "../configs",
    "config_path": str(root / "configs"),
    "config_name": "embeddings.yaml" # we use a different embedding main yaml
}

# set the datamodule and load the data
# set the embedding model and embed the data

# @utils.register_custom_resolvers(**_HYDRA_PARAMS)
@hydra.main(**_HYDRA_PARAMS)
def main(cfg):
    # replace datamodule
    cfg.datamodule._target_ = "dataset.embeddings.embedding_datamodule.EmbeddingDatamodule"
    print(OmegaConf.to_yaml(cfg))
    
    log.info(f"Seed everything with <{cfg.seed}>")
    L.seed_everything(cfg.seed)
    
    
    # Setup data
    log.info(f"Instantiate datamodule <{cfg.datamodule._target_}>")
    datamodule = hydra.utils.instantiate(cfg.datamodule)
    datamodule.prepare_data() # has to be called before model for len_traindataset!
    datamodule.setup()
    
    # Setup Embedding Module
    log.info(f"Instantiating Embedding Module")
    module = hydra.utils.instantiate(cfg.embedding_module)
    
    module.run(datamodule)
    
    utils.close_loggers()

if __name__ == "__main__":   
    disable_caching() 
    main()