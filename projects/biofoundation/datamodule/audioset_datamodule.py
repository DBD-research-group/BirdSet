from birdset.datamodule.components.transforms import BirdSetTransformsWrapper
from birdset.datamodule.base_datamodule import BaseDataModuleHF
from birdset.configs import DatasetConfig, LoadersConfig
from datasets import (
    load_dataset,
    IterableDataset,
    IterableDatasetDict,
    DatasetDict,
    Audio,
    Dataset,
)
from birdset.utils import pylogger
import logging

log = pylogger.get_pylogger(__name__)



class AS20DataModule(BaseDataModuleHF):
    def __init__(
        self,
        dataset: DatasetConfig = DatasetConfig(),
        loaders: LoadersConfig = LoadersConfig(),
        transforms: BirdSetTransformsWrapper = BirdSetTransformsWrapper(),
    ):
        super().__init__(dataset=dataset, loaders=loaders, transforms=transforms)

    

    def _preprocess_data(self, dataset):
        """
        Preprocess the data
        """

        return dataset
