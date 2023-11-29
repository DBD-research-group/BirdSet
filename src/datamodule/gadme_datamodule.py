from src.datamodule.components.transforms import TransformsWrapper
from src.utils.extraction import DefaultFeatureExtractor
from .base_datamodule import BaseDataModuleHF, DatasetConfig, LoadersConfig

class GADMEDataModule(BaseDataModuleHF):
    def __init__(
            self,
            mapper,
            dataset: DatasetConfig = DatasetConfig(),
            loaders: LoadersConfig = LoadersConfig(),
            transforms: TransformsWrapper = TransformsWrapper(),
            extractors: DefaultFeatureExtractor = DefaultFeatureExtractor()
    ):
        super().__init__(
            dataset=dataset,
            loaders=loaders,
            transforms=transforms,
            extractors=extractors,
            mapper=mapper
        )
