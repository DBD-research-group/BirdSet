from datasets import DatasetDict, Audio
from src.datamodule.components.transforms import TransformsWrapperN
from src.utils.extraction import DefaultFeatureExtractor
from .base_datamodule import BaseDataModuleHF, DatasetConfig, LoadersConfig

class ESC50(BaseDataModuleHF):
    def __init__(
            self,
            dataset: DatasetConfig = DatasetConfig(),
            loaders: LoadersConfig = LoadersConfig(),
            transforms: TransformsWrapperN = TransformsWrapperN(),
            extractors: DefaultFeatureExtractor = DefaultFeatureExtractor()
    ):
        super().__init__(
            dataset=dataset,
            loaders=loaders,
            transforms=transforms,
            extractors=extractors,
        )



