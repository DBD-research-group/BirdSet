from src.datamodule.components.transforms import TransformsWrapperN
from src.utils.extraction import DefaultFeatureExtractor
from .base_datamodule import BaseDataModuleHF, DatasetConfig, LoadersConfig

class GADMEDataModule(BaseDataModuleHF):
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

    @property
    def num_classes(self):
        return self.dataset.n_classes