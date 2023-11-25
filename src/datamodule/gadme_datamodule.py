from omegaconf import DictConfig
from .base_datamodule import BaseDataModuleHF

class GADMEDataModule(BaseDataModuleHF):
    def __init__(
            self,
            dataset: DictConfig,
            loaders: DictConfig,
            transforms: DictConfig,
            extractors: DictConfig,
            transforms_rene=None
    ):
        super().__init__(
            dataset=dataset,
            loaders=loaders,
            transforms=transforms,
            extractors=extractors,
            transforms_rene=transforms_rene
        )

    @property
    def num_classes(self):
        return self.dataset.n_classes