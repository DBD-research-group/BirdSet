from .base_datamodule import BaseDataModule
from omegaconf import DictConfig

class HighSierra(BaseDataModule):
    def __init__(
            self,
            dataset: DictConfig,
            loaders: DictConfig,
            transforms: DictConfig
    ):
        super().__init__(
            dataset=dataset,
            loaders=loaders,
            transforms=transforms
        )

    @property
    def num_classes(self):
        return self.dataset.n_classes
