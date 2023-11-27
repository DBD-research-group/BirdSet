from src.datamodule.components.transforms import TransformsWrapper
from src.utils.extraction import DefaultFeatureExtractor
from .base_datamodule import BaseDataModuleHF, DatasetConfig, LoadersConfig

class ESC50(BaseDataModuleHF):
    def __init__(
            self,
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
        )

    @property
    def num_classes(self):
        return self.dataset.n_classes

    def _create_splits(self, dataset):
        split_1 = dataset["train"].train_test_split(
            self.dataset.val_split, shuffle=True, seed=self.dataset.seed)
        split_2 = split_1["test"].train_test_split(
            0.5, shuffle=False, seed=self.dataset.seed)
        train_dataset = split_1["train"]
        val_dataset = split_2["train"]
        test_dataset = split_2["test"]
        return train_dataset, val_dataset, test_dataset
