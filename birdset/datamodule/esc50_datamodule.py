from datasets import Audio

from birdset.datamodule.components.transforms import BirdSetTransformsWrapper
from . import BaseDataModuleHF
from birdset.configs import DatasetConfig, LoadersConfig


class ESC50DataModule(BaseDataModuleHF):
    def __init__(
        self,
        dataset: DatasetConfig = DatasetConfig(),
        loaders: LoadersConfig = LoadersConfig(),
        transforms: BirdSetTransformsWrapper = BirdSetTransformsWrapper(),
    ):
        super().__init__(
            dataset=dataset,
            loaders=loaders,
            transforms=transforms,
        )

    def _preprocess_data(self, dataset):
        dataset = dataset.cast_column(
            column="audio",
            feature=Audio(
                sampling_rate=self.dataset_config.sampling_rate,
                mono=True,
                decode=True,
            ),
        )
        dataset = dataset.rename_column("target", "labels")
        dataset = dataset.select_columns(["audio", "labels"])
        return dataset
