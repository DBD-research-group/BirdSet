from datasets import Audio
from birdset.datamodule.components.transforms import GADMETransformsWrapper
from .base_datamodule import BaseDataModuleHF, DatasetConfig, LoadersConfig
from datasets import load_dataset, Audio

class ESC50DataModule(BaseDataModuleHF):
    def __init__(
            self,
            dataset: DatasetConfig = DatasetConfig(),
            loaders: LoadersConfig = LoadersConfig(),
            transforms: GADMETransformsWrapper = GADMETransformsWrapper(),
            mapper: None = None
    ):
        super().__init__(
            dataset=dataset,
            loaders=loaders,
            transforms=transforms,
            mapper=mapper
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
        if self.event_mapper is not None:
            dataset = dataset['train'].map(
                self.event_mapper,
                batched=True,
                batch_size=300,
                load_from_cache_file=True,
                num_proc=self.dataset_config.n_workers,
            )
        dataset = dataset.rename_column("target", "labels")
        dataset = dataset.select_columns(["audio", "labels"])
        return dataset

