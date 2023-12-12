from typing import Literal
from datasets import Audio
from src.datamodule.components.transforms import TransformsWrapper
from .base_datamodule import BaseDataModuleHF, DatasetConfig, LoadersConfig
from datasets import load_dataset, Audio, DatasetDict
from src.datamodule.components.event_mapping import Mapper

class ESC50(BaseDataModuleHF):
    def __init__(
            self,
            dataset: DatasetConfig = DatasetConfig(),
            loaders: LoadersConfig = LoadersConfig(),
            transforms: TransformsWrapper = TransformsWrapper(),
            mapper: Mapper | None = None
    ):
        super().__init__(
            dataset=dataset,
            loaders=loaders,
            transforms=transforms,
            mapper=mapper
        )


    def _preprocess_data(self, dataset, task_type: Literal['multiclass', 'multilabel']):
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

