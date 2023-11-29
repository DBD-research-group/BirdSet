from datasets import DatasetDict, Audio
from src.datamodule.components.event_mapping import EventMapping
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


    def _preprocess_multiclass(self, dataset: DatasetDict, select_range=None):
        """Preprocess the dataset for multiclass classification.
        Args:
            dataset (Dataset): Dataset to preprocess.
            split (str): Split to preprocess.
            select_range (list): Range of classes to select.
        Returns:
            Dataset: Preprocessed dataset.
        """
        for split in dataset.keys():
            dataset[split] = dataset[split].map(
                # TODO add to hydra
                EventMapping(with_noise_cluster=False, biggest_cluster=True, only_one=True),
                remove_columns=["audio"],
                batched=True,
                batch_size=100,
                load_from_cache_file=True,
                num_proc=self.dataset_config.n_workers,
            )
        dataset = dataset.cast_column("audio", Audio(self.transforms.sampling_rate, mono=True, decode=True))
            #dataset = dataset.select_columns(self.dataset.column_list)

        return dataset