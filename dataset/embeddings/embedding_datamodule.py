from datasets import Dataset, DatasetDict


from src.datamodule.components.transforms import BaseTransforms
from src.datamodule.components.event_mapping import XCEventMapping
from src.datamodule.gadme_datamodule import GADMEDataModule
from src.datamodule.base_datamodule import DatasetConfig, LoadersConfig

from dataset.embeddings.embedding_transforms import EmbeddingTransforms

class EmbeddingDatamodule(GADMEDataModule):
    def __init__(
        self,
        dataset: DatasetConfig = DatasetConfig(),
        loaders: LoadersConfig = LoadersConfig(),
        transforms: BaseTransforms = BaseTransforms(),
        mapper: XCEventMapping = XCEventMapping()
        ) -> None:
        super().__init__(
            dataset=dataset,
            loaders=loaders,
            transforms=EmbeddingTransforms(transforms),
            mapper=mapper
        )
    
    
    def _preprocess_data(self, dataset):
        dataset["train"] = dataset["train"].map(
                self.event_mapper,
                remove_columns=["audio"],
                batched=True,
                batch_size=300,
                load_from_cache_file=True,
                num_proc=self.dataset_config.n_workers,
            )
        
        if self.dataset_config.class_weights_loss or self.dataset_config.class_weights_sampler:
                self.num_train_labels = self._count_labels((dataset["train"]["ebird_code"]))
        
        dataset = dataset.rename_column("ebird_code", "labels")
        # dataset = dataset.rename_column("ebird_code_multilabel", "labels")

        dataset["train"] = dataset["train"].select_columns(
            ["filepath", "labels", "detected_events", "start_time", "end_time", "no_call_events", "ebird_code_multilabel"]
        )
        # maybe has to be added to test data to avoid two selections
        dataset["test"]= dataset["test"].select_columns(
            ["filepath", "labels", "detected_events", "start_time", "end_time", "ebird_code_multilabel"]
        )
        dataset["test_5s"]= dataset["test_5s"].select_columns(
            ["filepath", "ebird_code_multilabel", "detected_events", "start_time", "end_time"]
        )
        return dataset
    
    def _create_splits(self, dataset: DatasetDict | Dataset):
        if isinstance(dataset, DatasetDict):
            return dataset
        raise ValueError(f"Type of dataset is {type(dataset)}")
    
    def setup(self, stage=None):
        self.train_dataset = self._get_dataset("train")
        self.test_dataset = self._get_dataset("test")
        self.valid_dataset = self._get_dataset("test_5s")
        self.valid_dataset = self.valid_dataset.rename_column("ebird_code_multilabel", "labels")
        self.test_5s_dataset = self.valid_dataset
    
    def set_task(self, task):
        self.transforms.set_task(task)