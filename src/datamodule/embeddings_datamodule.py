from src.datamodule.base_datamodule import DatasetConfig, LoadersConfig
from src.datamodule.components.event_mapping import XCEventMapping
from src.datamodule.components.transforms import GADMETransformsWrapper, EmbeddingTransforms
from src.datamodule.gadme_datamodule import GADMEDataModule
from datasets import Dataset, DatasetDict

class EmbeddingsDataModule(GADMEDataModule):
    def __init__(
            self,
            dataset: DatasetConfig = DatasetConfig(),
            loaders: LoadersConfig = LoadersConfig(),
            transforms: GADMETransformsWrapper = GADMETransformsWrapper(),
            mapper: XCEventMapping = XCEventMapping()
    ):        
        super().__init__(
            dataset=dataset,
            loaders=loaders,
            transforms=transforms,
            mapper=mapper
        )
    
    def _configure_data(self, dataset, decode: bool = True):
        if self.dataset_config.task == "multilabel":
            dataset["test"] = dataset["test_5s"]
        dataset = self._ensure_train_test_splits(dataset)
        
        if self.dataset_config.subset:
            dataset = self._fast_dev_subset(dataset, self.dataset_config.subset)
               
        # if self.dataset_config.task == "multilabel":
        #     dataset["test"] = dataset["test_5s"]
        
        return dataset
    
    def _get_dataset(self, split):
        assert isinstance(self.transforms, EmbeddingTransforms)
        return super()._get_dataset(split)
    
    def _preprocess_data(self, dataset, data_column_name="embeddings"):
        return super()._preprocess_data(dataset, "embeddings")
    
    def _preprocess_multiclass(self, dataset):
        dataset = DatasetDict({split: dataset[split] for split in ["train", "test"]})

        if self.dataset_config.class_weights_loss or self.dataset_config.class_weights_sampler:
            self.num_train_labels = self._count_labels((dataset["train"]["ebird_code"]))

        dataset = dataset.rename_column("ebird_code", "labels")
        return dataset
    
    def _preprocess_multilabel(self, dataset):
        dataset = DatasetDict({split: dataset[split] for split in ["train", "test"]})
        
        dataset = dataset.rename_column("ebird_code_multilabel", "labels")
        
        dataset = dataset.map(
            self._classes_one_hot,
            batched=True,
            batch_size=300,
            load_from_cache_file=False,
            num_proc=self.dataset_config.n_workers#,
            # fn_kwargs={"column_name": "ebird_code_multilabel"}
        )
                
        if self.dataset_config.class_weights_loss or self.dataset_config.class_weights_sampler:
            self.num_train_labels = self._count_labels((dataset["train"]["ebird_code"]))

        # dataset["test"] = dataset["test_5s"]
        # dataset = dataset.drop_column("ebird_code_multilabel", "labels")
        
        return dataset