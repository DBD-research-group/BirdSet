from src import utils
from src.datamodule.components.transforms import GADMETransformsWrapper
from src.datamodule.components.event_mapping import XCEventMapping
from .base_datamodule import BaseDataModuleHF, DatasetConfig, LoadersConfig
from datasets import DatasetDict

log = utils.get_pylogger(__name__)

class GADMEDataModule(BaseDataModuleHF):
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

    def _load_and_configure_data(self, decode: bool = False):
        return super()._load_and_configure_data(decode=decode)

    def _preprocess_data(self, dataset):
        if self.dataset_config.task == "multiclass":
            # pick only train and test dataset
            dataset = DatasetDict({split: dataset[split] for split in ["train", "test"]})

            log.info("> Mapping data set.")
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
            
            if self.dataset_config.classlimit and not self.dataset_config.eventlimit:
                dataset["train"] = self._limit_classes(
                    dataset=dataset["train"],
                    label_name="ebird_code",
                    limit=self.dataset_config.classlimit
                )
            elif self.dataset_config.classlimit or self.dataset_config.eventlimit:
                dataset["train"] = self._smart_sampling(
                    dataset=dataset["train"],
                    label_name="ebird_code",
                    class_limit=self.dataset_config.classlimit,
                    event_limit=self.dataset_config.eventlimit
                )

            dataset = dataset.rename_column("ebird_code", "labels")

        elif self.dataset_config.task == "multilabel":
            # pick only train and test_5s dataset
            dataset = DatasetDict({split: dataset[split] for split in ["train", "test_5s"]})

            log.info(">> Mapping train data.")
            dataset["train"] = dataset["train"].map(
                self.event_mapper,
                remove_columns=["audio"],
                batched=True,
                batch_size=300,
                load_from_cache_file=True,
                num_proc=self.dataset_config.n_workers,
            ) # has to be deterministic for cache loading??

            dataset = dataset.rename_column("ebird_code_multilabel", "labels")

            if self.dataset_config.classlimit or self.dataset_config.eventlimit:
                log.info(">> Smart Sampling") #!TODO: implement custom caching?
                dataset["train"] = self._smart_sampling(
                    dataset=dataset["train"],
                    label_name="ebird_code",
                    class_limit=self.dataset_config.classlimit,
                    event_limit=self.dataset_config.eventlimit
                )
            log.info(">> One-hot-encode classes")
            dataset = dataset.map(
                self._classes_one_hot,
                batched=True,
                batch_size=500,
                load_from_cache_file=True,
                num_proc=self.dataset_config.n_workers,
            )

            if self.dataset_config.class_weights_loss or self.dataset_config.class_weights_sampler:
                self.num_train_labels = self._count_labels((dataset["train"]["ebird_code"]))

            dataset["test"] = dataset["test_5s"]

        dataset["train"] = dataset["train"].select_columns(
            ["filepath", "labels", "detected_events", "start_time", "end_time", "no_call_events"]
        )
        # maybe has to be added to test data to avoid two selections
        dataset["test"] = dataset["test"].select_columns(
            ["filepath", "labels", "detected_events", "start_time", "end_time"]
        )

        return dataset
