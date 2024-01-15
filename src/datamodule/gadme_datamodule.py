from typing import Literal
from collections import Counter
from src.datamodule.components.event_decoding import EventDecoding
from src.datamodule.components.transforms import GADMETransformsWrapper
from src.datamodule.components.event_mapping import XCEventMapping
from .base_datamodule import BaseDataModuleHF, DatasetConfig, LoadersConfig
from datasets import DatasetDict
import logging
import torch


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

    def _load_data(self, decode: bool = False):
        return super()._load_data(decode=decode)

    def _preprocess_data(self, dataset):
        if self.dataset_config.task == "multiclass":
            # pick only train and test dataset
            dataset = DatasetDict({split: dataset[split] for split in ["train", "valid", "test"]})

            logging.info("> Mapping data set.")
            dataset["train"] = dataset["train"].map(
                self.event_mapper,
                remove_columns=["audio"],
                batched=True,
                batch_size=300,
                load_from_cache_file=True,
                num_proc=self.dataset_config.n_workers,
            )

            dataset["valid"] = dataset["valid"].map(
                self.event_mapper,
                remove_columns=["audio"],
                batched=True,
                batch_size=300,
                load_from_cache_file=True,
                num_proc=self.dataset_config.n_workers,
            )

            if self.dataset_config.class_weights_loss or self.dataset_config.class_weights_sampler:
                self.num_train_labels = self._count_labels((dataset["train"]["ebird_code"]))
            
            # if self.dataset_config.classlimit and not self.dataset_config.eventlimit:
            #     print("class limiting")
            #     dataset["train"] = self._limit_classes(
            #         dataset=dataset["train"],
            #         label_name="ebird_code",
            #         limit=self.dataset_config.classlimit
            #     )

            if self.dataset_config.classlimit or self.dataset_config.eventlimit:
                print("smart sampling")
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

            logging.info("> Mapping data set.")
            dataset["train"] = dataset["train"].map(
                self.event_mapper,
                remove_columns=["audio"],
                batched=True,
                batch_size=300,
                load_from_cache_file=True,
                num_proc=self.dataset_config.n_workers,
            )

            dataset = dataset.rename_column("ebird_code_multilabel", "labels")

            dataset = dataset.map(
                self._classes_one_hot,
                batched=True,
                batch_size=300,
                load_from_cache_file=False,
                num_proc=self.dataset_config.n_workers,
            )

            if self.dataset_config.class_weights_loss or self.dataset_config.class_weights_sampler:
                self.num_train_labels = self._count_labels((dataset["train"]["ebird_code"]))

            dataset["test"] = dataset["test_5s"]

        dataset["train"] = dataset["train"].select_columns(
            ["filepath", "labels", "detected_events", "start_time", "end_time", "no_call_events"]
        )
        dataset["valid"] = dataset["valid"].select_columns(
            ["filepath", "labels", "detected_events", "start_time", "end_time", "no_call_events"]
        )
        # maybe has to be added to test data to avoid two selections
        dataset["test"] = dataset["test"].select_columns(
            ["filepath", "labels", "detected_events", "start_time", "end_time"]
        )

        return dataset
