from typing import Literal
from collections import Counter
from src.datamodule.components.event_decoding import EventDecoding
from src.datamodule.components.transforms import TransformsWrapper
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
            transforms: TransformsWrapper = TransformsWrapper(decoding=EventDecoding()),
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
            dataset = DatasetDict({split: dataset[split] for split in ["train", "test"]})

            logging.info("> Mapping data set.")
            dataset["train"] = dataset["train"].map(
                self.event_mapper,
                remove_columns=["audio"],
                batched=True,
                batch_size=300,
                load_from_cache_file=True,
                num_proc=self.dataset_config.n_workers,
            )
            if self.dataset_config.get("class_weights"):
                self.num_train_labels = self._count_labels((dataset["train"]["ebird_code"]))

            dataset = dataset.select_columns(
                ["filepath", "ebird_code", "detected_events", "start_time", "end_time"]
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
                load_from_cache_file=False,
                num_proc=self.dataset_config.n_workers,
            )

            dataset = dataset.map(
                self._classes_one_hot,
                batched=True,
                batch_size=300,
                load_from_cache_file=True,
                num_proc=self.dataset_config.n_workers,
            )
            
            if self.dataset_config.get("class_weights"):
                self.num_train_labels = self._count_labels((dataset["train"]["ebird_code"]))

            dataset["test"] = dataset["test_5s"]
            dataset = dataset.select_columns(
                ["filepath", "ebird_code_multilabel", "detected_events", "start_time", "end_time"]
            )

            dataset = dataset.rename_column("ebird_code_multilabel", "labels")
        return dataset
    
    def _count_labels(self,labels):
        # frequency
        label_counts = Counter(labels)

        if 0 not in label_counts:
            label_counts[0] = 0
        
        num_labels = max(label_counts)
        counts = [label_counts[i] for i in range(num_labels+1)]
        return counts
        
    def _classes_one_hot(self, batch):
        """
        Converts class labels to one-hot encoding.

        This method takes a batch of data and converts the class labels in the "ebird_code_multilabel" field to one-hot encoding.
        The one-hot encoding is a binary matrix representation of the class labels.

        Args:
            batch (dict): A batch of data. The batch should be a dictionary where the keys are the field names and the values are the field data.

        Returns:
            dict: The batch with the "ebird_code_multilabel" field converted to one-hot encoding. The keys are the field names and the values are the field data.
        """
        label_list = [y for y in batch["ebird_code_multilabel"]]
        class_one_hot_matrix = torch.zeros(
            (len(label_list), self.dataset_config.n_classes), dtype=torch.float
        )

        for class_idx, idx in enumerate(label_list):
            class_one_hot_matrix[class_idx, idx] = 1

        class_one_hot_matrix = torch.tensor(class_one_hot_matrix, dtype=torch.float32)
        return {"ebird_code_multilabel": class_one_hot_matrix}  