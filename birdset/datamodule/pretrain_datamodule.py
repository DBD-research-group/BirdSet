from collections import Counter
from birdset.datamodule.components.event_decoding import EventDecoding
from birdset.datamodule.components.transforms import GADMETransformsWrapper
from birdset.datamodule.components.event_mapping import XCEventMapping
from .base_datamodule import BaseDataModuleHF, DatasetConfig, LoadersConfig
from datasets import DatasetDict, Dataset
from datasets import load_dataset, Audio
import logging
import torch


class PretrainDataModule(BaseDataModuleHF):
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
        """
        Load audio dataset from Hugging Face Datasets.

        Returns HF dataset with audio column casted to Audio feature, containing audio data as numpy array and sampling rate.
        """
        logging.info("> Loading data set.")

        dataset = load_dataset(
            name=self.dataset_config.hf_name,
            path=self.dataset_config.hf_path,
            cache_dir=self.dataset_config.data_dir,
            num_proc=3,
        )

        if self.dataset_config.subset:
            dataset = self._fast_dev_subset(dataset, self.dataset_config.subset)


        dataset = dataset.cast_column(
            column="audio",
            feature=Audio(
                sampling_rate=self.dataset_config.sampling_rate,
                mono=True,
                decode=decode,
            ),
        )

        return dataset
    
    def _preprocess_data(self, dataset):
        if self.dataset_config.task == "multiclass":
            # we only have train data
            logging.info("> Mapping data set.")
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
            
            if self.dataset_config.classlimit:
                dataset["train"] = self._limit_classes(
                    dataset=dataset["train"],
                    label_name="ebird_code",
                    limit=self.dataset_config.classlimit)

            dataset = dataset.rename_column("ebird_code", "labels")

        elif self.dataset_config.task == "multilabel":
            # only train data
            
            logging.info("> Mapping data set.")
            dataset["train"] = dataset["train"].map(
                self.event_mapper,
                remove_columns=["audio"],
                batched=True,
                batch_size=350,
                load_from_cache_file=False,
                num_proc=1,
            )

            dataset = dataset.rename_column("ebird_code_multilabel", "labels")

            logging.info(">> One-hot-encode classes")
            dataset = dataset.map(
                self._classes_one_hot,
                batched=True,
                batch_size=350,
                load_from_cache_file=False,
                num_proc=1,
            )

            if self.dataset_config.class_weights_loss or self.dataset_config.class_weights_sampler:
                self.num_train_labels = self._count_labels((dataset["train"]["ebird_code"]))


        dataset["train"] = dataset["train"].select_columns(
            ["filepath", "labels", "detected_events", "start_time", "end_time", "no_call_events"]
        )

        return dataset

    def _create_splits(self, dataset: DatasetDict | Dataset):
        # no test set 
        split = dataset["train"].train_test_split(
            self.dataset_config.val_split, shuffle=True, seed=self.dataset_config.seed
        )

        return DatasetDict({
            "train": split["train"],
            "valid": split["test"]
        })

    
