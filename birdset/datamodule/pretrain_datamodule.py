import os
import logging
from copy import deepcopy
from datasets import load_dataset, Audio, load_from_disk

from birdset.utils import pylogger
from birdset.datamodule.components.event_mapping import XCEventMapping
from birdset.datamodule.components.transforms import BirdSetTransformsWrapper
from . import BaseDataModuleHF
from birdset.configs import DatasetConfig, LoadersConfig

log = pylogger.get_pylogger(__name__)


class PretrainDataModule(BaseDataModuleHF):
    def __init__(
            self,
            dataset: DatasetConfig = DatasetConfig(),
            loaders: LoadersConfig = LoadersConfig(),
            transforms: BirdSetTransformsWrapper = BirdSetTransformsWrapper(),
            mapper: XCEventMapping = XCEventMapping()
    ):
        super().__init__(
            dataset=dataset,
            loaders=loaders,
            transforms=transforms,
        )
        self.event_mapper = mapper

    def prepare_data(self):
        if self.dataset_config.direct_fingerprint:

            if self._prepare_done:
                log.info("Skip preparing.")
                return
            path = self.dataset_config.direct_fingerprint
            log.info(f"Loading an already sharded dataset from local path: {path}")
            dataset = load_from_disk(os.path.join(path, "train"))
            self.len_trainset = len(dataset)
            self._prepare_done = True
        else:
            return super().prepare_data()
    
    def _get_dataset(self, split):
        if self.dataset_config.direct_fingerprint:
            path = os.path.join(self.dataset_config.direct_fingerprint, split)
        else:
            path = os.path.join(self.disk_save_path, split)

        dataset = load_from_disk(path)

        transforms = deepcopy(self.transforms)
        transforms.set_mode(split)
        
        if split == "train":  # we need this for sampler, cannot be done later because set_transform
            if self.dataset_config.class_weights_sampler:
                self.train_label_list = dataset["labels"]

        if split == "valid":
            transforms.modes_to_skip.append("valid")
        dataset.set_transform(transforms, output_all_columns=False)

        return dataset

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
                desc="Event Mapping"
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
            # only train data
            
            logging.info("> Mapping data set.")
            dataset["train"] = dataset["train"].map(
                self.event_mapper,
                remove_columns=["audio"],
                batched=True,
                batch_size=300,
                load_from_cache_file=True,
                num_proc=1,
                desc="Event Mapping"
            )

            dataset = dataset.rename_column("ebird_code_multilabel", "labels")

            if self.dataset_config.classlimit or self.dataset_config.eventlimit:
                logging.info(">> Smart Sampling") #!TODO: implement custom caching?
                dataset["train"] = self._smart_sampling(
                    dataset=dataset["train"],
                    label_name="ebird_code",
                    class_limit=self.dataset_config.classlimit,
                    event_limit=self.dataset_config.eventlimit
                )

            logging.info(">> One-hot-encode classes")
            dataset = dataset.map(
                self._classes_one_hot,
                batched=True,
                batch_size=300,
                load_from_cache_file=True,
                num_proc=1,
                desc="One-hot-encoding"
            )

            if self.dataset_config.class_weights_loss or self.dataset_config.class_weights_sampler:
                self.num_train_labels = self._count_labels((dataset["train"]["ebird_code"]))

        dataset["train"] = dataset["train"].select_columns(
            ["filepath", "labels", "detected_events", "start_time", "end_time"]
        )
        dataset["valid"] = dataset["valid"].select_columns(
            ["filepath", "labels", "detected_events", "start_time", "end_time"]
        )

        return dataset
