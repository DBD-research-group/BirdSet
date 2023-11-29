from datasets import DatasetDict, Audio
from src.datamodule.components.transforms import TransformsWrapper
from src.utils.extraction import DefaultFeatureExtractor
from .base_datamodule import BaseDataModuleHF, DatasetConfig, LoadersConfig
import logging
from datasets import load_dataset, Audio, DatasetDict
import os

class ESC50(BaseDataModuleHF):
    def __init__(
            self,
            mapper = None,
            dataset: DatasetConfig = DatasetConfig(),
            loaders: LoadersConfig = LoadersConfig(),
            transforms: TransformsWrapper = TransformsWrapper(),
            extractors: DefaultFeatureExtractor = DefaultFeatureExtractor()
    ):
        super().__init__(
            dataset=dataset,
            loaders=loaders,
            transforms=transforms,
            extractors=extractors,
            mapper=None
        )


    def _create_splits(self, dataset):
        split_1 = dataset["train"].train_test_split(
            self.dataset.val_split, shuffle=True, seed=self.dataset.seed)
        split_2 = split_1["test"].train_test_split(
            0.5, shuffle=False, seed=self.dataset.seed)
        train_dataset = split_1["train"]
        val_dataset = split_2["train"]
        test_dataset = split_2["test"]
        return train_dataset, val_dataset, test_dataset
    
    def prepare_data(self):

        logging.info("Check if preparing has already been done.")

        if self._prepare_done:
            logging.info("Skip preparing.")
            return

        logging.info("> Loading data set.")

        dataset = load_dataset(
            name=self.dataset.hf_name,
            path=self.dataset.hf_path,
            cache_dir=self.dataset.data_dir,
            num_proc=3,
        )

        if self.dataset.subset:
            dataset = self._fast_dev_subset(dataset, self.dataset.subset)

        dataset = dataset.cast_column(
            column="audio",
            feature=Audio(
                sampling_rate=self.feature_extractor.sampling_rate,
                mono=True,
                decode=True,
            ),
        )

        logging.info("> Mapping data set.")
        dataset = dataset.rename_column("target", "labels")
        dataset = dataset.select_columns(["audio", "labels"])

        dataset.set_format("np")
        train_dataset, val_dataset, test_dataset = self._create_splits(dataset)

        complete = DatasetDict(
            {"train": train_dataset, "valid": val_dataset, "test": test_dataset}
        )
        data_path = os.path.join(
            self.dataset.data_dir,
            f"{self.dataset.dataset_name}_processed",
            train_dataset._fingerprint,
        )
        self.data_path = data_path
        self._prepare_done = True
        self.len_trainset = len(train_dataset)

        if os.path.exists(data_path):
            logging.info("Dataset exists on disk.")
            return

        logging.info(f"Saving to disk: {os.path.join(self.data_path)}")
        complete.save_to_disk(self.data_path)
