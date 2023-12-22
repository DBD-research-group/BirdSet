from dataclasses import asdict, dataclass, field
import logging
import torch
import random
import os
from typing import List, Literal

import lightning as L

from datasets import load_dataset, load_from_disk, Audio, DatasetDict, Dataset, IterableDataset, IterableDatasetDict
from torch.utils.data import DataLoader
from src.datamodule.components.event_mapping import XCEventMapping
from src.datamodule.components.transforms import TransformsWrapper

@dataclass
class DatasetConfig:
    data_dir: str = "/workspace/data_gadme"
    dataset_name: str = "esc50"
    hf_path: str = "ashraq/esc50"
    hf_name: str = ""
    seed: int = 42
    n_classes: int = 50
    n_workers: int = 1
    val_split: float = 0.2
    task: Literal["multiclass", "multilabel"] = "multiclass"
    subset: int | None = None
    sampling_rate: int = 32_000
    class_weights_loss = None
    class_weights_sampler = None


@dataclass
class LoaderConfig:
    batch_size: int = 32
    shuffle: bool = True
    num_workers: int = 1
    pin_memory: bool = True
    drop_last: bool = False
    persistent_workers: bool = True
    prefetch_factor: int = 2 

@dataclass
class LoadersConfig:
    train: LoaderConfig = LoaderConfig()
    valid: LoaderConfig = LoaderConfig(shuffle=False)
    test: LoaderConfig = LoaderConfig(shuffle=False)

class BaseDataModuleHF(L.LightningDataModule):
    """
    A base data module for handling datasets using Hugging Face's datasets library.

    Attributes:
        dataset (DatasetConfig): Configuration for the dataset. Defaults to an instance of `DatasetConfig`.
        loaders (LoadersConfig): Configuration for the data loaders. Defaults to an instance of `LoadersConfig`.
        transforms (TransformsWrapper): Configuration for the data transformations. Defaults to an instance of `TransformsWrapper`.
        extractors (DefaultFeatureExtractor): Configuration for the feature extraction. Defaults to an instance of `DefaultFeatureExtractor`.

    Methods:
        __init__(dataset, loaders, transforms, extractors): Initializes the `BaseDataModuleHF` instance.
        prepare_data(): Prepares the data for use.
        setup(stage): Sets up the data for use.
        train_dataloader(): Returns the data loader for the training data.
        val_dataloader(): Returns the data loader for the validation data.
        test_dataloader(): Returns the data loader for the test data.
    """

    def __init__(
        self, 
        mapper: XCEventMapping | None = None,
        dataset: DatasetConfig = DatasetConfig(),
        loaders: LoadersConfig = LoadersConfig(),
        transforms: TransformsWrapper = TransformsWrapper(),
        ):
        super().__init__()
        self.dataset_config = dataset
        self.loaders_config = loaders
        self.transforms = transforms
        self.event_mapper = mapper

        self.data_path = None
        self.train_dataset = None
        self.val_dataset = None
        self.test_dataset = None

        self._prepare_done = False
        self.len_trainset = None
        self.num_train_labels = None
        self.train_label_list = None

    @property
    def num_classes(self):
        return self.dataset_config.n_classes

    def prepare_data(self):
        """
        Prepares the data for use.

        This method checks if the data preparation has already been done. If not, it loads the dataset, applies transformations,
        creates train, validation, and test splits, and saves the processed data to disk. If the data has already been prepared,
        this method does nothing.

        The method supports both multilabel and multiclass tasks. For multilabel tasks, it selects a subset of the data,
        applies preprocessing, and selects the necessary columns. For multiclass tasks, it applies preprocessing and selects
        the necessary columns.

        If the feature extractor is configured to return an attention mask, this method adds 'attention_mask' to the list of
        columns to select from the dataset.

        The method saves the processed dataset to disk in the directory specified by the 'data_dir' attribute of the 'dataset'
        configuration, under a subdirectory named after the dataset and the fingerprint of the training data.

        After the data is prepared, this method sets the '_prepare_done' attribute to True and the 'len_trainset' attribute
        to the length of the training dataset.

        Outputs data with the following columns:
            - audio: The preprocessed audio data, containing:
                - 'array': The audio data as a numpy array.
                - 'sampling_rate': The sampling rate of the audio data.
            - labels: The label for the audio data

        """

        logging.info("Check if preparing has already been done.")
        if self._prepare_done:
            logging.info("Skip preparing.")
            return

        logging.info("Prepare Data")
        
        dataset = self._load_data()
        dataset = self._preprocess_data(dataset)
        dataset = self._create_splits(dataset)

        # set the length of the training set to be accessed by the model
        self.len_trainset = len(dataset["train"])        
        self._save_dataset_to_disk(dataset)

        # set to done so that lightning does not call it again
        self._prepare_done = True
       
    def _preprocess_data(self, dataset):
        """
        Preprocesses the dataset.
        This includes stuff that only needs to be done once.
        """
          
        return dataset

    def _save_dataset_to_disk(self, dataset):
        """
        Saves the dataset to disk.

        This method sets the format of the dataset to numpy, prepares the path where the dataset will be saved, and saves
        the dataset to disk. If the dataset already exists on disk, it does not save the dataset again.

        Args:
            dataset (datasets.Dataset): The dataset to be saved. The dataset should be a Hugging Face `datasets.Dataset` object.

        Returns:
            None
        """
        dataset.set_format("np")

        data_path = os.path.join(
            self.dataset_config.data_dir,
            f"{self.dataset_config.dataset_name}_processed",    
        )
        logging.info(f"Saving to disk: {data_path}")
        dataset.save_to_disk(data_path)

    def _ensure_train_test_splits(self, dataset: Dataset | DatasetDict) -> DatasetDict:
        if isinstance(dataset, Dataset):
            split_1 = dataset.train_test_split(
                self.dataset_config.val_split, shuffle=True, seed=self.dataset_config.seed
            )
            return DatasetDict({"train": split_1["train"], "test": split_1["test"]})
        else:
            if "train" in dataset.keys() and "test" in dataset.keys():
                return dataset
            elif "train" in dataset.keys() and "test" not in dataset.keys():
                return self._ensure_train_test_splits(dataset["train"])
            else:
                dataset = dataset[list(dataset.keys())[0]]
                return self._ensure_train_test_splits(dataset)
    
    def _create_splits(self, dataset: DatasetDict | Dataset):
        """
        Creates train, validation, and test splits for the dataset.

        This method creates train, validation, and test splits for the dataset. If the dataset is a `Dataset` object, it is
        split into train, validation, and test splits. If the dataset is a `DatasetDict` object, it checks if the dataset
        already has train, validation, and test splits. If not, it creates them.

        Args:
            dataset (Union[DatasetDict, Dataset]): The dataset to be split. The dataset should be a Hugging Face `datasets.Dataset` or `datasets.DatasetDict` object.

        Returns:
            DatasetDict: The dataset with train, validation, and test splits. The keys are the names of the splits and the values are the datasets for each split.
        """
        if isinstance(dataset, Dataset):
            split_1 = dataset.train_test_split(
                self.dataset_config.val_split, shuffle=True, seed=self.dataset_config.seed
            )
            split_2 = split_1["test"].train_test_split(
                0.2, shuffle=False, seed=self.dataset_config.seed)
            return DatasetDict({"train": split_1["train"], "valid": split_2["train"], "test": split_2["test"]})
        elif isinstance(dataset, DatasetDict):
            # check if dataset has train, valid, test splits
            if "train" in dataset.keys() and "valid" in dataset.keys() and "test" in dataset.keys():
                return dataset
            if "train" in dataset.keys() and "test" in dataset.keys():
                split = dataset["train"].train_test_split(
                    self.dataset_config.val_split, shuffle=True, seed=self.dataset_config.seed
                )
                return DatasetDict({"train": split["train"], "valid": split["test"], "test": dataset["test"]})
            # if dataset has only one key, split it into train, valid, test
            elif "train" in dataset.keys() and "test" not in dataset.keys():
                return self._create_splits(dataset["train"])
            else: 
                return self._create_splits(dataset[list(dataset.keys())[0]])

    def _load_data(self,decode: bool = True ):
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
        if isinstance(dataset, IterableDataset |IterableDatasetDict):
            logging.error("Iterable datasets not supported yet.")
            return
        assert isinstance(dataset, DatasetDict | Dataset)
        dataset = self._ensure_train_test_splits(dataset)


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
    
    def _fast_dev_subset(self, dataset: DatasetDict, size: int=500):
        """
        Selects a subset of the dataset for fast development.

        This method selects the first `size` examples from each split in the dataset. This can be useful for quickly testing
        code during development.

        Args:
            dataset (DatasetDict): A Hugging Face `datasets.DatasetDict` object containing the dataset splits.
            size (int, optional): The number of examples to select from each split. Default is 500.

        Returns:
            DatasetDict: The subsetted dataset. The keys are the names of the dataset splits and the values are the subsetted datasets.
        """
        for split in dataset.keys():
            random_indices = random.sample(range(len(dataset[split])), size)
            dataset[split] = dataset[split].select(random_indices)
        return dataset
    
 
    def _get_dataset(self, split):
        """
        Get Dataset from disk and add run-time transforms to a specified split.
        """
        
        dataset_path = os.path.join(
            self.dataset_config.data_dir,
            f"{self.dataset_config.dataset_name}_processed", 
            split
        )

        dataset = load_from_disk(dataset_path)

        self.transforms.set_mode(split)

        if split == "train": # we need this for sampler, cannot be done later because set_transform
            self.train_label_list = dataset["labels"]

        # add run-time transforms to dataset
        dataset.set_transform(self.transforms, output_all_columns=False) 
        
        return dataset
    
    def _create_weighted_sampler(self):
        label_counts = torch.tensor(self.num_train_labels)
        #calculate sample weights
        sample_weights = (label_counts / label_counts.sum())**(-0.5)    
        #when no_call = 0 --> 0 probability 
        sample_weights = torch.where(
            condition=torch.isinf(sample_weights), 
            input=torch.tensor(0), 
            other=sample_weights
        )

        if self.dataset_config.task == "multiclass":
            weight_list = [sample_weights[classes] for classes in self.train_label_list]
        elif self.dataset_config.task == "multilabel": # sum up weights if multilabel
            weight_list = torch.matmul(torch.tensor(self.train_label_list, dtype=torch.float32), sample_weights)

        weighted_sampler = torch.utils.data.WeightedRandomSampler(
            weight_list, len(weight_list)
        )

        return weighted_sampler
                
    
    def setup(self, stage=None):
        if not self.train_dataset and not self.val_dataset:
            if stage == "fit":
                logging.info("fit")
                self.train_dataset = self._get_dataset("train")
                self.val_dataset = self._get_dataset("valid")

        if not self.test_dataset:
            if stage == "test":
                logging.info("test")
                self.test_dataset = self._get_dataset("test")

    def train_dataloader(self):
        if self.dataset_config.class_weights_sampler is None: 
            return DataLoader(self.train_dataset, **asdict(self.loaders_config.train)) # type: ignore
        else: # change so that it works as a flag 
            weighted_sampler = self._create_weighted_sampler()
            self.loaders_config.train.shuffle = False # mutually exclusive!
            return DataLoader(self.train_dataset, sampler=weighted_sampler, **asdict(self.loaders_config.train))

    def val_dataloader(self):
        return DataLoader(self.val_dataset, **asdict(self.loaders_config.valid)) # type: ignore

    def test_dataloader(self):
        return DataLoader(self.test_dataset, **asdict(self.loaders_config.test)) # type: ignore
