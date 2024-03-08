from dataclasses import asdict, dataclass, field
import logging
import torch
import random
import os
from typing import List, Literal, Optional
from collections import Counter
from torch.utils.data import Subset
from tqdm import tqdm
import lightning as L
import numpy as np
import pandas as pd
import hydra 

from datasets import load_dataset, load_from_disk, Audio, DatasetDict, Dataset, IterableDataset, IterableDatasetDict
from torch.utils.data import DataLoader
from src.datamodule.components.event_mapping import XCEventMapping
from src.datamodule.components.transforms import GADMETransformsWrapper

@dataclass
class DatasetConfig:
    """
    A class used to configure the dataset for the model.

    Attributes
    ----------
    data_dir : str
        Specifies the directory where the dataset files are stored.  **Important**: The dataset uses a lot of disk space, so make sure you have enough storage available.
    dataset_name : str
        The name assigned to the dataset.
    hf_path : str
        The path to the dataset stored on HuggingFace.
    hf_name : str
        The name of the dataset on HuggingFace.
    seed : int
        A seed value for ensuring reproducibility across runs.
    n_classes : int
        The total number of distinct classes in the dataset.
    n_workers : int
        The number of worker processes used for data loading.
    val_split : float
        The proportion of the dataset reserved for validation.
    task : str
        Defines the type of task (e.g., 'multilabel' or 'multiclass').
    subset : int, optional
        A subset of the dataset to use. If None, the entire dataset is used.
    sampling_rate : int
        The sampling rate for audio data processing.
    class_weights_loss : bool, optional
        (Deprecated) Previously used for applying class weights in loss calculation.
    class_weights_sampler : bool, optional
        Indicates whether to use class weights in the sampler for handling imbalanced datasets.
    classlimit : int, optional
        The maximum number of samples per class. If None, all samples are used.
    eventlimit : int, optional
        Defines the maximum number of audio events processed per audio file, capping the quantity to ensure balance across files. If None, all events are processed.
    """
    data_dir: str = "/workspace/data_gadme"
    dataset_name: str = "esc50"
    hf_path: str = "ashraq/esc50"
    hf_name: str = ""
    seed: int = 42
    n_classes: int = 50
    n_workers: int = 1
    val_split: float = 0.2
    task: Literal["multiclass", "multilabel"] = "multiclass"
    subset: Optional[int] = None
    sampling_rate: int = 32_000
    class_weights_loss: Optional[bool] = None
    class_weights_sampler: Optional[bool] = None
    classlimit: Optional[int] = None
    eventlimit: Optional[int] = None


@dataclass
class LoaderConfig:
    """
    A class used to configure the data loader for the model.

    Attributes
    ----------
    batch_size : int
        Specifies the number of samples contained in each batch. This is a crucial parameter as it impacts memory utilization and model performance.
    shuffle : bool
        Determines whether the data is shuffled at the beginning of each epoch. Shuffling is typically used for training data to ensure model robustness and prevent overfitting.
    num_workers : int
        Sets the number of subprocesses to be used for data loading. More workers can speed up the data loading process but also increase memory consumption.
    pin_memory : bool
        When set to `True`, enables the DataLoader to copy Tensors into CUDA pinned memory before returning them. This can lead to faster data transfer to CUDA-enabled GPUs.
    drop_last : bool
        Determines whether to drop the last incomplete batch. Setting this to `True` is useful when the total size of the dataset is not divisible by the batch size.
    persistent_workers : bool
        Indicates whether the data loader should keep the workers alive for the next epoch. This can improve performance at the cost of memory.
    prefetch_factor : int
        Defines the number of samples loaded in advance by each worker. This parameter is commented out here and can be adjusted based on specific requirements.
    """
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
        transforms (GADMETransformsWrapper): Configuration for the data transformations. Defaults to an instance of `GADMETransformsWrapper`.
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
        transforms: GADMETransformsWrapper = GADMETransformsWrapper(),
        ):
        super().__init__()
        self.dataset_config = dataset
        self.loaders_config = loaders
        self.transforms = transforms #hier erst nach 15 iterations??
        self.event_mapper = mapper

        self.data_path = None
        self.train_dataset = None
        self.val_dataset = None
        self.test_dataset = None

        self._prepare_done = False
        self.len_trainset = None
        self.num_train_labels = None
        self.train_label_list = None
        self.disk_save_path = None

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
        fingerprint = dataset["train"]._fingerprint

        self.disk_save_path = os.path.join(
            self.dataset_config.data_dir,
            f"{self.dataset_config.dataset_name}_processed_{self.dataset_config.seed}_{fingerprint}",
        )

        if os.path.exists(self.disk_save_path):
            logging.info(f"Train fingerprint found in {self.disk_save_path}, saving to disk is skipped")
        else:
            logging.info(f"Saving to disk: {self.disk_save_path}")
            dataset.save_to_disk(self.disk_save_path)

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
        
        dataset = load_from_disk(os.path.join(self.disk_save_path, split))

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

    def _count_labels(self, labels):
        # frequency
        label_counts = Counter(labels)

        if 0 not in label_counts:
            label_counts[0] = 0
        
        num_labels = max(label_counts)
        counts = [label_counts[i] for i in range(num_labels+1)]
        
        return counts

    def _limit_classes(self, dataset, label_name, limit):
        # Count labels
        label_counts = Counter(dataset[label_name])

        # Gather indices for each class
        all_indices = {label: [] for label in label_counts.keys()}
        for idx, label in enumerate(dataset[label_name]):
            all_indices[label].append(idx)

        # Randomly select indices for classes exceeding the limit
        limited_indices = []
        for label, indices in all_indices.items():
            if label_counts[label] > limit:
                limited_indices.extend(random.sample(indices, limit))
            else:
                limited_indices.extend(indices)
        # Subset the dataset
        return dataset.select(limited_indices)

    def _unique_identifier(self, x, label_name):
        file = x["filepath"]
        label = x[label_name]
        return {"id": f"{file}-{label}"}

    def _smart_sampling(self, dataset, label_name, class_limit, event_limit):
        class_limit = class_limit if class_limit else -float("inf")
        dataset = dataset.map(lambda x: self._unique_identifier(x, label_name))
        df = pd.DataFrame(dataset)
        path_label_count = df.groupby(["id", label_name], as_index=False).size()
        path_label_count = path_label_count.set_index("id")
        class_sizes = df.groupby(label_name).size()

        for label in tqdm(class_sizes.index, desc="Processing labels"):
            current = path_label_count[path_label_count[label_name] == label]
            total = current["size"].sum()
            most = current["size"].max()

            while total > class_limit or most != event_limit:
                largest_count = current["size"].value_counts()[current["size"].max()]
                n_largest = current.nlargest(largest_count + 1, "size")
                to_del = (n_largest["size"].max() - n_largest["size"].min())

                idxs = n_largest[n_largest["size"] == n_largest["size"].max()].index
                if total - (to_del * largest_count) < class_limit or most == event_limit or most == 1:
                    break
                for idx in idxs:
                    current.at[idx, "size"] = current.at[idx, "size"] - to_del
                    path_label_count.at[idx, "size"] = path_label_count.at[idx, "size"] - to_del

                total = current["size"].sum()
                most = current["size"].max()

        event_counts = Counter(dataset["id"])

        all_file_indices = {label: [] for label in event_counts.keys()}
        for idx, label in enumerate(dataset["id"]):
            all_file_indices[label].append(idx)

        limited_indices = []
        for file, indices in all_file_indices.items():
            limit = path_label_count.loc[file]["size"]
            limited_indices.extend(random.sample(indices, limit))

        dataset = dataset.remove_columns("id")
        return dataset.select(limited_indices)

    def _classes_one_hot(self, batch):
        """
        Converts class labels to one-hot encoding.

        This method takes a batch of data and converts the class labels to one-hot encoding.
        The one-hot encoding is a binary matrix representation of the class labels.

        Args:
            batch (dict): A batch of data. The batch should be a dictionary where the keys are the field names and the values are the field data.

        Returns:
            dict: The batch with the "labels" field converted to one-hot encoding. The keys are the field names and the values are the field data.
        """
        label_list = [y for y in batch["labels"]]
        class_one_hot_matrix = torch.zeros(
            (len(label_list), self.dataset_config.n_classes), dtype=torch.float
        )

        for class_idx, idx in enumerate(label_list):
            class_one_hot_matrix[class_idx, idx] = 1

        class_one_hot_matrix = torch.tensor(class_one_hot_matrix, dtype=torch.float32)
        return {"labels": class_one_hot_matrix}

    def train_dataloader(self):
        if self.dataset_config.class_weights_sampler:
            weighted_sampler = self._create_weighted_sampler()
            self.loaders_config.train.shuffle = False  # Mutually exclusive with sampler!
            # Use the weighted_sampler in the DataLoader
            return DataLoader(self.train_dataset, sampler=weighted_sampler, **asdict(self.loaders_config.train))
        else:
            # If class_weights_sampler is not True, return a regular DataLoader without the weighted sampler
            return DataLoader(self.train_dataset, **asdict(self.loaders_config.train))  # type: ignore

    def val_dataloader(self):
        return DataLoader(self.val_dataset, **asdict(self.loaders_config.valid)) # type: ignore

    def test_dataloader(self):
        return DataLoader(self.test_dataset, **asdict(self.loaders_config.test)) # type: ignore