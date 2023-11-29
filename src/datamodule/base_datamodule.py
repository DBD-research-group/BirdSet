from dataclasses import asdict, dataclass, field
import logging
import torch
import os
from typing import List, Literal

import lightning as L

from datasets import load_dataset, load_from_disk, Audio, DatasetDict, Dataset
from src.utils.extraction import DefaultFeatureExtractor
from torch.utils.data import DataLoader

from src.datamodule.components.bird_premapping import AudioPreprocessor
#from src.datamodule.components.event_mapping import EventMapping
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
    # the columns of the dataset that should be used
    column_list: List[str] = field(default_factory=lambda: ["audio", "target"])
    val_split: float = 0.2
    task: Literal["multiclass", "multilabel"] = "multiclass"
    subset: int|None = None

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
        mapper,
        dataset: DatasetConfig = DatasetConfig(),
        loaders: LoadersConfig = LoadersConfig(),
        transforms: TransformsWrapper = TransformsWrapper(),
        extractors: DefaultFeatureExtractor = DefaultFeatureExtractor(),
        ):
        super().__init__()
        self.dataset_config = dataset
        self.loaders_config = loaders
        self.transforms = transforms
        self.feature_extractor = extractors
        self.event_mapper = mapper

        self.data_path = None
        self.train_dataset = None
        self.val_dataset = None
        self.test_dataset = None

        self._prepare_done = False
        self._setup_done = False
        self.data_path = None
        self.len_trainset = None
    
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

        logging.info("> Loading data set.")

        dataset = load_dataset(
            name=self.dataset_config.hf_name,
            path=self.dataset_config.hf_path,
            cache_dir=self.dataset_config.data_dir,
            num_proc=3,
        )

        if isinstance(dataset, DatasetDict | Dataset):
            dataset = self._create_splits(dataset)
        else:
            logging.error("Dataset is not a DatasetDict or Dataset, Iterabel Dataset not supported yet.")
            return

        if self.dataset_config.subset:
            dataset = self._fast_dev_subset(dataset, self.dataset_config.subset)

        dataset = dataset.cast_column(
            column="audio",
            feature=Audio(
                sampling_rate=self.feature_extractor.sampling_rate,
                mono=True,
                decode=True,
            ),
        )

        logging.info("> Mapping data set.")

        if self.dataset.task == "multiclass": #and self.dataset.dataset_name != "esc50":
            dataset = DatasetDict({split: dataset[split] for split in ["train", "test"]})

            dataset["train"] = dataset["train"].map(
                self.event_mapper,
                remove_columns=["audio"],
                batched=True,
                batch_size=300,
                load_from_cache_file=True,
                num_proc=self.dataset.n_workers,
            )

            dataset = dataset.select_columns(
                ["filepath", "ebird_code", "detected_events", "start_time", "end_time"]
            )

            dataset = dataset.rename_column("ebird_code", "labels")

            # if self.dataset.column_list[1] != "labels" and self.dataset.dataset_name != "esc50":
            #     dataset = dataset.rename_column("ebird_code", "labels")

        elif self.dataset.task == "multilabel":
            dataset = DatasetDict({split: dataset[split] for split in ["train", "test_5s"]})
            
            dataset["train"] = dataset["train"].map(
                self.event_mapper,
                remove_columns=["audio"],
                batched=True,
                batch_size=300,
                load_from_cache_file=True,
                num_proc=self.dataset.n_workers,
            )

            dataset = dataset.map(
                self._classes_one_hot,
                batched=True,
                batch_size=300,
                load_from_cache_file=False,
                num_proc=self.dataset.n_workers,
            )

            dataset["test"] = dataset["test_5s"]
            dataset = dataset.select_columns(
                ["filepath", "ebird_code_multilabel", "detected_events", "start_time", "end_time"]
            )

            dataset = dataset.rename_column("ebird_code_multilabel", "labels")
            
        # # TODO: esc50 specific
        # if self.dataset.dataset_name == "esc50":
        #     dataset = dataset.rename_column("target", "labels")

        # if self.feature_extractor.return_attention_mask:
        #     self.dataset.column_list.append("attention_mask")

        self._save_dataset_to_disk(dataset)
       

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
            dataset['train']._fingerprint,
        )
        self.data_path = data_path
        self._prepare_done = True

        if os.path.exists(data_path):
            logging.warn("Dataset exists on disk.")
            return

        logging.info(f"Saving to disk: {os.path.join(self.data_path)}")
        dataset.save_to_disk(self.data_path)

    def _select_and_rename_columns(self, dataset):

        dataset = dataset.select_columns(self.dataset_config.column_list)
        # if audio not in dataset
        if "audio" not in dataset['train'].column_names:
            dataset = dataset.rename_column(self.dataset_config.column_list[0], "audio")
        if 'labels' not in dataset['train'].column_names:
            dataset = dataset.rename_column(self.dataset_config.column_list[1], "labels")
        return dataset

    
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
            self.dataset_config.val_split, shuffle=True, seed=self.dataset_config.seed)
            split_2 = split_1["test"].train_test_split(
                0.5, shuffle=False, seed=self.dataset_config.seed)
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
            else:
                return self._create_splits(dataset[list(dataset.keys())[0]])


    
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
            dataset[split] = dataset[split].select(range(size))
        return dataset
    
    def _classes_one_hot(self, batch):
        label_list = [y for y in batch["ebird_code_multilabel"]]
        class_one_hot_matrix = torch.zeros(
            (len(label_list), self.dataset.n_classes), dtype=torch.float
        )

        for class_idx, idx in enumerate(label_list):
            class_one_hot_matrix[class_idx, idx] = 1

        class_one_hot_matrix = torch.tensor(class_one_hot_matrix, dtype=torch.float32)
        return {"ebird_code_multilabel": class_one_hot_matrix}   

    def _preprocess_multilabel(self, dataset, split, preprocessor, select_range=None):
        """
        Preprocesses a multilabel dataset.

        This method applies preprocessing to each split in the dataset. The preprocessing includes feature extraction,
        selection of a range of data, and mapping of the preprocessing function to the data. The "audio" column is removed
        from the dataset, and only the "input_values" and "labels" columns are kept.

        Args:
            dataset (dict): A dictionary where the keys are the names of the dataset splits and the values are the datasets.
            select_range (list, optional): A list specifying the range of data to select from each dataset split. If None,
                all data is selected. Default is None.

        Returns:
            dict: The preprocessed dataset. The keys are the names of the dataset splits and the values are the preprocessed datasets.
        """
        preprocessor = AudioPreprocessor(
            feature_extractor=self.feature_extractor,
            n_classes=self.dataset_config.n_classes,
            window_length=5,
        )
        for split in dataset.keys():
            # map through dataset split and apply preprocessing
            dataset[split] = dataset[split].map(
                preprocessor.preprocess_multilabel,
                remove_columns=["audio"],
                batched=True,
                batch_size=100,
                load_from_cache_file=True,
                num_proc=self.dataset_config.n_workers,
            )
            dataset[split] = dataset[split].select_columns(["input_values", "labels"])  
        return dataset
    
    def _preprocess_multiclass(self, dataset):
        return dataset

    def _get_dataset(self, split):
        dataset = load_from_disk(
            os.path.join(self.data_path, split)
        )
        self.transforms.set_mode(split)
        # add run-time transforms to dataset
        dataset.set_transform(self.transforms, output_all_columns=False) 
        
        return dataset

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
        # TODO: nontype objects in hf dataset
        return DataLoader(self.train_dataset, **asdict(self.loaders_config.train)) # type: ignore

    def val_dataloader(self):
        return DataLoader(self.val_dataset, **asdict(self.loaders_config.valid)) # type: ignore

    def test_dataloader(self):
        return DataLoader(self.test_dataset, **asdict(self.loaders_config.test)) # type: ignore
