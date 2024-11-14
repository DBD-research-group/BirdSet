from datasets import Audio

from birdset.datamodule.components.transforms import BirdSetTransformsWrapper
#from . import BaseDataModuleHF
from birdset.datamodule.base_datamodule import BaseDataModuleHF
from birdset.configs import DatasetConfig, LoadersConfig
from datasets import load_dataset, load_from_disk, Audio, DatasetDict, Dataset, IterableDataset, IterableDatasetDict
from birdset.utils import pylogger
from datasets import Dataset, DatasetDict, concatenate_datasets

log = pylogger.get_pylogger(__name__)

class ESC50DataModule(BaseDataModuleHF):
    def __init__(
            self,
            dataset: DatasetConfig = DatasetConfig(),
            loaders: LoadersConfig = LoadersConfig(),
            transforms: BirdSetTransformsWrapper = BirdSetTransformsWrapper(),
            cross_valid: bool = False,
            fold: int = 1
    ):
        """
        Args: 
        cross_valid (bool, optional): If a cross_valid set should be used or not. Defaults to False which means that the normal split of dataset is used.
        """
        super().__init__(
            dataset=dataset,
            loaders=loaders,
            transforms=transforms,
        )
        self.cross_valid = cross_valid
        self.fold = fold
    

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

        log.info("Check if preparing has already been done.")
        if self._prepare_done:
            log.info("Skip preparing.")
            return

        log.info("Prepare Data")

        dataset = self._load_data()
        print("dataset type:", type(dataset))
        dataset = self._preprocess_data(dataset)
        dataset = self._create_splits(dataset)

        # set the length of the training set to be accessed by the model
        self.len_trainset = len(dataset["train"])
        self._save_dataset_to_disk(dataset)

        # set to done so that lightning does not call it again
        self._prepare_done = True

    def _preprocess_data(self, dataset):
        dataset = dataset.cast_column(
            column="audio",
            feature=Audio(
                sampling_rate=self.dataset_config.sampling_rate,
                mono=True,
                decode=True,
            ),
        )
        dataset = dataset.rename_column("target", "labels")
        dataset = dataset.select_columns(["audio", "labels","fold"])
        return dataset
    
    def _cross_validation(self, dataset, fold):
        # Define fold configurations for cross-validation
        
        fold_combinations = [
            {"test": [1], "train": [2, 3, 4], "validation": [5]},
            {"test": [2], "train": [3, 4, 5], "validation": [1]},
            {"test": [3], "train": [1, 4, 5], "validation": [2]},
            {"test": [4], "train": [1, 2, 5], "validation": [3]},
            {"test": [5], "train": [1, 2, 3], "validation": [4]},
        ]

        # Select the specific fold configuration
        selected_fold = fold_combinations[fold - 1]

        # Filter the datasets for each split (train, validation, test)
        train_splits = [dataset[split_name].filter(lambda example: example["fold"] in selected_fold["train"]) for split_name in dataset]
        val_splits = [dataset[split_name].filter(lambda example: example["fold"] in selected_fold["validation"]) for split_name in dataset]
        test_splits = [dataset[split_name].filter(lambda example: example["fold"] in selected_fold["test"]) for split_name in dataset]

        # Concatenate all filtered datasets into a single Dataset for each split
        train_set = concatenate_datasets(train_splits)
        val_set = concatenate_datasets(val_splits)
        test_set = concatenate_datasets(test_splits)

        # Create a new DatasetDict with the final splits
        cv_dataset = DatasetDict({
            "train": train_set,
            "valid": val_set,
            "test": test_set
        })

        # Optionally, set the format for each split (e.g., to numpy arrays)
        cv_dataset["train"].set_format("np")
        cv_dataset["valid"].set_format("np")
        cv_dataset["test"].set_format("np")

        # Return the cross-validated dataset
        return cv_dataset


    def _create_splits(self, dataset: DatasetDict | Dataset) -> DatasetDict:
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
        if self.cross_valid:
            if isinstance(dataset, DatasetDict):
                # Get cross-validation dataset without wrapping it in another DatasetDict
                dataset = self._cross_validation(dataset, self.fold)
                # `dataset` is already a DatasetDict at this point, so no need to wrap it
                return dataset
   
        else:
            if isinstance(dataset, Dataset):
                split_1 = dataset.train_test_split(
                    self.dataset_config.val_split, shuffle=True, seed=self.dataset_config.seed
                )
                split_2 = split_1["test"].train_test_split(
                    0.2, shuffle=False, seed=self.dataset_config.seed)
                return DatasetDict({"train": split_1["train"], "valid": split_2["train"], "test": split_2["test"]})
            elif isinstance(dataset, DatasetDict):
                # check if dataset has train, valid, test splits
                if "train" in dataset.keys() and "valid" in dataset.keys():
                    return dataset
                if "train" in dataset.keys() and "valid" in dataset.keys() and "test" in dataset.keys():
                    return dataset
                if "train" in dataset.keys() and "test" in dataset.keys():
                    split = dataset["train"].train_test_split(
                        self.dataset_config.val_split, shuffle=True, seed=self.dataset_config.seed
                    )
                    return DatasetDict({"train": split["train"], "valid": split["test"], "test": dataset["test"]})
                else:
                    return self._create_splits(dataset[list(dataset.keys())[0]])
    
   