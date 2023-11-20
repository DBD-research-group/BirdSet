import torch
import hydra
import logging
import os 

import lightning as L
import torch_audiomentations

from datasets import load_dataset, load_from_disk, Audio, DatasetDict
from torch.utils.data import DataLoader
from omegaconf import DictConfig
from src.datamodule.components.eventMapping import EventSegmenting, EventMapping
from src.datamodule.components.bird_premapping import AudioPreprocessor

class BaseDataModuleHF(L.LightningDataModule):

    def __init__(
            self, 
            dataset: DictConfig,
            loaders: DictConfig,
            transforms: DictConfig
    ):
        
        super().__init__()
        self.dataset = dataset
        self.loaders = loaders
        self.transforms = transforms
        self.feature_extractor = self.dataset.feature_extractor
        
        self.data_path = None
        self.train_dataset = None
        self.val_dataset = None
        self.test_dataset = None

        self._prepare_done = False
        self._setup_done = False
        self.data_path = None
        self.len_trainset = None
    

    def _create_splits(self, dataset):
        logging.info("> Creating Splits.")
        split = dataset["train"].train_test_split(
            self.dataset.val_split, 
            shuffle=True, 
            seed=self.dataset.seed)
        train_dataset = split["train"]
        val_dataset = split["test"]
        test_dataset = dataset["test"]
        return train_dataset, val_dataset, test_dataset
    
    def _get_dataset_(self, split_name, dataset_name):
        pass
    
    # prepare data is 
    def prepare_data(self):
        """
        Prepares the data for use.
        This method loads the dataset, applies transformations, creates
        train, validation, and test splits,
        and saves the processed data to disk. If the data has already been
        prepared, this method does nothing.
        """
        logging.info("Check if preparing has already been done.")

        if self._prepare_done:
            logging.info("Skip preparing.")
            return 
        
        logging.info("> Loading data set.")

        dataset = load_dataset(
            name=self.dataset.hf_name,
            path=self.dataset.hf_path,
            cache_dir=self.dataset.data_dir,
            num_proc=3
        )
        
        dataset = dataset.cast_column(
            column="audio",
            feature=Audio(
                sampling_rate=self.feature_extractor.sampling_rate,
                mono=True,
                decode=True
            )
        )

        logging.info("> Mapping data set.")

        preprocessor = AudioPreprocessor(
            feature_extractor=self.feature_extractor,
            n_classes=self.dataset.n_classes,
            window_length=5
        )

        if self.dataset.task == "multilabel":
            dataset["test_5s"] = dataset["test_5s"].select(range(1000))
            dataset["test"] = dataset["test_5s"].map(
                preprocessor.preprocess_multilabel,
                remove_columns=["audio"],
                batched=True,
                batch_size=100,
                load_from_cache_file=True,
                num_proc=1,
                #num_proc=self.dataset.n_workers,
            )       
            dataset["test"] = dataset["test"].select_columns(["input_values", "labels"])

            dataset["train"] = dataset["train"].select(range(1000))
            dataset["train"] = dataset["train"].map(
                preprocessor.preprocess_multilabel,
                remove_columns=["audio"],
                batched=True,
                batch_size=100,
                load_from_cache_file=True,
                num_proc=1
               #num_proc=self.dataset.n_workers,
            )         
            dataset["train"]=dataset["train"].select_columns(["input_values", "labels"])
            #dataset["train"]=dataset["train"].rename_column("ebird_code", "labels")  

            dataset = DatasetDict(dict(list(dataset.items())[:2]))

        elif self.dataset.task == "multiclass":
            dataset = DatasetDict(dict(list(dataset.items())[:2]))
            dataset = dataset.map(
                preprocessor.preprocess_train,
                remove_columns=["audio"],
                batched=True,
                batch_size=100,
                load_from_cache_file=True,
                num_proc=self.dataset.n_workers,
            )             
            if self.dataset.column_list[1] != "labels":
                dataset = dataset.rename_column(self.dataset.column_list[1], "labels")
   
        # dataset["train"] = dataset["train"].map(
        #     EventMapping(),
        #     batch_size=64,
        #     batched=True,
        #     load_from_cache_file=True,
        #     num_proc=self.dataset.n_workers
        # )
        #dataset = dataset.select_columns(self.dataset.column_list)
        # if self.dataset.column_list[1] != "labels":
        #     dataset = dataset.rename_column(self.dataset.column_list[1], "labels")

        if self.feature_extractor.return_attention_mask:
            self.dataset.column_list.append("attention_mask")
        
        dataset.set_format("np")
        train_dataset, val_dataset, test_dataset = self._create_splits(dataset)
        complete = DatasetDict({
            "train": train_dataset,
            "valid": val_dataset,
            "test": test_dataset
        })
        data_path = os.path.join(
            self.dataset.data_dir, 
            f"{self.dataset.dataset_name}_processed", 
            train_dataset._fingerprint
        )
        self.data_path = data_path
        self._prepare_done = True
        self.len_trainset = len(train_dataset)

        if os.path.exists(data_path):
            logging.info("Dataset exists on disk.")
            return 

        logging.info(f"Saving to disk: {os.path.join(self.data_path)}")
        complete.save_to_disk(self.data_path)
        
    def setup(self, stage=None):
        if not self.train_dataset and not self.val_dataset:
            if stage == "fit":
                logging.info("fit")
                self.train_dataset = load_from_disk(
                    os.path.join(self.data_path,"train")
                )
                self.val_dataset = load_from_disk(
                    os.path.join(self.data_path,"valid")
                )

        if not self.test_dataset:
            if stage == "test":
                logging.info("test")
                self.test_dataset = load_from_disk(
                    os.path.join(self.data_path, "test")
                )

        if self.transforms:
            self.train_dataset.set_transform(
                self.transforms, 
                output_all_columns=False
            )
            self.val_dataset.set_transforms(
                self.transforms, 
                output_all_columns=False
            )
            self.transforms.set_transforms(self.augmentation, 
                output_all_columns=False
            )
    
    # def _preprocess_function(self, batch, task):
    #     audio_arrays = [x["array"] for x in batch["audio"]]
    #     inputs = self.feature_extractor(
    #         audio_arrays,
    #         sampling_rate=self.feature_extractor.sampling_rate,
    #         padding=True,
    #         max_length=self.feature_extractor.sampling_rate*5,
    #         truncation=True,
    #         return_tensors="pt",
    #     )
    #     #check if y is a label list. if so: one-hot encode for multilabel
 
    #     if isinstance(label_list[0], list):
    #         labels = self._classes_one_hot(label_list)
    #         return inputs, labels
        
    #     return inputs

    def _eval_transform(self):
        pass

    def _train_transform(self):
        transform = torch_audiomentations.Compose(
            transforms=[
                torch_audiomentations.Gain(
                    min_gain_in_db=-15.0,
                    max_gain_in_db=5.0,
                    p=0.5,
                    output_type="tensor"
                ),
                torch_audiomentations.AddColoredNoise(
                    p=0.5,
                    sample_rate=32_000,
                    output_type="tensor"
                ),
                torch_audiomentations.PolarityInversion(
                    p=0.5,
                    output_type="tensor"
                )
            ],
            output_type="tensor"
        )
        return transform
    
    def augmentation(self, batch):
        audio = torch.Tensor(batch["input_values"].unsqueeze(1))
        labels = torch.Tensor(batch["primary"])

        augmented = [self._train_transform(raw_audio).squeeze() for raw_audio in audio.unsqueeze(1)]
        batch["input_values"] = augmented
        batch["labels"] = labels
        return batch
    
    def train_dataloader(self):
        #TODO: nontype objects in hf dataset 
        return DataLoader(
            self.train_dataset,
            **self.loaders.get("train")
        )
    
    def val_dataloader(self):
        return DataLoader(
            self.val_dataset,
            **self.loaders.get("valid")
        )

    def test_dataloader(self):
        return DataLoader(
            self.test_dataset,
            **self.loaders.get("test")
        )
