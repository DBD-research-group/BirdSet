import torch
import hydra
import logging
import os 

import lightning as L
import torch_audiomentations

from datasets import load_dataset, load_from_disk, Audio, DatasetDict
from torch.utils.data import DataLoader
from transformers import AutoFeatureExtractor


class BaseDataModule(L.LightningDataModule):

    def __init__(
            self,
            data_dir,
            dataset_name,
            feature_extractor_name,
            hf_path,
            hf_name,
            seed, 
            train_batch_size, 
            eval_batch_size, 
            val_split,
            column_list=None, 
            transforms=None,
            num_workers=1
    ):
        
        super().__init__()
        self.data_dir = data_dir
        self.dataset_name = dataset_name
        self.feature_extractor = AutoFeatureExtractor.from_pretrained(
            feature_extractor_name
        )
        self.hf_path = hf_path
        self.hf_name = hf_name
        self.seed = seed
        self.train_batch_size = train_batch_size
        self.eval_batch_size = eval_batch_size
        self.val_split = val_split
        self.column_list = column_list
        self.num_workers = num_workers

        self.transforms = transforms
        
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
            self.val_split, 
            shuffle=True, 
            seed=self.seed)
        train_dataset = split["train"]
        val_dataset = split["test"]
        test_dataset = dataset["test"]
        return train_dataset, val_dataset, test_dataset
    
    def _get_dataset_(self, split_name, dataset_name):
        pass
    
    # prepare data is 
    def prepare_data(self):
        logging.info("Check if preparing has already been done.")

        if self._prepare_done:
            logging.info("Skip preparing.")
            return 
        
        logging.info("> Loading data set.")

        dataset = load_dataset(
            name=self.hf_name,
            path=self.hf_path,
            cache_dir=self.data_dir
        )
        
        dataset = dataset.cast_column(
            column="audio",
            feature=Audio(
                sampling_rate=self.feature_extractor.sampling_rate,
                mono=True,
                decode=True
            )
        )
        #os.cpu_count() = 32 # num proc only works when prepare is called manually?
        logging.info("> Mapping data set.")
        dataset = dataset.map(
            self._preprocess_function,
            remove_columns=["audio"],
            batched=True,
            batch_size=100,
            load_from_cache_file=True,
            num_proc=self.num_workers,
        )
        if self.feature_extractor.return_attention_mask:
            self.column_list.append("attention_mask")

        dataset = dataset.select_columns(self.column_list)
        
        if self.column_list[1] != "labels":
            dataset = dataset.rename_column(self.column_list[1], "labels")

        dataset.set_format("np")
        train_dataset, val_dataset, test_dataset = self._create_splits(dataset)
        complete = DatasetDict({
            "train": train_dataset,
            "valid": val_dataset,
            "test": test_dataset
        })
        data_path = os.path.join(
            self.data_dir, 
            f"{self.dataset_name}_processed", 
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
                    os.path.join(self.data_path,"test")
                )

        if self.transforms:
            self.train_dataset.set_transform(
                self.augmentation, 
                output_all_columns=False
            )
            self.val_dataset.set_transforms(
                self.augmentation, 
                output_all_columns=False
            )
            self.test_dataset.set_transforms(self.augmentation, 
                output_all_columns=False
            )
            
    def _preprocess_function(self, batch):
        audio_arrays = [x["array"] for x in batch["audio"]]
        inputs = self.feature_extractor(
            audio_arrays,
            sampling_rate=self.feature_extractor.sampling_rate,
            padding=True,
            max_length=self.feature_extractor.sampling_rate*5,
            truncation=True,
            return_tensors="pt",
        )
        return inputs

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
            batch_size=self.train_batch_size,
            shuffle=False,
            num_workers=4)
    
    def val_dataloader(self):
        return DataLoader(
            self.val_dataset,
            batch_size=self.eval_batch_size,
            shuffle=False,
            num_workers=4
        )

    def test_dataloader(self):
        return DataLoader(
            self.test_dataset,
            batch_size=self.eval_batch_size,
            shuffle=False,
            num_workers=4
        )