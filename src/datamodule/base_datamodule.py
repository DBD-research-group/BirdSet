import logging
import os

import hydra
import lightning as L

from datasets import load_dataset, load_from_disk, Audio, DatasetDict
from torch.utils.data import DataLoader
from omegaconf import DictConfig

from src.datamodule.components.bird_premapping import AudioPreprocessor
from src.datamodule.components.event_mapping import EventMapping
from src.datamodule.components.transforms import TransformsWrapperN
from src.datamodule.components.transforms import TransformsWrapper
import transformers

class BaseDataModuleHF(L.LightningDataModule):
    def __init__(
        self, 
        dataset: DictConfig, 
        loaders: DictConfig, 
        transforms: DictConfig,
        extractors: DictConfig,
        transforms_rene: DictConfig
    ):
        super().__init__()
        self.dataset = dataset
        self.loaders = loaders
        self.transforms = TransformsWrapperN(transforms)
        self.transforms_rene = transforms_rene
        self.feature_extractor = hydra.utils.instantiate(extractors)

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
            self.dataset.val_split, shuffle=True, seed=self.dataset.seed
        )
        train_dataset = split["train"]
        val_dataset = split["test"]
        test_dataset = dataset["test"]
        return train_dataset, val_dataset, test_dataset

    # prepare data is
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

        dataset = dataset.cast_column(
            column="audio",
            feature=Audio(
                sampling_rate=self.feature_extractor.sampling_rate,
                mono=True,
                decode=True,
            ),
        )

        logging.info("> Mapping data set.")

        preprocessor = AudioPreprocessor(
            feature_extractor=self.feature_extractor,
            n_classes=self.dataset.n_classes,
            window_length=5,
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
                # num_proc=self.dataset.n_workers,
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
                # num_proc=self.dataset.n_workers,
            )
            dataset["train"] = dataset["train"].select_columns(
                ["input_values", "labels"]
            )
            # dataset["train"]=dataset["train"].rename_column("ebird_code", "labels")

            dataset = DatasetDict(dict(list(dataset.items())[:2]))

        elif self.dataset.task == "multiclass" and self.dataset.dataset_name != "esc50":
            dataset = DatasetDict(dict(list(dataset.items())[:2]))
            dataset["train"] = dataset["train"].map(
                # TODO add to hydra
                EventMapping(with_noise_cluster=False, biggest_cluster=True, only_one=True),
                remove_columns=["audio"],
                batched=True,
                batch_size=100,
                load_from_cache_file=True,
                num_proc=self.dataset.n_workers,
            )
            dataset = dataset.cast_column("audio", Audio(self.transforms.sampling_rate, mono=True, decode=False))
            #dataset = dataset.select_columns(self.dataset.column_list)

            if self.dataset.column_list[1] != "labels" and self.dataset.dataset_name != "esc50":
                dataset = dataset.rename_column("ebird_code", "labels")

        # TODO: esc50 specific
        if self.dataset.dataset_name == "esc50":
            dataset = dataset.rename_column("target", "labels")

        # dataset["train"] = dataset["train"].map(
        #     EventMapping(),
        #     batch_size=64,
        #     batched=True,
        #     load_from_cache_file=True,
        #     num_proc=self.dataset.n_workers
        # )
        # dataset = dataset.select_columns(self.dataset.column_list)
        # if self.dataset.column_list[1] != "labels":
        #     dataset = dataset.rename_column(self.dataset.column_list[1], "labels")

        if self.feature_extractor.return_attention_mask:
            self.dataset.column_list.append("attention_mask")

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

    def _get_dataset(self, split):
        dataset = load_from_disk(
            os.path.join(self.data_path, split)
        )
        self.transforms.set_mode(split)
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

        # if self.transforms:
        #     if stage == "fit":
        #         self.train_dataset.set_transform(
        #             self.transforms, output_all_columns=False
        #         )
        #         self.val_dataset.set_transform(
        #             self._valid_test_predict_transform, output_all_columns=False
        #         )

            # if stage == "test":
            #     self.test_dataset.set_transform(
            #         self._valid_test_predict_transform, output_all_columns=False
            #     )

    # def _preprocess_function(self, batch):
    #     audio_arrays = [x["array"] for x in batch["audio"]]
    #     inputs = self.feature_extractor(
    #         audio_arrays,
    #         sampling_rate=self.feature_extractor.sampling_rate,
    #         padding=True,
    #         max_length=self.feature_extractor.sampling_rate*5,
    #         truncation=True,
    #         return_tensors="pt",
    #     )
    #     return inputs
    #     #check if y is a label list. if so: one-hot encode for multilabel

    #     if isinstance(label_list[0], list):
    #         labels = self._classes_one_hot(label_list)
    #         return inputs, labels

    #     return inputs

    # def _train_transform(self, examples):
    #     train_transform = hydra.utils.instantiate(
    #         config=self.transforms,
    #         _target_=TransformsWrapper,
    #         mode="train",
    #         sample_rate=self.feature_extractor.sampling_rate,
    #     )

    #     return train_transform(examples)

    # def _valid_test_predict_transform(self, examples):
    #     valid_test_predict_transform = hydra.utils.instantiate(
    #         config=self.transforms_rene,
    #         _target_=TransformsWrapper,
    #         mode="test",
    #         sample_rate=self.feature_extractor.sampling_rate,
    #     )

    #     return valid_test_predict_transform(examples)

    def train_dataloader(self):
        # TODO: nontype objects in hf dataset
        return DataLoader(self.train_dataset, **self.loaders.get("train"))

    def val_dataloader(self):
        return DataLoader(self.val_dataset, **self.loaders.get("valid"))

    def test_dataloader(self):
        return DataLoader(self.test_dataset, **self.loaders.get("test"))
