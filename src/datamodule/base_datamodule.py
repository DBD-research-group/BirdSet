import logging
import os

import lightning as L

from datasets import load_dataset, load_from_disk, Audio, DatasetDict
from torch.utils.data import DataLoader

from omegaconf import DictConfig

from src.datamodule.components.transforms import TransformsWrapper


class BaseDataModule(L.LightningDataModule):
    def __init__(
        self, dataset: DictConfig, loaders: DictConfig, transforms: DictConfig
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
            self.dataset.val_split, shuffle=True, seed=self.dataset.seed
        )
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
        # os.cpu_count() = 32 # num proc only works when prepare is called manually?
        logging.info("> Mapping data set.")
        dataset = dataset.map(
            self._preprocess_function,
            remove_columns=["audio"],
            batched=True,
            batch_size=100,
            load_from_cache_file=True,
            num_proc=self.dataset.n_workers,
        )
        if self.feature_extractor.return_attention_mask:
            self.dataset.column_list.append("attention_mask")

        dataset = dataset.select_columns(self.dataset.column_list)

        if self.dataset.column_list[1] != "labels":
            dataset = dataset.rename_column(self.dataset.column_list[1], "labels")

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

    def setup(self, stage=None):
        if not self.train_dataset and not self.val_dataset:
            if stage == "fit":
                logging.info("fit")
                self.train_dataset = load_from_disk(
                    os.path.join(self.data_path, "train")
                )
                self.val_dataset = load_from_disk(os.path.join(self.data_path, "valid"))

        if not self.test_dataset:
            if stage == "test":
                logging.info("test")
                self.test_dataset = load_from_disk(os.path.join(self.data_path, "test"))

        if self.transforms:
            self.train_dataset.set_transform(
                self._train_transforms, output_all_columns=False
            )
            self.val_dataset.set_transforms(
                self._valid_test_predict_transforms, output_all_columns=False
            )

            self.test_dataset.set_transforms(
                self._valid_test_predict_transforms, output_all_columns=False
            )

    def _preprocess_function(self, batch):
        audio_arrays = [x["array"] for x in batch["audio"]]
        inputs = self.feature_extractor(
            audio_arrays,
            sampling_rate=self.feature_extractor.sampling_rate,
            padding=True,
            max_length=self.feature_extractor.sampling_rate * 5,
            truncation=True,
            return_tensors="pt",
        )
        return inputs

    def _train_transforms(self):
        train_transforms = TransformsWrapper(
            mode="train",
            normalize=self.transforms.normalize,
            use_spectrogram=self.transforms.use_spectrogram,
            n_fft = self.transforms.n_fft,
            hop_length = self.transforms.hop_length,
            n_mels = self.transforms.n_mels,
            db_scale = self.transforms.db_scale,
            waveform_augmentations = self.transforms.waveform_augmentations,
            spectrogram_augmentations = self.transforms.spectrogram_augmentations,
        )
        return train_transforms

    def _valid_test_predict_transforms(self):
        valid_test_predict_transforms = TransformsWrapper(
            mode="test",
            normalize=self.transforms.normalize,
            use_spectrogram=self.transforms.use_spectrogram,
            n_fft = self.transforms.n_fft,
            hop_length = self.transforms.hop_length,
            n_mels = self.transforms.n_mels,
            db_scale = self.transforms.db_scale,
            waveform_augmentations = None,
            spectrogram_augmentations = None,
        )
        return valid_test_predict_transforms

    def train_dataloader(self):
        # TODO: nontype objects in hf dataset
        return DataLoader(self.train_dataset, **self.loaders.get("train"))

    def val_dataloader(self):
        return DataLoader(self.val_dataset, **self.loaders.get("valid"))

    def test_dataloader(self):
        return DataLoader(self.test_dataset, **self.loaders.get("test"))
