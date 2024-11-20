from datasets import Audio

from birdset.datamodule.components.transforms import BirdSetTransformsWrapper
from birdset.datamodule.embedding_datamodule import EmbeddingDataModule, EmbeddingModuleConfig
from birdset.datamodule.esc50_datamodule import ESC50DataModule
from birdset.configs import DatasetConfig, LoadersConfig
from datasets import load_dataset, IterableDataset, IterableDatasetDict, DatasetDict, Audio, Dataset, load_from_disk
from birdset.utils import pylogger
import os

log = pylogger.get_pylogger(__name__)


class ESC50EmbeddingDataModule(EmbeddingDataModule, ESC50DataModule):
    def __init__(
            self,
            dataset: DatasetConfig = DatasetConfig(),
            loaders: LoadersConfig = LoadersConfig(),
            transforms: BirdSetTransformsWrapper = BirdSetTransformsWrapper(),
            k_samples: int = 0,
            val_batches: int = None, # Should val set be created
            test_ratio: float = 0.5, # Ratio of test set if val set is also created
            low_train: bool = False, # If low train set is used
            embedding_model: EmbeddingModuleConfig = EmbeddingModuleConfig(),
            average: bool = True,
            gpu_to_use: int = 0,
            cross_valid: bool = False,
            fold : int = 1,
            
            
    ):
           
        
        """
        DataModule for using BEANS and extracting embeddings.

        Args:
            dataset (DatasetConfig, optional): The config of the dataset to use. Defaults to DatasetConfig().
            loaders (LoadersConfig, optional): Loader config. Defaults to LoadersConfig().
            transforms (BirdSetTransformsWrapper, optional): uses EmbeddingsTransform so no Embeddings are cut off. Defaults to EmbeddingTransforms().
            k_samples (int, optional): The amount of samples per class that should be used. Defaults to 0 where the predefined sets are used.
            val_batches (int, optional): If a validation set should be used or not. Defaults to None which means that the normal validation split or amount is used.
            test_ratio (float, optional): Ratio of test set if val set is also created. Defaults to 0.5.
            low_train (bool, optional): If low train set is used (Exists in BEANS for example). Defaults to False.
            embedding_model (EmbeddingModuleConfig, optional): Model for extracting the embeddings. Defaults to EmbeddingModuleConfig().
            average (bool, optional): If embeddings should be averaged if the audio clip is too long. Defaults to True.
            gpu_to_use (int, optional): Which GPU should be used for extracting the embeddings. Defaults to 0.
            cross_valid (bool, optional): If a cross_valid set should be used or not. Defaults to False which means that the normal split of dataset is used.
            fold (int, optional) : fold defines which combination of data should assume as subsection in cross_validation structure. Defult 1, it can be 1 to 5.
        """
        ESC50DataModule.__init__(
            self,
            dataset=dataset,
            loaders=loaders,
            transforms=transforms,
            cross_valid= cross_valid,
            fold= fold
            
            
        )
        

        EmbeddingDataModule.__init__(
            self,
            dataset=dataset,
            loaders=loaders,
            transforms=transforms,
            k_samples = k_samples,
            val_batches = val_batches,
            test_ratio = test_ratio,
            low_train = low_train,
            embedding_model = embedding_model,
            average = average,
            gpu_to_use = gpu_to_use
            
        )

        self.cross_valid = cross_valid
        self.fold = fold

    def prepare_data(self):
        """
        Same as prepare_data in esc50_embedding_datamodule but checks if path exists and skips rest otherwise

        """
        log.info("Check if preparing has already been done.")
        if self._prepare_done:
            log.info("Skip preparing.")
            return
                 # Check if the embeddings for the dataset have already been computed
        if os.path.exists(self.embeddings_save_path):
            log.info(f"Embeddings found in {self.embeddings_save_path}, loading from disk")
            dataset = load_from_disk(self.embeddings_save_path)
        else:
            log.info("Prepare Data")
            dataset = self._load_data()
            dataset = self._compute_embeddings(dataset)


            
        print("dataset type:", type(dataset))
        dataset = self._preprocess_data(dataset)
        dataset = self._create_splits(dataset)

        # set the length of the training set to be accessed by the model
        self.len_trainset = len(dataset["train"])
        self._save_dataset_to_disk(dataset)

        # set to done so that lightning does not call it again
        self._prepare_done = True
        
    def _preprocess_data(self, dataset: Dataset|DatasetDict):
    
        dataset = dataset.rename_column("target", "labels")
        dataset = dataset.select_columns(["embedding", "labels","fold"])

        dataset = EmbeddingDataModule._preprocess_data(self, dataset)
        
            
        return dataset
    
    
    def _save_dataset_to_disk(self, dataset: Dataset | DatasetDict):
        """
        Saves the dataset to disk.

        This method sets the format of the dataset to numpy, prepares the path where the dataset will be saved, and saves
        the dataset to disk. If the dataset already exists on disk, it does not save the dataset again.

        Args:
            dataset (datasets.DatasetDict): The dataset to be saved. The dataset should be a Hugging Face `datasets.DatasetDict` object.

        Returns:
            None
        """
        dataset.set_format("np")
        if self.cross_valid:
            fingerprint = f"{dataset[next(iter(dataset))]._fingerprint}_{self.fold}" if isinstance(dataset, DatasetDict) else f"{dataset._fingerprint}_{self.fold}"  # changed to next_iter to be more robust
        else:
            fingerprint = dataset[next(iter(dataset))]._fingerprint if isinstance(dataset, DatasetDict) else dataset._fingerprint  # changed to next_iter to be more robust

        self.disk_save_path = os.path.join(
            self.dataset_config.data_dir,
            f"{self.dataset_config.dataset_name}_processed_{self.dataset_config.seed}_{fingerprint}",
        )

        if os.path.exists(self.disk_save_path):
            log.info(f"Train fingerprint found in {self.disk_save_path}, saving to disk is skipped")
        else:
            log.info(f"Saving to disk: {self.disk_save_path}")
            dataset.save_to_disk(self.disk_save_path)
        
        
        
        
          
       