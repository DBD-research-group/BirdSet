
import os
from birdset import utils
from birdset.datamodule.birdset_datamodule import BirdSetDataModule
from birdset.datamodule.components.transforms import BirdSetTransformsWrapper
from birdset.datamodule.components.event_mapping import XCEventMapping
from birdset.datamodule.embedding_datamodule import EmbeddingDataModule, EmbeddingModuleConfig
from birdset.configs import DatasetConfig, LoadersConfig
from datasets import load_from_disk

log = utils.get_pylogger(__name__)


class BirdSetEmbeddingDataModule(EmbeddingDataModule, BirdSetDataModule):
    def __init__(
            self,
            dataset: DatasetConfig = DatasetConfig(
                dataset_name='HSN',
                hf_path='DBD-research-group/BirdSet',
                hf_name='HSN',
                n_workers=3,
                val_split=0.2,
                task="multilabel",
                classlimit=500,
                eventlimit=5,
                sampling_rate=32000,
            ),
            loaders: LoadersConfig = LoadersConfig(),
            transforms: BirdSetTransformsWrapper = BirdSetTransformsWrapper(),
            mapper: XCEventMapping = XCEventMapping(),
            k_samples: int = 0,
            val_batches: int = None,  # Should val set be created
            test_ratio: float = 0.5,  # Ratio of test set if val set is also created
            low_train: bool = False,  # If low train set is used
            embedding_model: EmbeddingModuleConfig = EmbeddingModuleConfig(),
            average: bool = True,
            gpu_to_use: int = 0,
    ):
        BirdSetDataModule.__init__(
            self,
            dataset=dataset,
            loaders=loaders,
            transforms=transforms,
            mapper=mapper
        )

        EmbeddingDataModule.__init__(
            self,
            dataset=dataset,
            k_samples=k_samples,
            val_batches=val_batches,
            test_ratio=test_ratio,
            low_train=low_train,
            embedding_model=embedding_model,
            average=average,
            gpu_to_use=gpu_to_use
        )
    
    @property
    def num_classes(self):
        return super(BirdSetDataModule, self).num_classes
    
    def prepare_data(self):
        """
        Same as prepare_data in BaseDataModuleHF but checks if path exists and skips rest otherwise
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
            dataset = self._load_data(decode=True)
            dataset = BirdSetDataModule._preprocess_data(self, dataset)
            dataset = self._compute_embeddings(dataset)

        dataset = self._preprocess_data(dataset)

        # set the length of the training set to be accessed by the model
        self.len_trainset = len(dataset["train"])
        self._save_dataset_to_disk(dataset)

        # set to done so that lightning does not call it again
        self._prepare_done = True
    
    def _preprocess_data(self, dataset):
        return EmbeddingDataModule._preprocess_data(dataset)