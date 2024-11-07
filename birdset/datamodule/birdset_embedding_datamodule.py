
import os
from birdset import utils
from birdset.datamodule.birdset_datamodule import BirdSetDataModule
from birdset.datamodule.components.transforms import EmbeddingTransforms
from birdset.datamodule.components.event_mapping import XCEventMapping
from birdset.datamodule.components.event_decoding import EventDecoding
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
            transforms: EmbeddingTransforms = EmbeddingTransforms(),
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
        decoder = EventDecoding(min_len=0, max_len=embedding_model.length, sampling_rate=embedding_model.sampling_rate)
        EmbeddingDataModule.__init__(
            self,
            dataset=dataset,
            loaders=loaders,
            transforms=transforms,
            k_samples=k_samples,
            val_batches=val_batches,
            test_ratio=test_ratio,
            low_train=low_train,
            embedding_model=embedding_model,
            decoder = decoder,
            average=average,
            gpu_to_use=gpu_to_use
        )
    
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
            dataset = self._load_data(decode=False)
            dataset = BirdSetDataModule._preprocess_data(self, dataset)
            if "test_5s" in dataset: # Can be removed as it is copied to 'test' split in _preprocess_data
                del dataset["test_5s"]
            dataset = self._compute_embeddings(dataset)

        dataset = self._preprocess_data(dataset)

        # set the length of the training set to be accessed by the model
        self.len_trainset = len(dataset["train"])
        self._save_dataset_to_disk(dataset)

        # set to done so that lightning does not call it again
        self._prepare_done = True
    
    def _preprocess_data(self, dataset):
        # Remove so that the decoder from embedding_transform is not used (Decoding is done before in the embedding_datamodule)
        dataset["train"] = dataset["train"].remove_columns("filepath")
        dataset["test"] = dataset["test"].remove_columns("filepath")
        return EmbeddingDataModule._preprocess_data(self,dataset)
    
    def _concatenate_dataset(self, dataset):
        # We need to cast the start_time and end_time to float64 as train dataset only has None there
        #dataset["train"] = dataset["train"].cast_column("start_time", Sequence(Value("float64")))
        #dataset["train"] = dataset["train"].cast_column("end_time", Sequence(Value("float64")))

        # Should probably only use training data (Non-soundscapes) as otherwise hard to track class presence

        return dataset['train']