from birdset.datamodule.components.transforms import BirdSetTransformsWrapper
from birdset.datamodule.components.event_mapping import XCEventMapping
from .base_datamodule import BaseDataModuleHF, DatasetConfig, LoadersConfig
from datasets import load_dataset, Audio
from birdset.utils import pylogger

log = pylogger.get_pylogger(__name__)

class InferenceDataModule(BaseDataModuleHF):
    def __init__(
            self,
            dataset: DatasetConfig = DatasetConfig(),
            loaders: LoadersConfig = LoadersConfig(),
            transforms: BirdSetTransformsWrapper = BirdSetTransformsWrapper(),
            mapper: XCEventMapping = XCEventMapping()
    ):
        super().__init__(
            dataset=dataset,
            loaders=loaders,
            transforms=transforms,
            mapper=mapper
        )

    def _load_data(self, decode: bool = False):
        """
        Load audio dataset from Hugging Face Datasets.

        Returns HF dataset with audio column casted to Audio feature, containing audio data as numpy array and sampling rate.
        """
        log.info("> Loading data set.")

        dataset = load_dataset(
            name=self.dataset_config.hf_name,
            path=self.dataset_config.hf_path,
            cache_dir=self.dataset_config.data_dir,
            num_proc=3,
        )

        if self.dataset_config.subset:
            print("Fast dev subset not supported for only inference")

        dataset = dataset.cast_column(
            column="audio",
            feature=Audio(
                sampling_rate=self.dataset_config.sampling_rate,
                mono=True,
                decode=decode,
            ),
        )

        return dataset
    
    def _preprocess_data(self, dataset):
        if self.dataset_config.task == "multiclass":
            dataset = dataset.rename_column("ebird_code", "labels")
            dataset = dataset["test"]

        elif self.dataset_config.task == "multilabel":
            dataset = dataset.rename_column("ebird_code_multilabel", "labels")
            del dataset["train"]
            del dataset["test"]
            dataset["test"] = dataset["test_5s"]
            del dataset["test_5s"]

            log.info(">> One-hot-encode classes")
            dataset = dataset.map(
                self._classes_one_hot,
                batched=True,
                batch_size=500,
                load_from_cache_file=False,
                num_proc=1,
            )

        dataset["test"] = dataset["test"].select_columns(
            ["filepath", "labels", "detected_events", "start_time", "end_time"]
        )
        return dataset

    def prepare_data(self):
            
        log.info("Check if preparing has already been done.")
        if self._prepare_done:
            log.info("Skip preparing.")
            return
    
        log.info("Prepate Data for Inference!")
        dataset = self._load_data()
        dataset = self._preprocess_data(dataset)
        # no split creation
        self.len_trainset = 0

        self._save_dataset_to_disk(dataset)
        self._prepare_done = True

    
