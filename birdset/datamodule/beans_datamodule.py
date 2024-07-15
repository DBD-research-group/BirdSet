from birdset.datamodule.components.transforms import BirdSetTransformsWrapper
from .base_datamodule import BaseDataModuleHF, DatasetConfig, LoadersConfig
from datasets import load_dataset, IterableDataset, IterableDatasetDict, DatasetDict, Audio, Dataset
import logging
from birdset.utils import pylogger
log = pylogger.get_pylogger(__name__)

detection_sets = ['beans_dcase', 'beans_enabirds', 'beans_hiceas', 'beans_rfcx', 'beans_gibbons']

class BEANSDataModule(BaseDataModuleHF):
    def __init__(
            self,
            dataset: DatasetConfig = DatasetConfig(),
            loaders: LoadersConfig = LoadersConfig(),
            transforms: BirdSetTransformsWrapper = BirdSetTransformsWrapper(),
    ):
        super().__init__(
            dataset=dataset,
            loaders=loaders,
            transforms=transforms
        )

    def _preprocess_data(self, dataset):
        """
        Preprocess the data. If multilabel is the task we will one hot encode.
        """

        if self.dataset_config.task == 'multilabel':
            log.info(">> One-hot-encode classes")
            dataset = dataset.map(
                self._classes_one_hot,
                batched=True,
                batch_size=500,
                load_from_cache_file=True,
                num_proc=self.dataset_config.n_workers,
            )

        return dataset



    def _load_data(self,decode: bool = True):    
        """
        Load audio dataset from Hugging Face Datasets. For BEANS the audio column is named path so we will rename it to audio and also rename labels column and remove unamed.

        Returns HF dataset with audio column casted to Audio feature, containing audio data as numpy array and sampling rate.
        """
        logging.info("> Loading data set.")
        dataset = load_dataset(
            name=self.dataset_config.hf_name,
            path=self.dataset_config.hf_path,
            cache_dir=self.dataset_config.data_dir,
            num_proc=3,
        )
        # Leave it here just in case
        if isinstance(dataset, IterableDataset |IterableDatasetDict):
            logging.error("Iterable datasets not supported yet.")
            return
        assert isinstance(dataset, DatasetDict | Dataset)
        dataset = self._ensure_train_test_splits(dataset)

        if self.dataset_config.subset:
            dataset = self._fast_dev_subset(dataset, self.dataset_config.subset)


        if self.dataset_config.dataset_name not in detection_sets:
            # Rename some columns and remove unnamed column (Leftover from BEANS processing)
            
            dataset = dataset.rename_column("label", "labels")
            dataset = dataset.remove_columns('Unnamed: 0')

            # Then we have to map the label to integers if they are strings
            #if isinstance(dataset['train'][0]['labels'],str):
            
            labels = set()
            for split in dataset.keys():
                labels.update(dataset[split]["labels"])

            label_to_id = {lbl: i for i, lbl in enumerate(labels)}

            def label_to_id_fn(batch):
                for i in range(len(batch['labels'])):
                    batch['labels'][i] = label_to_id[batch['labels'][i]]
                return batch

        
            dataset = dataset.map(
                label_to_id_fn,
                batched=True,
                batch_size=500,
                load_from_cache_file=True,
                num_proc=self.dataset_config.n_workers,
            )


        dataset = dataset.rename_column("path", "audio")
        # Normal casting
        dataset = dataset.cast_column(
            column="audio",
            feature=Audio(
                sampling_rate=self.dataset_config.sampling_rate,
                mono=True,
                decode=decode,
            ),
        )
        
        return dataset