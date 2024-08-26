from birdset.datamodule.components.transforms import BirdSetTransformsWrapper
from birdset.datamodule.embeddings_datamodule import EmbeddingDataModule, EmbeddingModuleConfig
from birdset.configs import DatasetConfig, LoadersConfig
from datasets import load_dataset, IterableDataset, IterableDatasetDict, DatasetDict, Audio, Dataset
from birdset.utils import pylogger
import logging
log = pylogger.get_pylogger(__name__)

detection_sets = ['beans_dcase', 'beans_enabirds', 'beans_hiceas', 'beans_rfcx', 'beans_gibbons']

class BEANSDataModule(EmbeddingDataModule):
    def __init__(
            self,
            dataset: DatasetConfig = DatasetConfig(),
            loaders: LoadersConfig = LoadersConfig(),
            transforms: BirdSetTransformsWrapper = BirdSetTransformsWrapper(),
            mapper: None = None,
            k_samples: int = 0,
            val_batches: int = None, # Should val set be created
            test_ratio: float = 0.5, # Ratio of test set if val set is also created
            low_train: bool = False, # If low train set is used
            embedding_model: EmbeddingModuleConfig = EmbeddingModuleConfig(),
            average: bool = True,
            gpu_to_use: int = 0
            
    ):
        super().__init__(
            dataset=dataset,
            loaders=loaders,
            transforms=transforms,
            mapper = mapper,
            k_samples = k_samples,
            val_batches = val_batches,
            test_ratio = test_ratio,
            low_train = low_train,
            embedding_model = embedding_model,
            average = average,
            gpu_to_use = gpu_to_use
        )


    def _load_data(self,decode: bool = True):    
        """
        Load audio dataset from Hugging Face Datasets. For BEANS the audio column is named path so we will rename it to audio and also rename labels column and remove unamed.

        Returns HF dataset with audio column casted to Audio feature, containing audio data as numpy array and sampling rate.
        """
        logging.info("> Loading data set.")
        print(self.embeddings_save_path)
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
            self.id_to_label = {value: key for key, value in label_to_id.items()} # Save id_to_label to get names later on 

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
    
    
    def _preprocess_data(self, dataset):
        """
        Preprocess the data. This calls the _ksamples function to select k samples per class and calls the embedding extraction function. If multilabel is the task we will one hot encode. 
        """

        # Check if actually a dict
        dataset = self._compute_embeddings(dataset)
        dataset = self._ksamples(dataset)

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


    