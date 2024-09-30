from datasets import DatasetDict

from birdset import utils
from birdset.datamodule.components.transforms import BirdSetTransformsWrapper
from birdset.datamodule.components.event_mapping import XCEventMapping
from . import BaseDataModuleHF
from birdset.configs import DatasetConfig, LoadersConfig

log = utils.get_pylogger(__name__)


class BirdSetDataModule(BaseDataModuleHF):
    """
    A data module for the BirdSet dataset.

    This class handles the loading, preprocessing, and transformation of the BirdSet dataset. It extends the BaseDataModuleHF class.

    Attributes:
        dataset (DatasetConfig): The configuration for the dataset.
        loaders (LoadersConfig): The configuration for the loaders.
        transforms (BirdSetTransformsWrapper): The transforms to be applied to the data.
        mapper (XCEventMapping): The mapping for the events.

    Methods:
        _load_data(decode: bool = False): Loads the data.
        _preprocess_data(dataset): Preprocesses the data.
    """

    def __init__(
        self,
        dataset: DatasetConfig = DatasetConfig(
            data_dir='data_birdset/HSN',
            hf_path='anonymous-birder/BirdSet',
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
        mapper: XCEventMapping = XCEventMapping()
    ):
        """
        Initializes the data module.

        This method initializes the data module with the specified dataset, loaders, transforms, and mapper. 
        It then calls the superclass's __init__ method with these arguments.

        Args:
            dataset (DatasetConfig, optional): The configuration for the dataset. Defaults to a DatasetConfig with specific values.
            loaders (LoadersConfig, optional): The configuration for the loaders. Defaults to an empty LoadersConfig.
            transforms (BirdSetTransformsWrapper, optional): The transforms to be applied to the data. Defaults to an empty BirdSetTransformsWrapper.
            mapper (XCEventMapping, optional): The mapping for the events. Defaults to an empty XCEventMapping.

        Returns:
            None
        """
        
        super().__init__(
            dataset=dataset,
            loaders=loaders,
            transforms=transforms,
        )
        self.event_mapper = mapper

    def _load_data(self, decode: bool = False):
        """
        Loads the data.
        This method loads the data by calling the superclass's _load_data method.
        Args:
            decode (bool, optional): Whether to decode the data. Defaults to False.
        Returns:
            The loaded data.
        """
        return super()._load_data(decode=decode)

    def _preprocess_data(self, dataset):
        """
        Preprocesses the data.

        This method preprocesses the data based on the task specified in the dataset configuration. 
        It handles both multiclass and multilabel tasks.

        Args:
            dataset: The dataset to preprocess.

        Returns:
            The preprocessed dataset.
        """
        if self.dataset_config.task == "multiclass":
            # pick only train and test dataset
            dataset = DatasetDict({split: dataset[split] for split in ["train", "test"]})

            log.info("> Mapping data set.")
            dataset["train"] = dataset["train"].map(
                self.event_mapper,
                remove_columns=["audio"],
                batched=True,
                batch_size=300,
                load_from_cache_file=True,
                num_proc=self.dataset_config.n_workers,
                desc="Train event mapping"
            )

            if self.dataset_config.class_weights_loss or self.dataset_config.class_weights_sampler:
                self.num_train_labels = self._count_labels((dataset["train"]["ebird_code"]))
            
            if self.dataset_config.classlimit and not self.dataset_config.eventlimit:
                log.info(f">> Limiting classes to {self.dataset_config.classlimit}")
                dataset["train"] = self._limit_classes(
                    dataset=dataset["train"],
                    label_name="ebird_code",
                    limit=self.dataset_config.classlimit
                )
            elif self.dataset_config.classlimit or self.dataset_config.eventlimit:
                log.info(f">> Smart sampling to {self.dataset_config.classlimit=}, {self.dataset_config.eventlimit=}")
                dataset["train"] = self._smart_sampling(
                    dataset=dataset["train"],
                    label_name="ebird_code",
                    class_limit=self.dataset_config.classlimit,
                    event_limit=self.dataset_config.eventlimit
                )
            dataset = dataset.rename_column("ebird_code", "labels")

        elif self.dataset_config.task == "multilabel":
            # pick only train and test_5s dataset
            dataset = DatasetDict({split: dataset[split] for split in ["train", "test_5s"]})

            log.info(">> Mapping train data.")
            dataset["train"] = dataset["train"].map(
                self.event_mapper,
                remove_columns=["audio"],
                batched=True,
                batch_size=300,
                num_proc=self.dataset_config.n_workers,
                desc="Train event mapping"
            )

            dataset = dataset.rename_column("ebird_code_multilabel", "labels")

            if self.dataset_config.classlimit or self.dataset_config.eventlimit:
                log.info(">> Smart Sampling") #!TODO: implement custom caching?
                dataset["train"] = self._smart_sampling(
                    dataset=dataset["train"],
                    label_name="ebird_code",
                    class_limit=self.dataset_config.classlimit,
                    event_limit=self.dataset_config.eventlimit
                )

            log.info(">> One-hot-encode classes")
            for split in ["train", "test_5s"]:
                dataset[split] = dataset[split].map(
                    self._classes_one_hot,
                    batched=True,
                    batch_size=1500,
                    load_from_cache_file=True,
                    num_proc=self.dataset_config.n_workers,
                    desc=f"One-hot-encoding {split} labels."
                )

            if self.dataset_config.class_weights_loss or self.dataset_config.class_weights_sampler:
                self.num_train_labels = self._count_labels((dataset["train"]["ebird_code"]))

            dataset_test = dataset.pop("test_5s")
            dataset["test"] = dataset_test
        else:
            raise f"{self.dataset_config.task=} is not supported, choose (multilabel, multiclass)"
        
        for split in ["train", "test"]:
            dataset[split] = dataset[split].select_columns(
                ["filepath", "labels", "detected_events", "start_time", "end_time"]
            )

        return dataset