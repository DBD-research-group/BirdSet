from src.datamodule.base_datamodule import DatasetConfig, LoadersConfig
from src.datamodule.components.event_mapping import XCEventMapping
from src.datamodule.components.transforms import GADMETransformsWrapper
from .gadme_datamodule import GADMEDataModule


class HSNDataModule(GADMEDataModule):
    """A GADMEDataModule for the HSN (high_sierras) dataset."""

    def __init__(self,
                 n_workers: int = 3,
                 val_split: float = 0.2,
                 task: str = "multilabel",
                 sampling_rate: int = 32000):
        """Initializes the HSNDataModule.

        Args:
            n_workers (int, optional): The number of worker processes used for data loading. Defaults to 3.
            val_split (float, optional): The proportion of the dataset reserved for validation. Defaults to 0.2.
            task (str, optional): Defines the type of task (e.g., 'multilabel' or 'multiclass'). Defaults to "multilabel".
            sampling_rate (int, optional): The sampling rate for audio data processing. Defaults to 32000.
        """
        
        super().__init__(
            DatasetConfig=DatasetConfig(dataset_name='HSN',
                                        hf_path='DBD-research-group/gadme',
                                        hf_name='HSN',
                                        n_classes=21,
                                        n_workers=n_workers,
                                        val_split=val_split,
                                        task=task,
                                        classlimit=500,
                                        eventlimit=5,
                                        sampling_rate=sampling_rate))


class NBPDataModule(GADMEDataModule):
    """A GADMEDataModule for the NBP (nips) dataset."""
    
    def __init__(self,
                 n_workers: int = 3,
                 val_split: float = 0.2,
                 task: str = "multilabel",
                 sampling_rate: int = 32000):
        """Initializes the NBPDataModule.
        
        
        Args:
            n_workers (int, optional): The number of worker processes used for data loading. Defaults to 3.
            val_split (float, optional): The proportion of the dataset reserved for validation. Defaults to 0.2.
            task (str, optional): Defines the type of task (e.g., 'multilabel' or 'multiclass'). Defaults to "multilabel".
            sampling_rate (int, optional): The sampling rate for audio data processing. Defaults to 32000.
        """
        
        super().__init__(
            DatasetConfig=DatasetConfig(dataset_name='NBP',
                                        hf_path='DBD-research-group/gadme',
                                        hf_name='NBP',
                                        n_classes=51,
                                        n_workers=n_workers,
                                        val_split=val_split,
                                        task=task,
                                        classlimit=500,
                                        eventlimit=5,
                                        sampling_rate=sampling_rate))


class NESDataModule(GADMEDataModule):
    """A GADMEDataModule for the NES (columbia_costa_rica) dataset."""
    
    def __init__(self,
                 n_workers: int = 3,
                 val_split: float = 0.2,
                 task: str = "multilabel",
                 sampling_rate: int = 32000):
        """Initializes the NESDataModule.
        
        
        Args:
            n_workers (int, optional): The number of worker processes used for data loading. Defaults to 3.
            val_split (float, optional): The proportion of the dataset reserved for validation. Defaults to 0.2.
            task (str, optional): Defines the type of task (e.g., 'multilabel' or 'multiclass'). Defaults to "multilabel".
            sampling_rate (int, optional): The sampling rate for audio data processing. Defaults to 32000.
        """
        
        super().__init__(
            DatasetConfig=DatasetConfig(dataset_name='NES',
                                        hf_path='DBD-research-group/gadme',
                                        hf_name='NES',
                                        n_classes=89,
                                        n_workers=n_workers,
                                        val_split=val_split,
                                        task=task,
                                        classlimit=500,
                                        eventlimit=5,
                                        sampling_rate=sampling_rate))
        
        
class PERDataModule(GADMEDataModule):
    """A GADMEDataModule for the PER (amazon_basin) dataset."""
    
    def __init__(self,
                 n_workers: int = 3,
                 val_split: float = 0.2,
                 task: str = "multilabel",
                 sampling_rate: int = 32000):
        """Initializes the PERDataModule.
        
        
        Args:
            n_workers (int, optional): The number of worker processes used for data loading. Defaults to 3.
            val_split (float, optional): The proportion of the dataset reserved for validation. Defaults to 0.2.
            task (str, optional): Defines the type of task (e.g., 'multilabel' or 'multiclass'). Defaults to "multilabel".
            sampling_rate (int, optional): The sampling rate for audio data processing. Defaults to 32000.
        """
        
        super().__init__(
            DatasetConfig=DatasetConfig(dataset_name='PER',
                                        hf_path='DBD-research-group/gadme',
                                        hf_name='PER',
                                        n_classes=132,
                                        n_workers=n_workers,
                                        val_split=val_split,
                                        task=task,
                                        classlimit=500,
                                        eventlimit=5,
                                        sampling_rate=sampling_rate))
        
        
class POWDataModule(GADMEDataModule):
    """A GADMEDataModule for the POW (powdermill_nature) dataset."""
    
    def __init__(self,
                 n_workers: int = 3,
                 val_split: float = 0.2,
                 task: str = "multilabel",
                 sampling_rate: int = 32000):
        """Initializes the POWDataModule.
        
        
        Args:
            n_workers (int, optional): The number of worker processes used for data loading. Defaults to 3.
            val_split (float, optional): The proportion of the dataset reserved for validation. Defaults to 0.2.
            task (str, optional): Defines the type of task (e.g., 'multilabel' or 'multiclass'). Defaults to "multilabel".
            sampling_rate (int, optional): The sampling rate for audio data processing. Defaults to 32000.
        """
        
        super().__init__(
            DatasetConfig=DatasetConfig(dataset_name='POW',
                                        hf_path='DBD-research-group/gadme',
                                        hf_name='POW',
                                        n_classes=48,
                                        n_workers=n_workers,
                                        val_split=val_split,
                                        task=task,
                                        classlimit=500,
                                        eventlimit=5,
                                        sampling_rate=sampling_rate))