from birdset.datamodule.base_datamodule import DatasetConfig
from .birdset_datamodule import BirdSetDataModule

HF_PATH = 'DBD-research-group/BirdSet'

class HSNDataModule(BirdSetDataModule):
    """A BirdSetDataModule for the HSN (high_sierras) dataset."""

    def __init__(self,
                 n_workers: int = 1,
                 val_split: float = 0.2,
                 task: str = "multilabel",
                 sampling_rate: int = 32000,
                 classlimit: int = 500,
                 eventlimit: int = 5):
        """Initializes the HSNDataModule.

        Args:
            n_workers (int, optional): The number of worker processes used for data loading. Defaults to 3.
            val_split (float, optional): The proportion of the dataset reserved for validation. Defaults to 0.2.
            task (str, optional): Defines the type of task (e.g., 'multilabel' or 'multiclass'). Defaults to "multilabel".
            sampling_rate (int, optional): The sampling rate for audio data processing. Defaults to 32000.
            classlimit (int, optional): The maximum number of samples per class. If None, all samples are used. Defaults to 500.
            eventlimit (int, optional): Defines the maximum number of audio events processed per audio file, capping the quantity to ensure balance across files. If None, all events are processed. Defaults to 5.
        """

        super().__init__(
            DatasetConfig=DatasetConfig(
                                        data_dir='/workspace/data_birdset/HSN',
                                        dataset_name='HSN',
                                        hf_path=HF_PATH,
                                        hf_name='HSN',
                                        n_classes=21,
                                        n_workers=n_workers,
                                        val_split=val_split,
                                        task=task,
                                        classlimit=classlimit,
                                        eventlimit=eventlimit,
                                        sampling_rate=sampling_rate))


class NBPDataModule(BirdSetDataModule):
    """A BirdSetDataModule for the NBP (nips) dataset."""

    def __init__(self,
                 n_workers: int = 1,
                 val_split: float = 0.2,
                 task: str = "multilabel",
                 sampling_rate: int = 32000,
                 classlimit: int = 500,
                 eventlimit: int = 5):
        """Initializes the NBPDataModule.
        
        
        Args:
            n_workers (int, optional): The number of worker processes used for data loading. Defaults to 3.
            val_split (float, optional): The proportion of the dataset reserved for validation. Defaults to 0.2.
            task (str, optional): Defines the type of task (e.g., 'multilabel' or 'multiclass'). Defaults to "multilabel".
            sampling_rate (int, optional): The sampling rate for audio data processing. Defaults to 32000.
            classlimit (int, optional): The maximum number of samples per class. If None, all samples are used. Defaults to 500.
            eventlimit (int, optional): Defines the maximum number of audio events processed per audio file, capping the quantity to ensure balance across files. If None, all events are processed. Defaults to 5.
        """

        super().__init__(
            DatasetConfig=DatasetConfig(
                                        data_dir='/workspace/data_birdset/NBP',
                                        dataset_name='NBP',
                                        hf_path=HF_PATH,
                                        hf_name='NBP',
                                        n_classes=51,
                                        n_workers=n_workers,
                                        val_split=val_split,
                                        task=task,
                                        classlimit=classlimit,
                                        eventlimit=eventlimit,
                                        sampling_rate=sampling_rate))


class NESDataModule(BirdSetDataModule):
    """A BirdSetDataModule for the NES (columbia_costa_rica) dataset."""

    def __init__(self,
                 n_workers: int = 1,
                 val_split: float = 0.2,
                 task: str = "multilabel",
                 sampling_rate: int = 32000,
                 classlimit: int = 500,
                 eventlimit: int = 5):
        """Initializes the NESDataModule.
        
        
        Args:
            n_workers (int, optional): The number of worker processes used for data loading. Defaults to 3.
            val_split (float, optional): The proportion of the dataset reserved for validation. Defaults to 0.2.
            task (str, optional): Defines the type of task (e.g., 'multilabel' or 'multiclass'). Defaults to "multilabel".
            sampling_rate (int, optional): The sampling rate for audio data processing. Defaults to 32000.
            classlimit (int, optional): The maximum number of samples per class. If None, all samples are used. Defaults to 500.
            eventlimit (int, optional): Defines the maximum number of audio events processed per audio file, capping the quantity to ensure balance across files. If None, all events are processed. Defaults to 5.
        """

        super().__init__(
            DatasetConfig=DatasetConfig(
                                        data_dir='/workspace/data_birdset/NES',
                                        dataset_name='NES',
                                        hf_path=HF_PATH,
                                        hf_name='NES',
                                        n_classes=89,
                                        n_workers=n_workers,
                                        val_split=val_split,
                                        task=task,
                                        classlimit=classlimit,
                                        eventlimit=eventlimit,
                                        sampling_rate=sampling_rate))


class PERDataModule(BirdSetDataModule):
    """A BirdSetDataModule for the PER (amazon_basin) dataset."""

    def __init__(self,
                 n_workers: int = 1,
                 val_split: float = 0.2,
                 task: str = "multilabel",
                 sampling_rate: int = 32000,
                 classlimit: int = 500,
                 eventlimit: int = 5):
        """Initializes the PERDataModule.
        
        
        Args:
            n_workers (int, optional): The number of worker processes used for data loading. Defaults to 3.
            val_split (float, optional): The proportion of the dataset reserved for validation. Defaults to 0.2.
            task (str, optional): Defines the type of task (e.g., 'multilabel' or 'multiclass'). Defaults to "multilabel".
            sampling_rate (int, optional): The sampling rate for audio data processing. Defaults to 32000.
            classlimit (int, optional): The maximum number of samples per class. If None, all samples are used. Defaults to 500.
            eventlimit (int, optional): Defines the maximum number of audio events processed per audio file, capping the quantity to ensure balance across files. If None, all events are processed. Defaults to 5.
        """

        super().__init__(
            DatasetConfig=DatasetConfig(
                                        data_dir='/workspace/data_birdset/PER',
                                        dataset_name='PER',
                                        hf_path=HF_PATH,
                                        hf_name='PER',
                                        n_classes=132,
                                        n_workers=n_workers,
                                        val_split=val_split,
                                        task=task,
                                        classlimit=classlimit,
                                        eventlimit=eventlimit,
                                        sampling_rate=sampling_rate))


class POWDataModule(BirdSetDataModule):
    """A BirdSetDataModule for the POW (powdermill_nature) dataset."""

    def __init__(self,
                 n_workers: int = 1,
                 val_split: float = 0.2,
                 task: str = "multilabel",
                 sampling_rate: int = 32000,
                 classlimit: int = 500,
                 eventlimit: int = 5):
        """Initializes the POWDataModule.
        
        
        Args:
            n_workers (int, optional): The number of worker processes used for data loading. Defaults to 3.
            val_split (float, optional): The proportion of the dataset reserved for validation. Defaults to 0.2.
            task (str, optional): Defines the type of task (e.g., 'multilabel' or 'multiclass'). Defaults to "multilabel".
            sampling_rate (int, optional): The sampling rate for audio data processing. Defaults to 32000.
            classlimit (int, optional): The maximum number of samples per class. If None, all samples are used. Defaults to 500.
            eventlimit (int, optional): Defines the maximum number of audio events processed per audio file, capping the quantity to ensure balance across files. If None, all events are processed. Defaults to 5.
        """

        super().__init__(
            DatasetConfig=DatasetConfig(
                                        data_dir='/workspace/data_birdset/POW',
                                        dataset_name='POW',
                                        hf_path=HF_PATH,
                                        hf_name='POW',
                                        n_classes=48,
                                        n_workers=n_workers,
                                        val_split=val_split,
                                        task=task,
                                        classlimit=classlimit,
                                        eventlimit=eventlimit,
                                        sampling_rate=sampling_rate))


class SNEDataModule(BirdSetDataModule):
    """A BirdSetDataModule for the SNE (sierra_nevada) dataset."""

    def __init__(self,
                 n_workers: int = 1,
                 val_split: float = 0.2,
                 task: str = "multilabel",
                 sampling_rate: int = 32000,
                 classlimit: int = 500,
                 eventlimit: int = 5):
        """Initializes the SNEDataModule.
        
        
        Args:
            n_workers (int, optional): The number of worker processes used for data loading. Defaults to 3.
            val_split (float, optional): The proportion of the dataset reserved for validation. Defaults to 0.2.
            task (str, optional): Defines the type of task (e.g., 'multilabel' or 'multiclass'). Defaults to "multilabel".
            sampling_rate (int, optional): The sampling rate for audio data processing. Defaults to 32000.
            classlimit (int, optional): The maximum number of samples per class. If None, all samples are used. Defaults to 500.
            eventlimit (int, optional): Defines the maximum number of audio events processed per audio file, capping the quantity to ensure balance across files. If None, all events are processed. Defaults to 5.
        """

        super().__init__(
            DatasetConfig=DatasetConfig(
                                        data_dir='/workspace/data_birdset/SNE',
                                        dataset_name='SNE',
                                        hf_path=HF_PATH,
                                        hf_name='SNE',
                                        n_classes=56,
                                        n_workers=n_workers,
                                        val_split=val_split,
                                        task=task,
                                        classlimit=classlimit,
                                        eventlimit=eventlimit,
                                        sampling_rate=sampling_rate))


class SSWDataModule(BirdSetDataModule):
    """A BirdSetDataModule for the SSW (sapsucker_woods) dataset."""

    def __init__(self,
                 n_workers: int = 1,
                 val_split: float = 0.2,
                 task: str = "multilabel",
                 sampling_rate: int = 32000,
                 classlimit: int = 500,
                 eventlimit: int = 5):
        """Initializes the SSWDataModule.
        
        
        Args:
            n_workers (int, optional): The number of worker processes used for data loading. Defaults to 3.
            val_split (float, optional): The proportion of the dataset reserved for validation. Defaults to 0.2.
            task (str, optional): Defines the type of task (e.g., 'multilabel' or 'multiclass'). Defaults to "multilabel".
            sampling_rate (int, optional): The sampling rate for audio data processing. Defaults to 32000.
            classlimit (int, optional): The maximum number of samples per class. If None, all samples are used. Defaults to 500.
            eventlimit (int, optional): Defines the maximum number of audio events processed per audio file, capping the quantity to ensure balance across files. If None, all events are processed. Defaults to 5.
        """

        super().__init__(
            DatasetConfig=DatasetConfig(
                                        data_dir='/workspace/data_birdset/SSW',
                                        dataset_name='SSW',
                                        hf_path=HF_PATH,
                                        hf_name='SSW',
                                        n_classes=81,
                                        n_workers=n_workers,
                                        val_split=val_split,
                                        task=task,
                                        classlimit=classlimit,
                                        eventlimit=eventlimit,
                                        sampling_rate=sampling_rate))


class XCMDataModule(BirdSetDataModule):
    """A BirdSetDataModule for the XCM (xenocanto) dataset."""

    def __init__(self,
                 n_workers: int = 3,
                 val_split: float = 0.05,
                 task: str = "multilabel",
                 sampling_rate: int = 32000,
                 classlimit: int = 500,
                 eventlimit: int = 5):
        """Initializes the XCMDataModule.
        
        
        Args:
            n_workers (int, optional): The number of worker processes used for data loading. Defaults to 3.
            val_split (float, optional): The proportion of the dataset reserved for validation. Defaults to 0.2.
            task (str, optional): Defines the type of task (e.g., 'multilabel' or 'multiclass'). Defaults to "multilabel".
            sampling_rate (int, optional): The sampling rate for audio data processing. Defaults to 32000.
            classlimit (int, optional): The maximum number of samples per class. If None, all samples are used. Defaults to 500.
            eventlimit (int, optional): Defines the maximum number of audio events processed per audio file, capping the quantity to ensure balance across files. If None, all events are processed. Defaults to 5.
        """

        super().__init__(
            DatasetConfig=DatasetConfig(
                                        data_dir='/workspace/data_birdset/XCM',
                                        dataset_name='XCM',
                                        hf_path=HF_PATH,
                                        hf_name='XCM',
                                        n_classes=409,
                                        n_workers=n_workers,
                                        val_split=val_split,
                                        task=task,
                                        classlimit=classlimit,
                                        eventlimit=eventlimit,
                                        sampling_rate=sampling_rate))


class XCLDataModule(BirdSetDataModule):
    """A BirdSetDataModule for the XCL (xenocanto) dataset."""

    def __init__(self,
                 n_workers: int = 1,
                 val_split: float = 0.05,
                 task: str = "multilabel",
                 sampling_rate: int = 32000,
                 classlimit: int = 500,
                 eventlimit: int = 5):
        """Initializes the XCLDataModule.
        
        
        Args:
            n_workers (int, optional): The number of worker processes used for data loading. Defaults to 3.
            val_split (float, optional): The proportion of the dataset reserved for validation. Defaults to 0.2.
            task (str, optional): Defines the type of task (e.g., 'multilabel' or 'multiclass'). Defaults to "multilabel".
            sampling_rate (int, optional): The sampling rate for audio data processing. Defaults to 32000.
            classlimit (int, optional): The maximum number of samples per class. If None, all samples are used. Defaults to 500.
            eventlimit (int, optional): Defines the maximum number of audio events processed per audio file, capping the quantity to ensure balance across files. If None, all events are processed. Defaults to 5.
        """

        super().__init__(
            DatasetConfig=DatasetConfig(
                                        data_dir='/workspace/data_birdset/SSW',
                                        dataset_name='XCL',
                                        hf_path=HF_PATH,
                                        hf_name='XCL',
                                        n_classes=9734,
                                        n_workers=n_workers,
                                        val_split=val_split,
                                        task=task,
                                        classlimit=classlimit,
                                        eventlimit=eventlimit,
                                        sampling_rate=sampling_rate))


class UHHDataModule(BirdSetDataModule):
    """A BirdSetDataModule for the UHH (hawaiian_islands) dataset."""

    def __init__(self,
                 n_workers: int = 1,
                 val_split: float = 0.2,
                 task: str = "multilabel",
                 sampling_rate: int = 32000,
                 classlimit: int = 500,
                 eventlimit: int = 5):
        """Initializes the UHUDataModule.
        
        
        Args:
            n_workers (int, optional): The number of worker processes used for data loading. Defaults to 3.
            val_split (float, optional): The proportion of the dataset reserved for validation. Defaults to 0.2.
            task (str, optional): Defines the type of task (e.g., 'multilabel' or 'multiclass'). Defaults to "multilabel".
            sampling_rate (int, optional): The sampling rate for audio data processing. Defaults to 32000.
            classlimit (int, optional): The maximum number of samples per class. If None, all samples are used. Defaults to 500.
            eventlimit (int, optional): Defines the maximum number of audio events processed per audio file, capping the quantity to ensure balance across files. If None, all events are processed. Defaults to 5.
        """

        super().__init__(
            DatasetConfig=DatasetConfig(
                                        data_dir='/workspace/data_birdset/SSW',
                                        dataset_name='UHH',
                                        hf_path=HF_PATH,
                                        hf_name='UHH',
                                        n_classes=25, # TODO UHH (hawaiian_islands) has a strange number of classes. "25 tr, 27 te" probably stands for training and test classes. But if so, what to use here? Config said 27 classes.
                                        n_workers=n_workers,
                                        val_split=val_split,
                                        task=task,
                                        classlimit=classlimit,
                                        eventlimit=eventlimit,
                                        sampling_rate=sampling_rate))

