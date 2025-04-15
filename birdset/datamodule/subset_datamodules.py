from . import BirdSetDataModule
from birdset.configs import DatasetConfig
from typing import Literal

HF_PATH = "DBD-research-group/BirdSet"


class HSNDataModule(BirdSetDataModule):
    """A BirdSetDataModule for the HSN (high_sierras) dataset."""

    def __init__(
        self,
        n_workers: int = 1,
        val_split: float = 0.2,
        task: Literal["multiclass", "multilabel"] = "multilabel",
        sample_rate: int = 32000,
        classlimit: int = 500,
        eventlimit: int = 5,
    ):
        """Initializes the HSNDataModule.

        Args:
            n_workers (int, optional): The number of worker processes used for data loading. Defaults to 3.
            val_split (float, optional): The proportion of the dataset reserved for validation. Defaults to 0.2.
            task (str, optional): Defines the type of task (e.g., 'multilabel' or 'multiclass'). Defaults to "multilabel".
            sample_rate (int, optional): The sample rate for audio data processing. Defaults to 32000.
            classlimit (int, optional): The maximum number of samples per class. If None, all samples are used. Defaults to 500.
            eventlimit (int, optional): Defines the maximum number of audio events processed per audio file, capping the quantity to ensure balance across files. If None, all events are processed. Defaults to 5.
        """

        super().__init__(
            dataset=DatasetConfig(
                data_dir="/workspace/data_birdset/HSN",
                hf_path=HF_PATH,
                hf_name="HSN",
                n_workers=n_workers,
                val_split=val_split,
                task=task,
                classlimit=classlimit,
                eventlimit=eventlimit,
                sample_rate=sample_rate,
            )
        )


class NBPDataModule(BirdSetDataModule):
    """A BirdSetDataModule for the NBP (nips) dataset."""

    def __init__(
        self,
        n_workers: int = 1,
        val_split: float = 0.2,
        task: Literal["multiclass", "multilabel"] = "multilabel",
        sample_rate: int = 32000,
        classlimit: int = 500,
        eventlimit: int = 5,
    ):
        """Initializes the NBPDataModule.


        Args:
            n_workers (int, optional): The number of worker processes used for data loading. Defaults to 3.
            val_split (float, optional): The proportion of the dataset reserved for validation. Defaults to 0.2.
            task (str, optional): Defines the type of task (e.g., 'multilabel' or 'multiclass'). Defaults to "multilabel".
            sample_rate (int, optional): The sample rate for audio data processing. Defaults to 32000.
            classlimit (int, optional): The maximum number of samples per class. If None, all samples are used. Defaults to 500.
            eventlimit (int, optional): Defines the maximum number of audio events processed per audio file, capping the quantity to ensure balance across files. If None, all events are processed. Defaults to 5.
        """

        super().__init__(
            dataset=DatasetConfig(
                data_dir="/workspace/data_birdset/NBP",
                hf_path=HF_PATH,
                hf_name="NBP",
                n_workers=n_workers,
                val_split=val_split,
                task=task,
                classlimit=classlimit,
                eventlimit=eventlimit,
                sample_rate=sample_rate,
            )
        )


class NESDataModule(BirdSetDataModule):
    """A BirdSetDataModule for the NES (columbia_costa_rica) dataset."""

    def __init__(
        self,
        n_workers: int = 1,
        val_split: float = 0.2,
        task: Literal["multiclass", "multilabel"] = "multilabel",
        sample_rate: int = 32000,
        classlimit: int = 500,
        eventlimit: int = 5,
    ):
        """Initializes the NESDataModule.


        Args:
            n_workers (int, optional): The number of worker processes used for data loading. Defaults to 3.
            val_split (float, optional): The proportion of the dataset reserved for validation. Defaults to 0.2.
            task (str, optional): Defines the type of task (e.g., 'multilabel' or 'multiclass'). Defaults to "multilabel".
            sample_rate (int, optional): The sample rate for audio data processing. Defaults to 32000.
            classlimit (int, optional): The maximum number of samples per class. If None, all samples are used. Defaults to 500.
            eventlimit (int, optional): Defines the maximum number of audio events processed per audio file, capping the quantity to ensure balance across files. If None, all events are processed. Defaults to 5.
        """

        super().__init__(
            dataset=DatasetConfig(
                data_dir="/workspace/data_birdset/NES",
                hf_path=HF_PATH,
                hf_name="NES",
                n_workers=n_workers,
                val_split=val_split,
                task=task,
                classlimit=classlimit,
                eventlimit=eventlimit,
                sample_rate=sample_rate,
            )
        )


class PERDataModule(BirdSetDataModule):
    """A BirdSetDataModule for the PER (amazon_basin) dataset."""

    def __init__(
        self,
        n_workers: int = 1,
        val_split: float = 0.2,
        task: Literal["multiclass", "multilabel"] = "multilabel",
        sample_rate: int = 32000,
        classlimit: int = 500,
        eventlimit: int = 5,
    ):
        """Initializes the PERDataModule.


        Args:
            n_workers (int, optional): The number of worker processes used for data loading. Defaults to 3.
            val_split (float, optional): The proportion of the dataset reserved for validation. Defaults to 0.2.
            task (str, optional): Defines the type of task (e.g., 'multilabel' or 'multiclass'). Defaults to "multilabel".
            sample_rate (int, optional): The sample rate for audio data processing. Defaults to 32000.
            classlimit (int, optional): The maximum number of samples per class. If None, all samples are used. Defaults to 500.
            eventlimit (int, optional): Defines the maximum number of audio events processed per audio file, capping the quantity to ensure balance across files. If None, all events are processed. Defaults to 5.
        """

        super().__init__(
            dataset=DatasetConfig(
                data_dir="/workspace/data_birdset/PER",
                hf_path=HF_PATH,
                hf_name="PER",
                n_workers=n_workers,
                val_split=val_split,
                task=task,
                classlimit=classlimit,
                eventlimit=eventlimit,
                sample_rate=sample_rate,
            )
        )


class POWDataModule(BirdSetDataModule):
    """A BirdSetDataModule for the POW (powdermill_nature) dataset."""

    def __init__(
        self,
        n_workers: int = 1,
        val_split: float = 0.2,
        task: Literal["multiclass", "multilabel"] = "multilabel",
        sample_rate: int = 32000,
        classlimit: int = 500,
        eventlimit: int = 5,
    ):
        """Initializes the POWDataModule.


        Args:
            n_workers (int, optional): The number of worker processes used for data loading. Defaults to 3.
            val_split (float, optional): The proportion of the dataset reserved for validation. Defaults to 0.2.
            task (str, optional): Defines the type of task (e.g., 'multilabel' or 'multiclass'). Defaults to "multilabel".
            sample_rate (int, optional): The sample rate for audio data processing. Defaults to 32000.
            classlimit (int, optional): The maximum number of samples per class. If None, all samples are used. Defaults to 500.
            eventlimit (int, optional): Defines the maximum number of audio events processed per audio file, capping the quantity to ensure balance across files. If None, all events are processed. Defaults to 5.
        """

        super().__init__(
            dataset=DatasetConfig(
                data_dir="/workspace/data_birdset/POW",
                hf_path=HF_PATH,
                hf_name="POW",
                n_workers=n_workers,
                val_split=val_split,
                task=task,
                classlimit=classlimit,
                eventlimit=eventlimit,
                sample_rate=sample_rate,
            )
        )


class SNEDataModule(BirdSetDataModule):
    """A BirdSetDataModule for the SNE (sierra_nevada) dataset."""

    def __init__(
        self,
        n_workers: int = 1,
        val_split: float = 0.2,
        task: Literal["multiclass", "multilabel"] = "multilabel",
        sample_rate: int = 32000,
        classlimit: int = 500,
        eventlimit: int = 5,
    ):
        """Initializes the SNEDataModule.


        Args:
            n_workers (int, optional): The number of worker processes used for data loading. Defaults to 3.
            val_split (float, optional): The proportion of the dataset reserved for validation. Defaults to 0.2.
            task (str, optional): Defines the type of task (e.g., 'multilabel' or 'multiclass'). Defaults to "multilabel".
            sample_rate (int, optional): The sample rate for audio data processing. Defaults to 32000.
            classlimit (int, optional): The maximum number of samples per class. If None, all samples are used. Defaults to 500.
            eventlimit (int, optional): Defines the maximum number of audio events processed per audio file, capping the quantity to ensure balance across files. If None, all events are processed. Defaults to 5.
        """

        super().__init__(
            dataset=DatasetConfig(
                data_dir="/workspace/data_birdset/SNE",
                hf_path=HF_PATH,
                hf_name="SNE",
                n_workers=n_workers,
                val_split=val_split,
                task=task,
                classlimit=classlimit,
                eventlimit=eventlimit,
                sample_rate=sample_rate,
            )
        )


class SSWDataModule(BirdSetDataModule):
    """A BirdSetDataModule for the SSW (sapsucker_woods) dataset."""

    def __init__(
        self,
        n_workers: int = 1,
        val_split: float = 0.2,
        task: Literal["multiclass", "multilabel"] = "multilabel",
        sample_rate: int = 32000,
        classlimit: int = 500,
        eventlimit: int = 5,
    ):
        """Initializes the SSWDataModule.


        Args:
            n_workers (int, optional): The number of worker processes used for data loading. Defaults to 3.
            val_split (float, optional): The proportion of the dataset reserved for validation. Defaults to 0.2.
            task (str, optional): Defines the type of task (e.g., 'multilabel' or 'multiclass'). Defaults to "multilabel".
            sample_rate (int, optional): The sample rate for audio data processing. Defaults to 32000.
            classlimit (int, optional): The maximum number of samples per class. If None, all samples are used. Defaults to 500.
            eventlimit (int, optional): Defines the maximum number of audio events processed per audio file, capping the quantity to ensure balance across files. If None, all events are processed. Defaults to 5.
        """

        super().__init__(
            dataset=DatasetConfig(
                data_dir="/workspace/data_birdset/SSW",
                hf_path=HF_PATH,
                hf_name="SSW",
                n_workers=n_workers,
                val_split=val_split,
                task=task,
                classlimit=classlimit,
                eventlimit=eventlimit,
                sample_rate=sample_rate,
            )
        )


class XCMDataModule(BirdSetDataModule):
    """A BirdSetDataModule for the XCM (xenocanto) dataset."""

    def __init__(
        self,
        n_workers: int = 3,
        val_split: float = 0.05,
        task: Literal["multiclass", "multilabel"] = "multilabel",
        sample_rate: int = 32000,
        classlimit: int = 500,
        eventlimit: int = 5,
    ):
        """Initializes the XCMDataModule.


        Args:
            n_workers (int, optional): The number of worker processes used for data loading. Defaults to 3.
            val_split (float, optional): The proportion of the dataset reserved for validation. Defaults to 0.2.
            task (str, optional): Defines the type of task (e.g., 'multilabel' or 'multiclass'). Defaults to "multilabel".
            sample_rate (int, optional): The sample rate for audio data processing. Defaults to 32000.
            classlimit (int, optional): The maximum number of samples per class. If None, all samples are used. Defaults to 500.
            eventlimit (int, optional): Defines the maximum number of audio events processed per audio file, capping the quantity to ensure balance across files. If None, all events are processed. Defaults to 5.
        """

        super().__init__(
            dataset=DatasetConfig(
                data_dir="/workspace/data_birdset/XCM",
                hf_path=HF_PATH,
                hf_name="XCM",
                n_workers=n_workers,
                val_split=val_split,
                task=task,
                classlimit=classlimit,
                eventlimit=eventlimit,
                sample_rate=sample_rate,
            )
        )


class XCLDataModule(BirdSetDataModule):
    """A BirdSetDataModule for the XCL (xenocanto) dataset."""

    def __init__(
        self,
        n_workers: int = 1,
        val_split: float = 0.05,
        task: Literal["multiclass", "multilabel"] = "multilabel",
        sample_rate: int = 32000,
        classlimit: int = 500,
        eventlimit: int = 5,
    ):
        """Initializes the XCLDataModule.


        Args:
            n_workers (int, optional): The number of worker processes used for data loading. Defaults to 3.
            val_split (float, optional): The proportion of the dataset reserved for validation. Defaults to 0.2.
            task (str, optional): Defines the type of task (e.g., 'multilabel' or 'multiclass'). Defaults to "multilabel".
            sample_rate (int, optional): The sample rate for audio data processing. Defaults to 32000.
            classlimit (int, optional): The maximum number of samples per class. If None, all samples are used. Defaults to 500.
            eventlimit (int, optional): Defines the maximum number of audio events processed per audio file, capping the quantity to ensure balance across files. If None, all events are processed. Defaults to 5.
        """

        super().__init__(
            dataset=DatasetConfig(
                data_dir="/workspace/data_birdset/SSW",
                hf_path=HF_PATH,
                hf_name="XCL",
                n_workers=n_workers,
                val_split=val_split,
                task=task,
                classlimit=classlimit,
                eventlimit=eventlimit,
                sample_rate=sample_rate,
            )
        )


class UHHDataModule(BirdSetDataModule):
    """A BirdSetDataModule for the UHH (hawaiian_islands) dataset."""

    def __init__(
        self,
        n_workers: int = 1,
        val_split: float = 0.2,
        task: Literal["multiclass", "multilabel"] = "multilabel",
        sample_rate: int = 32000,
        classlimit: int = 500,
        eventlimit: int = 5,
    ):
        """Initializes the UHUDataModule.


        Args:
            n_workers (int, optional): The number of worker processes used for data loading. Defaults to 3.
            val_split (float, optional): The proportion of the dataset reserved for validation. Defaults to 0.2.
            task (str, optional): Defines the type of task (e.g., 'multilabel' or 'multiclass'). Defaults to "multilabel".
            sample_rate (int, optional): The sample rate for audio data processing. Defaults to 32000.
            classlimit (int, optional): The maximum number of samples per class. If None, all samples are used. Defaults to 500.
            eventlimit (int, optional): Defines the maximum number of audio events processed per audio file, capping the quantity to ensure balance across files. If None, all events are processed. Defaults to 5.
        """

        super().__init__(
            dataset=DatasetConfig(
                data_dir="/workspace/data_birdset/SSW",
                hf_path=HF_PATH,
                hf_name="UHH",
                n_workers=n_workers,
                val_split=val_split,
                task=task,
                classlimit=classlimit,
                eventlimit=eventlimit,
                sample_rate=sample_rate,
            )
        )
