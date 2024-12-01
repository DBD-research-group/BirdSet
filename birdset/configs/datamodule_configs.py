from dataclasses import dataclass
from typing import Literal, Optional


@dataclass
class FewShotConfig:
    """
    Configuration for few-shot learning.

    Attributes:
        k_samples (int): Number of samples per class. Default is 32.
        use_train (bool): Whether to use training data. Default is True.
        use_valid (bool): Whether to use validation data. Default is False.
        use_test (bool): Whether to use test data. Default is False.
    """

    k_samples: int = 32
    use_train: bool = True
    use_valid: bool = False
    use_test: bool = False


@dataclass
class DatasetConfig:
    """
    A class used to configure the dataset for the model.

    Attributes
    ----------
    data_dir : str
        Specifies the directory where the dataset files are stored.  **Important**: The dataset uses a lot of disk space, so make sure you have enough storage available.
    dataset_name : str
        The name assigned to the dataset.
    hf_path : str
        The path to the dataset stored on HuggingFace.
    hf_name : str
        The name of the dataset on HuggingFace.
    seed : int
        A seed value for ensuring reproducibility across runs.
    n_workers : int
        The number of worker processes used for data loading.
    val_split : float
        The proportion of the dataset reserved for validation.
    task : str
        Defines the type of task (e.g., 'multilabel' or 'multiclass').
    subset : int, optional
        A subset of the dataset to use. If None, the entire dataset is used.
    sampling_rate : int
        The sampling rate for audio data processing.
    class_weights_loss : bool, optional
        (Deprecated) Previously used for applying class weights in loss calculation.
    class_weights_sampler : bool, optional
        Indicates whether to use class weights in the sampler for handling imbalanced datasets.
    classlimit : int, optional
        The maximum number of samples per class. If None, all samples are used.
    eventlimit : int, optional
        Defines the maximum number of audio events processed per audio file, capping the quantity to ensure balance across files. If None, all events are processed.
    direct_fingerprint: int, optional
        Only works with PretrainDatamodule. Path to a saved preprocessed dataset path
    fewshot: FewShotConfig, optional
        Configuration for few-shot learning.
    """

    data_dir: str = "/workspace/data_birdset"
    dataset_name: str = "esc50"
    hf_path: str = "ashraq/esc50"
    hf_name: str = ""
    seed: int = 42
    n_classes: Optional[int] = 50
    n_workers: int = 1
    val_split: float = 0.2
    task: Literal["multiclass", "multilabel"] = "multilabel"
    subset: Optional[int] = None
    sampling_rate: int = 32_000
    class_weights_loss: Optional[bool] = None
    class_weights_sampler: Optional[bool] = None
    classlimit: Optional[int] = None
    eventlimit: Optional[int] = None
    direct_fingerprint: Optional[str] = (
        None  # TODO only supported in PretrainDatamodule
    )
    fewshot: Optional[FewShotConfig] = None


@dataclass
class LoaderConfig:
    """
    A class used to configure the data loader for the model.

    Attributes
    ----------
    batch_size : int
        Specifies the number of samples contained in each batch. This is a crucial parameter as it impacts memory utilization and model performance.
    shuffle : bool
        Determines whether the data is shuffled at the beginning of each epoch. Shuffling is typically used for training data to ensure model robustness and prevent overfitting.
    num_workers : int
        Sets the number of subprocesses to be used for data loading. More workers can speed up the data loading process but also increase memory consumption.
    pin_memory : bool
        When set to `True`, enables the DataLoader to copy Tensors into CUDA pinned memory before returning them. This can lead to faster data transfer to CUDA-enabled GPUs.
    drop_last : bool
        Determines whether to drop the last incomplete batch. Setting this to `True` is useful when the total size of the dataset is not divisible by the batch size.
    persistent_workers : bool
        Indicates whether the data loader should keep the workers alive for the next epoch. This can improve performance at the cost of memory.
    prefetch_factor : int
        Defines the number of samples loaded in advance by each worker. This parameter is commented out here and can be adjusted based on specific requirements.
    """

    batch_size: int = 32
    shuffle: bool = True
    num_workers: int = 1
    pin_memory: bool = True
    drop_last: bool = False
    persistent_workers: bool = True
    prefetch_factor: int = 2


@dataclass
class LoadersConfig:
    train: LoaderConfig = LoaderConfig()
    valid: LoaderConfig = LoaderConfig(shuffle=False)
    test: LoaderConfig = LoaderConfig(shuffle=False)
