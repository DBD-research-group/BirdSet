from typing import Dict, Optional, List, Tuple

import datasets
from datasets import load_dataset
import numpy.typing as npt
import torch

from src.augmentations.augmentations import AudioAugmentor


def standardize_tensor(
    x: torch.Tensor, mean: Tuple[float], std: Tuple[float]
) -> torch.Tensor:
    """
    Preprocesses a tensor of shape (N, C, H, W) by normalizing it along the channel axis using the given mean and
    standard deviation.

    Args:
        x (torch.Tensor): The input tensor of shape (N, C, H, W).
        mean (Tuple[float]): The mean values for each channel, as a tuple of floats.
        std (Tuple[float]): The standard deviation values for each channel, as a tuple of floats.

    Returns:
        torch.Tensor: The standardized tensor of shape (N, C, H, W).

    Raises:
        AssertionError: If x does not have shape (N, C, H, W) or if the length of mean or std does not match the number
        of channels.
    """
    # Check if the input tensor has the correct shape
    assert len(x.shape) == 4 and x.size(1) == len(mean) == len(
        std
    ), "The input tensor must have shape (N, C, H, W), and mean and std must be of length C"

    num_channels = x.size(1)
    y = torch.zeros_like(x)
    for i in range(num_channels):
        # Normalize the tensor along the channel axis using mean and standard deviation
        y[:, i, :, :] = (x[:, i, :, :] - mean[i]) / std[i]
    return y


def undo_standardize_tensor(
    x: torch.Tensor, mean: Tuple[float], std: Tuple[float]
) -> torch.Tensor:
    """
    Undoes the preprocessing applied by the standardize_tensor() function by scaling and shifting the tensor back to
    its original range.

    Args:
        x (torch.Tensor): The standardized tensor of shape (N, C, H, W).
        mean (Tuple[float]): The mean values for each channel, as a tuple of floats.
        std (Tuple[float]): The standard deviation values for each channel, as a tuple of floats.

    Returns:
        torch.Tensor: The original tensor of shape (N, C, H, W).

    Raises:
        AssertionError: If x does not have shape (N, C, H, W) or if the length of mean or std does not match the number
        of channels.
    """
    # Check if the input tensor has the correct shape
    assert len(x.shape) == 4 and x.size(1) == len(mean) == len(
        std
    ), "The input tensor must have shape (N, C, H, W), and mean and std must be of length C"

    num_channels = x.size(1)
    y = torch.zeros_like(x)
    for i in range(num_channels):
        # Undo the normalization applied in standardize_tensor() by scaling and shifting
        y[:, i, :, :] = x[:, i, :, :] * std[i] + mean[i]
    return y


def calculate_mean_std_dataset(
    dataset: datasets.Dataset,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Calculates the mean and standard deviation of a Hugging Face dataset.

    Args:
        dataset (datasets.Dataset): The Hugging Face dataset containing input tensors.

    Returns:
        Tuple[Tensor, Tensor]: A tuple containing the mean tensor and the standard deviation tensor.

    Raises:
        AssertionError: If the feature tensor does not have the shape (N, C, H, W).
    """
    features = torch.Tensor([])

    for example in dataset:
        # Create batch dimension
        example = example["input_values"].unsqueeze(dim=0)

        # Concatenate feature tensors
        features = torch.cat((features, example), dim=0)

    assert (
        len(features.shape) == 4
    ), "The feature tensor must have the shape (N, C, H, W)"

    mean = features.mean(dim=(0, 2, 3))
    std = features.std(dim=(0, 2, 3))
    return mean, std


def preprocess(
    waveform: Dict[str, torch.Tensor],
    use_spectrogram: bool,
    spectrogram_augmentations: Optional[Dict] = None,
    waveform_augmentations: Optional[Dict] = None,
    n_fft: Optional[int] = 1024,
    hop_length: Optional[int] = 512,
    n_mels: Optional[int] = None,
    db_scale: bool = False,
    normalize: bool = False,
    mean: Optional[Tuple[float]] = None,
    std: Optional[Tuple[float]] = None,
) -> torch.Tensor:
    """
    Preprocesses an audio waveform.

    Args:
        waveform (Dict[str, torch.Tensor]): A dictionary containing the audio waveform and its metadata.
        use_spectrogram (bool): Whether to convert the audio waveform into a spectrogram.
        spectrogram_augmentations (Optional[Dict]): Dictionary of spectrogram augmentations to apply.
        waveform_augmentations (Optional[Dict]): Dictionary of waveform augmentations to apply.
        n_fft (Optional[int]): The number of points for the FFT. Default is 1024. Only needed if use_spectrogram=True.
        hop_length (Optional[int]): The number of samples between successive frames. Default is 512. Only needed if
        use_spectrogram=True.
        n_mels (Optional[int]): The number of Mel filter banks. If not specified, the spectrogram will not be converted
         to a Mel spectrogram. Only needed if use_spectrogram=True.
        db_scale (bool): Flag indicating whether to convert spectrograms to decibel (dB) units. Only required if
        use_spectrogram=True.
        normalize (bool): Whether to normalize the audio or not. Default is False.
        mean (Optional[Tuple[float]]): The mean values for normalization. Default is None.
        std (Optional[Tuple[float]]): The standard deviation values for normalization. Default is None.

    Returns:
        torch.Tensor: The preprocessed audio waveform or spectrogram as a tensor.
    """
    audio_augmentor = AudioAugmentor(
        sample_rate=waveform["sampling_rate"],
        use_spectrogram=use_spectrogram,
        spectrogram_augmentations=spectrogram_augmentations,
        waveform_augmentations=waveform_augmentations,
        n_fft=n_fft,
        hop_length=hop_length,
        n_mels=n_mels,
        db_scale=db_scale,
    )

    audio_augmented = audio_augmentor.combined_augmentations(waveform["array"])

    if normalize:
        # Expand the first dimension so that we get a PyTorch tensor, with shape
        # [batch_size, num_channels, heigth, width]
        audio_augmented = audio_augmented.unsqueeze(0)

        # Normalize the audio using mean and std
        audio_augmented = standardize_tensor(x=audio_augmented, mean=mean, std=std)

        # Squeeze the first dimension so that we get a tensor with the shape [num_channels, heigth, width]
        audio_augmented = audio_augmented.squeeze(0)

    return audio_augmented


def apply_preprocessing(
    examples: Dict[str, List[torch.Tensor]],
    use_spectrogram: bool,
    waveform_augmentations: Optional[Dict] = None,
    spectrogram_augmentations: Optional[Dict] = None,
    n_fft: Optional[int] = None,
    hop_length: Optional[int] = None,
    n_mels: Optional[int] = None,
    db_scale: bool = False,
    normalize: bool = False,
    mean: Optional[Tuple[float]] = None,
    std: Optional[Tuple[float]] = None,
) -> Dict[str, List[torch.Tensor]]:
    """
    Apply preprocessing transforms and augmentations to a list of audio waveforms.

    Args:
        examples (Dict[str, List[torch.Tensor]]): A dictionary containing a list of audio waveforms.
        use_spectrogram (bool): Whether to convert the audio waveform into a spectrogram.
        waveform_augmentations (Optional[Dict]): Dictionary of waveform augmentations to apply.
        spectrogram_augmentations (Optional[Dict]): Dictionary of spectrogram augmentations to apply.
        n_fft (Optional[int]): The number of points for the FFT. Only needed if use_spectrogram=True.
        hop_length (Optional[int]): The number of samples between successive frames. Only needed if
        use_spectrogram=True.
        n_mels (Optional[int]): The number of Mel filter banks. If not specified, the spectrogram will not be converted
        to a Mel spectrogram. Only needed if use_spectrogram=True.
        db_scale (bool): Flag indicating whether to convert spectrograms to decibel (dB) units. Only required if
        use_spectrogram=True.
        normalize (bool): Whether to normalize the audio or not. Default is False.
        mean (Optional[Tuple[float]]): The mean values for normalization. Only needed if normalize=True.
        std (Optional[Tuple[float]]): The standard deviation values for normalization. Only needed if normalize=True.

    Returns:
        Dict[str, List[torch.Tensor]]: A dictionary containing a list of preprocessed and augmented audio waveforms or
        spectrograms.

    Notes:
        This function applies preprocessing transforms and augmentations to each audio waveform in the 'audio' list.

        If use_spectrogram=True, the audio waveform will be converted into a spectrogram and additional parameters
        (n_fft, hop_length, n_mels) will be used for spectrogram conversion.

        If normalize=True, the audio will be normalized using mean and std.

        The audio waveforms are expected to be stored in the 'audio' key of the 'examples' dictionary, and the
        preprocessed results will be stored in the 'input_values' key.
    """
    # Preprocess and augment each audio waveform in the 'audio' list and store the results in 'input_values'
    examples["input_values"] = [
        preprocess(
            waveform=audio,
            use_spectrogram=use_spectrogram,
            waveform_augmentations=waveform_augmentations,
            spectrogram_augmentations=spectrogram_augmentations,
            n_fft=n_fft,
            hop_length=hop_length,
            n_mels=n_mels,
            db_scale=db_scale,
            normalize=normalize,
            mean=mean,
            std=std,
        )
        for audio in examples["audio"]
    ]
    return examples


def create_dataset(
    path: str = "ashraq/esc50",
    split: str = "train",
    columns: List[str] = ["audio", "target"],
    test_size: float = 0.2,
    use_spectrogram: bool = True,
    waveform_augmentations: Optional[Dict] = None,
    spectrogram_augmentations: Optional[Dict] = None,
    n_fft: Optional[int] = 1024,
    hop_length: Optional[int] = 512,
    n_mels: Optional[int] = None,
    normalize_test_data: bool = True,
) -> Tuple[
    datasets.Dataset, datasets.Dataset, datasets.Dataset, Tuple[float], Tuple[float]
]:
    """
    Load and preprocess audio dataset for training and testing.

    Args:
        path (str): Path to the dataset root directory.
        split (str): Split to load from the dataset (e.g., "train", "test").
        columns (List[str]): List of columns to retain from the dataset.
        test_size (float): Proportion of examples to use for the test set.
        use_spectrogram (bool): Whether to convert the audio waveform into a spectrogram.
        waveform_augmentations (Optional[Dict]): Dictionary of waveform augmentations to apply.
        spectrogram_augmentations (Optional[Dict]): Dictionary of spectrogram augmentations to apply.
        n_fft (Optional[int]): The number of points for the FFT. Only needed if use_spectrogram=True.
        hop_length (Optional[int]): The number of samples between successive frames. Only needed if
        use_spectrogram=True.
        n_mels (Optional[int]): The number of Mel filter banks. If not specified, the spectrogram will not be converted
                                to a Mel spectrogram. Only needed if use_spectrogram=True.
        normalize_test_data (bool): Flag indicating whether the test data should be normalized.

    Returns:
        Tuple[datasets.Dataset, datasets.Dataset, datasets.Dataset, Tuple[float], Tuple[float]]:
            - train_dataset: Preprocessed training dataset.
            - train_push_dataset: Preprocessed training dataset for pushing the prototypes.
            - test_dataset: Preprocessed test dataset.
            - train_mean: Tuple containing mean values for normalization.
            - train_std: Tuple containing standard deviation values for normalization.
    """
    # Load the dataset and select the specified columns
    dataset = load_dataset(path, split=split)
    dataset = dataset.select_columns(columns)

    # Split the dataset into train and test sets
    train_test_split = dataset.train_test_split(test_size=test_size, seed=42)
    train_dataset = train_test_split["train"]
    test_dataset = train_test_split["test"]

    # Calculate mean and standard deviation for normalization from the training dataset
    train_dataset.set_transform(
        lambda x: apply_preprocessing(
            examples=x,
            use_spectrogram=use_spectrogram,
            waveform_augmentations=None,
            spectrogram_augmentations=None,
            n_fft=n_fft,
            hop_length=hop_length,
            n_mels=n_mels,
            db_scale=True,
            normalize=False,
            mean=None,
            std=None,
        )
    )

    train_mean, train_std = calculate_mean_std_dataset(dataset=train_dataset)
    train_mean = tuple(train_mean.tolist())
    train_std = tuple(train_std.tolist())

    train_dataset.reset_format()

    # Preprocess training dataset for pushing the prototypes
    train_push_dataset = train_dataset.with_transform(
        lambda x: apply_preprocessing(
            examples=x,
            use_spectrogram=use_spectrogram,
            waveform_augmentations=None,
            spectrogram_augmentations=None,
            n_fft=n_fft,
            hop_length=hop_length,
            n_mels=n_mels,
            db_scale=True,
            normalize=False,
            mean=None,
            std=None,
        )
    )

    # Apply augmentations, conversion to spectrogram, and normalization to the training dataset
    train_dataset.set_transform(
        lambda x: apply_preprocessing(
            examples=x,
            use_spectrogram=use_spectrogram,
            waveform_augmentations=waveform_augmentations,
            spectrogram_augmentations=spectrogram_augmentations,
            n_fft=n_fft,
            hop_length=hop_length,
            n_mels=n_mels,
            db_scale=True,
            normalize=True,
            mean=train_mean,
            std=train_std,
        )
    )

    # Apply conversion to spectrogram and normalization to the test dataset
    test_dataset.set_transform(
        lambda x: apply_preprocessing(
            examples=x,
            use_spectrogram=use_spectrogram,
            waveform_augmentations=None,
            spectrogram_augmentations=None,
            n_fft=n_fft,
            hop_length=hop_length,
            n_mels=n_mels,
            db_scale=True,
            normalize=normalize_test_data,
            mean=train_mean,
            std=train_std,
        )
    )

    return train_dataset, train_push_dataset, test_dataset, train_mean, train_std


def collate_batch(
    batch: List[Dict[str, torch.Tensor]], return_category: bool = True
) -> Tuple[torch.Tensor, torch.Tensor, npt.NDArray[str]]:
    """
    Collate a batch of data samples.

    Args:
        batch (List[Dict[str, torch.Tensor]]): A list of data samples, where each sample is a dictionary containing
                                               'input_values', 'target', and 'category' tensors.

    Returns:
        Tuple[torch.Tensor, torch.Tensor, numpy.ndarray]: A tuple containing input features and targets as
        tensors. While 'targets' specifies the ground truth class as an integer, 'category' specifies the ground truth
        class as a string.
    """
    # Extract 'input_values' tensor from each sample and stack them to create a batch tensor
    input_features = [sample["input_values"].unsqueeze(0).unsqueeze(0) for sample in batch]
    input_features = torch.cat(input_features, 0)

    # Extract 'target' tensor from each sample and convert it to a tensor of integers
    targets = [sample["labels"] for sample in batch]
    targets = torch.tensor(targets)
    targets = targets.to(torch.int64)

    if return_category:
        # Extract 'category' tensor from each sample
        categories = [sample["category"] for sample in batch]

        return input_features, targets, categories

    return input_features, targets
