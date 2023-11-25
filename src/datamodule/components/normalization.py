from typing import Optional, Tuple

import datasets
import torch
from omegaconf import DictConfig
from torch.utils.data import DataLoader


# TODO: Also add logs to this function
class NormalizationWrapper:
    def __init__(
        self, config: DictConfig, dataset: Optional[datasets.Dataset] = None
    ) -> None:
        """
        Initializes the NormalizationWrapper module.

        Args:
            config (DictConfig): Configuration for normalization, including mean, std, and whether to use spectrogram.
            dataset (Optional[datasets.Dataset]): The dataset to be used for calculating mean and std, if not provided
            in config. Defaults to None.

        Attributes:
            config (DictConfig): Configuration dictionary.
            dataset (Optional[datasets.Dataset]): Dataset for normalization.
            dataloader (DataLoader): DataLoader, initialized as None and can be set later for mean and std calculation.
            use_spectrogram (bool): Flag to determine if spectrogram or waveform data should be used.
            mean (float): Mean value for standardization.
            std (float): Standard deviation for standardization.
        """

        self.config = config
        self.dataset = dataset
        self.dataloader = None  # Set as None initially; can be assigned later

        # Extract the use_spectrogram flag from config; ensure it's a boolean
        self.use_spectrogram = config.get("use_spectrogram", False)

        # Retrieve or calculate mean and standard deviation
        self.mean, self.std = self.get_mean_std()

    def standardize_tensor(self, tensor: torch.Tensor) -> torch.Tensor:
        """
        Standardizes a given tensor using the pre-calculated mean and standard deviation.

        Args:
            tensor (torch.Tensor): The input tensor to be standardized.

        Returns:
            torch.Tensor: The standardized tensor.

        Note:
            The standardization formula used is: (tensor - mean) / std.
        """

        # Standardize the tensor using the pre-calculated mean and standard deviation
        return (tensor - self.mean) / self.std

    def get_mean_std(self) -> Tuple[float, float]:
        """
        Retrieves the mean and standard deviation for normalization.

        This function attempts to fetch the mean and standard deviation in the following order:
        1. From the configuration (using `get_mean_std_from_config`).
        2. From the dataset (using `get_mean_std_from_dataset`).
        3. If both values are still undefined, it raises an error.

        Raises:
            ValueError: If both mean and standard deviation are undefined in the configuration
                        and dataset.
            ValueError: If only the mean is undefined in the configuration and dataset.
            ValueError: If only the standard deviation is undefined in the configuration and dataset.

        Returns:
            Tuple[float, float]: A tuple containing the mean and standard deviation.
        """

        # Try to get mean and std from config
        mean, std = self.get_mean_std_from_config(self.config)

        # If not found in config, try to get them from the dataset
        if mean is None or std is None:
            mean, std = self.get_mean_std_from_dataset(self.dataset)

        # Separate checks for mean and standard deviation
        if mean is None and std is None:
            raise ValueError(
                "Both mean and standard deviation are undefined in the configuration and dataset."
            )
        if mean is None:
            raise ValueError("Mean is undefined in the configuration and dataset.")
        if std is None:
            raise ValueError(
                "Standard deviation is undefined in the configuration and dataset."
            )

        return mean, std

    def get_mean_std_from_config(
        self, config: DictConfig
    ) -> Tuple[Optional[float], Optional[float]]:
        """
        Attempts to retrieve the mean and standard deviation from the provided configuration object.

        Args:
            config (DictConfig): Configuration object containing potential mean and standard deviation values.

        Returns:
            Tuple[Optional[float], Optional[float]]: A tuple containing the mean and standard deviation.
            Returns (None, None) if they are not specified in the config.
        """

        mean = config.get("mean")
        std = config.get("std")

        return mean, std

    def get_mean_std_from_dataset(
        self, dataset: datasets.Dataset
    ) -> Tuple[Optional[float], Optional[float]]:
        """
        Attempts to retrieve the mean and standard deviation from the provided dataset.

        Args:
            dataset (datasets.Dataset): The dataset which might contain mean and standard deviation values.

        Returns:
            Tuple[Optional[float], Optional[float]]: A tuple containing the mean and standard deviation.
            Returns (None, None) if they are not specified in the dataset.
        """

        # Depending on the flag 'use_spectrogram', fetch the appropriate mean and std keys
        mean_key, std_key = (
            ("spectrogram_mean", "spectrogram_std")
            if self.use_spectrogram
            else ("waveform_mean", "waveform_std")
        )

        mean = dataset.get(mean_key)
        std = dataset.get(std_key)

        return mean, std

    def calculate_mean_std_from_dataloader(
        self, dataloader: DataLoader
    ) -> Tuple[float, float]:
        """
        Calculates the mean and standard deviation from a PyTorch DataLoader.

        Args:
            dataloader (DataLoader): The DataLoader containing the dataset for calculation.

        Returns:
            Tuple[float, float]: The calculated mean and standard deviation.
        """

        sum_, sum_of_squares, num_elements = 0.0, 0.0, 0

        for batch in dataloader:
            input_values = batch["input_values"]
            sum_ += input_values.sum()
            sum_of_squares += (input_values**2).sum()
            num_elements += input_values.nelement()

        mean = sum_ / num_elements
        std = (sum_of_squares / num_elements - mean**2) ** 0.5
        return mean, std
