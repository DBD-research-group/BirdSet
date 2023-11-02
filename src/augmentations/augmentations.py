import random
from typing import Optional, TypedDict

import audiomentations
import librosa
import numpy as np
import torch
import torchvision
import torch_audiomentations

from torchaudio import transforms


def pad_spectrogram_width(
    spectrogram: torch.Tensor, target_width: int, value: float
) -> torch.Tensor:
    """
    Pad the input spectrogram along the width axis to reach the desired width.

    Args:
        spectrogram (torch.Tensor): Input spectrogram tensor with shape (C, H, W),
            where C is the number of channels, H is the height, and W is the original width.
        target_width (int): The desired width of the padded spectrogram.
        value (float): The value to fill the padding with.

    Returns:
        torch.Tensor: Padded spectrogram tensor with shape (C, H, target_width).
    """
    spectrogram_width = spectrogram.shape[2]

    # Calculate the amount of padding required on each side
    pad_left = (target_width - spectrogram_width) // 2
    pad_right = target_width - spectrogram_width - pad_left

    # Pad only along the width axis
    # Note: torch.nn.functional.pad pads in the format (padding_left, padding_right, padding_top, padding_bottom)
    padded_spectrogram = torch.nn.functional.pad(
        spectrogram, (pad_left, pad_right, 0, 0), mode="constant", value=value
    )

    return padded_spectrogram


class RandomTimeStretch:
    def __init__(
        self,
        prob: float = 0.33,
        n_freq: int = 96,
        min_rate: float = 0.8,
        max_rate: float = 1.25,
        hop_length: Optional[int] = None,
    ):
        """Random Time Stretch (is applied randomly and selects the rate randomly).

        Args:
            prob (float): Probability of the spectrogram being time-stretched. Default value is 0.33.
            n_freq (int): Number of filter banks from stft.
            min_rate (float): Minimum rate to speed up or slow down by.
            max_rate (float): Maximum rate to speed up or slow down by.
            hop_length (Optional[int]): Hop length for time-stretching. Default is None.
        """
        self.prob = prob
        self.n_freq = n_freq
        self.min_rate = min_rate
        self.max_rate = max_rate
        self.hop_length = hop_length

    def __call__(self, spectrogram: torch.Tensor) -> torch.Tensor:
        """
        Time-stretch the input spectrogram with a random rate.

        Args:
            spectrogram (torch.Tensor): Spectrogram to be time-stretched. A tensor of dimension (…, freq, num_frame)
            with complex dtype.

        Returns:
            torch.Tensor: Time-stretched spectrogram if time stretch is applied; else, the original spectrogram.
        """
        if random.random() < self.prob:
            time_stretch = transforms.TimeStretch(
                n_freq=self.n_freq, hop_length=self.hop_length
            )

            # Generate a random rate to speed up or slow down by
            rate = np.random.uniform(low=self.min_rate, high=self.max_rate)

            # Apply time-stretching with the generated rate
            spectrogram_time_stretched = time_stretch(spectrogram, rate)

            # Convert the complex-valued STFT returned by TimeStretch into a power spectrogram
            spectrogram_time_stretched = torch.abs(
                spectrogram_time_stretched
            )  # Compute the magnitude of the STFT
            spectrogram_time_stretched = spectrogram_time_stretched.pow(
                2
            )  # Compute the power spectrogram

            # Pad the spectrogram along the width axis
            # so that it retains its original width, and the time-stretched audio is in the middle
            original_width = spectrogram.shape[2]
            spectrogram_time_stretched = pad_spectrogram_width(
                spectrogram=spectrogram_time_stretched,
                target_width=original_width,
                value=1e-10,
            )

            return spectrogram_time_stretched

        # If time-stretching is not applied, return the original complex-valued STFT as power spectrogram
        spectrogram = torch.abs(spectrogram)  # Compute the magnitude of the STFT
        spectrogram = spectrogram.pow(2)  # Compute the power spectrogram

        return spectrogram

    def __repr__(self):
        return f"{self.__class__.__name__}(p={self.prob})"


class SpecAugmentations(TypedDict):
    """A class representing the configuration for spectrogram augmentations.

    Attributes:
        time_masking (dict): Time masking parameters.
        frequency_masking (dict): Frequency masking parameters.
        time_stretch (dict): Time stretching parameters.
    """

    time_masking: dict
    frequency_masking: dict
    time_stretch: dict


class WaveAugmentations(TypedDict):
    """
    A class representing the configuration for waveform augmentations.

    Attributes:
        colored_noise (dict): Colored noise augmentation parameters.
        background_noise (dict): Background noise augmentation parameters.
        pitch_shift (dict): Pitch shifting parameters.
        time_mask (dict): Time masking parameters.
        time_stretch (dict): Time stretching parameters.
    """

    colored_noise: dict
    background_noise: dict
    pitch_shift: dict
    time_mask: dict
    time_stretch: dict


class AudioAugmentor:
    def __init__(
        self,
        sample_rate: int = 16000,
        use_spectrogram: bool = False,
        waveform_augmentations: Optional[WaveAugmentations] = None,
        spectrogram_augmentations: Optional[SpecAugmentations] = None,
        n_fft: Optional[int] = 2048,
        hop_length: Optional[int] = 1024,
        n_mels: Optional[int] = None,
        db_scale: bool = False,
    ):
        """
        Initialize the AudioAugmentor, which is used for data augmentation of waveforms and spectrograms.

        Args:
            sample_rate (int): Sample rate of the audio signal.
            use_spectrogram (bool): Flag indicating whether to convert waveforms to spectrograms.
            waveform_augmentations (Optional[WaveAugmentations]): Configuration for waveform augmentations.
            spectrogram_augmentations (Optional[SpecAugmentations]): Configuration for spectrogram augmentations.
            n_fft (Optional[int]): Size of the FFT, only required if use_spectrogram=True.
            hop_length (Optional[int]): Length of hop between STFT windows, only required if use_spectrogram=True.
            n_mels (Optional[int]): Number of Mel filter banks. If not specified, the spectrogram will not be converted
             to a Mel spectrogram. Only required if use_spectrogram=True.
            db_scale (bool): Flag indicating whether to convert spectrograms to decibel (dB) units. Only required if
            use_spectrogram=True.
        """
        self.sample_rate = sample_rate
        self.use_spectrogram = use_spectrogram
        self.waveform_augmentations = waveform_augmentations
        self.spectrogram_augmentations = spectrogram_augmentations
        self.n_fft = n_fft
        self.hop_length = hop_length
        self.n_mels = n_mels
        self.db_scale = db_scale

    def augment_waveform(self, waveform: torch.Tensor) -> torch.Tensor:
        """
        Apply waveform augmentations if specified in the configuration.

        Args:
            waveform (torch.Tensor): Input waveform.

        Returns:
            torch.Tensor: Augmented waveform if waveform_augmentations are specified; else, the original waveform.
        """
        if self.waveform_augmentations:
            augmentations_torch_audiomentations = []
            augmentations_audiomentations = []

            # audiomentations expects 1d numpy arrays, with shape [num_samples,],
            waveform = np.array(waveform)

            # Make a randomly chosen part of the audio silent.
            if "time_mask" in self.waveform_augmentations:
                min_band_part = self.waveform_augmentations["time_mask"][
                    "min_band_part"
                ]
                max_band_part = self.waveform_augmentations["time_mask"][
                    "max_band_part"
                ]
                prob = self.waveform_augmentations["time_mask"]["prob"]
                time_mask = audiomentations.TimeMask(
                    min_band_part=min_band_part, max_band_part=max_band_part, p=prob
                )
                augmentations_audiomentations.append(time_mask)

            # Change the speed or duration of the signal without changing the pitch.
            if "time_stretch" in self.waveform_augmentations:
                min_rate = self.waveform_augmentations["time_stretch"]["min_rate"]
                max_rate = self.waveform_augmentations["time_stretch"]["max_rate"]
                prob = self.waveform_augmentations["time_stretch"]["prob"]
                time_stretch = audiomentations.TimeStretch(
                    min_rate=min_rate, max_rate=max_rate, p=prob
                )
                augmentations_audiomentations.append(time_stretch)

            waveform_transforms_audiomentations = audiomentations.Compose(
                transforms=augmentations_audiomentations
            )

            waveform_augmented = waveform_transforms_audiomentations(
                samples=waveform, sample_rate=self.sample_rate
            )

            # torch-audiomentations expects 3d PyTorch tensors, with shape [batch_size, num_channels, num_samples],
            # so we expand the first two dimensions
            waveform_augmented = torch.Tensor(waveform_augmented)
            waveform_augmented = waveform_augmented.unsqueeze(0).unsqueeze(0)

            # Add colored noises to the input audio.
            if "colored_noise" in self.waveform_augmentations:
                prob = self.waveform_augmentations["colored_noise"]["prob"]
                min_snr_in_db = self.waveform_augmentations["colored_noise"][
                    "min_snr_in_db"
                ]
                max_snr_in_db = self.waveform_augmentations["colored_noise"][
                    "max_snr_in_db"
                ]
                min_f_decay = self.waveform_augmentations["colored_noise"][
                    "min_f_decay"
                ]
                max_f_decay = self.waveform_augmentations["colored_noise"][
                    "max_f_decay"
                ]
                colored_noise = torch_audiomentations.AddColoredNoise(
                    p=prob,
                    min_snr_in_db=min_snr_in_db,
                    max_snr_in_db=max_snr_in_db,
                    min_f_decay=min_f_decay,
                    max_f_decay=max_f_decay,
                )
                augmentations_torch_audiomentations.append(colored_noise)

            # Add background noise to the input audio.
            if "background_noise" in self.waveform_augmentations:
                prob = self.waveform_augmentations["background_noise"]["prob"]
                min_snr_in_db = self.waveform_augmentations["background_noise"][
                    "min_snr_in_db"
                ]
                max_snr_in_db = self.waveform_augmentations["background_noise"][
                    "max_snr_in_db"
                ]
                background_paths = self.waveform_augmentations["background_noise"][
                    "background_paths"
                ]
                background_noise = torch_audiomentations.AddBackgroundNoise(
                    background_paths=background_paths,
                    p=prob,
                    min_snr_in_db=min_snr_in_db,
                    max_snr_in_db=max_snr_in_db,
                )
                augmentations_torch_audiomentations.append(background_noise)

            # Pitch-shift sounds up or down without changing the tempo.
            if "pitch_shift" in self.waveform_augmentations:
                prob = self.waveform_augmentations["pitch_shift"]["prob"]
                min_transpose_semitones = self.waveform_augmentations["pitch_shift"][
                    "min_transpose_semitones"
                ]
                max_transpose_semitones = self.waveform_augmentations["pitch_shift"][
                    "max_transpose_semitones"
                ]
                pitch_shift = torch_audiomentations.PitchShift(
                    p=prob,
                    sample_rate=self.sample_rate,
                    min_transpose_semitones=min_transpose_semitones,
                    max_transpose_semitones=max_transpose_semitones,
                )
                augmentations_torch_audiomentations.append(pitch_shift)

            waveform_transforms_torch_audiomentations = torch_audiomentations.Compose(
                transforms=augmentations_torch_audiomentations
            )

            waveform_augmented = waveform_transforms_torch_audiomentations(
                waveform_augmented, sample_rate=self.sample_rate
            )

            # Squeeze the first dimension so that we get a tensor with the shape [num_channels, num_samples]
            waveform_augmented = waveform_augmented.squeeze(0)

            return waveform_augmented

        # Expand the first dimension so that we get a PyTorch tensor, with shape [num_channels, num_samples]
        waveform = torch.Tensor(waveform)
        waveform = waveform.unsqueeze(0)

        return waveform

    def augment_spectrogram(self, spectrogram: torch.Tensor) -> torch.Tensor:
        """
        Apply spectrogram augmentations if specified in the configuration.

        Args:
            spectrogram (torch.Tensor): Input spectrogram.

        Returns:
            torch.Tensor: Augmented spectrogram if spectrogram_augmentations are specified; else, the original
            spectrogram.
        """
        if self.spectrogram_augmentations:
            all_augmentations = []

            # Stretch stft in time without modifying pitch for a given rate.
            # Note: Time stretching must be the first augmentation technique applied because it takes a complex-valued
            # STFT and returns a power spectrogram, while all other augmentation techniques take power spectrograms.
            if "time_stretch" in self.spectrogram_augmentations:
                n_freq = self.n_fft // 2 + 1
                hop_length = self.hop_length

                min_rate = self.spectrogram_augmentations["time_stretch"]["min_rate"]
                max_rate = self.spectrogram_augmentations["time_stretch"]["max_rate"]
                prob = self.spectrogram_augmentations["time_stretch"]["prob"]

                time_stretch = RandomTimeStretch(
                    prob=prob,
                    n_freq=n_freq,
                    min_rate=min_rate,
                    max_rate=max_rate,
                    hop_length=hop_length,
                )
                all_augmentations.append(time_stretch)

            # Apply masking to a spectrogram in the time domain.
            if "time_masking" in self.spectrogram_augmentations:
                time_mask_param = self.spectrogram_augmentations["time_masking"][
                    "time_mask_param"
                ]
                prob = self.spectrogram_augmentations["time_masking"]["prob"]
                time_masking = torchvision.transforms.RandomApply(
                    [transforms.TimeMasking(time_mask_param=time_mask_param)], p=prob
                )
                all_augmentations.append(time_masking)

            # Apply masking to a spectrogram in the frequency domain.
            if "frequency_masking" in self.spectrogram_augmentations:
                freq_mask_param = self.spectrogram_augmentations["frequency_masking"][
                    "freq_mask_param"
                ]
                prob = self.spectrogram_augmentations["frequency_masking"]["prob"]
                frequency_masking = torchvision.transforms.RandomApply(
                    [transforms.FrequencyMasking(freq_mask_param=freq_mask_param)],
                    p=prob,
                )
                all_augmentations.append(frequency_masking)

            spectrogram_transforms = torchvision.transforms.Compose(all_augmentations)

            spectrogram_augmented = spectrogram_transforms(spectrogram)

            return spectrogram_augmented

        return spectrogram

    def transform_to_spectrogram(
        self, waveform: torch.Tensor, power: Optional[float] = 2.0
    ) -> torch.Tensor:
        """
        Transform waveform to spectrogram.

        Args:
            waveform (torch.Tensor): Input waveform.
            power (Optional[float]): Exponent for the magnitude spectrogram, (must be > 0) e.g., 1 for magnitude, 2 for
             power, etc. If None, then the complex spectrum is returned instead.

        Returns:
            torch.Tensor: Spectrogram representation of the input waveform.
        """
        transform = transforms.Spectrogram(
            n_fft=self.n_fft, hop_length=self.hop_length, power=power
        )
        spectrogram = transform(waveform)
        return spectrogram

    def transform_to_mel_scale(self, spectrogram: torch.Tensor) -> torch.Tensor:
        """
        Transform spectrogram to mel scale.

        Args:
            spectrogram (torch.Tensor): Input spectrogram.

        Returns:
            torch.Tensor: Mel spectrogram representation of the input spectrogram.
        """
        transform = transforms.MelScale(
            n_mels=self.n_mels, sample_rate=self.sample_rate, n_stft=self.n_fft // 2 + 1
        )
        mel_spectrogram = transform(spectrogram)
        return mel_spectrogram

    def combined_augmentations(self, waveform: torch.Tensor) -> torch.Tensor:
        """
        Apply combined augmentations to the input waveform.

        Args:
            waveform (torch.Tensor): Input waveform.

        Returns:
            torch.Tensor: Augmented waveform (or spectrogram if use_spectrogram=True).
        """

        waveform_augmented = self.augment_waveform(waveform)

        if self.use_spectrogram:
            if (self.spectrogram_augmentations is not None) and (
                "time_stretch" in self.spectrogram_augmentations
            ):
                # Time stretching requires a complex-valued STFT, but returns a power spectrogram.
                spectrogram = self.transform_to_spectrogram(
                    waveform_augmented, power=None
                )
            else:
                # When time stretching is not used, we work directly with power spectrograms.
                spectrogram = self.transform_to_spectrogram(
                    waveform_augmented, power=2.0
                )

            spectrogram_augmented = self.augment_spectrogram(spectrogram)

            # Transform spectrogram to mel scale
            if self.n_mels:
                spectrogram_augmented = self.transform_to_mel_scale(
                    spectrogram_augmented
                )

            if self.db_scale:
                # Convert spectrogram to decibel (dB) units.
                spectrogram_augmented = spectrogram_augmented.numpy()
                spectrogram_augmented = librosa.power_to_db(spectrogram_augmented)
                spectrogram_augmented = torch.from_numpy(spectrogram_augmented)

            return spectrogram_augmented

        return waveform_augmented
