from typing import Dict, List, Optional

import numpy as np
from omegaconf import DictConfig
import torch

from src.datamodule.components.augmentations import AudioAugmentor, WaveAugmentations, SpecAugmentations


class TransformsWrapper:
    def __init__(self,
                 mode: str,
                 sample_rate: int,
                 normalize: bool = False,
                 use_spectrogram: bool = False,
                 n_fft: Optional[int] = 2048,
                 hop_length: Optional[int] = 1024,
                 n_mels: Optional[int] = None,
                 db_scale: Optional[bool] = False,
                 waveform_augmentations: Optional[WaveAugmentations] = None,
                 spectrogram_augmentations: Optional[SpecAugmentations] = None,
                 ) -> None:
        """TransformsWrapper module.

        Args:
            transforms_config (DictConfig): Transforms config.
        """

        self.normalize = normalize
        self.sample_rate = sample_rate

        self.use_spectrogram = use_spectrogram
        self.n_fft = n_fft
        self.hop_length = hop_length
        self.n_mels = n_mels
        self.db_scale = db_scale

        if mode == "train":
            self.waveform_augmentations = waveform_augmentations
            self.spectrogram_augmentations = spectrogram_augmentations

        elif mode in ("valid", "test", "predict"):
            self.waveform_augmentations = None
            self.spectrogram_augmentations = None
        else:
            raise NotImplementedError(f"The mode {mode} is not implemented.")

    def _transform_function(
        self,
        waveform: Dict[str, torch.Tensor],
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
            sample_rate=self.sample_rate,
            use_spectrogram=self.use_spectrogram,
            spectrogram_augmentations=self.spectrogram_augmentations,
            waveform_augmentations=self.waveform_augmentations,
            n_fft=self.n_fft,
            hop_length=self.hop_length,
            n_mels=self.n_mels,
            db_scale=self.db_scale,
        )

        waveform = np.array(waveform)
        print(waveform.shape)

        audio_augmented = audio_augmentor.combined_augmentations(waveform)

        if self.normalize:
            raise NotImplementedError("Normalizations are not implemented yet!")

        return audio_augmented

    def __call__(
        self,
        examples: Dict[str, List[torch.Tensor]],
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

        print(np.array(examples["input_values"]).shape)

        examples["input_values"] = [
            self._transform_function(
                waveform=audio,
            )
            for audio in examples["input_values"]
        ]
        return examples
