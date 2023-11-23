from typing import Dict, List, Optional

import numpy as np
from omegaconf import DictConfig
import torch

from src.datamodule.components.augmentations import (
    AudioAugmentor,
    WaveAugmentations,
    SpecAugmentations,
)
from src.datamodule.components.resize import Resizer
import hydra
import torch_audiomentations
import torchaudio
import librosa
from transformers import AutoFeatureExtractor
import torchvision

class TransformsWrapperN:
    def __init__(self, transforms_cfg: DictConfig):
        """TransformsWrapper module.

        Args:
            transforms_config (DictConfig): Transforms config.
        """

        self.mode = "train"
        self.sampling_rate = transforms_cfg.get("sampling_rate")
        self.model_type = transforms_cfg.get("model_type")

        self.preprocessing = transforms_cfg.get("preprocessing")
        self.waveform_augmentations = transforms_cfg.get("waveform_augmentations")
        self.spectrogram_augmentations = transforms_cfg.get("spectrogram_augmentations")
        self.event_extractions = transforms_cfg.get("event_extractions")
        self.resizer = Resizer(
            use_spectrogram=self.preprocessing.use_spectrogram,
            db_scale=self.preprocessing.db_scale
        )

        if self.mode == "train":
            # waveform augmentations
            wave_aug = []
            for wave_aug_name in self.waveform_augmentations:
                aug = hydra.utils.instantiate(
                    self.waveform_augmentations.get(wave_aug_name), _convert_="object"
                )
                wave_aug.append(aug)

            self.wave_aug = torch_audiomentations.Compose(
                transforms=wave_aug,
                output_type="tensor")

            # spectrogram augmentations
            spec_aug = []
            for spec_aug_name in self.spectrogram_augmentations:
                aug = hydra.utils.instantiate(
                    self.spectrogram_augmentations.get(spec_aug_name), _convert_="object"
                )
                spec_aug.append(aug)
            
            self.spec_aug = torchvision.transforms.Compose(
                transforms=spec_aug)
            
        elif self.mode in ("valid", "test", "predict"):
            self.wave_aug = None
            self.spec_aug = None
        
    def set_mode(self, mode):
        self.mode = mode

    def _spectrogram_conversion(self, waveform):

        if "time_stretch" in self.spectrogram_augmentations:
            spectrogram_transform = torchaudio.transforms.Spectrogram(
                n_fft=self.preprocessing.n_fft,
                hop_length=self.preprocessing.hop_length,
                power=0.0
                )     
        else:
            spectrogram_transform = torchaudio.transforms.Spectrogram(
                n_fft=self.preprocessing.n_fft,
                hop_length=self.preprocessing.hop_length,
                power=2.0 # hard coded?
                )     
        
        spectrograms = [spectrogram_transform(waveform) for waveform in waveform]

        return spectrograms

    def _transform_function(self, waveform: Dict[str, torch.Tensor]):
        # !TODO: event decoding
        #waveform = np.array(waveform)
        waveform = torch.Tensor(waveform)
        waveform = waveform.unsqueeze(1)
        audio_augmented = self.wave_aug(
            samples=waveform, sample_rate=self.sampling_rate
        )

        if self.model_type == "vision":
            spectrograms = self._spectrogram_conversion(audio_augmented)
            #spectrograms_augmented = self.spec_aug(spectrograms)
            spectrograms_augmented = [self.spec_aug(spectrogram) for spectrogram in spectrograms]

            if self.preprocessing.n_mels:
                melscale_transform = torchaudio.transforms.MelScale(
                    n_mels=self.preprocessing.n_mels,
                    sample_rate=self.sampling_rate,
                    n_stft=self.preprocessing.n_fft//2+1
                )
                spectrograms_augmented = [melscale_transform(spectrograms) for spectrograms in spectrograms_augmented]
        
            if self.preprocessing.db_scale:
                # list with 1 x 128 x 2026
                spectrograms_augmented = [spectrogram.numpy() for spectrogram in spectrograms_augmented]
                spectrograms_augmented = torch.from_numpy(librosa.power_to_db(spectrograms_augmented))

            audio_augmented = self.resizer.resize_spectrogram_batch(
                spectrograms_augmented,
                target_height=self.preprocessing.target_height,
                target_width=self.preprocessing.target_width
            )

            # batch_size x 1 x height x width
            if self.preprocessing.normalize:
                audio_augmented = (audio_augmented - (-4.268)) / (4.569 * 2)
            
        if self.model_type == "hf":
            pass
            # waveform_augmented_list = waveform_augmented.unsqueeze(1)
            # waveform_augmented_list = [waveform.numpy() for waveform in waveform_augmented_list]
            # extracted = extractor(waveform_augmented_list)
            
        
        return audio_augmented

    def _transform_valid_test_predict(self, waveform):
        pass

    def __call__(self, audio_samples, **kwargs):
        audio_samples["input_values"] = self._transform_function(
            audio_samples["input_values"]
        )

        return audio_samples

class TransformsWrapper:
    def __init__(
        self,
        mode: str,
        sample_rate: int,
        normalize: bool = False,
        use_spectrogram: bool = False,
        n_fft: Optional[int] = 2048,
        hop_length: Optional[int] = 1024,
        n_mels: Optional[int] = None,
        db_scale: Optional[bool] = False,
        target_height: Optional[int] = None,
        target_width: Optional[int] = None,
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

        self.resizer = Resizer(use_spectrogram=use_spectrogram)
        self.target_height = target_height
        self.target_width = target_width

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

        audio_augmented = audio_augmentor.combined_augmentations(waveform)

        # resize the data
        audio_augmented = self.resizer.resize(
            audio_augmented,
            target_height=self.target_height,
            target_width=self.target_width,
        )

        if self.normalize:
            # TODO: currently hardcoded, here we need a normalization module!
            audio_augmented = (audio_augmented - (-4.268)) / (4.569 * 2)

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

        examples["input_values"] = [
            self._transform_function(
                waveform=audio,
            )
            for audio in examples["input_values"]
        ]
        return examples