from dataclasses import dataclass
from typing import Dict, Literal

import numpy as np
from omegaconf import DictConfig
from src.datamodule.components.feature_extraction import DefaultFeatureExtractor
import torch

from src.datamodule.components.resize import Resizer
import torch_audiomentations
import torchaudio
import librosa
import torchvision

@dataclass
class PreprocessingConfig:
    use_spectrogram: bool = True
    n_fft: int = 1024
    hop_length: int = 79
    n_mels: int = 128
    db_scale: bool = True
    target_height: int = None
    target_width: int = 1024
    normalize: bool = True

class TransformsWrapper:
    """
    A class to handle audio transformations for different model types and modes.

    Attributes:
        mode (str): The mode in which the class is operating. Can be "train", "valid", "test", or "predict".
        sampling_rate (int): The sampling rate of the audio data.
        model_type (str): The type of model being used. Can be "vision" or "waveform".
        preprocessing (PreprocessingConfig): Configuration for preprocessing the audio data.
        waveform_augmentations (DictConfig): Configuration for augmentations to apply to the waveform.
        spectrogram_augmentations (DictConfig): Configuration for augmentations to apply to the spectrogram.
        event_extractions (DefaultFeatureExtractor): Configuration for extracting events from the audio data.
        resizer (Resizer): An instance of the Resizer class for resizing the spectrogram.
    """
    def __init__(self,
                task: str = "multiclass",
                sampling_rate: int = 32000,
                model_type: Literal['vision', 'waveform'] = "waveform",
                preprocessing: PreprocessingConfig = PreprocessingConfig(),
                spectrogram_augmentations: DictConfig = DictConfig({}),
                waveform_augmentations: DictConfig = DictConfig({}),
                decoding: DictConfig = DictConfig({}), #@raphael
                feature_extractor: DefaultFeatureExtractor = DefaultFeatureExtractor()
            ):

        self.mode = "train"
        self.feature_extractor = feature_extractor
        self.task = task
        self.sampling_rate = sampling_rate 
        self.model_type = model_type

        self.preprocessing = preprocessing
        self.waveform_augmentations = waveform_augmentations
        self.spectrogram_augmentations = spectrogram_augmentations
        self.feature_extractor = feature_extractor

        self.resizer = Resizer(
            use_spectrogram=self.preprocessing.use_spectrogram,
            db_scale=self.preprocessing.db_scale
        )
        self.event_decoder = decoding

        if self.mode == "train":
            # waveform augmentations
            wave_aug = []
            for wave_aug_name in self.waveform_augmentations:
                aug = self.waveform_augmentations.get(wave_aug_name)
                wave_aug.append(aug)

            self.wave_aug = torch_audiomentations.Compose(
                transforms=wave_aug,
                output_type="tensor")

            # spectrogram augmentations
            spec_aug = []
            for spec_aug_name in self.spectrogram_augmentations:
                aug = self.spectrogram_augmentations.get(spec_aug_name)
                spec_aug.append(aug)
            
            self.spec_aug = torchvision.transforms.Compose(
                transforms=spec_aug)
            
        elif self.mode in ("valid", "test", "predict"):
            self.wave_aug = None
            self.spec_aug = None
        
    def set_mode(self, mode):
        self.mode = mode

    def _spectrogram_conversion(self, waveform):
        """
        Converts a waveform to a spectrogram.

        This method applies a spectrogram transformation to a waveform. If "time_stretch" is in the 
        `spectrogram_augmentations` attribute, the power of the spectrogram transformation is set to 0.0. 
        Otherwise, the power is set to 2.0.

        Args:
            waveform (torch.Tensor): The waveform to be converted to a spectrogram.

        Returns:
            list: A list of spectrograms corresponding to the input waveform.
        """

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
                power=2.0 # TODO: hard coded?
                )     
        
        spectrograms = [spectrogram_transform(waveform) for waveform in waveform]

        return spectrograms

    def _transform_function(self, batch):
        """
        Applies transformations to a batch.

        This method applies a series of transformations to a batch, including waveform augmentations,
        spectrogram conversion, Mel scale transformation, decibel scaling, resizing, and normalization.
        The specific transformations applied depend on the `model_type` and `preprocessing` attributes.

        Args:
            waveform (Dict[str, torch.Tensor]): A dictionary where the keys are the names of the audio samples
                and the values are the waveforms of the audio samples as PyTorch tensors.

        Returns:
            torch.Tensor: The transformed waveform. If `model_type` is "vision", the waveform is transformed
            into a spectrogram and further processed. If `model_type` is "waveform", the waveform is returned as is.
        """
        # we overwrite the feature extractor with None because we can do this here manually 
        # this is quite complicated if we want to make adjustments to non bird methods
        if self.event_decoder is not None: 
            batch = self.event_decoder(batch)
        waveform_batch = [audio["array"] for audio in batch["audio"]]

        # extract/pad/truncate
        waveform_batch = self.feature_extractor(
            waveform_batch,
            padding="max_length",
            max_length=self.sampling_rate*5,
            truncation=True,
            return_attention_mask=False
        )
        
        waveform_batch = waveform_batch["input_values"].unsqueeze(1)

        if self.wave_aug is not None:
            audio_augmented = self.wave_aug(
                samples=waveform_batch, sample_rate=self.sampling_rate
            )

        else:
            audio_augmented = waveform_batch
        
        if self.model_type == "raw":
            # normalize 
            audio_augmented = self._zero_mean_unit_var_norm(
                input_values=audio_augmented,
                attention_mask=None
            )

        elif self.model_type == "vision":
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
        
        if self.task == "multiclass":
            labels = batch["labels"]
        
        elif self.task == "multilabel":
            labels = torch.tensor(batch["labels"], dtype=torch.float32)

        return {"input_values": audio_augmented, "labels": labels}
    
    def _zero_mean_unit_var_norm(
            self, input_values, attention_mask, padding_value=0.0
    ):
        if attention_mask is not None:
            attention_mask = np.array(attention_mask, np.int32)
            normed_input_values = []

            for vector, length in zip(input_values, attention_mask.sum(-1)):
                normed_slice = (vector - vector[:length].mean()) / np.sqrt(vector[:length].var() + 1e-7)
                if length < normed_slice.shape[0]:
                    normed_slice[length:] = padding_value

                normed_input_values.append(normed_slice)
        else:
            normed_input_values = [(x - x.mean()) / np.sqrt(x.var() + 1e-7) for x in input_values]

        return torch.stack(normed_input_values)
        

    def __call__(self, batch, **kwargs):
        batch = self._transform_function(batch)

        return batch