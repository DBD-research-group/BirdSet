from dataclasses import dataclass
from typing import Dict, Literal

import datasets
import numpy as np
import transformers
from omegaconf import DictConfig
from src.utils.extraction import DefaultFeatureExtractor
import torch

from src.datamodule.components.augmentations import (
    AudioAugmentor,
    WaveAugmentations,
    SpecAugmentations,
)
from src.datamodule.components.event_decoding import EventDecoding
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
                decoding=None #@raphael
):

        self.mode = "train"
        self.task = task
        self.sampling_rate = sampling_rate 
        self.model_type = model_type

        self.preprocessing = preprocessing
        self.waveform_augmentations = waveform_augmentations
        self.spectrogram_augmentations = spectrogram_augmentations
        #self.event_extractions = event_extractions
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

        # audio collating and padding
        waveform_batch = [audio["array"] for audio in batch["audio"]]
        waveform_batch = transformers.BatchFeature({"input_values": waveform_batch})

        sequence_feature_extractor = transformers.SequenceFeatureExtractor(
            feature_size=1,
            sampling_rate=self.sampling_rate,
            padding_value=0.0,
            model_input_names=["input_values"]
        )
        #!TODO: what about the attention mask and padding values?!
        waveform_batch = sequence_feature_extractor.pad(
            waveform_batch, 
            padding="max_length", 
            max_length=self.sampling_rate*5,
            truncation=True,
            return_tensors="pt",
            return_attention_mask=False)
        
        waveform_batch = waveform_batch["input_values"].unsqueeze(1)
        if self.wave_aug is not None:

        audio_augmented = self.wave_aug(
            samples=waveform_batch, sample_rate=self.sampling_rate
        )
            audio_augmented = self.wave_aug(
            samples=waveform, sample_rate=self.sampling_rate
            )
        else:
            audio_augmented = waveform

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
            
            #print(np.array([spec.shape for spec in spectrograms_augmented]))

            audio_augmented = self.resizer.resize_spectrogram_batch(
                spectrograms_augmented,
                target_height=self.preprocessing.target_height,
                target_width=self.preprocessing.target_width
            )
            #print(np.array([spec.shape for spec in audio_augmented]))
            # batch_size x 1 x height x width
            if self.preprocessing.normalize:
                audio_augmented = (audio_augmented - (-4.268)) / (4.569 * 2)
            
        if self.model_type == "waveform":
            pass
            # waveform_augmented_list = waveform_augmented.unsqueeze(1)
            # waveform_augmented_list = [waveform.numpy() for waveform in waveform_augmented_list]
            # extracted = extractor(waveform_augmented_list)
        
        if self.task == "multiclass":
            labels = batch["labels"]
        
        elif self.task == "multilabel":
            labels = torch.tensor(batch["labels"], dtype=torch.float32)

        return {"input_values": audio_augmented, "labels": labels}
    
    def _transform_valid_test_predict(self, waveform):
        pass

    def __call__(self, batch, **kwargs):
        batch = self._transform_function(batch)

        return batch

# class TransformsWrapper:
#     def __init__(
#         self,
#         mode: str,
#         sample_rate: int,
#         normalize: bool = False,
#         use_spectrogram: bool = False,
#         n_fft: Optional[int] = 2048,
#         hop_length: Optional[int] = 1024,
#         n_mels: Optional[int] = None,
#         db_scale: Optional[bool] = False,
#         target_height: Optional[int] = None,
#         target_width: Optional[int] = None,
#         waveform_augmentations: Optional[WaveAugmentations] = None,
#         spectrogram_augmentations: Optional[SpecAugmentations] = None,
#     ) -> None:
#         """TransformsWrapper module.

#         Args:
#             transforms_config (DictConfig): Transforms config.
#         """

#         self.normalize = normalize
#         self.sample_rate = sample_rate

#         self.use_spectrogram = use_spectrogram
#         self.n_fft = n_fft
#         self.hop_length = hop_length
#         self.n_mels = n_mels
#         self.db_scale = db_scale

#         self.resizer = Resizer(use_spectrogram=use_spectrogram)
#         self.target_height = target_height
#         self.target_width = target_width

#         if mode == "train":
#             self.waveform_augmentations = waveform_augmentations
#             self.spectrogram_augmentations = spectrogram_augmentations

#         elif mode in ("valid", "test", "predict"):
#             self.waveform_augmentations = None
#             self.spectrogram_augmentations = None
#         else:
#             raise NotImplementedError(f"The mode {mode} is not implemented.")

#     def _transform_function(
#         self,
#         waveform: Dict[str, torch.Tensor],
#     ) -> torch.Tensor:
#         """
#         Preprocesses an audio waveform.

#         Args:
#             waveform (Dict[str, torch.Tensor]): A dictionary containing the audio waveform and its metadata.
#             use_spectrogram (bool): Whether to convert the audio waveform into a spectrogram.
#             spectrogram_augmentations (Optional[Dict]): Dictionary of spectrogram augmentations to apply.
#             waveform_augmentations (Optional[Dict]): Dictionary of waveform augmentations to apply.
#             n_fft (Optional[int]): The number of points for the FFT. Default is 1024. Only needed if use_spectrogram=True.
#             hop_length (Optional[int]): The number of samples between successive frames. Default is 512. Only needed if
#             use_spectrogram=True.
#             n_mels (Optional[int]): The number of Mel filter banks. If not specified, the spectrogram will not be converted
#              to a Mel spectrogram. Only needed if use_spectrogram=True.
#             db_scale (bool): Flag indicating whether to convert spectrograms to decibel (dB) units. Only required if
#             use_spectrogram=True.
#             normalize (bool): Whether to normalize the audio or not. Default is False.
#             mean (Optional[Tuple[float]]): The mean values for normalization. Default is None.
#             std (Optional[Tuple[float]]): The standard deviation values for normalization. Default is None.

#         Returns:
#             torch.Tensor: The preprocessed audio waveform or spectrogram as a tensor.
#         """
#         audio_augmentor = AudioAugmentor(
#             sample_rate=self.sample_rate,
#             use_spectrogram=self.use_spectrogram,
#             spectrogram_augmentations=self.spectrogram_augmentations,
#             waveform_augmentations=self.waveform_augmentations,
#             n_fft=self.n_fft,
#             hop_length=self.hop_length,
#             n_mels=self.n_mels,
#             db_scale=self.db_scale,
#         )

#         waveform = np.array(waveform)

#         audio_augmented = audio_augmentor.combined_augmentations(waveform)

#         # resize the data
#         audio_augmented = self.resizer.resize(
#             audio_augmented,
#             target_height=self.target_height,
#             target_width=self.target_width,
#         )

#         if self.normalize:
#             # TODO: currently hardcoded, here we need a normalization module!
#             audio_augmented = (audio_augmented - (-4.268)) / (4.569 * 2)

#         return audio_augmented

#     def __call__(
#         self,
#         examples: Dict[str, List[torch.Tensor]],
#     ) -> Dict[str, List[torch.Tensor]]:
#         """
#         Apply preprocessing transforms and augmentations to a list of audio waveforms.

#         Args:
#             examples (Dict[str, List[torch.Tensor]]): A dictionary containing a list of audio waveforms.
#             use_spectrogram (bool): Whether to convert the audio waveform into a spectrogram.
#             waveform_augmentations (Optional[Dict]): Dictionary of waveform augmentations to apply.
#             spectrogram_augmentations (Optional[Dict]): Dictionary of spectrogram augmentations to apply.
#             n_fft (Optional[int]): The number of points for the FFT. Only needed if use_spectrogram=True.
#             hop_length (Optional[int]): The number of samples between successive frames. Only needed if
#             use_spectrogram=True.
#             n_mels (Optional[int]): The number of Mel filter banks. If not specified, the spectrogram will not be converted
#             to a Mel spectrogram. Only needed if use_spectrogram=True.
#             db_scale (bool): Flag indicating whether to convert spectrograms to decibel (dB) units. Only required if
#             use_spectrogram=True.
#             normalize (bool): Whether to normalize the audio or not. Default is False.
#             mean (Optional[Tuple[float]]): The mean values for normalization. Only needed if normalize=True.
#             std (Optional[Tuple[float]]): The standard deviation values for normalization. Only needed if normalize=True.

#         Returns:
#             Dict[str, List[torch.Tensor]]: A dictionary containing a list of preprocessed and augmented audio waveforms or
#             spectrograms.

#         Notes:
#             This function applies preprocessing transforms and augmentations to each audio waveform in the 'audio' list.

#             If use_spectrogram=True, the audio waveform will be converted into a spectrogram and additional parameters
#             (n_fft, hop_length, n_mels) will be used for spectrogram conversion.

#             If normalize=True, the audio will be normalized using mean and std.

#             The audio waveforms are expected to be stored in the 'audio' key of the 'examples' dictionary, and the
#             preprocessed results will be stored in the 'input_values' key.
#         """
#         # Preprocess and augment each audio waveform in the 'audio' list and store the results in 'input_values'

#         examples["input_values"] = [
#             self._transform_function(
#                 waveform=audio,
#             )
#             for audio in examples["input_values"]
#         ]
#         return examples