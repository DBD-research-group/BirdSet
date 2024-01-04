from dataclasses import dataclass
from typing import Dict, Literal

import numpy as np
from omegaconf import DictConfig
from src.datamodule.components.feature_extraction import DefaultFeatureExtractor
from src.datamodule.components.event_decoding import EventDecoding
from src.datamodule.components.augmentations import BackgroundNoise
from src.datamodule.components.augmentations import Compose

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
    target_height: int | None = None
    target_width: int | None = 1024
    normalize_spectorgram: bool = True
    normalize_waveform: Literal['instance_normalization', 'instance_min_max'] | None  = None

class BaseTransforms:
    """
    Base class to handle audio transformations for different model types and modes.

    Attributes:
        mode (str): The mode in which the class is operating. Can be "train", "valid", "test", or "predict".
        sampling_rate (int): The sampling rate of the audio data.
        max_length (int): Maximum segment lenght in seconds
        decoding (EventDecoding): Detecting events in sample (EventDecoding if None given)
        feature_extractor (DefaultFeatureExtractor): Configuration for extracting events from the audio data (DefaultFeatureExtractor id None given)
    """
    
    def __init__(self, 
                 task: Literal['multiclass', 'multilabel'] = "multiclass", 
                 sampling_rate:int = 3200, 
                 max_length:int = 5, 
                 decoding: EventDecoding | None = None,
                 feature_extractor : DefaultFeatureExtractor | None = None) -> None:
        self.mode = "train"
        self.task = task
        self.sampling_rate = sampling_rate
        self.max_length = max_length
        self.event_decoder = decoding
        if self.event_decoder is None:
            self.event_decoder = EventDecoding(min_len=0,
                                          max_len=self.max_length,
                                          sampling_rate=self.sampling_rate)
        self.feature_extractor = feature_extractor
        if self.feature_extractor is None:
            self.feature_extractor = DefaultFeatureExtractor(feature_size=1,
                                                             sampling_rate=self.sampling_rate,
                                                             padding_value=0.0,
                                                             return_attention_mask=False)
    
    def _transform(self, batch):
        """
        Called when tansformer is called
        Applies transformations to a batch of data.
        
        1. Applies Event Decoding (almost always)
        2. Applies feature extraction with FeatureExtractor
        """
        # we overwrite the feature extractor with None because we can do this here manually 
        # this is quite complicated if we want to make adjustments to non bird methods
        if self.event_decoder is not None: 
            batch = self.event_decoder(batch)

        #----
        # Feature extractor
        #----
        
        if not "audio" in batch.keys():
            raise ValueError(f"There is no audio in batch {batch.keys()}")
        
        # audio collating and padding
        waveform_batch = [audio["array"] for audio in batch["audio"]]

        # extract/pad/truncate
        # max_length determains the difference with input waveforms as factor 5 (embedding)
        max_length = int(int(self.sampling_rate) * int(self.max_length)) #!TODO: how to determine 5s
        waveform_batch = self.feature_extractor(
            waveform_batch,
            padding="max_length",
            max_length=max_length, 
            truncation=True,
            return_attention_mask=True
        )
        # print(batch)
        
        attention_mask = waveform_batch["attention_mask"]
        # i dont know why it was unsqueezed earlier, but this solves the problem of dimensionality (is now the same, if you augment further or not...)
        # waveform_batch = waveform_batch["input_values"].unsqueeze(1)
        waveform_batch = waveform_batch["input_values"]
        
        audio_augmented = self.augment_waveform_batch(waveform_batch, attention_mask, batch)
        
        if self.task == "multiclass":
            labels = batch["labels"]
        
        else:
            # self.task == "multilabel"
            # datatype of labels must be float32 to support BCEWithLogitsLoss
            labels = torch.tensor(batch["labels"], dtype=torch.float32)

        return {"input_values": audio_augmented, "labels": labels}
    
    def augment_waveform_batch(self, waveform_batch, attention_mask, batch):
        """
        Do your augmentations in derived class here
        """
        
        return waveform_batch
    
    def set_mode(self, mode):
        self.mode = mode
    
    def _prepare_call(self):
        """
        Overwrite this to prepare the call
        """
        return
    
    
    def __call__(self, batch, **kwargs):
        self._prepare_call()
        batch = self._transform(batch)

        return batch

class TransformsWrapper(BaseTransforms):
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
                spectrogram_augmentations: DictConfig = DictConfig({}), # TODO: typing is wrong, can also be List of Augmentations
                waveform_augmentations: DictConfig = DictConfig({}), # TODO: typing is wrong, can also be List of Augmentations
                decoding: EventDecoding | None = None,
                feature_extractor: DefaultFeatureExtractor = DefaultFeatureExtractor()
            ):
        max_length = 5
        super().__init__(task, sampling_rate, max_length, decoding, feature_extractor)

        self.model_type = model_type
        self.preprocessing = preprocessing
        self.waveform_augmentations = waveform_augmentations
        self.spectrogram_augmentations = spectrogram_augmentations

        # waveform augmentations
        wave_aug = []
        for wave_aug_name in waveform_augmentations:
            aug = self.waveform_augmentations.get(wave_aug_name)
            wave_aug.append(aug)

        if self.mode == "train":
            # waveform augmentations
            wave_aug = []
            if self.waveform_augmentations:
                for wave_aug_name in self.waveform_augmentations:
                    aug = self.waveform_augmentations.get(wave_aug_name)
                    wave_aug.append(aug)

            self.wave_aug = torch_audiomentations.Compose(
                transforms=wave_aug,
                output_type="tensor")

            # spectrogram augmentations
            spec_aug = []
            if self.spectrogram_augmentations:
                for spec_aug_name in self.spectrogram_augmentations:
                    aug = self.spectrogram_augmentations.get(spec_aug_name)
                    spec_aug.append(aug)
                
            self.spec_aug = torchvision.transforms.Compose(
                transforms=spec_aug)
                
        elif self.mode in ("valid", "test", "predict"):
            self.wave_aug = None
            self.spec_aug = None
        
        # self.wave_aug_background = Compose(
        #     transforms=[BackgroundNoise(p=0.5)]
        # )

        self.background_noise = BackgroundNoise(p=0.5)

        # spectrogram augmentations
        spec_aug = []
        for spec_aug_name in self.spectrogram_augmentations:
            aug = self.spectrogram_augmentations.get(spec_aug_name)
            spec_aug.append(aug)
        
        self.spec_aug = torchvision.transforms.Compose(
            transforms=spec_aug)

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

    def augment_waveform_batch(self, waveform_batch, attention_mask, batch):
        """
        Applies transformations to a batch of data.
        1. Applies Event Decoding if specified / needed
        2. Applies feature extraction with FeatureExtractor
        3. Applies augmentations to waveform
        4. Applies conversions to spectrogram and augmentations to spectrogram if task is vision
        5. Convert labels type to float32 if task is multilabel

        """

        if self.event_decoder is not None: 
            batch = self.event_decoder(batch)

        #----
        # Feature extractor
        #----

        # audio collating and padding
        waveform_batch = [audio["array"] for audio in batch["audio"]]

        # extract/pad/truncate
        waveform_batch = self.feature_extractor(
            waveform_batch,
            padding="max_length",
            max_length=self.sampling_rate*5, #!TODO: how to determine 5s
            truncation=True,
            return_attention_mask=True
        )
        
        waveform_batch = waveform_batch.unsqueeze(1)

        if self.wave_aug is not None:
            audio_augmented = self.wave_aug(
                samples=waveform_batch, sample_rate=self.sampling_rate
            )
        else:
            audio_augmented = waveform_batch
        
        # shape: batch x 1 x sample_rate
        if self.background_noise:
            noise_events = {key: batch[key] for key in ["filepath", "no_call_events"]}
            self.background_noise.noise_events = noise_events
            audio_augmented = self.background_noise(audio_augmented)
                
        if self.model_type == "waveform":
            #TODO vectorize this
            if self.preprocessing.normalize_waveform == "instance_normalization":
                # normalize #!TODO: do we have to normalize before spectrogram? '#TODO Implement normalizaton module
                audio_augmented = self._zero_mean_unit_var_norm(
                    input_values=audio_augmented,
                    attention_mask=attention_mask
                )
            elif self.preprocessing.normalize_waveform == "instance_min_max":
                audio_augmented = self._min_max_scaling(
                    input_values=audio_augmented,
                    attention_mask=attention_mask
                )

        if self.model_type == "vision":
            # spectrogram conversion and augmentation 
            audio_augmented = self._vision_augmentations(audio_augmented) #!TODO: its conversion + augmentation
            
        return audio_augmented
    
    def _zero_mean_unit_var_norm(
            self, input_values, attention_mask, padding_value=0.0
    ):
        # instance normalization taken from huggingface
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

    def _min_max_scaling(
            self, input_values, attention_mask=None, padding_value=0.0
    ):
        input_values = input_values.squeeze(1) #?
        # instance normalization to [-1,1]
        normed_input_values = []

        if attention_mask is not None: 
            attention_mask = np.array(attention_mask, np.int32)

            for vector, mask in zip(input_values, attention_mask):
                # 0 vector! 
                masked_vector = vector[mask==1]

                # check if masked vector is empty
                if masked_vector.size == 0:
                    normed_vector = np.full(vector.shape, padding_value)
                    #!TODO: check 0 length soundscape files

                min_val = masked_vector.min()
                max_val = masked_vector.max()

                normed_vector = 2 * ((vector - min_val) / (max_val - min_val + 1e-7)) - 1
                normed_vector[mask==0] = padding_value

                normed_input_values.append(normed_vector)
        else:
            for x in input_values:
                min_val = x.min()
                max_val = x.max()
                normed_vector = 2 * ((x - min_val) / (max_val - min_val + 1e-7)) - 1
                
                normed_input_values.append(normed_vector)
        return torch.stack(normed_input_values)
   
    def _vision_augmentations(self, audio_augmented):
        spectrograms = self._spectrogram_conversion(audio_augmented)
        if self.spec_aug is not None:
            spectrograms_augmented = [self.spec_aug(spectrogram) for spectrogram in spectrograms]
        else:
            spectrograms_augmented = spectrograms

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

        resizer = Resizer(
            use_spectrogram=self.preprocessing.use_spectrogram,
            db_scale=self.preprocessing.db_scale
        )

        audio_augmented = resizer.resize_spectrogram_batch(
            spectrograms_augmented,
            target_height=self.preprocessing.target_height,
            target_width=self.preprocessing.target_width
        )
        # batch_size x 1 x height x width
        if self.preprocessing.normalize_spectrogram:
            audio_augmented = (audio_augmented - (-4.268)) / (4.569 * 2)
        return audio_augmented
   
    def _prepare_call(self):
        if self.mode in ("test", "predict"):
            self.wave_aug = None
            self.spec_aug = None
            self.background_noise = None
        return