from typing import Literal
from birdset import utils

import numpy as np
from omegaconf import DictConfig
from birdset.datamodule.components.feature_extraction import DefaultFeatureExtractor
from birdset.datamodule.components.event_decoding import EventDecoding
from birdset.datamodule.components.augmentations import Compose, NoCallMixer, PowerToDB
from birdset.modules.models.embedding_abstract import EmbeddingModel
from birdset.modules.embedding_module import EmbeddingModuleConfig

import torch
import torchaudio

from birdset.datamodule.components.resize import Resizer
import torch_audiomentations
from torchaudio.transforms import Spectrogram, MelScale
import torchvision

log = utils.get_pylogger(__name__)


# TODO not a dataclass
class PreprocessingConfig:
    """
    A class used to configure the preprocessing steps for the audio data.

    Attributes
    ----------
    spectrogram_conversion : Spectrogram
        The configuration for converting the audio waveform to a spectrogram. If not provided, a default Spectrogram configuration is used.
    resizer : Resizer
        The configuration for resizing the spectrogram. If not provided, a default Resizer configuration is used.
    melscale_conversion : MelScale
        The configuration for converting the spectrogram to a Mel scale. If not provided, a default MelScale configuration is used.
    dbscale_conversion : PowerToDB
        The configuration for converting the spectrogram to a dB scale. If not provided, a default PowerToDB configuration is used.
    normalize_spectrogram : bool
        Whether to normalize the spectrogram. Defaults to True.
    normalize_waveform : bool, optional
        Whether to normalize the audio waveform. If not provided, no normalization is applied to the waveform.
    mean : float
        The mean value used for normalization. Defaults to -4.268 (calculated on AudioSet).
    std : float
        The standard deviation used for normalization. Defaults to -4.569 (calculated on AudioSet).
    """
    def __init__(
        self,
        spectrogram_conversion: Spectrogram | None = Spectrogram(
            n_fft=1024,
            hop_length=320,
            power=2.0,
        ),
        resizer: Resizer | None = Resizer(
            db_scale=True, # manually adjusted!! False when not using it
        ),
        melscale_conversion: MelScale | None = MelScale(
            n_mels=128,
            sample_rate=32000,
            n_stft=513, # n_fft//2+1!!! how to include in code
        ),
        dbscale_conversion: PowerToDB | None = PowerToDB(),
        normalize_spectrogram=True,
        normalize_waveform=None,
        mean=-4.268,  # calculated on AudioSet
        std=-4.569,  # calculated on AudioSet
    ):
        self.spectrogram_conversion = spectrogram_conversion
        self.resizer = resizer
        self.melscale_conversion = melscale_conversion
        self.dbscale_conversion = dbscale_conversion
        self.normalize_spectrogram = normalize_spectrogram
        self.normalize_waveform = normalize_waveform
        self.mean = mean
        self.std = std


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
                 sampling_rate:int = 32000, 
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
        assert type(self.feature_extractor) == DefaultFeatureExtractor, "Feature Extractor must be of type DefaultFeatureExtractor"
    
    def _transform(self, batch):
        """
        Called when tansformer is called
        Applies transformations to a batch of data.
        
        1. Applies Event Decoding (almost always)
        2. Applies feature extraction with FeatureExtractor
        """
        batch = self.decode_batch(batch)
        
        input_values, labels = self.transform_values(batch)
        
        labels = self.transform_labels(labels)

        return {"input_values": input_values, "labels": labels}
    
    def decode_batch(self, batch):
        # we overwrite the feature extractor with None because we can do this here manually 
        # this is quite complicated if we want to make adjustments to non bird methods
        if self.event_decoder is not None: 
            batch = self.event_decoder(batch)
        
        return batch
    
    def transform_values(self, batch):
        if not "audio" in batch.keys():
            raise ValueError(f"There is no audio in batch {batch.keys()}")
        
        # audio collating and padding
        waveform_batch = [audio["array"] for audio in batch["audio"]]
        
        # extract/pad/truncate
        # max_length determains the difference with input waveforms as factor 5 (embedding)
        max_length = int(int(self.sampling_rate) * int(self.max_length)) #!TODO: how to determine 5s
        assert type(self.feature_extractor) == DefaultFeatureExtractor,        "Feature Extractor must be of type DefaultFeatureExtractor"
        waveform_batch = self.feature_extractor(
            waveform_batch,
            padding='max_length',
            max_length=max_length, 
            truncation=True,
            return_attention_mask=True
        )
        
        # i dont know why it was unsqueezed earlier, but this solves the problem of dimensionality (is now the same, if you augment further or not...)
        # waveform_batch = waveform_batch["input_values"].unsqueeze(1)
        waveform_batch = waveform_batch["input_values"]
        
        return waveform_batch, batch["labels"]
    
    def transform_labels(self, labels):
        if self.task == "multilabel": #for bcelosswithlogits
            labels = torch.tensor(labels, dtype=torch.float16)
        
        elif self.task =="multiclass":
            labels = labels
        
        return labels
    
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

class BirdSetTransformsWrapper(BaseTransforms):
    """
    A class to handle audio transformations for different model types and modes.

    Attributes
    ----------
    task : str
        Specifies the type of task (e.g., 'multiclass' or 'multilabel').
    sampling_rate : int
        The sampling rate at which the audio data should be processed.
    model_type : str
        Indicates the type of model (e.g. 'vision' for spectrogram-based models or 'waveform' for waveform-based models).
    spectrogram_augmentations : DictConfig
        The set of augmentations to be applied to the spectrogram data.
    waveform_augmentations : DictConfig
        The set of augmentations to be applied to the waveform data.
    decoding : EventDecoding | None
        The component responsible for data decoding.
    feature_extractor : DefaultFeatureExtractor
        The component responsible for feature extraction.
    max_length : int
        The maximum length for the processed data segments in seconds.
    nocall_sampler : NoCallMixer | None
        The no-call sampler component, if configured.
    preprocessing : PreprocessingConfig | None
        A preprocessing configuration or None if no preprocessing is required.
    """
    def __init__(self,
                task: Literal['multiclass', 'multilabel'] = "multilabel",
                sampling_rate: int = 32000,
                model_type: Literal['vision', 'waveform'] = "vision",
                spectrogram_augmentations: DictConfig = DictConfig({}), # TODO: typing is wrong, can also be List of Augmentations
                waveform_augmentations: DictConfig = DictConfig({}), # TODO: typing is wrong, can also be List of Augmentations
                decoding: EventDecoding | None = None,
                feature_extractor: DefaultFeatureExtractor = DefaultFeatureExtractor(),
                max_length: int = 5,
                nocall_sampler: NoCallMixer | None = None, 
                preprocessing: PreprocessingConfig | None = PreprocessingConfig()
            ):
        super().__init__(task, sampling_rate, max_length, decoding, feature_extractor)

        self.modes_to_skip = ["test", "predict"]

        self.model_type = model_type
        self.preprocessing = preprocessing
        self.waveform_augmentations = waveform_augmentations
        self.spectrogram_augmentations = spectrogram_augmentations
        self.nocall_sampler = nocall_sampler

        # waveform augmentations
        wave_aug = []
        for wave_aug_name in waveform_augmentations:
            aug = self.waveform_augmentations.get(wave_aug_name)
            wave_aug.append(aug)

        self.wave_aug = torch_audiomentations.Compose(
            transforms=wave_aug,
            output_type="object_dict")

        # spectrogram augmentations
        spec_aug = []
        for spec_aug_name in self.spectrogram_augmentations:
            aug = self.spectrogram_augmentations.get(spec_aug_name)
            spec_aug.append(aug)
        # if i set a debug point here, it takes ~15 skips to get to the value we need from the yaml file 
        self.spec_aug = torchvision.transforms.Compose(
            transforms=spec_aug)
    
    def transform_values(self, batch):
        if not "audio" in batch.keys():
            raise ValueError(f"There is no audio in batch {batch.keys()}")
        
        # audio collating and padding
        waveform_batch = self._get_waveform_batch(batch)
        
        attention_mask = waveform_batch["attention_mask"]
        input_values = waveform_batch["input_values"]
        input_values = input_values.unsqueeze(1)
        labels = torch.tensor(batch["labels"])

        if self.wave_aug: 
            input_values, labels = self._waveform_augmentation(input_values, labels)
        
        if self.nocall_sampler and self.task == "multilabel":
            input_values, labels = self.nocall_sampler(input_values, labels) 
        
        if self.preprocessing is not None:
            input_values = self._preprocess(input_values, attention_mask)
        
        return input_values, labels

    def _preprocess(self, input_values: torch.Tensor, attention_mask: torch.Tensor) -> torch.Tensor:
        if self.preprocessing.normalize_waveform:
            input_values = self._waveform_scaling(input_values, attention_mask)
        
        if self.model_type == "vision" and self.preprocessing.spectrogram_conversion is not None:
            spectrograms = self.preprocessing.spectrogram_conversion(input_values)

            if self.spec_aug:
                spectrograms = self.spec_aug(spectrograms)

            if self.preprocessing.melscale_conversion:
                spectrograms = self.preprocessing.melscale_conversion(spectrograms)

            if self.preprocessing.dbscale_conversion:
                # spectrograms = [spectrogram.numpy() for spectrogram in spectrograms]
                # spectrograms = torch.from_numpy(librosa.power_to_db(spectrograms)) #list to tensor!
                spectrograms = self.preprocessing.dbscale_conversion(spectrograms) # different results??

            if self.preprocessing.resizer:
                spectrograms = self.preprocessing.resizer.resize_spectrogram_batch(spectrograms)
                
            if self.preprocessing.normalize_spectrogram:
                spectrograms = (spectrograms - self.preprocessing.mean) / self.preprocessing.std
            
            input_values = spectrograms
        return input_values

    
    def _waveform_augmentation(self, input_values, labels):
        if self.task == "multilabel":
            labels = labels.unsqueeze(1).unsqueeze(1)
            output_dict = self.wave_aug(
                samples=input_values, 
                sample_rate=self.sampling_rate,
                targets=labels
            )
            labels = output_dict.targets.squeeze(1).squeeze(1)

        elif self.task == "multiclass": #multilabel mix is questionable
            output_dict = self.wave_aug(
                    samples=input_values, 
                    sample_rate=self.sampling_rate,
                )
            
        input_values = output_dict.samples
        
        return input_values, labels

    def _get_waveform_batch(self, batch):
        waveform_batch = [audio["array"] for audio in batch["audio"]]
        
        # extract/pad/truncate
        # max_length determains the difference with input waveforms as factor 5 (embedding)
        max_length = int(int(self.sampling_rate) * int(self.max_length)) #!TODO: how to determine 5s
        waveform_batch = self.feature_extractor(
            waveform_batch,
            padding='max_length',
            max_length=max_length, 
            truncation=True,
            return_attention_mask=True
        )
        
        return waveform_batch

    def _zero_mean_unit_var_norm(
            self, input_values, attention_mask, padding_value=0.0
    ):
        #w2v2
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

    def _zero_center_and_peak_normalization(self, input_values: torch.Tensor, target_peak: float = 0.25,) -> torch.Tensor:
        """Zero-centers and peak normalizes the input values to a specified target peak amplitude.

        Args:
            input_values (torch.Tensor): The input tensor with shape [..., T].
            target_peak (float): The target peak value for normalization.

        Returns:
            torch.Tensor: The normalized and reshaped tensor.
        """

        # Clone to avoid modifying the original tensor
        input_values = input_values.clone()

        # Subtract mean along the last dimension for zero-centering
        input_values -= torch.mean(input_values, dim=-1, keepdim=True)

        # Calculate the peak normalization factor
        peak_norm = torch.max(torch.abs(input_values), dim=-1, keepdim=True)[0]

        # Normalize the tensor to the peak value, avoiding division by zero
        input_values = torch.where(
            peak_norm > 0.0,
            input_values / peak_norm,
            input_values
        )

        # Scale to the target peak amplitude
        input_values *= target_peak

        # Reshape the tensor
        #input_values = input_values.view(-1, input_values.shape[-1])

        return input_values
    
    def _waveform_scaling(self, audio_augmented, attention_mask):
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
        elif self.preprocessing.normalize_waveform == "instance_peak_normalization":
            audio_augmented = self._zero_center_and_peak_normalization(
                input_values=audio_augmented,
            )
        return audio_augmented
   
    def _prepare_call(self):
        if self.mode in self.modes_to_skip:
            self.wave_aug = None
            self.spec_aug = None
            self.nocall_sampler = None
        return
    
class EmbeddingTransforms(BirdSetTransformsWrapper):
    def __init__(self,
                task: Literal['multiclass', 'multilabel'] = "multilabel",
                sampling_rate: int = 32000,
                model_type: Literal['vision', 'waveform'] = "vision",
                spectrogram_augmentations: DictConfig = DictConfig({}), # TODO: typing is wrong, can also be List of Augmentations
                waveform_augmentations: DictConfig = DictConfig({}), # TODO: typing is wrong, can also be List of Augmentations
                decoding: EventDecoding | None = None,
                feature_extractor: DefaultFeatureExtractor = DefaultFeatureExtractor(),
                max_length: int = 5,
                nocall_sampler: NoCallMixer | None = None, 
                preprocessing: PreprocessingConfig | None = PreprocessingConfig()) -> None:
        super().__init__(task, sampling_rate, model_type, spectrogram_augmentations, waveform_augmentations, decoding, feature_extractor, max_length, nocall_sampler, preprocessing)
    
    
    def _get_waveform_batch(self, batch):
        waveform_batch = [audio["array"] for audio in batch["audio"]]
        
        # Idea: Padd to longest audio file in batch, then do average embedding to model size 
        
        # Find the longest audio in the batch
        max_length = max(len(audio["array"]) for audio in batch["audio"])
        
        #! Implement resampling here 

        # max_length determains the difference with input waveforms as factor 5 (embedding)
        #max_length = int(int(self.sampling_rate) * int(self.max_length))
        waveform_batch = self.feature_extractor(
            waveform_batch,
            padding='max_length',
            max_length=max_length, 
            truncation=True,
            return_attention_mask=True
        )
        
        #print(waveform_batch['input_values'].shape)
        
        return waveform_batch

    # Resample function
    def _resample_audio(self, audio, orig_sr):
        resampler = torchaudio.transforms.Resample(orig_freq=orig_sr, new_freq=self.sampling_rate)
        return resampler(audio)

    def _resample_batch(self, batch, target_sample_rate):
            resampled_batch = []
            for audio in batch["audio"]:
                waveform = torch.tensor(audio["array"], dtype=torch.float32)
                orig_sample_rate = audio["sampling_rate"]
                resampler = torchaudio.transforms.Resample(orig_freq=orig_sample_rate, new_freq=target_sample_rate)
                resampled_waveform = resampler(waveform)
                audio["array"] = resampled_waveform.numpy()
                audio["sampling_rate"] = target_sample_rate
                resampled_batch.append(audio)
            return resampled_batch
