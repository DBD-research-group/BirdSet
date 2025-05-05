from birdset.datamodule.components.transforms import BirdSetTransformsWrapper, PreprocessingConfig
from birdset.datamodule.components.feature_extraction import DefaultFeatureExtractor
from birdset.datamodule.components.event_decoding import EventDecoding
from birdset.datamodule.components.augmentations import NoCallMixer
from transformers import ClapProcessor
from typing import Literal
from omegaconf import DictConfig

class BiolingualTransforms(BirdSetTransformsWrapper):
    """
    Biolingual Transforms wrapper for BirdSet.
    """

    checkpoint = "laion/clap-htsat-unfused"
    processor = ClapProcessor.from_pretrained(checkpoint)

    def __init__(
        self,
        task: Literal["multiclass", "multilabel"] = "multilabel",
        sampling_rate: int = 32000,
        model_type: Literal["vision", "waveform"] = "vision",
        spectrogram_augmentations: DictConfig = DictConfig(
            {}
        ),  # TODO: typing is wrong, can also be List of Augmentations
        waveform_augmentations: DictConfig = DictConfig(
            {}
        ),  # TODO: typing is wrong, can also be List of Augmentations
        decoding: EventDecoding | None = None,
        feature_extractor: DefaultFeatureExtractor = DefaultFeatureExtractor(),
        max_length: int = 5,
        nocall_sampler: NoCallMixer | None = None,
        preprocessing: PreprocessingConfig | None = PreprocessingConfig(),
    ):
        super().__init__(task, sampling_rate, model_type, spectrogram_augmentations, waveform_augmentations, decoding, feature_extractor, max_length, nocall_sampler, preprocessing)

    def transform_values(self, batch):
        input_values, labels = super().transform_values(batch)
        input_values = input_values.squeeze(1)
        input_values = self.processor(
                audios=input_values.cpu().numpy(),
                return_tensors="pt",
                sampling_rate=48000,
            ).input_features
        return input_values, labels