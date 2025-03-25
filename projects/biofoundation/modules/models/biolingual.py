import torch
import torch.nn as nn
from transformers import ClapModel, ClapProcessor

from typing import Tuple
from birdset.configs import PretrainInfoConfig
from typing import Optional

from biofoundation.modules.models.birdset_model import BirdSetModel


class BioLingualClassifier(BirdSetModel):
    """
    Pretrained model for audio classification using the Biolingual model.

    This file includes code from BioLingual by David Robinson, licensed under the Apache-2.0 License
    Github-Repository: https://github.com/david-rx/BioLingual
    Paper: https://arxiv.org/abs/2308.04978

    Important Parameters:
    ---------------------
    checkpoint: Path to the AVES model checkpoint.
    n_last_hidden_layer: Number of last hidden layer (from the back) to extract the embeddings from. Default is 1.
    train_classifier: If True, the model will output the embeddings and freeze the feature extractor. Default is False.
    """

    EMBEDDING_SIZE = 512

    def __init__(
        self,
        num_classes: int = None,
        embedding_size: int = EMBEDDING_SIZE,
        checkpoint: str = "laion/clap-htsat-unfused",
        local_checkpoint: str = None,
        load_classifier_checkpoint: bool = True,
        freeze_backbone: bool = False,
        preprocess_in_model: bool = False,
        classifier: nn.Module = None,
        pretrain_info: PretrainInfoConfig = None,
        device: int | str = "cuda",
    ):
        """
        Note: Either num_classes or pretrain_info must be given
        Args:
            checkpoint: huggingface checkpoint path of any model of correct type
            num_classes: number of classification heads to be used in the model
            local_checkpoint: local path to checkpoint file
            cache_dir: specified cache dir to save model files at
            pretrain_info: hf_path and hf_name of info will be used to infer if num_classes is None
        """
        super().__init__(
            num_classes=num_classes,
            embedding_size=embedding_size,
            local_checkpoint=local_checkpoint,
            load_classifier_checkpoint=load_classifier_checkpoint,
            freeze_backbone=freeze_backbone,
            preprocess_in_model=preprocess_in_model,
            pretrain_info=pretrain_info
        )

        self.checkpoint = checkpoint
        self.device = device

        self.model = ClapModel.from_pretrained(checkpoint).to(self.device)

        if classifier is None:
            self.classifier = nn.Linear(embedding_size, num_classes)
        else:
            self.classifier = classifier

        #self.processor = ClapProcessor.from_pretrained(checkpoint) This takes too much memory if activated

        if local_checkpoint:
            self._load_local_checkpoint()

        if freeze_backbone:
            for param in self.model.parameters():
                param.requires_grad = False

    def _preprocess(self, input_values: torch.Tensor) -> torch.Tensor:
        """
        Preprocessing for the input values is done in BETAs.py
        The waveform gets resampled to 16kHz, transformed into a fbank and then normalized.
        """
        if self.preprocess_in_model:
            input_values = input_values .squeeze(1)
            return self.processor(
                audios=input_values.cpu().numpy(),
                return_tensors="pt",
                sampling_rate=48000,
            ).input_features.to(input_values.device)
        else:
            return input_values

    def forward(
        self, input_values: torch.Tensor, labels: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        embeddings = self.get_embeddings(input_values)

        return self.classifier(embeddings)

    def get_embeddings(self, input_tensor) -> torch.Tensor:
        inputs = self._preprocess(input_tensor)
        audio_embed = self.model.get_audio_features(
            inputs, output_hidden_states=True, return_dict=True
        )
        # audio_embed doesnt return hidden states for some reason
        return audio_embed
