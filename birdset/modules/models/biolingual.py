import torch
import torch.nn as nn
from transformers import ClapModel, ClapProcessor

# from transformers import pipeline
import datasets
from typing import Tuple
from birdset.configs import PretrainInfoConfig
from typing import Optional


class BioLingualClassifier(nn.Module):
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
        checkpoint: str,
        num_classes: int = None,
        local_checkpoint: str = None,
        cache_dir: str = None,
        pretrain_info: PretrainInfoConfig = None,
        n_last_hidden_layer: int = 1,
        train_classifier: bool = False,
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
        super(BioLingualClassifier, self).__init__()

        self.checkpoint = checkpoint
        self.n_last_hidden_layer = n_last_hidden_layer

        if pretrain_info:
            self.hf_path = pretrain_info.hf_path
            self.hf_name = (
                pretrain_info.hf_name
                if not pretrain_info.hf_pretrain_name
                else pretrain_info.hf_pretrain_name
            )
            if self.hf_path == "DBD-research-group/BirdSet":
                self.num_classes = len(
                    datasets.load_dataset_builder(self.hf_path, self.hf_name)
                    .info.features["ebird_code"]
                    .names
                )
            else:
                self.num_classes = num_classes
        else:
            self.hf_path = None
            self.hf_name = None
            self.num_classes = num_classes

        self.cache_dir = cache_dir

        state_dict = None
        if local_checkpoint:
            state_dict = torch.load(local_checkpoint)["state_dict"]
            state_dict = {
                key.replace("model.model.", ""): weight
                for key, weight in state_dict.items()
            }

        self.model = ClapModel.from_pretrained(checkpoint)
        self.processor = ClapProcessor.from_pretrained(checkpoint)

        self.train_classifier = train_classifier
        # Define a linear classifier to use on top of the embeddings
        if self.train_classifier:
            self.classifier = nn.Sequential(
                nn.Linear(self.EMBEDDING_SIZE, 128),
                nn.ReLU(),
                nn.Dropout(0.5),
                nn.Linear(128, 64),
                nn.ReLU(),
                nn.Linear(64, self.num_classes),
            )

            for param in self.model.parameters():
                param.requires_grad = False

    def forward(
        self, input_values: torch.Tensor, labels: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        embeddings = self.get_embeddings(input_values)[0]
        if self.train_classifier:
            flattend_embeddings = embeddings.reshape(embeddings.size(0), -1)
            # Pass embeddings through the classifier to get the final output
            output = self.classifier(flattend_embeddings)
        else:
            output = embeddings

        return output

    def get_embeddings(self, input_tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        input_tensor = input_tensor.squeeze(1)
        device = input_tensor.device
        inputs = self.processor(
            audios=input_tensor.cpu().numpy(), return_tensors="pt", sampling_rate=48000
        ).to(device)
        audio_embed = self.model.get_audio_features(
            **inputs, output_hidden_states=True, return_dict=True
        )
        # audio_embed doesnt return hidden states for some reason
        return audio_embed, None
