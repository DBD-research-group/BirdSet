import torch
import datasets
import torch.nn as nn
from typing import Tuple
from birdset.configs import PretrainInfoConfig
from torchaudio.models import wav2vec2_model
import json
from typing import Optional



class AvesClassifier(nn.Module):
    """
    Pretrained model for audio classification using the AVES model.
    
    This file includes code from AVES by Masato Hagiwara, licensed under the MIT License
    Copyright (c) 2022 Earth Species Project
    Github-Repository: https://github.com/earthspecies/aves 
    Paper: https://arxiv.org/abs/2210.14493

    Important Parameters:
    ---------------------
    model_path: Path to the AVES model checkpoint.
    n_last_hidden_layer: Number of last hidden layer (from the back) to extract the embeddings from. Default is 1.
    train_classifier: If True, the model will output the embeddings and freeze the feature extractor. Default is False. 
    """
    EMBEDDING_SIZE = 768
    
    def __init__(self, 
                 model_path: str, 
                 config_path: str,
                 num_classes: int = None, 
                 local_checkpoint: str = None,
                 pretrain_info: PretrainInfoConfig = None,
                 n_last_hidden_layer: int = 1,
                 train_classifier: bool = False
                 ):

        super().__init__()
        
        self.n_last_hidden_layer = n_last_hidden_layer

        if pretrain_info:
            self.hf_path = pretrain_info.hf_path
            self.hf_name = (
                pretrain_info.hf_name
                if not pretrain_info.hf_pretrain_name
                else pretrain_info.hf_pretrain_name
            )
            if self.hf_path == 'DBD-research-group/BirdSet':
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

        state_dict = None
        if local_checkpoint:
            state_dict = torch.load(local_checkpoint)["state_dict"]
            state_dict = {key.replace('model.model.', ''): weight for key, weight in state_dict.items()}

        
        self.config = self.load_config(config_path)
        self.model = wav2vec2_model(**self.config, aux_num_out=None)
        self.model.load_state_dict(torch.load(model_path))
        self.model.feature_extractor.requires_grad_(True) #! Taken out
        
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

    def load_config(self, config_path):
        with open(config_path, 'r') as ff:
            obj = json.load(ff)

        return obj
        
    def forward(self, input_values: torch.Tensor, labels: Optional[torch.Tensor] = None
        ) -> torch.Tensor:
        embeddings = self.get_embeddings(input_values)[0]
        if self.train_classifier:
            flattend_embeddings = embeddings.reshape(embeddings.size(0), -1)
            # Pass embeddings through the classifier to get the final output
            output = self.classifier(flattend_embeddings)
        else:
            output = embeddings

        return output
    
    
    def get_embeddings(
        self, input_tensor
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        input_tensor = input_tensor.squeeze(1)
        last_hidden_state = self.model.extract_features(input_tensor)[0][-self.n_last_hidden_layer]
        cls_state = last_hidden_state[:,0,:]

        return cls_state, None
        