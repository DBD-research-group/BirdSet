import torch
import torch.nn as nn
from transformers import ClapModel, ClapProcessor
#from transformers import pipeline
import datasets
from typing import Tuple
from birdset.configs import PretrainInfoConfig

class BioLingualClassifier(nn.Module):
    def __init__(self,
                 checkpoint: str,
                 num_classes: int = None,
                 local_checkpoint: str = None,
                 cache_dir: str = None,
                 pretrain_info: PretrainInfoConfig = None,
                 n_last_hidden_layer: int = 1):
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
                

        self.cache_dir = cache_dir

        state_dict = None
        if local_checkpoint:
            state_dict = torch.load(local_checkpoint)["state_dict"]
            state_dict = {key.replace('model.model.', ''): weight for key, weight in state_dict.items()}

        
        self.model = ClapModel.from_pretrained(checkpoint)
        self.processor = ClapProcessor.from_pretrained(checkpoint)
        
    def get_embeddings(self, input_tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        input_tensor = input_tensor.squeeze(1)
        device = input_tensor.device
        inputs = self.processor(audios=input_tensor.cpu().numpy(), return_tensors="pt", sampling_rate=48000).to(device)
        audio_embed = self.model.get_audio_features(**inputs, output_hidden_states = True, return_dict = True)
        # audio_embed doesnt return hidden states for some reason
        return audio_embed, None      