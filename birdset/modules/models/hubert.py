import torch
import torch.nn as nn
from transformers import (
    AutoModelForAudioClassification,
    AutoConfig,
    HubertModel,
    Wav2Vec2FeatureExtractor,
    HubertModel,
    HubertConfig,
)
import datasets
from typing import Tuple
from birdset.configs import PretrainInfoConfig
from birdset.modules.models.birdset_model import BirdSetModel


class HubertSequenceClassifier(BirdSetModel):
    
    EMBEDDING_SIZE = 768
    
    def __init__(
        self,
        num_classes: int = None,
        embedding_size: int = EMBEDDING_SIZE,
        checkpoint: str = "facebook/hubert-base-ls960",
        local_checkpoint: str = None,
        load_classifier_checkpoint: bool = True,
        freeze_backbone: bool = False,
        preprocess_in_model: bool = False,
        classifier: nn.Module | None = None,
        cache_dir: str = None,
        pretrain_info: PretrainInfoConfig = None
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
            pretrain_info=pretrain_info,
        )

        self.checkpoint = checkpoint
        self.classifier = classifier

        self.cache_dir = cache_dir

        state_dict = None
        model_state_dict = None
        if local_checkpoint:
            state_dict = torch.load(local_checkpoint)['state_dict']
            model_state_dict = {
                key.replace("model.model.hubert.", ""): weight
                for key, weight in state_dict.items() if key.startswith("model.model.")
            }

            # Process the keys for the classifier
            if self.classifier:
                if self.load_classifier_checkpoint:
                    try:
                        classifier_state_dict = {
                            key.replace("model.classifier.", ""): weight
                            for key, weight in state_dict.items() if key.startswith("model.classifier.")
                        }
                        self.classifier.load_state_dict(classifier_state_dict)
                    except Exception as e:
                        print(f"Could not load classifier state dict from local checkpoint: {e}")  

        self.model = AutoModelForAudioClassification.from_pretrained(
            self.checkpoint,
            num_labels=self.num_classes,
            cache_dir=self.cache_dir,
            state_dict=model_state_dict,
            ignore_mismatched_sizes=True,
        )
        
        if freeze_backbone:
            for param in self.model.parameters():
                param.requires_grad = False

    def forward(
        self, input_values, attention_mask=None, labels=None, return_hidden_state=False
    ):
        """
        This method processes the input tensor, primarily by adjusting its dimensions to match the expected
        format of the model. It first squeezes out the channel dimension, assuming it's of size one. The processed tensor is then passed through the model to
        generate outputs.
        Parameters:
        - input_values (Tensor): The main input tensor of shape (channel, height, width).
        - attention_mask (Tensor, optional): An optional mask applied to the input, used in models that
          focus on specific parts of the input like transformers. Defaults to None.
        - labels (Tensor, optional): Labels used for supervised learning, typically for computing loss.
          Defaults to None.
        - return_hidden_state (bool, optional): A flag to determine whether to return hidden states of the
          model. Defaults to False.
        """
        # Squeeze the channel dimension so that the tensor has shape (batch size, wavelength)
        input_values = input_values.squeeze(1)
        outputs = self.model(
            input_values,
            attention_mask,
            output_attentions=False,
            output_hidden_states=True,
            return_dict=True,
            labels=None,
        )
        logits = outputs["logits"]

        last_hidden_state = outputs["hidden_states"][-1]  # (batch, sequence, dim)
        cls_state = last_hidden_state[:, 0, :]  # (batch, dim)
        
        if self.classifier is None:
            if return_hidden_state:
                output = (logits, cls_state)

            else:
                output = logits
        else:
            output = self.classifier(cls_state)
            
        return output

    def get_embeddings(self, input_tensor) -> torch.Tensor:
        input_tensor = input_tensor.squeeze(1)

        outputs = self.model(
            input_tensor,
            output_attentions=False,
            output_hidden_states=True,
            return_dict=True,
            # labels=None
        )
        logits = outputs["logits"]
        # There are 13 layers
        last_hidden_state = outputs["hidden_states"][
            -self.n_last_hidden_layer
        ]  # (batch, sequence, dim)
        cls_state = last_hidden_state[:, 0, :]

        return cls_state

    @torch.inference_mode()
    def get_logits(self, dataloader, device):
        pass

    @torch.inference_mode()
    def get_probas(self, dataloader, device):
        pass

    @torch.inference_mode()
    def get_representations(self, dataloader, device):
        pass


class HubertSequenceClassifierRandomInit(HubertSequenceClassifier):

    def __init__(self, *args, **kwargs):
        super(HubertSequenceClassifierRandomInit, self).__init__(*args, **kwargs)

        config = AutoConfig.from_pretrained(
            self.checkpoint, num_labels=kwargs["num_classes"]
        )
        self.model = AutoModelForAudioClassification.from_config(config)
