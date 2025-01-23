import torch
import torch.nn as nn
from typing import Optional, Tuple
from transformers import AutoModelForAudioClassification, AutoConfig
import datasets

from birdset.utils import pylogger
from birdset.configs import PretrainInfoConfig
from birdset.modules.models.birdset_model import BirdSetModel

log = pylogger.get_pylogger(__name__)


class Wav2vec2SequenceClassifier(BirdSetModel):
    EMBEDDING_SIZE = 768

    def __init__(
        self,
        num_classes: int = None,
        embedding_size: int = EMBEDDING_SIZE,
        checkpoint: str = None,
        freeze_backbone: bool = False,
        preprocess_in_model: bool = False,
        classifier: nn.Module = None,
        local_checkpoint: str = None,
        load_classifier_checkpoint: bool = True,
        cache_dir: str = None,
        pretrain_info: PretrainInfoConfig = None,
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
            classifier=classifier,
            local_checkpoint=local_checkpoint,
            load_classifier_checkpoint=load_classifier_checkpoint,
            freeze_backbone=freeze_backbone,
            preprocess_in_model=preprocess_in_model,
            pretrain_info=pretrain_info,
        )
        self.checkpoint = checkpoint

        self.cache_dir = cache_dir

        state_dict = None
        if local_checkpoint:
            log.info(f">> Loading state dict from local checkpoint: {local_checkpoint}")
            state_dict = torch.load(local_checkpoint)["state_dict"]
            state_dict = {
                key.replace("model.model.", ""): weight
                for key, weight in state_dict.items()
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
                        log.error(f"Could not load classifier state dict from local checkpoint: {e}")      

        self.model = AutoModelForAudioClassification.from_pretrained(
            self.checkpoint,
            num_labels=self.num_classes,
            cache_dir=self.cache_dir,
            state_dict=state_dict,
            ignore_mismatched_sizes=True,
        )

        if freeze_backbone:
            for param in self.model.parameters():
                param.requires_grad = False

    def forward(self, input_values: torch.Tensor, labels: Optional[torch.Tensor] = None) -> torch.Tensor:

        if self.classifier is None:
            # Squeeze the channel dimension so that the tensor has shape (batch size, wavelength)
            input_values = input_values.squeeze(1)

            outputs = self.model(
                input_values,
                attention_mask=None,
                output_attentions=False,
                output_hidden_states=False,
                return_dict=True,
            )

            return outputs["logits"]
        
        else:
            embeddings = self.get_embeddings(input_values)
            return self.classifier(embeddings)


    def get_embeddings(self, input_tensor) -> torch.Tensor:
        input_tensor = input_tensor.squeeze(1)

        outputs = self.model(
            input_tensor,
            output_attentions=False,
            output_hidden_states=True,
            return_dict=True,
            # labels=None
        )
        last_hidden_state = outputs["hidden_states"][-1]  # (batch, sequence, dim)
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


class Wav2vec2SequenceClassifierRandomInit(Wav2vec2SequenceClassifier):

    def __init__(self, *args, **kwargs):
        super(Wav2vec2SequenceClassifierRandomInit, self).__init__(*args, **kwargs)

        config = AutoConfig.from_pretrained(
            self.checkpoint, num_labels=kwargs["num_classes"]
        )
        self.model = AutoModelForAudioClassification.from_config(config)


class Wav2vec2SequenceClassifierFreezeExtractor(Wav2vec2SequenceClassifier):

    def __init__(self, *args, **kwargs):
        super(Wav2vec2SequenceClassifierFreezeExtractor, self).__init__(*args, **kwargs)
        self.model.freeze_feature_extractor()


class Wav2vec2SequenceClassifierFreezeBase(Wav2vec2SequenceClassifier):

    def __init__(self, *args, **kwargs):
        super(Wav2vec2SequenceClassifierFreezeBase, self).__init__(*args, **kwargs)
        self.model.freeze_base_model()
