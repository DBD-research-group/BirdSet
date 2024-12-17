import torch
import torch.nn as nn
from transformers import ASTConfig, ASTForAudioClassification
from birdset.utils import pylogger
from birdset.configs import PretrainInfoConfig

from biofoundation.modules.models.birdset_model import BirdSetModel

log = pylogger.get_pylogger(__name__)


class ASTSequenceClassifier(BirdSetModel):
    EMBEDDING_SIZE = 768

    def __init__(
        self,
        num_classes: int = None,
        embedding_size: int = EMBEDDING_SIZE,
        checkpoint: str = "MIT/ast-finetuned-audioset-10-10-0.4593",
        local_checkpoint: str = None,
        freeze_backbone: bool = False,
        preprocess_in_model: bool = False,  # This isn't implemented for this model (yet?!)
        classifier: nn.Module | None = None,
        cache_dir: str = None,
        pretrain_info: PretrainInfoConfig = None,
    ):

        super().__init__(
            num_classes=num_classes,
            embedding_size=embedding_size,
            local_checkpoint=local_checkpoint,
            freeze_backbone=freeze_backbone,
            preprocess_in_model=preprocess_in_model,
            pretrain_info=pretrain_info,
        )
        self.checkpoint = checkpoint
        self.cache_dir = cache_dir
        self.classifier = classifier

        if (
            local_checkpoint
        ):  # TODO only loads a pretrained model from a local checkpoint else a randomly init model???
            log.info(f">> Loading state dict from local checkpoint: {local_checkpoint}")
            state_dict = torch.load(local_checkpoint)["state_dict"]
            state_dict = {
                key.replace("model.model.", ""): weight
                for key, weight in state_dict.items()
            }

            self.model = ASTForAudioClassification.from_pretrained(
                self.checkpoint,
                num_labels=self.num_classes,
                cache_dir=self.cache_dir,
                state_dict=state_dict,
                ignore_mismatched_sizes=True,
            )
        else:
            print(f"Loading only HF model from {self.checkpoint}")
            self.model = ASTForAudioClassification.from_pretrained(
                self.checkpoint,
                num_labels=self.num_classes,
                cache_dir=self.cache_dir,
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
        format of the model. It first squeezes out the channel dimension, assuming it's of size one, and then
        transposes the height and width dimensions. The processed tensor is then passed through the model to
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

        input_values = input_values.squeeze(1)

        # Swap the height and width dimensions so that the tensor has shape (width, height)
        # 6,1,128,1024
        input_values = input_values.transpose(1, 2)
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

        if return_hidden_state:

            embeddings = (logits, cls_state)

        else:
            embeddings = logits

        if self.train_classifier:
            output = self.classifier(embeddings)
        else:
            output = embeddings

        return output

    def get_embeddings(
        self, input_tensor: torch.Tensor, attention_mask=None, return_hidden_state=False
    ):
        """
        Get the embeddings and logits from the model.

        Args:
            input_tensor (torch.Tensor): The input tensor for the model.

        Returns:
            torch.Tensor: The embeddings from the model.
        """
        # Ensure input tensor has the correct dimensions
        print("shaaaaaaaap", input_tensor.shape)

        input_tensor = input_tensor.squeeze(1)
        print("shaaaaaaaap", input_tensor.shape)
        input_values = input_tensor.transpose(1, 2)  # Swap sequence and feature dims

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

    @torch.inference_mode()
    def get_logits(self, dataloader, device):
        pass

    @torch.inference_mode()
    def get_probas(self, dataloader, device):
        pass

    @torch.inference_mode()
    def get_representations(self, dataloader, device):
        pass


class ASTSequenceClassifierRandomInit(ASTSequenceClassifier):
    def __init__(self, *args, **kwargs):
        super(ASTSequenceClassifierRandomInit, self).__init__(*args, **kwargs)

        config = ASTConfig.from_pretrained(
            self.checkpoint, num_labels=kwargs["num_classes"]
        )
        self.model = ASTForAudioClassification.from_config(config)
