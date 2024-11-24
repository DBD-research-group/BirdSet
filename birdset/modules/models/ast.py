import torch
import torch.nn as nn
import datasets
from transformers import ASTConfig, ASTForAudioClassification
from birdset.utils import pylogger
from birdset.configs import PretrainInfoConfig

log = pylogger.get_pylogger(__name__)


class ASTSequenceClassifier(nn.Module):
    def __init__(
        self,
        checkpoint: str,
        num_classes: int = None,
        local_checkpoint: str = None,
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
        super(ASTSequenceClassifier, self).__init__()

        self.checkpoint = checkpoint
        if pretrain_info:  # either num_classes if provided or pretrain info
            self.hf_path = pretrain_info.hf_path
            self.hf_name = (
                pretrain_info.hf_name
                if not pretrain_info.hf_pretrain_name
                else pretrain_info.hf_pretrain_name
            )
            self.num_classes = len(
                datasets.load_dataset_builder(self.hf_path, self.hf_name)
                .info.features["ebird_code"]
                .names
            )
        else:
            self.num_classes = num_classes

        self.cache_dir = cache_dir

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

        # Squeeze the channel dimension so that the tensor has shape (height, width)
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
            output = (logits, cls_state)

        else:
            output = logits

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
