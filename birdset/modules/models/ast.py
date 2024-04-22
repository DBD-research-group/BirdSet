import torch
import torch.nn as nn
import datasets
from transformers import AutoModelForAudioClassification, AutoConfig
from transformers import ASTForAudioClassification
from birdset.utils import pylogger

log = pylogger.get_pylogger(__name__)

class ASTSequenceClassifier(nn.Module):
    def __init__(self, checkpoint: str, local_checkpoint: str | None, num_classes: int, cache_dir: str | None, pretrain_info):
        super(ASTSequenceClassifier, self).__init__()

        self.checkpoint = checkpoint
        # self.num_classes = num_classes

        self.hf_path = pretrain_info.hf_path
        self.hf_name = pretrain_info.hf_name if not pretrain_info.hf_pretrain_name else pretrain_info.hf_pretrain_name
        self.num_classes = len(
            datasets.load_dataset_builder(self.hf_path, self.hf_name).info.features["ebird_code"].names)

        self.cache_dir = cache_dir

        state_dict = None

        if self.checkpoint:
            if local_checkpoint:
                log.info(f">> Loading state dict from local checkpoint: {local_checkpoint}")
                state_dict = torch.load(local_checkpoint)["state_dict"]
                state_dict = {key.replace('model.model.', ''): weight for key, weight in state_dict.items()}

            self.model = AutoModelForAudioClassification.from_pretrained(
                self.checkpoint,
                num_labels=self.num_classes,
                cache_dir=self.cache_dir,
                state_dict=state_dict,
                ignore_mismatched_sizes=True
            )
        else:
            config = AutoConfig.from_pretrained(self.checkpoint, num_labels=self.num_classes)
            self.model = ASTForAudioClassification(config)
        
    def forward(self, input_values, attention_mask=None, labels=None, return_hidden_state=False):
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
        #6,1,128,1024
        input_values = input_values.transpose(1, 2)

        outputs = self.model(
            input_values, 
            attention_mask,
            output_attentions=False,
            output_hidden_states=True,
            return_dict=True,
            labels=None
        )

        logits = outputs["logits"]

        last_hidden_state = outputs["hidden_states"][-1] #(batch, sequence, dim)
        cls_state = last_hidden_state[:,0,:] #(batch, dim)

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

        config = AutoConfig.from_pretrained(self.checkpoint, num_labels=kwargs["num_classes"])
        self.model = AutoModelForAudioClassification.from_config(config)
