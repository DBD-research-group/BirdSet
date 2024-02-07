import torch
import torch.nn as nn
from transformers import AutoModelForAudioClassification


class Wav2vec2SequenceClassifier(nn.Module):
    def __init__(self, checkpoint, num_classes, cache_dir):
        super(Wav2vec2SequenceClassifier, self).__init__()

        self.checkpoint = checkpoint
        self.num_classes = num_classes
        self.cache_dir = cache_dir
        
        self.model = AutoModelForAudioClassification.from_pretrained(
            self.checkpoint,
            num_labels=self.num_classes,
            cache_dir=self.cache_dir
        )

    def forward(self, input_values, attention_mask=None, labels=None, return_hidden_state=False):
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
            labels=labels
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
    



