import torch
import torch.nn as nn
from transformers import AutoModelForAudioClassification


class ResNetClassifier(nn.Module):
    def __init__(self, model, weights, num_classes):
        super(ResNetClassifier, self).__init__()

        self.num_classes = num_classes
        self.model = model(weights=weights)

    def forward(self, input_values, attention_mask=None, labels=None, return_hidden_state=False):

        prediction = self.model(input_values)

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
    