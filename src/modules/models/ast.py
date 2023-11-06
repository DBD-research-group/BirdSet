import torch
import torch.nn as nn
from transformers import AutoModelForAudioClassification


class ASTSequenceClassifier(nn.Module):
    def __init__(self, checkpoint, num_classes):
        super(ASTSequenceClassifier, self).__init__()

        self.checkpoint = checkpoint
        self.num_classes = num_classes
        self.model = AutoModelForAudioClassification.from_pretrained(
            self.checkpoint,
            num_labels=self.num_classes,
            ignore_mismatched_sizes=True
        )
        #TODO! problem type not really used, only bcewithlogits on logits
        
    def forward(self, input_values, attention_mask=None, labels=None, return_hidden_state=False):

        input_values = input_values.transpose(1, 2)

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