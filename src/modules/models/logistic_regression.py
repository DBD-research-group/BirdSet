
import torch.nn as nn

class DeterministicLogisticRegression(nn.Module):
    def __init__(self, embedding_size, num_classes , **kwargs) -> None:
        super().__init__(**kwargs)
        self.linear = nn.Linear(in_features=embedding_size, out_features=num_classes)
    
    def forward(self, input_values, **kwargs):
        logits = self.linear(input_values)
        return logits
        