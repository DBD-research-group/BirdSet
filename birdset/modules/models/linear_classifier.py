import torch
from torch import nn
from typing import Optional

class LinearClassifier(nn.Module):
    
    def __init__(
        self,
        num_classes: int,
        in_features: int,
        state_dict: Optional[dict] = None,   
        first = True
    ) -> None:
        """
        Initialize the LinearClassifier.
        
        Args:
            num_classes (int): The number of output classes for the classifier.
            in_features (int): The size of the input for the linear classifier.
        """
        super().__init__()
        self.classifier = nn.Linear(in_features, num_classes)
        self.first = first
        if state_dict is not None:
            print("LOADED")
            print("Current model parameter names:", self.state_dict().keys())
            state_dict = torch.load(state_dict)["state_dict"]
            #print(state_dict)
            state_dict = {key: weight for key, weight in state_dict.items() if key.startswith('model.')}
            state_dict = {key.replace('model.', ''): weight for key, weight in state_dict.items()}
            print("Modified state_dict keys:", state_dict.keys())
            print(state_dict)
            self.load_state_dict(state_dict,strict=True)
         

    def forward(
        self, input_values: torch.Tensor, labels: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        if self.first:
            print("First")
            print("Current model parameter names:", self.state_dict().keys())
            print(self.state_dict())
            self.first = False
        return self.classifier(input_values).squeeze(1) 