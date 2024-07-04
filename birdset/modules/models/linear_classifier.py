import torch
from torch import nn
from typing import Optional

class LinearClassifier(nn.Module):
    
    def __init__(
        self,
        num_classes: int,
        in_features: int   
    ) -> None:
        """
        Initialize the LinearClassifier.
        
        Args:
            num_classes (int): The number of output classes for the classifier.
            in_features (int): The size of the input for the linear classifier.
        """
        super().__init__()
        self.classifier = nn.Linear(in_features, num_classes)
        '''self.classifier = nn.Sequential(
            nn.Linear(in_features, 128),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Linear(64, num_classes),
        )'''
        
    
    def forward(
        self, input_values: torch.Tensor, labels: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        return self.classifier(input_values) 