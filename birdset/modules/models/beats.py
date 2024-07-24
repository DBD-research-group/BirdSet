from typing import Optional, Tuple

import sys
import os

SCRIPT_DIR = os.path.dirname(os.path.abspath("/workspace/beats"))
sys.path.append(os.path.dirname(SCRIPT_DIR))

from workspace.beats.BEATs import BEATs, BEATsConfig
import torch
from torch import nn
import torch.nn.functional as F
import torchaudio.transforms as T



class BEATsModel(nn.Module):
    """
    Pretrained model for audio classification using the BEATs model.
    The model expects a 1D audio signal sampled with 16kHz and a length of 10s.
    """
    EMBEDDING_SIZE = 768

    def __init__(
            self,
            num_classes: int,
            train_classifier: bool = False,
        ) -> None:
        super().__init__()
        self.model = None  # Placeholder for the loaded model
        self.load_model()
        self.num_classes = num_classes
        self.train_classifier = train_classifier
         # Define a linear classifier to use on top of the embeddings
        # self.classifier = nn.Linear(
        #     in_features=self.EMBEDDING_SIZE, out_features=num_classes
        # )
        self.classifier = nn.Sequential(
            nn.Linear(self.EMBEDDING_SIZE, 128),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Linear(64, self.num_classes),
        )


    def load_model(self) -> None:
        """
        Load the model from Huggingface.
        """
        # load the pre-trained checkpoints
        checkpoint = torch.load('/workspace/BEATs.pt')

        cfg = BEATsConfig(checkpoint['cfg'])
        self.model = BEATs(cfg)
        self.model.load_state_dict(checkpoint['model'])
        self.model.eval()


    
    def forward(
            self, input_values: torch.Tensor, labels: Optional[torch.Tensor] = None
        ) -> torch.Tensor:
            """
            Forward pass through the model.

            Args:
                input_values (torch.Tensor): The input tensor for the classifier.
                labels (Optional[torch.Tensor]): The true labels for the input values. Default is None.

            Returns:
                torch.Tensor: The output of the classifier.
            """
            #print("input_values",input_values.size())
            embeddings = self.get_embeddings(input_values)
            #print("embeddings",embeddings.size())
            if self.train_classifier:
                # Pass embeddings through the classifier to get the final output
                output = self.classifier(embeddings)
            else:
                output = embeddings

            return output

    def get_embeddings(
        self, input_values: torch.Tensor
    ) -> torch.Tensor:
        """
        Get the embeddings and logits from the AUDIOMAE model.

        Args:
            input_tensor (torch.Tensor): The input tensor for the model.

        Returns:
            torch.Tensor: The embeddings from the model.
        """
        embeddings = self.model.extract_features(input_values)[0]
        return embeddings[:,-1,:]