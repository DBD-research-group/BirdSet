from typing import Optional


from birdset.modules.models.BEATs import BEATs, BEATsConfig
import torch
from torch import nn
from typing import Tuple


class BEATsModel(nn.Module):
    """
    Pretrained model for audio classification using the BEATs model.
    """

    EMBEDDING_SIZE = 768

    def __init__(
        self,
        num_classes: int,
        local_checkpoint: str = None,
        train_classifier: bool = False,
    ) -> None:
        super().__init__()
        self.model = None  # Placeholder for the loaded model
        self.load_model()

        if local_checkpoint:
            state_dict = torch.load(local_checkpoint)["state_dict"]
            state_dict = {
                key.replace("model.model.", ""): weight
                for key, weight in state_dict.items()
            }
            self.model.load_state_dict(state_dict)

        self.num_classes = num_classes
        self.train_classifier = train_classifier
        # Define a linear classifier to use on top of the embeddings
        # self.classifier = nn.Linear(
        #     in_features=self.EMBEDDING_SIZE, out_features=num_classes
        # )
        if self.train_classifier:
            self.classifier = nn.Sequential(
                nn.Linear(self.EMBEDDING_SIZE, 128),
                nn.ReLU(),
                nn.Dropout(0.5),
                nn.Linear(128, 64),
                nn.ReLU(),
                nn.Linear(64, self.num_classes),
            )
            # freeze the model
            for param in self.model.parameters():
                param.requires_grad = False

    def load_model(self) -> None:
        """
        Load the model from shared storage.
        """
        # load the pre-trained checkpoints
        checkpoint = torch.load("/workspace/models/beats/BEATs_iter3_plus_AS2M.pt")

        cfg = BEATsConfig(checkpoint["cfg"])
        self.model = BEATs(cfg)
        self.model.load_state_dict(checkpoint["model"])
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
        embeddings = self.get_embeddings(input_values)[0]
        if self.train_classifier:
            flattend_embeddings = embeddings.reshape(embeddings.size(0), -1)
            # Pass embeddings through the classifier to get the final output
            output = self.classifier(flattend_embeddings)
        else:
            output = embeddings

        return output

    def get_embeddings(
        self, input_values: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Get the embeddings and logits from the BEATs model.

        Args:
            input_tensor (torch.Tensor): The input tensor for the model.

        Returns:
            torch.Tensor: The embeddings from the model.
        """
        embeddings = self.model.extract_features(input_values)[
            0
        ]  # outputs a tensor of size 496x768
        cls_state = embeddings[:, 0, :]

        return cls_state, None
