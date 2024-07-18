import torch

from typing import Dict, Optional, Tuple
import warnings

import datasets
import pandas as pd
import tensorflow as tf
import tensorflow_hub as hub
import torch
from torch import nn
from ...utils.label_utils import get_label_to_class_mapping_from_metadata

from .embedding_abstract import EmbeddingModel

class LTU(nn.Module):
    """
    
    Attributes:
        PERCH_TF_HUB_URL (str): URL to the TensorFlow Hub model for bird vocalization.
        EMBEDDING_SIZE (int): The size of the embeddings produced by the TensorFlow Hub model.
        num_classes (int): The number of classes to classify into.
        tfhub_version (str): The version of the TensorFlow Hub model to use.
        train_classifier (bool): Whether to train a classifier on top of the embeddings.
        restrict_logits (bool): Whether to restrict output logits to target classes only.
        dataset_info_path (Optional[str]): Path to the dataset information file for target class filtering.
        task (Optional[str]): The type of classification task ('multiclass' or 'multilabel').
        model: The loaded TensorFlow Hub model (loaded dynamically).
        class_mapping (Optional[np.ndarray]): Classes from the TensorFlow Hub model.
        classifier (Optional[nn.Linear]): A linear classifier layer on top of the embeddings.
    """

    # Constants for the model URL and embedding size


    def __init__(
        self,
        num_classes: int,
        tfhub_version: str,
        train_classifier: bool = False,
        restrict_logits: bool = False,
        label_path: Optional[str] = None,
        pretrain_info: Optional[Dict] = None,
        task: Optional[str] = None,
        gpu_to_use: int = 0,
    ) -> None:
        """
        Initializes the PerchModel with configuration for loading the TensorFlow Hub model,
        an optional classifier, and setup for target class restriction based on dataset info.

        Args:
            num_classes: The number of output classes for the classifier.
            tfhub_version: The version identifier of the TensorFlow Hub model to load.
            label_path: Path to a CSV file containing the class information for the Perch model.
            train_classifier: If True, a classifier is added on top of the model embeddings.
            restrict_logits: If True, output logits are restricted to target classes based on dataset info.
            dataset_info_path: Optional path to a JSON file containing target class information.
            task: The classification task type ('multiclass' or 'multilabel'), used with `dataset_info_path`.
        """
        super().__init__()
        self.model = None  # Placeholder for the loaded model
        self.class_mapping = None

        self.train_classifier = train_classifier 
        self.restrict_logits = restrict_logits
        self.label_path = label_path
        self.task = task
        self.target_indices = None

        self.num_classes = num_classes
        self.tfhub_version = tfhub_version
        self.hf_path = pretrain_info.hf_path
        self.hf_name = pretrain_info.hf_name if not pretrain_info.hf_pretrain_name else pretrain_info.hf_pretrain_name
        """if pretrain_info:
            self.hf_path = pretrain_info["hf_path"]
            self.hf_name = pretrain_info["hf_name"]
        else:
            self.hf_path = None
            self.hf_name = None"""
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
            nn.Linear(64, self.num_classes)           
        )

        self.load_model()

    def load_model(self) -> None:
        """
        Load the model from a .pth checkpoint file.
        """

        self.load_state_dict(torch.load("LTU.pth")) #loads weights on GPU by default
    
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
        # If there's an extra channel dimension, remove it
        if input_values.dim() > 2:
            input_values = input_values.squeeze(1)

        device = input_values.device  # Get the device of the input tensor

        # Move the tensor to the CPU and convert it to a NumPy array.
        input_values = input_values.cpu().numpy()

        # Get embeddings from the Perch model and move to the same device as input_values
        embeddings, logits = self.get_embeddings()

        if self.train_classifier:
            embeddings = embeddings.to(device)
            # Pass embeddings through the classifier to get the final output
            output = self.classifier(embeddings)
        else:
            output = logits.to(device)

        return output

    def get_embeddings(
        self
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Get the embeddings and logits from the AudioMAE model.

        Args:
            input_tensor (tf.Tensor): The input tensor for the model.

        Returns:
            Tuple[torch.Tensor, torch.Tensor]: A tuple of two tensors (embeddings, logits).
        """

        # Normalize the input tensor
        input_tensor = self.normalize_audio(input_tensor)

        input_tensor = input_tensor.reshape([-1, input_tensor.shape[-1]])

        # Run the model and get the outputs using the optimized TensorFlow function

        # Extract embeddings and logits, convert them to PyTorch tensors
        embeddings = torch.nn.Embedding
        logits = torch.logit

        if self.restrict_logits:
            logits = logits[:, self.target_indices]

        return embeddings, logits