import logging
from typing import Optional, Tuple

import datasets
import pandas as pd
import tensorflow as tf
import tensorflow_hub as hub
import torch
from torch import nn

from birdset.configs import PretrainInfoConfig

class PerchModel(nn.Module):
    """
    A PyTorch model for bird vocalization classification, integrating a TensorFlow Hub model.

    Attributes:
        PERCH_TF_HUB_URL (str): URL to the TensorFlow Hub model for bird vocalization.
        EMBEDDING_SIZE (int): The size of the embeddings produced by the TensorFlow Hub model.
        num_classes (int): The number of classes to classify into.
        tfhub_version (str): The version of the TensorFlow Hub model to use.
        train_classifier (bool): Whether to train a classifier on top of the embeddings.
        restrict_logits (bool): Whether to restrict output logits to target classes only.
        dataset_info_path (Optional[str]): Path to the dataset information file for target class filtering.
        model: The loaded TensorFlow Hub model (loaded dynamically).
        classifier (Optional[nn.Linear]): A linear classifier layer on top of the embeddings.
    """

    # Constants for the model URL and embedding size
    PERCH_TF_HUB_URL = "https://tfhub.dev/google/bird-vocalization-classifier"
    EMBEDDING_SIZE = 1280

    def __init__(
        self,
        num_classes: int,
        tfhub_version: str,
        train_classifier: bool = False,
        restrict_logits: bool = False,
        label_path: Optional[str] = None,
        pretrain_info: Optional[PretrainInfoConfig] = None,
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
            pretrain_info: A dictionary containing information about the pretraining of the model.
            gpu_to_use: The GPU index to use for the model.
        """
        super().__init__()
        self.model = None  # Placeholder for the loaded model
        self.class_mask = None
        self.class_indices = None

        self.num_classes = num_classes
        self.tfhub_version = tfhub_version
        self.train_classifier = train_classifier
        self.restrict_logits = restrict_logits
        self.label_path = label_path
        self.gpu_to_use = gpu_to_use

        if pretrain_info:
            self.hf_path = pretrain_info.hf_path
            self.hf_name = pretrain_info.hf_name
        else:
            self.hf_path = None
            self.hf_name = None

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
        self.load_model()

    def load_model(self) -> None:
        """
        Load the model from TensorFlow Hub.
        """

        model_url = f"{self.PERCH_TF_HUB_URL}/{self.tfhub_version}"
        #self.model = hub.load(model_url)
        #with tf.device('/CPU:0'):
            #self.model = hub.load(model_url)
        physical_devices = tf.config.list_physical_devices('GPU')
        tf.config.experimental.set_visible_devices(physical_devices[self.gpu_to_use], 'GPU')
        tf.config.experimental.set_memory_growth(physical_devices[self.gpu_to_use], True)

        tf.config.optimizer.set_jit(True)
        self.model = hub.load(model_url)

        if self.restrict_logits:
            # Load the class list from the CSV file
            pretrain_classlabels = pd.read_csv(self.label_path)
            # Extract the 'ebird2021' column as a list
            pretrain_classlabels = pretrain_classlabels["ebird2021"].tolist()

            # Load dataset information
            dataset_info = datasets.load_dataset_builder(
                self.hf_path, self.hf_name
            ).info
            dataset_classlabels = dataset_info.features["ebird_code"].names

            # Create the class mask
            self.class_mask = [
                pretrain_classlabels.index(label)
                for label in dataset_classlabels
                if label in pretrain_classlabels
            ]
            self.class_indices = [
                i
                for i, label in enumerate(dataset_classlabels)
                if label in pretrain_classlabels
            ]

            # Log missing labels
            missing_labels = [
                label
                for label in dataset_classlabels
                if label not in pretrain_classlabels
            ]
            if missing_labels:
                logging.warning(f"Missing labels in pretrained model: {missing_labels}")

    @tf.function  # Decorate with tf.function to compile into a callable TensorFlow graph
    def run_tf_model(self, input_tensor: tf.Tensor) -> dict:
        """
        Run the TensorFlow model and get outputs.

        Args:
            input_tensor (tf.Tensor): The input tensor for the model.

        Returns:
            dict: A dictionary of model outputs.
        """

        return self.model.signatures["serving_default"](inputs=input_tensor)

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
        #input_values = input_values.cpu().numpy()

        # Get embeddings from the Perch model and move to the same device as input_values
        embeddings, logits = self.get_embeddings(input_tensor=input_values)

        if self.train_classifier:
            embeddings = embeddings.to(device)
            # Pass embeddings through the classifier to get the final output
            output = self.classifier(embeddings)
        else:
            output = logits.to(device)

        return output

    def get_embeddings(
        self, input_tensor: tf.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Get the embeddings and logits from the Perch model.

        Args:
            input_tensor (tf.Tensor): The input tensor for the model.

        Returns:
            Tuple[torch.Tensor, torch.Tensor]: A tuple of two tensors (embeddings, logits).
        """
        device = input_tensor.device # Get the device of the input tensor 
        input_tensor = input_tensor.cpu().numpy()  # Move the tensor to the CPU and convert it to a NumPy array.

        input_tensor = input_tensor.reshape([-1, input_tensor.shape[-1]])

        # Run the model and get the outputs using the optimized TensorFlow function
        outputs = self.run_tf_model(input_tensor=input_tensor)

        # Extract embeddings and logits, convert them to PyTorch tensors
        embeddings = torch.from_numpy(outputs["output_1"].numpy())
        logits = torch.from_numpy(outputs["output_0"].numpy())
        embeddings = embeddings.to(device)
        logits = logits.to(device)
        
        if self.class_mask:
            # Initialize full_logits to a large negative value for penalizing non-present classes
            full_logits = torch.full(
                (logits.shape[0], self.num_classes),
                -10.0,
                device=logits.device,
                dtype=logits.dtype,
            )
            # Extract valid logits using indices from class_mask and directly place them
            full_logits[:, self.class_indices] = logits[:, self.class_mask]
            logits = full_logits

        return embeddings, logits
