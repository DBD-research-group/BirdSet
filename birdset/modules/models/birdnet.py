from typing import Optional, Tuple

import torch
import tensorflow as tf
from torch import nn


class BirdNetTFLiteModel(nn.Module):
    EMBEDDING_SIZE = 1024  # Assuming the embedding size for BirdNet TFLite is 1024

    def __init__(
        self, model_path: str, num_classes: int, train_classifier: bool, num_threads: int = None
    ) -> None:
        """
        Initialize the BirdNetTFLiteModel.

        Args:
            model_path (str): The file path to the TensorFlow Lite model.
            num_classes (int): The number of output classes for the classifier.
            num_threads (int, optional): The number of threads to use for TFLite model. Defaults to None.
        """
        super().__init__()
        self.interpreter = tf.lite.Interpreter(
            model_path=model_path, num_threads=num_threads
        )
        self.interpreter.allocate_tensors()
        self.train_classifier = train_classifier
        # Get input and output details
        self.input_details = self.interpreter.get_input_details()
        self.output_details = self.interpreter.get_output_details()

        # Determine indices for embeddings and logits
        self.logits_idx = self.output_details[-1]["index"]
        self.embeddings_idx = (
            self.logits_idx - 1
        )  # Assuming embeddings are just before logits

        # Define a linear classifier to use on top of the embeddings
        self.classifier = nn.Linear(
            in_features=self.EMBEDDING_SIZE, out_features=num_classes
        )

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

        # Get embeddings and logits from the TFLite model and move to the same device as input_values
        embeddings, logits = self.get_embeddings(input_values)
        
        # Pass embeddings through the classifier to get the final output if training the classifier
        if self.train_classifier:
            embeddings = embeddings.to(device)
            output = self.classifier(embeddings)
        else:
            output = logits.to(device)    
        
        return output

    def get_embeddings(
        self, input_tensor: tf.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Get predictions from the BirdNet TFLite model including both embeddings and logits.

        Args:
            input_tensor (torch.Tensor): The input waveform batch as a PyTorch tensor.

        Returns:
            Tuple[torch.Tensor, torch.Tensor]: A tuple containing two PyTorch tensors, (embeddings, logits).
        """
        # Ensure it's the correct shape for the TFLite model
        input_shape = self.input_details[0]["shape"]
        batch_size = input_tensor.shape[0]

        # Resize input tensor if necessary
        if input_shape[0] != batch_size:
            input_shape[0] = batch_size
            self.interpreter.resize_tensor_input(
                self.input_details[0]["index"], input_shape
            )
            self.interpreter.allocate_tensors()

        # Set the input tensor and run inference
        self.interpreter.set_tensor(self.input_details[0]["index"], input_tensor)
        self.interpreter.invoke()

        # Get the output tensors
        embeddings = torch.from_numpy(self.interpreter.get_tensor(self.embeddings_idx))
        logits = torch.from_numpy(self.interpreter.get_tensor(self.logits_idx))

        return embeddings, logits


class BirdNetModel(nn.Module):
    # Constants for the model embedding size
    EMBEDDING_SIZE = 1024

    def __init__(self, num_classes: int, model_path: str, train_classifier: bool) -> None:
        """
        Initialize the BirdNetModel.

        Args:
            num_classes (int): The number of output classes for the classifier.
            model_path (str): The path to the TensorFlow BirdNet model/checkpoint.
        """
        super().__init__()
        self.birdnet_model = tf.saved_model.load(model_path)  # Load the BirdNet model
        self.num_classes = num_classes
        self.train_classifier = train_classifier
        # Define a linear classifier to use on top of the embeddings
        self.classifier = nn.Linear(
            in_features=self.EMBEDDING_SIZE, out_features=num_classes
        )

    @tf.function  # Decorate with tf.function
    def run_tf_model(self, input_tensor: tf.Tensor) -> dict:
        """
        Run the TensorFlow BirdNet model and get outputs.

        Args:
            input_tensor (tf.Tensor): The input tensor for the BirdNet model in TensorFlow format.

        Returns:
            dict: A dictionary containing 'embeddings' and 'logits' TensorFlow tensors.
        """
        print(input_tensor.shape)
        logits = self.birdnet_model.signatures["basic"](inputs=input_tensor)["scores"]
        embeddings = self.birdnet_model.signatures["embeddings"](inputs=input_tensor)[
            "embeddings"
        ]
        return {"embeddings": embeddings, "logits": logits}

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

        # Get embeddings from the BirdNet model and move to the same device as input_values
        embeddings, logits = self.get_embeddings(input_tensor=input_values)

        # Pass embeddings through the classifier to get the final output if training the classifier
        if self.train_classifier:
            embeddings = embeddings.to(device)
            output = self.classifier(embeddings)
        else:
            output = logits.to(device)    
        
        return output

    def get_embeddings(
        self, input_tensor: tf.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Get predictions from BirdNet model including both logits and embeddings.
        ...
        """
        
        input_tensor = input_tensor.reshape([-1, input_tensor.shape[-1]])
        # Run the TensorFlow BirdNet model using the optimized function
        outputs = self.run_tf_model(input_tensor=input_tensor)

        # Convert the TensorFlow tensors to PyTorch tensors.
        embeddings = torch.from_numpy(outputs["embeddings"].numpy())
        logits = torch.from_numpy(outputs["logits"].numpy())

        return embeddings, logits