from typing import List, Literal, Tuple

import torch
import torch.nn as nn
import torchvision

ResNetVersion = Literal["resnet18", "resnet34", "resnet50", "resnet101", "resnet152"]

class ResNetClassifier(nn.Module):
    """
    A ResNet classifier for image classification tasks.

    Attributes
    ----------
    baseline_architecture : ResNetVersion
        The version of ResNet architecture to use (e.g., ResNet18, ResNet34, ResNet50, etc.).
    num_classes : int
        The number of classes in the classification task.
    num_channels : int, optional
        The number of input channels in the images, by default 1.
    pretrained : bool, optional
        Whether to use a pretrained model, by default False.

    Methods
    -------
    forward(x):
        Performs a forward pass through the network.
    """

    def __init__(
            self,
            baseline_architecture: ResNetVersion,
            num_classes: int,
            num_channels: int = 1,
            pretrained: bool = False,
            **kwargs):
        """
        Constructs all the necessary attributes for the ResNetClassifier object.

        Parameters
        ----------
            baseline_architecture : ResNetVersion
                The version of ResNet architecture to use (e.g., ResNet18, ResNet34, ResNet50, etc.).
            num_classes : int
                The number of classes in the classification task.
            num_channels : int, optional
                The number of input channels in the images, by default 1.
            pretrained : bool, optional
                Whether to use a pretrained model, by default False.
        """
        super(ResNetClassifier, self).__init__()
        self.baseline_architecture = baseline_architecture
        self.num_classes = num_classes
        self.num_channels = num_channels

        # Available resnet versions
        resnet_versions = {
            "resnet18": torchvision.models.resnet18,
            "resnet34": torchvision.models.resnet34,
            "resnet50": torchvision.models.resnet50,
            "resnet101": torchvision.models.resnet101,
            "resnet152": torchvision.models.resnet152,
        }

        # Using the resnet backbone
        # TODO: Customize this to read pre-trained weights from a specified directory
        if pretrained:
            resnet_weights = {
                "resnet18": torchvision.models.ResNet18_Weights.DEFAULT,
                "resnet34": torchvision.models.ResNet34_Weights.DEFAULT,
                "resnet50": torchvision.models.ResNet50_Weights.DEFAULT,
                "resnet101": torchvision.models.ResNet101_Weights.DEFAULT,
                "resnet152": torchvision.models.ResNet152_Weights.DEFAULT,
            }
            weights = resnet_weights[baseline_architecture]
        else:
            weights = None

        resnet_model = resnet_versions[baseline_architecture](weights=weights)

        # Replace the old FC layer with Identity, so we can train our own
        linear_size = list(resnet_model.children())[-1].in_features
        # Replace the final layer for fine-tuning (classification into num_classes classes)
        resnet_model.fc = nn.Linear(linear_size, num_classes)

        # Manually set the number of channels in the first Conv layer according to the shape of our input
        resnet_model.conv1 = nn.Conv2d(
            num_channels, 64, kernel_size=7, stride=2, padding=3, bias=False
        )
        resnet_model.bn1 = nn.BatchNorm2d(64)

        self.model = resnet_model
    def forward(self, input_values: torch.Tensor, **kwargs):
        """
        Performs a forward pass through the network.

        Parameters
        ----------
        input_values : torch.Tensor
            The input tensor to the network.
        kwargs : dict, optional
            Additional keyword arguments are not used.

        Returns
        -------
        torch.Tensor
            The output of the network.
        """
        return self.model(input_values)

    @torch.inference_mode()
    def get_logits(self, dataloader, device):
        pass

    @torch.inference_mode()
    def get_probas(self, dataloader, device):
        pass

    @torch.inference_mode()
    def get_representations(self, dataloader, device):
        pass
    