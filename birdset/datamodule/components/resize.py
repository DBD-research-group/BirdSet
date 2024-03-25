from torch import Tensor
import torch.nn.functional as F

class Resizer:
    def __init__(self,
                 db_scale: bool = False,
                 target_height: int | None = None,
                 target_width: int = 1024 ) -> None:
        """
        Initializes the Resizer object.

        Args:
            db_scale (bool): Flag indicating whether spectrograms in decibel (dB) units are used. Only required if
            use_spectrogram=True.
        """
        self.target_height = target_height
        self.target_width = target_width

        if db_scale:
            self.padding_value = -80.
        else:
            self.padding_value = 0.

    def pad_spectrogram_height(self, spectrogram: Tensor) -> Tensor:
        """
        Pads the height of a 3D spectrogram to a given target height with the specified padding value.

        Args:
            spectrogram (Tensor): The input 3D spectrogram to be padded.
            target_height (int): The desired height for the 3D spectrogram.

        Returns:
            Tensor: The padded 3D spectrogram.
        """
        if self.target_height is not None:
            difference = self.target_height - spectrogram.shape[2]
            padding = (0, 0, 0, difference)
            return F.pad(spectrogram, padding, value=self.padding_value)
        return spectrogram

    def pad_spectrogram_width(self, spectrogram: Tensor) -> Tensor:
        """
        Pads the width of a 3D spectrogram to a given target width with the specified padding value.

        Args:
            spectrogram (Tensor): The input 3D spectrogram to be padded.
            target_width (int): The desired width for the 3D spectrogram.

        Returns:
            Tensor: The padded 3D spectrogram.
        """
        difference = self.target_width - spectrogram.shape[3]
        if difference > 0:
            padding = (0, difference, 0, 0)
            return F.pad(spectrogram, padding, value=self.padding_value)
        return spectrogram

    def truncate_spectrogram_height(self,spectrogram: Tensor) -> Tensor:
        """
        Truncates the height of a 3D spectrogram to a given target height.

        Args:
            spectrogram (Tensor): The input 3D spectrogram to be truncated.
            target_height (int): The desired height for the truncated 3D spectrogram.

        Returns:
            Tensor: The truncated 3D spectrogram.
        """
        return spectrogram[:, :, :self.target_height, :]

    def truncate_spectrogram_width(self,spectrogram: Tensor) -> Tensor:
        """
        Truncates the width of a 3D spectrogram to a given target width.

        Args:
            spectrogram (Tensor): The input 3D spectrogram to be truncated.
            target_width (int): The desired width for the truncated 3D spectrogram.

        Returns:
            Tensor: The truncated 3D spectrogram.
        """
        return spectrogram[:, :, :, :self.target_width]
    
    def resize_spectrogram_batch(self, spectrogram: Tensor) -> Tensor:
        """
        Resizes a 3D spectrogram to a given maximum height and width by either padding or truncating.

        Args:
            spectrogram (Tensor): The input 3D spectrogram to be resized.
            target_height (int, optional): The maximum height for the 3D spectrogram. If None, the height will not be changed.
            target_width (int, optional): The maximum width for the 3D spectrogram. If None, the width will not be changed.

        Returns:
            Tensor: The resized 3D spectrogram.
        """
        if self.target_height:
            if spectrogram.shape[2] < self.target_height:
                spectrogram = self.pad_spectrogram_height(spectrogram)
            elif spectrogram.shape[2] > self.target_height:
                spectrogram = self.truncate_spectrogram_height(spectrogram)

        if self.target_width:
            if spectrogram.shape[3] < self.target_width:
                spectrogram = self.pad_spectrogram_width(spectrogram)
            elif spectrogram.shape[3] > self.target_width:
                spectrogram = self.truncate_spectrogram_width(spectrogram)

        return spectrogram