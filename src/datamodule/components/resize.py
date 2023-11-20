from torch import Tensor
import torch.nn.functional as F

class Resizer:
    def __init__(self,
                 use_spectrogram: bool = False,
                 db_scale: bool = False,) -> None:
        """
        Initializes the Resizer object.

        Args:
            use_spectrogram (bool): If True, the resizer will work with 3D spectrograms of shape (channels, height, width). If False, it will work with waveforms.
            db_scale (bool): Flag indicating whether spectrograms in decibel (dB) units are used. Only required if
            use_spectrogram=True.
        """
        self.use_spectrogram = use_spectrogram

        if db_scale:
            self.padding_value = -80.
        else:
            self.padding_value = 0.

    def pad_spectrogram_height(self, spectrogram: Tensor, target_height: int) -> Tensor:
        """
        Pads the height of a 3D spectrogram to a given target height with the specified padding value.

        Args:
            spectrogram (Tensor): The input 3D spectrogram to be padded.
            target_height (int): The desired height for the 3D spectrogram.

        Returns:
            Tensor: The padded 3D spectrogram.
        """
        difference = target_height - spectrogram.shape[1]
        if difference > 0:
            padding = (0, 0, 0, difference)
            return F.pad(spectrogram, padding, value=self.padding_value)
        return spectrogram

    def pad_spectrogram_width(self, spectrogram: Tensor, target_width: int) -> Tensor:
        """
        Pads the width of a 3D spectrogram to a given target width with the specified padding value.

        Args:
            spectrogram (Tensor): The input 3D spectrogram to be padded.
            target_width (int): The desired width for the 3D spectrogram.

        Returns:
            Tensor: The padded 3D spectrogram.
        """
        difference = target_width - spectrogram.shape[2]
        if difference > 0:
            padding = (0, difference, 0, 0)
            return F.pad(spectrogram, padding, value=self.padding_value)
        return spectrogram

    @staticmethod
    def truncate_spectrogram_height(spectrogram: Tensor, target_height: int) -> Tensor:
        """
        Truncates the height of a 3D spectrogram to a given target height.

        Args:
            spectrogram (Tensor): The input 3D spectrogram to be truncated.
            target_height (int): The desired height for the truncated 3D spectrogram.

        Returns:
            Tensor: The truncated 3D spectrogram.
        """
        return spectrogram[:, :target_height, :]

    @staticmethod
    def truncate_spectrogram_width(spectrogram: Tensor, target_width: int) -> Tensor:
        """
        Truncates the width of a 3D spectrogram to a given target width.

        Args:
            spectrogram (Tensor): The input 3D spectrogram to be truncated.
            target_width (int): The desired width for the truncated 3D spectrogram.

        Returns:
            Tensor: The truncated 3D spectrogram.
        """
        return spectrogram[:, :, :target_width]

    def resize_spectrogram(self, spectrogram: Tensor, target_height: int = None, target_width: int = None) -> Tensor:
        """
        Resizes a 3D spectrogram to a given maximum height and width by either padding or truncating.

        Args:
            spectrogram (Tensor): The input 3D spectrogram to be resized.
            target_height (int, optional): The maximum height for the 3D spectrogram. If None, the height will not be changed.
            target_width (int, optional): The maximum width for the 3D spectrogram. If None, the width will not be changed.

        Returns:
            Tensor: The resized 3D spectrogram.
        """
        if target_height:
            if spectrogram.shape[1] < target_height:
                spectrogram = self.pad_spectrogram_height(spectrogram, target_height)
            elif spectrogram.shape[1] > target_height:
                spectrogram = self.truncate_spectrogram_height(spectrogram, target_height)

        if target_width:
            if spectrogram.shape[2] < target_width:
                spectrogram = self.pad_spectrogram_width(spectrogram, target_width)
            elif spectrogram.shape[2] > target_width:
                spectrogram = self.truncate_spectrogram_width(spectrogram, target_width)

        return spectrogram

    def resize_waveform(self, waveform: Tensor) -> Tensor:
        """
        Resizes a waveform. Currently, this method is not implemented.

        Args:
            waveform (Tensor): The input waveform to be resized.

        Raises:
            NotImplementedError: This method is not implemented yet.
        """

        #raise NotImplementedError("Resizing for waveforms is not implemented yet.")

        #TODO: Implement resizing for waveforms
        return waveform

    def resize(self, data: Tensor, target_height: int = None, target_width: int = None) -> Tensor:
        """
        Resizes data (either 3D spectrogram or waveform) based on the mode set during initialization.

        Args:
            data (Tensor): The input data to be resized.
            target_height (int, optional): The maximum height for the data. If None, the height will not be changed.
            target_width (int, optional): The maximum width for the data. If None, the width will not be changed.

        Returns:
            Tensor: The resized data.
        """
        if self.use_spectrogram:
            return self.resize_spectrogram(data, target_height, target_width)
        else:
            return self.resize_waveform(data)
