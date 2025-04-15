import glob
import math
import random
import warnings
from pathlib import Path
from typing import List, Text, Union
from torch import nn

# Related third-party imports
import librosa
import numpy as np
import soundfile as sf
import torch.nn.functional as F
import torchaudio
from torchaudio import transforms
import torch_audiomentations
from torch_audiomentations.core.transforms_interface import EmptyPathException
from torch_audiomentations.utils.file import find_audio_files_in_paths

from typing import Optional
import torch
import os
from torch import Tensor
from torch_audiomentations.core.transforms_interface import BaseWaveformTransform
from torch_audiomentations.utils.dsp import calculate_rms
from torch_audiomentations.utils.object_dict import ObjectDict


def pad_spectrogram_width(
    spectrogram: torch.Tensor, target_width: int, value: float
) -> torch.Tensor:
    """
    Pad the input spectrogram along the width axis to reach the desired width.

    Args:
        spectrogram (torch.Tensor): Input spectrogram tensor with shape (C, H, W),
            where C is the number of channels, H is the height, and W is the original width.
        target_width (int): The desired width of the padded spectrogram.
        value (float): The value to fill the padding with.

    Returns:
        torch.Tensor: Padded spectrogram tensor with shape (C, H, target_width).
    """
    spectrogram_width = spectrogram.shape[2]

    # Calculate the amount of padding required on each side
    pad_left = (target_width - spectrogram_width) // 2
    pad_right = target_width - spectrogram_width - pad_left

    # Pad only along the width axis
    # Note: torch.nn.functional.pad pads in the format (padding_left, padding_right, padding_top, padding_bottom)
    padded_spectrogram = torch.nn.functional.pad(
        spectrogram, (pad_left, pad_right, 0, 0), mode="constant", value=value
    )

    return padded_spectrogram


def _choose_original_labels(target, background_target, snr):
    return target


def _make_union_labels(target, background_target, snr):
    return torch.maximum(target, background_target)


class RandomTimeStretch:
    def __init__(
        self,
        prob: float = 0.33,
        n_freq: int = 96,
        min_rate: float = 0.8,
        max_rate: float = 1.25,
        hop_length: Optional[int] = None,
    ):
        """Random Time Stretch (is applied randomly and selects the rate randomly).

        Args:
            prob (float): Probability of the spectrogram being time-stretched. Default value is 0.33.
            n_freq (int): Number of filter banks from stft.
            min_rate (float): Minimum rate to speed up or slow down by.
            max_rate (float): Maximum rate to speed up or slow down by.
            hop_length (Optional[int]): Hop length for time-stretching. Default is None.
        """
        self.prob = prob
        self.n_freq = n_freq
        self.min_rate = min_rate
        self.max_rate = max_rate
        self.hop_length = hop_length
        self.time_stretch = transforms.TimeStretch(
            n_freq=self.n_freq, hop_length=self.hop_length
        )

    def __call__(self, spectrogram: torch.Tensor) -> torch.Tensor:
        """
        Time-stretch the input spectrogram with a random rate.

        Args:
            spectrogram (torch.Tensor): Spectrogram to be time-stretched. A tensor of dimension (…, freq, num_frame)
            with complex dtype.

        Returns:
            torch.Tensor: Time-stretched spectrogram if time stretch is applied; else, the original spectrogram.
        """
        if random.random() < self.prob:
            # Generate a random rate to speed up or slow down by
            rate = np.random.uniform(low=self.min_rate, high=self.max_rate)

            # Apply time-stretching with the generated rate
            spectrogram_time_stretched = self.time_stretch(spectrogram, rate)

            # Convert the complex-valued STFT returned by TimeStretch into a power spectrogram
            spectrogram_time_stretched = torch.abs(
                spectrogram_time_stretched
            )  # Compute the magnitude of the STFT
            spectrogram_time_stretched = spectrogram_time_stretched.pow(
                2
            )  # Compute the power spectrogram

            # Pad the spectrogram along the width axis
            # so that it retains its original width, and the time-stretched audio is in the middle
            original_width = spectrogram.shape[2]
            spectrogram_time_stretched = pad_spectrogram_width(
                spectrogram=spectrogram_time_stretched,
                target_width=original_width,
                value=1e-10,
            )

            return spectrogram_time_stretched

        # If time-stretching is not applied, return the original complex-valued STFT as power spectrogram
        spectrogram = torch.abs(spectrogram)  # Compute the magnitude of the STFT
        spectrogram = spectrogram.pow(2)  # Compute the power spectrogram

        return spectrogram

    def __repr__(self) -> str:
        """
        Return a string representation of the RandomTimeStretch object.

        Returns:
            str: A string representation of the object.
        """
        return f"{self.__class__.__name__}(p={self.prob})"


class AddBackgroundNoiseHf(torch_audiomentations.AddBackgroundNoise):
    def __init__(
        self,
        noise_data,
        min_snr_in_db: float = 3.0,
        max_snr_in_db: float = 30.0,
        p: float = 0.5,
    ):

        self.noise_data = noise_data
        self.min_snr_in_db = min_snr_in_db
        self.max_snr_in_db = max_snr_in_db
        self.p = p

    # def random_background(self, noise_data, target_num_samples):
    #     pieces = []
    #     while missing_num_samples > 0:
    #         background_path = random.choice(self.background_paths)
    #         background_num_samples = audio.get_num_samples(background_path)

    #         if background_num_samples > missing_num_samples:
    #             sample_offset = random.randint(
    #                 0, background_num_samples - missing_num_samples
    #             )
    #             num_samples = missing_num_samples
    #             background_samples = audio(
    #                 background_path, sample_offset=sample_offset, num_samples=num_samples
    #             )
    #             missing_num_samples = 0
    #         else:
    #             background_samples = audio(background_path)
    #             missing_num_samples -= background_num_samples

    #         pieces.append(background_samples)

    #     # the inner call to rms_normalize ensures concatenated pieces share the same RMS (1)
    #     # the outer call to rms_normalize ensures that the resulting background has an RMS of 1
    #     # (this simplifies "apply_transform" logic)
    #     return audio.rms_normalize(
    #         torch.cat([audio.rms_normalize(piece) for piece in pieces], dim=1)
    #     )


class Compose:
    def __init__(self, transforms):
        self.transforms = transforms

    def __call__(self, inputs, *args, **kwargs):
        for transform in self.transforms:
            inputs = transform(inputs, *args, **kwargs)
        return inputs


class AudioTransforms:
    def __init__(self, p=0.5):
        self.p = p

    def __call__(
        self,
        inputs,
    ):
        if np.random.rand() < self.p:
            return self.apply(inputs)

        return inputs


class BackgroundNoise(AudioTransforms):
    def __init__(self, noise_events=None, p=0.5):
        super().__init__(p=p)

        self.noise_events = noise_events
        self.p = p
        self.max_length = 5
        self.min_length = 1

    def _decode(self, path, start, end):
        sr = sf.info(path).samplerate
        if start is not None and end is not None:
            if end - start < self.min_length:
                end = start + self.min_length
            if self.max_length and end - start > self.max_length:
                end = start + self.max_length
            start, end = int(start * sr), int(end * sr)
        if not end:
            end = int(self.max_length * sr)

        audio, _ = sf.read(path, start=start, stop=end)

        if audio.ndim != 1:
            audio = audio.swapaxes(1, 0)
            audio = librosa.to_mono(audio)

        return audio

    def apply(self, inputs):

        augmented_audio = []
        for i, sample in enumerate(inputs):
            random_idx = random.randint(
                0, len(self.noise_events["filepath"]) - 1
            )  # exclude the file itself?
            noise_path = self.noise_events["filepath"][random_idx]
            if self.noise_events["no_call_events"][random_idx]:
                noise_event = random.choice(
                    self.noise_events["no_call_events"][random_idx]
                )
            else:  # can be empty! 16 times in training high sierras (how?)
                noise_event = [0, 0]

            noise = self._decode(
                path=noise_path, start=noise_event[0], end=noise_event[1]
            )

            noise = torch.Tensor(noise).unsqueeze(0)

            if sample.size(1) != noise.size(1):
                padding_length = sample.size(1) - noise.size(1)
                noise = F.pad(noise, (0, padding_length), "constant", 0)

            augmented = torchaudio.functional.add_noise(
                waveform=sample,
                noise=noise,
                snr=torch.Tensor([15]),  # should also work on a batch??
                # expects no_samples x length
            )
            augmented_audio.append(augmented)

        augmented_audio = torch.stack(augmented_audio)

        return augmented_audio


# Mix not officially released yet
class MultilabelMix(BaseWaveformTransform):
    supported_modes = {"per_example", "per_channel"}

    supports_multichannel = True
    requires_sample_rate = False

    supports_target = True
    requires_target = False

    def __init__(
        self,
        min_snr_in_db: float = 0.0,
        max_snr_in_db: float = 5.0,
        mix_target: str = "union",
        max_samples: int = 1,
        mode: str = "per_example",
        p: float = 0.5,
        p_mode: str = None,
        sample_rate: int = None,
        target_rate: int = None,
        output_type: Optional[str] = None,
    ):
        super().__init__(
            mode=mode,
            p=p,
            p_mode=p_mode,
            sample_rate=sample_rate,
            target_rate=target_rate,
            output_type=output_type,
        )
        self.min_snr_in_db = min_snr_in_db
        self.max_snr_in_db = max_snr_in_db
        if self.min_snr_in_db > self.max_snr_in_db:
            raise ValueError("min_snr_in_db must not be greater than max_snr_in_db")

        self.mix_target = mix_target
        if mix_target == "original":
            self._mix_target = _choose_original_labels

        elif mix_target == "union":
            self._mix_target = _make_union_labels
        else:
            raise ValueError("mix_target must be one of 'original' or 'union'.")

        self.max_samples = max_samples

    def randomize_parameters(
        self,
        samples: Tensor = None,
        sample_rate: Optional[int] = None,
        targets: Optional[Tensor] = None,
        target_rate: Optional[int] = None,
    ):

        batch_size, num_channels, num_samples = samples.shape

        snr_distribution = torch.distributions.Uniform(
            low=torch.tensor(
                self.min_snr_in_db,
                dtype=torch.float32,
                device=samples.device,
            ),
            high=torch.tensor(
                self.max_snr_in_db,
                dtype=torch.float32,
                device=samples.device,
            ),
            validate_args=True,
        )  # sample uniformly from this distribution (low and high values)

        # randomize SNRs
        self.transform_parameters["snr_in_db"] = snr_distribution.sample(
            sample_shape=(batch_size,)
        )

        # randomize number of samples to mix for the entire batch
        num_mixes = torch.randint(
            1, self.max_samples + 1, (1,), device=samples.device
        ).item()

        self.transform_parameters["num_mixes"] = num_mixes

        # Ensure the number of mixes is smaller than the batch size
        # if num_mixes >= batch_size:
        #     raise ValueError("The number of mixes must be smaller than the batch size.")
        # randomize indices of samples to mix
        # self.transform_parameters["sample_indices"] = torch.randint(
        #     0,
        #     batch_size,
        #     (batch_size, num_mixes),
        #     device=samples.device,
        # )
        # Generate random indices with the constraint
        sample_indices = torch.empty((batch_size, num_mixes), dtype=torch.long)

        for i in range(batch_size):
            possible_indices = list(range(batch_size))

            if len(possible_indices) > 1:  # avoid error if only one sample is chosen
                possible_indices.remove(
                    i
                )  # Remove the current index to avoid self-mixing
                sample_indices[i] = torch.tensor(
                    [
                        possible_indices[
                            torch.randint(0, len(possible_indices), (1,)).item()
                        ]
                    ]
                )
            else:
                # If there's only one sample, we can set the index to a default value or skip
                sample_indices[i] = torch.tensor([0])
        self.transform_parameters["sample_indices"] = sample_indices

    def apply_transform(
        self,
        samples: Tensor = None,
        sample_rate: Optional[int] = None,
        targets: Optional[Tensor] = None,
        target_rate: Optional[int] = None,
    ) -> ObjectDict:

        snr = self.transform_parameters["snr_in_db"]
        # idx = self.transform_parameters["sample_idx"]
        num_mixes = self.transform_parameters["num_mixes"]
        sample_indices = self.transform_parameters["sample_indices"]

        mixed_samples = samples.clone()
        if targets is not None:
            mixed_targets = targets.clone()
        else:
            mixed_targets = None

        batch_size, _, waveform_length = mixed_samples.shape

        for i in range(num_mixes):
            current_indices = sample_indices[:, i]
            background_samples = Audio.rms_normalize(samples[current_indices])

            idx = torch.randint(
                0, waveform_length, (batch_size,), device=background_samples.device
            )
            arange = (
                torch.arange(waveform_length, device=background_samples.device)
                .unsqueeze(0)
                .expand(batch_size, -1)
            )
            rolled_indices = (arange + idx.unsqueeze(1)) % waveform_length
            background_samples = background_samples.squeeze(1)[
                torch.arange(batch_size).unsqueeze(1), rolled_indices
            ].unsqueeze(1)
            background_rms = calculate_rms(mixed_samples) / (
                10 ** (snr.unsqueeze(dim=-1) / 20)
            )

            mixed_samples += background_rms.unsqueeze(-1) * background_samples

            if mixed_targets is not None:
                background_targets = targets[current_indices]
                mixed_targets = self._mix_target(mixed_targets, background_targets, snr)

        return ObjectDict(
            samples=mixed_samples,
            sample_rate=sample_rate,
            targets=mixed_targets,
            target_rate=target_rate,
        )


class NoCallMixer:
    """
    A class used to mix no-call samples into the dataset.

    Attributes
    ----------
    _target_ : str
        Specifies the no-call sampler component in the pipeline.
    directory : str
        The directory containing the no-call data. The directory should contain audio files in a format that can be read by torchaudio (e.g. .wav).
    p : float
        The probability of a sample being replaced with a no-call sample. This parameter allows you to control the frequency of no-call samples in your dataset.
    sample_rate : int
        The sample rate at which the audio data should be processed. This parameter should align with the rest of your dataset and model configuration.
    length : int
        The length of the audio samples. This parameter should align with the rest of your dataset and model configuration.
    """

    def __init__(self, directory, p, sample_rate, length=5, *args, **kwargs):
        self.p = p
        self.sample_rate = sample_rate
        self.length = length

        self.paths = self.get_all_file_paths(directory)

    def get_all_file_paths(self, directory):
        pattern = os.path.join(directory, "**", "*")
        file_paths = [
            path for path in glob.glob(pattern, recursive=True) if os.path.isfile(path)
        ]

        absolute_file_paths = [os.path.abspath(path) for path in file_paths]

        return absolute_file_paths

    def __call__(self, input_values, labels):
        b, c = labels.shape
        for idx in range(len(input_values)):
            if random.random() < self.p:
                selected_path = random.choice(self.paths)
                info = sf.info(selected_path)
                sr = info.samplerate
                duration = info.duration

                if duration >= self.length:
                    latest_start = int(duration - self.length) * sr

                    start_frame = int(random.randint(0, latest_start))
                    end_frame = start_frame + self.length * sr
                    audio, sr = sf.read(
                        selected_path, start=start_frame, stop=end_frame
                    )

                else:
                    audio, sr = sf.read(selected_path)

                if sr != self.sample_rate:
                    audio = librosa.resample(
                        audio, orig_sr=sr, target_sr=self.sample_rate
                    )
                    sr = self.sample_rate

                audio = torch.tensor(audio)

                if audio.numel() < input_values[idx].numel():
                    padding = input_values[idx].numel() - audio.numel()
                    audio = torch.nn.functional.pad(audio, (0, padding))

                input_values[idx] = audio
                labels[idx] = torch.zeros(c)
        return input_values, labels


AudioFile = Union[Path, Text, dict]
"""
Audio files can be provided to the Audio class using different types:
    - a "str" instance: "/path/to/audio.wav"
    - a "Path" instance: Path("/path/to/audio.wav")
    - a dict with a mandatory "audio" key (mandatory) and an optional "channel" key:
        {"audio": "/path/to/audio.wav", "channel": 0}
    - a dict with mandatory "samples" and "sample_rate" keys and an optional "channel" key:
        {"samples": (channel, time) torch.Tensor, "sample_rate": 44100}

The optional "channel" key can be used to indicate a specific channel.
"""

# TODO: Remove this when it is the default
# torchaudio.USE_SOUNDFILE_LEGACY_INTERFACE = False
# torchaudio.set_audio_backend("soundfile")


class Audio:
    """Audio IO with on-the-fly resampling

    Parameters
    ----------
    sample_rate: int
        Target sample rate.
    mono : int, optional
        Convert multi-channel to mono. Defaults to True.

    Usage
    -----
    >>> audio = Audio(sample_rate=16000)
    >>> samples = audio("/path/to/audio.wav")

    # on-the-fly resampling
    >>> original_sample_rate = 44100
    >>> two_seconds_stereo = torch.rand(2, 2 * original_sample_rate)
    >>> samples = audio({"samples": two_seconds_stereo, "sample_rate": original_sample_rate})
    >>> assert samples.shape[1] == 2 * 16000
    """

    @staticmethod
    def is_valid(file: AudioFile) -> bool:

        if isinstance(file, dict):

            if "samples" in file:

                samples = file["samples"]
                if len(samples.shape) != 2 or samples.shape[0] > samples.shape[1]:
                    raise ValueError(
                        "'samples' must be provided as a (channel, time) torch.Tensor."
                    )

                sample_rate = file.get("sample_rate", None)
                if sample_rate is None:
                    raise ValueError(
                        "'samples' must be provided with their 'sample_rate'."
                    )
                return True

            elif "audio" in file:
                return True

            else:
                # TODO improve error message
                raise ValueError("either 'audio' or 'samples' key must be provided.")

        return True

    @staticmethod
    def rms_normalize(samples: Tensor) -> Tensor:
        """Power-normalize samples

        Parameters
        ----------
        samples : (..., time) Tensor
            Single (or multichannel) samples or batch of samples

        Returns
        -------
        samples: (..., time) Tensor
            Power-normalized samples
        """
        rms = samples.square().mean(dim=-1, keepdim=True).sqrt()
        return samples / (rms + 1e-8)

    @staticmethod
    def get_audio_metadata(file_path) -> tuple:
        """Return (num_samples, sample_rate)."""
        info = torchaudio.info(file_path)
        # Deal with backwards-incompatible signature change.
        # See https://github.com/pytorch/audio/issues/903 for more information.
        if type(info) is tuple:
            si, ei = info
            num_samples = si.length
            sample_rate = si.rate
        else:
            num_samples = info.num_frames
            sample_rate = info.sample_rate
        return num_samples, sample_rate

    def get_num_samples(self, file: AudioFile) -> int:
        """Number of samples (in target sample rate)

        :param file: audio file

        """

        self.is_valid(file)

        if isinstance(file, dict):

            # file = {"samples": torch.Tensor, "sample_rate": int, [ "channel": int ]}
            if "samples" in file:
                num_samples = file["samples"].shape[1]
                sample_rate = file["sample_rate"]

            # file = {"audio": str or Path, [ "channel": int ]}
            else:
                num_samples, sample_rate = self.get_audio_metadata(file["audio"])

        #  file = str or Path
        else:
            num_samples, sample_rate = self.get_audio_metadata(file)

        return math.ceil(num_samples * self.sample_rate / sample_rate)

    def __init__(self, sample_rate: int, mono: bool = True):
        super().__init__()
        self.sample_rate = sample_rate
        self.mono = mono

    def downmix_and_resample(self, samples: Tensor, sample_rate: int) -> Tensor:
        """Downmix and resample

        Parameters
        ----------
        samples : (channel, time) Tensor
            Samples.
        sample_rate : int
            Original sample rate.

        Returns
        -------
        samples : (channel, time) Tensor
            Remixed and resampled samples
        """

        # downmix to mono
        if self.mono and samples.shape[0] > 1:
            samples = samples.mean(dim=0, keepdim=True)

        # resample
        if self.sample_rate != sample_rate:
            samples = samples.numpy()
            if self.mono:
                # librosa expects mono audio to be of shape (n,), but we have (1, n).
                samples = librosa.core.resample(
                    samples[0], orig_sr=sample_rate, target_sr=self.sample_rate
                )[None]
            else:
                samples = librosa.core.resample(
                    samples.T, orig_sr=sample_rate, target_sr=self.sample_rate
                ).T

            samples = torch.tensor(samples)

        return samples

    def __call__(
        self, file: AudioFile, sample_offset: int = 0, num_samples: int = None
    ) -> Tensor:
        """

        Parameters
        ----------
        file : AudioFile
            Audio file.
        sample_offset : int, optional
            Start loading at this `sample_offset` sample. Defaults ot 0.
        num_samples : int, optional
            Load that many samples. Defaults to load up to the end of the file.

        Returns
        -------
        samples : (time, channel) torch.Tensor
            Samples

        """

        self.is_valid(file)

        original_samples = None

        if isinstance(file, dict):

            # file = {"samples": torch.Tensor, "sample_rate": int, [ "channel": int ]}
            if "samples" in file:
                original_samples = file["samples"]
                original_sample_rate = file["sample_rate"]
                original_total_num_samples = original_samples.shape[1]
                channel = file.get("channel", None)

            # file = {"audio": str or Path, [ "channel": int ]}
            else:
                audio_path = str(file["audio"])
                (
                    original_total_num_samples,
                    original_sample_rate,
                ) = self.get_audio_metadata(audio_path)
                channel = file.get("channel", None)

        #  file = str or Path
        else:
            audio_path = str(file)
            original_total_num_samples, original_sample_rate = self.get_audio_metadata(
                audio_path
            )
            channel = None

        original_sample_offset = round(
            sample_offset * original_sample_rate / self.sample_rate
        )
        if num_samples is None:
            original_num_samples = original_total_num_samples - original_sample_offset
        else:
            original_num_samples = round(
                num_samples * original_sample_rate / self.sample_rate
            )

        if original_sample_offset + original_num_samples > original_total_num_samples:
            original_sample_offset = original_total_num_samples - original_num_samples
            # raise ValueError() # rounding error i guess

        if original_samples is None:
            try:
                original_data, _ = torchaudio.load(
                    audio_path,
                    frame_offset=original_sample_offset,
                    num_frames=original_num_samples,
                )
            except TypeError:
                raise Exception(
                    "It looks like you are using an unsupported version of torchaudio."
                    " If you have 0.6 or older, please upgrade to a newer version."
                )

        else:
            original_data = original_samples[
                :,
                original_sample_offset : original_sample_offset + original_num_samples,
            ]

        if channel is not None:
            original_data = original_data[channel - 1 : channel, :]

        result = self.downmix_and_resample(original_data, original_sample_rate)

        if num_samples is not None:
            # If there is an off-by-one error in the length (e.g. due to resampling), fix it.
            if result.shape[-1] > num_samples:
                result = result[:, :num_samples]
            elif result.shape[-1] < num_samples:
                diff = num_samples - result.shape[-1]
                result = torch.nn.functional.pad(result, (0, diff))

        return result


class AddBackgroundNoise(BaseWaveformTransform):
    """
    Add background noise to the input audio.
    """

    supported_modes = {"per_batch", "per_example", "per_channel"}

    # Note: This transform has only partial support for multichannel audio. Noises that are not
    # mono get mixed down to mono before they are added to all channels in the input.
    supports_multichannel = True
    requires_sample_rate = True

    supports_target = True
    requires_target = False

    def __init__(
        self,
        background_paths: Union[List[Path], List[str], Path, str],
        min_snr_in_db: float = 3.0,
        max_snr_in_db: float = 30.0,
        mode: str = "per_example",
        p: float = 0.5,
        p_mode: str = None,
        sample_rate: int = None,
        target_rate: int = None,
        output_type: Optional[str] = None,
    ):
        """

        :param background_paths: Either a path to a folder with audio files or a list of paths
            to audio files.
        :param min_snr_in_db: minimum SNR in dB.
        :param max_snr_in_db: maximum SNR in dB.
        :param mode:
        :param p:
        :param p_mode:
        :param sample_rate:
        """

        super().__init__(
            mode=mode,
            p=p,
            p_mode=p_mode,
            sample_rate=sample_rate,
            target_rate=target_rate,
            output_type=output_type,
        )

        # TODO: check that one can read audio files
        self.background_paths = find_audio_files_in_paths(background_paths)

        if sample_rate is not None:
            self.audio = Audio(sample_rate=sample_rate, mono=True)

        if len(self.background_paths) == 0:
            raise EmptyPathException("There are no supported audio files found.")

        self.min_snr_in_db = min_snr_in_db
        self.max_snr_in_db = max_snr_in_db
        if self.min_snr_in_db > self.max_snr_in_db:
            raise ValueError("min_snr_in_db must not be greater than max_snr_in_db")

    def random_background(self, audio: Audio, target_num_samples: int) -> torch.Tensor:
        pieces = []

        # TODO: support repeat short samples instead of concatenating from different files

        missing_num_samples = target_num_samples
        while missing_num_samples > 0:
            background_path = random.choice(self.background_paths)
            background_num_samples = audio.get_num_samples(background_path)

            # If the background sample is longer than what we need, extract the exact amount
            if background_num_samples >= missing_num_samples:
                sample_offset = random.randint(
                    0, background_num_samples - missing_num_samples
                )
                background_samples = audio(
                    background_path,
                    sample_offset=sample_offset,
                    num_samples=missing_num_samples,
                )
                pieces.append(background_samples)
                # background_samples matches missing_num_samples, break out of while loop
                break

            background_samples = audio(background_path)
            pieces.append(background_samples)
            missing_num_samples -= background_num_samples

        # the inner call to rms_normalize ensures concatenated pieces share the same RMS (1)
        # the outer call to rms_normalize ensures that the resulting background has an RMS of 1
        # (this simplifies "apply_transform" logic)
        return audio.rms_normalize(
            torch.cat([audio.rms_normalize(piece) for piece in pieces], dim=1)
        )

    def randomize_parameters(
        self,
        samples: Tensor = None,
        sample_rate: Optional[int] = None,
        targets: Optional[Tensor] = None,
        target_rate: Optional[int] = None,
    ):
        """

        :params samples: (batch_size, num_channels, num_samples)
        """

        batch_size, _, num_samples = samples.shape

        # (batch_size, num_samples) RMS-normalized background noise
        audio = self.audio if hasattr(self, "audio") else Audio(sample_rate, mono=True)
        self.transform_parameters["background"] = torch.stack(
            [self.random_background(audio, num_samples) for _ in range(batch_size)]
        )

        # (batch_size, ) SNRs
        if self.min_snr_in_db == self.max_snr_in_db:
            self.transform_parameters["snr_in_db"] = torch.full(
                size=(batch_size,),
                fill_value=self.min_snr_in_db,
                dtype=torch.float32,
                device=samples.device,
            )
        else:
            snr_distribution = torch.distributions.Uniform(
                low=torch.tensor(
                    self.min_snr_in_db, dtype=torch.float32, device=samples.device
                ),
                high=torch.tensor(
                    self.max_snr_in_db, dtype=torch.float32, device=samples.device
                ),
                validate_args=True,
            )
            self.transform_parameters["snr_in_db"] = snr_distribution.sample(
                sample_shape=(batch_size,)
            )

    def apply_transform(
        self,
        samples: Tensor = None,
        sample_rate: Optional[int] = None,
        targets: Optional[Tensor] = None,
        target_rate: Optional[int] = None,
    ) -> ObjectDict:
        batch_size, num_channels, num_samples = samples.shape

        # (batch_size, num_samples)
        background = self.transform_parameters["background"].to(samples.device)

        # (batch_size, num_channels)
        background_rms = calculate_rms(samples) / (
            10 ** (self.transform_parameters["snr_in_db"].unsqueeze(dim=-1) / 20)
        )

        return ObjectDict(
            samples=samples
            + background_rms.unsqueeze(-1)
            * background.view(batch_size, 1, num_samples).expand(-1, num_channels, -1),
            sample_rate=sample_rate,
            targets=targets,
            target_rate=target_rate,
        )


class PowerToDB(nn.Module):
    def __init__(self, ref=1.0, amin=1e-10, top_db=80.0):
        super(PowerToDB, self).__init__()
        # Initialize parameters
        self.ref = ref
        self.amin = amin
        self.top_db = top_db

    def forward(self, S):
        # Convert S to a PyTorch tensor if it is not already
        S = torch.as_tensor(S, dtype=torch.float32)

        if self.amin <= 0:
            raise ValueError("amin must be strictly positive")

        if torch.is_complex(S):
            warnings.warn(
                "power_to_db was called on complex input so phase "
                "information will be discarded. To suppress this warning, "
                "call power_to_db(S.abs()**2) instead.",
                stacklevel=2,
            )
            magnitude = S.abs()
        else:
            magnitude = S

        # Check if ref is a callable function or a scalar
        if callable(self.ref):
            ref_value = self.ref(magnitude)
        else:
            ref_value = torch.abs(torch.tensor(self.ref, dtype=S.dtype))

        # Compute the log spectrogram
        log_spec = 10.0 * torch.log10(
            torch.maximum(magnitude, torch.tensor(self.amin, device=magnitude.device))
        )
        log_spec -= 10.0 * torch.log10(
            torch.maximum(ref_value, torch.tensor(self.amin, device=magnitude.device))
        )

        # Apply top_db threshold if necessary
        if self.top_db is not None:
            if self.top_db < 0:
                raise ValueError("top_db must be non-negative")
            log_spec = torch.maximum(log_spec, log_spec.max() - self.top_db)

        return log_spec
