import os
import tempfile
import torch
import torchaudio

from gadme.augmentations.augmentations import AudioAugmentor, pad_spectrogram_width, RandomTimeStretch


def test_padding_equal_width(sample_spectrogram):
    """
    Test if the function pads the spectrogram correctly when the target width is equal to the original width.
    """
    original_width = sample_spectrogram.shape[2]
    target_width = original_width
    value = 0.0
    padded_spectrogram = pad_spectrogram_width(sample_spectrogram, target_width, value)
    assert padded_spectrogram.shape == sample_spectrogram.shape
    assert torch.all(padded_spectrogram == sample_spectrogram)


def test_padding_increased_width_odd(sample_spectrogram):
    """
    Test if the function pads the spectrogram correctly when the target width is increased unevenly.
    """
    original_width = sample_spectrogram.shape[2]
    target_width = original_width + 3
    value = 0.0
    padded_spectrogram = pad_spectrogram_width(sample_spectrogram, target_width, value)
    assert padded_spectrogram.shape[2] == target_width


def test_padding_increased_width_even(sample_spectrogram):
    """
    Test if the function pads the spectrogram correctly when the target width is increased evenly.
    """
    original_width = sample_spectrogram.shape[2]
    target_width = original_width + 4
    value = 0.0
    padded_spectrogram = pad_spectrogram_width(sample_spectrogram, target_width, value)
    assert padded_spectrogram.shape[2] == target_width


def test_padding_decreased_width_even(sample_spectrogram):
    """
    Test if the function pads the spectrogram correctly when the target width is decreased evenly.
    """
    original_width = sample_spectrogram.shape[2]
    target_width = original_width - 2
    value = 0.0
    padded_spectrogram = pad_spectrogram_width(sample_spectrogram, target_width, value)
    assert padded_spectrogram.shape[2] == target_width


def test_padding_decreased_width_odd(sample_spectrogram):
    """
    Test if the function pads the spectrogram correctly when the target width is decreased unevenly.
    """
    original_width = sample_spectrogram.shape[2]
    target_width = original_width - 1
    value = 0.0
    padded_spectrogram = pad_spectrogram_width(sample_spectrogram, target_width, value)
    assert padded_spectrogram.shape[2] == target_width


def test_padding_value(sample_spectrogram):
    """
    Test if the function pads the spectrogram with the correct value.
    """
    original_width = sample_spectrogram.shape[2]
    target_width = original_width + 4
    value = 5.0
    padded_spectrogram = pad_spectrogram_width(sample_spectrogram, target_width, value)
    assert torch.all(padded_spectrogram[:, :, :2] == value)
    assert torch.all(padded_spectrogram[:, :, -2:] == value)


def test_time_stretch_probability(sample_spectrogram):
    """
    Test if the time-stretching is applied with the specified probability.
    """
    time_stretcher = RandomTimeStretch(
        prob=1.0, n_freq=201
    )  # Set the probability to 100% for testing
    time_stretched_spectrogram = time_stretcher(sample_spectrogram)
    assert time_stretched_spectrogram.shape == sample_spectrogram.shape
    assert torch.all(time_stretched_spectrogram != sample_spectrogram)


def test_time_stretch_no_probability(sample_spectrogram):
    """
    Test if the time-stretching is not applied when the probability is set to 0.
    """
    time_stretcher = RandomTimeStretch(
        prob=0.0
    )  # Set the probability to 0% for testing
    original_spectrogram = sample_spectrogram.clone()
    time_stretched_spectrogram = time_stretcher(sample_spectrogram)
    assert torch.all(time_stretched_spectrogram == original_spectrogram.abs().pow(2))


def test_time_stretch_pad_width(sample_spectrogram):
    """
    Test if the time-stretched spectrogram is properly padded along the width axis.
    """
    time_stretcher = RandomTimeStretch(prob=1.0, n_freq=201)
    time_stretched_spectrogram = time_stretcher(sample_spectrogram)

    original_width = sample_spectrogram.shape[2]
    assert time_stretched_spectrogram.shape[2] == original_width


def test_time_stretch_complex_to_float(sample_spectrogram):
    """
    Test if the time-stretched spectrogram is converted from complex to float values.
    """
    time_stretcher = RandomTimeStretch(prob=1.0, n_freq=201)
    time_stretched_spectrogram = time_stretcher(sample_spectrogram)

    assert time_stretched_spectrogram.dtype == torch.float32


def test_audioaugmentor_augment_waveform_no_augmentation(sample_waveform):
    """
    Test if the waveform remains unchanged when no waveform augmentations are specified.
    """
    audio_augmentor = AudioAugmentor(waveform_augmentations=None)
    augmented_waveform = audio_augmentor.augment_waveform(sample_waveform)
    assert torch.all(augmented_waveform == sample_waveform)
    assert augmented_waveform.shape[0] == 1
    assert augmented_waveform.shape[1] == sample_waveform.shape[0]


def test_audioaugmentor_augment_waveform_time_mask(sample_waveform):
    """
    Test if the waveform is augmented with time mask when specified in the configuration.
    """
    waveform_augmentations = {
        "time_mask": {"prob": 1.0, "min_band_part": 0.0, "max_band_part": 0.5}
    }
    audio_augmentor = AudioAugmentor(waveform_augmentations=waveform_augmentations)
    augmented_waveform = audio_augmentor.augment_waveform(sample_waveform)
    assert augmented_waveform is not sample_waveform
    assert augmented_waveform.shape[0] == 1
    assert augmented_waveform.shape[1] == sample_waveform.shape[0]


def test_audioaugmentor_augment_waveform_time_stretch(sample_waveform):
    """
    Test if the waveform is augmented with time stretch when specified in the configuration.
    """
    waveform_augmentations = {
        "time_stretch": {"prob": 1.0, "min_rate": 0.8, "max_rate": 1.25}
    }
    audio_augmentor = AudioAugmentor(waveform_augmentations=waveform_augmentations)
    augmented_waveform = audio_augmentor.augment_waveform(sample_waveform)
    assert torch.all(augmented_waveform != sample_waveform)
    assert augmented_waveform.shape[0] == 1
    assert augmented_waveform.shape[1] == sample_waveform.shape[0]


def test_audioaugmentor_augment_waveform_colored_noise(sample_waveform):
    """
    Test if the waveform is augmented with colored noise when specified in the configuration.
    """
    waveform_augmentations = {
        "colored_noise": {
            "prob": 1.0,
            "min_snr_in_db": 3.0,
            "max_snr_in_db": 30.0,
            "min_f_decay": -2.0,
            "max_f_decay": 2.0,
        }
    }
    audio_augmentor = AudioAugmentor(waveform_augmentations=waveform_augmentations)
    augmented_waveform = audio_augmentor.augment_waveform(sample_waveform)
    assert torch.all(augmented_waveform != sample_waveform)
    assert augmented_waveform.shape[0] == 1
    assert augmented_waveform.shape[1] == sample_waveform.shape[0]


def test_audioaugmentor_augment_waveform_background_noise(sample_waveform):
    """
    Test if the waveform is augmented with background noise when specified in the configuration.
    """
    waveform_augmentations = {
        "background_noise": {
            "prob": 1.0,
            "background_paths": [],
            "min_snr_in_db": 3.0,
            "max_snr_in_db": 30.0,
        }
    }

    # Create a temporary directory and save a sample WAV file for background noise
    with tempfile.TemporaryDirectory() as tmpdir:
        background_noise = torch.randn(1, 16000, dtype=torch.float32)
        background_path = os.path.join(tmpdir, "background.wav")
        torchaudio.save(background_path, background_noise, 16000)
        waveform_augmentations["background_noise"]["background_paths"].append(
            background_path
        )

        audio_augmentor = AudioAugmentor(waveform_augmentations=waveform_augmentations)
        augmented_waveform = audio_augmentor.augment_waveform(sample_waveform)
        assert torch.all(augmented_waveform != sample_waveform)
        assert augmented_waveform.shape[0] == 1
        assert augmented_waveform.shape[1] == sample_waveform.shape[0]


def test_audioaugmentor_augment_waveform_pitch_shift(sample_waveform):
    """
    Test if the waveform is augmented with pitch shift when specified in the configuration.
    """
    waveform_augmentations = {
        "pitch_shift": {
            "prob": 1.0,
            "min_transpose_semitones": -4.0,
            "max_transpose_semitones": 4.0,
        }
    }
    audio_augmentor = AudioAugmentor(waveform_augmentations=waveform_augmentations)
    augmented_waveform = audio_augmentor.augment_waveform(sample_waveform)
    assert torch.all(augmented_waveform != sample_waveform)
    assert augmented_waveform.shape[0] == 1
    assert augmented_waveform.shape[1] == sample_waveform.shape[0]


def test_audioaugmentor_augment_waveform_multiple_augmentations(sample_waveform):
    """
    Test if the waveform is augmented with multiple augmentations when specified in the configuration.
    """
    waveform_augmentations = {
        "colored_noise": {
            "prob": 1.0,
            "min_snr_in_db": 3.0,
            "max_snr_in_db": 30.0,
            "min_f_decay": -2.0,
            "max_f_decay": 2.0,
        },
        "background_noise": {
            "prob": 1.0,
            "background_paths": [],
            "min_snr_in_db": 3.0,
            "max_snr_in_db": 30.0,
        },
        "pitch_shift": {
            "prob": 1.0,
            "min_transpose_semitones": -4.0,
            "max_transpose_semitones": 4.0,
        },
        "time_mask": {"prob": 1.0, "min_band_part": 0.0, "max_band_part": 0.5},
        "time_stretch": {"prob": 1.0, "min_rate": 0.8, "max_rate": 1.25},
    }

    # Create a temporary directory and save a sample WAV file for background noise
    with tempfile.TemporaryDirectory() as tmpdir:
        background_noise = torch.randn(1, 16000, dtype=torch.float32)
        background_path = os.path.join(tmpdir, "background.wav")
        torchaudio.save(background_path, background_noise, 16000)
        waveform_augmentations["background_noise"]["background_paths"].append(
            background_path
        )

        audio_augmentor = AudioAugmentor(waveform_augmentations=waveform_augmentations)
        augmented_waveform = audio_augmentor.augment_waveform(sample_waveform)
        assert torch.all(augmented_waveform != sample_waveform)
        assert augmented_waveform.shape[0] == 1
        assert augmented_waveform.shape[1] == sample_waveform.shape[0]


def test_audioaugmentor_augment_spectrogram_no_augmentation(sample_spectrogram):
    """
    Test if the spectrogram remains unchanged when no spectrogram augmentations are specified.
    """
    spectrogram_augmentations = None
    audio_augmentor = AudioAugmentor(
        spectrogram_augmentations=spectrogram_augmentations
    )
    augmented_spectrogram = audio_augmentor.augment_spectrogram(sample_spectrogram)
    assert augmented_spectrogram.shape == sample_spectrogram.shape
    assert torch.all(augmented_spectrogram == sample_spectrogram)


def test_audioaugmentor_augment_spectrogram_time_masking(sample_spectrogram):
    """
    Test if the spectrogram is augmented with time masking when specified in the configuration.
    """
    spectrogram_augmentations = {"time_masking": {"prob": 1.0, "time_mask_param": 20}}
    audio_augmentor = AudioAugmentor(
        spectrogram_augmentations=spectrogram_augmentations
    )
    augmented_spectrogram = audio_augmentor.augment_spectrogram(sample_spectrogram)
    assert augmented_spectrogram.shape == sample_spectrogram.shape
    assert augmented_spectrogram is not sample_spectrogram


def test_audioaugmentor_augment_spectrogram_frequency_masking(sample_spectrogram):
    """
    Test if the spectrogram is augmented with frequency masking when specified in the configuration.
    """
    spectrogram_augmentations = {
        "frequency_masking": {"prob": 1.0, "freq_mask_param": 10}
    }
    audio_augmentor = AudioAugmentor(
        spectrogram_augmentations=spectrogram_augmentations
    )
    augmented_spectrogram = audio_augmentor.augment_spectrogram(sample_spectrogram)
    assert augmented_spectrogram.shape == sample_spectrogram.shape
    assert augmented_spectrogram is not sample_spectrogram


def test_audioaugmentor_augment_spectrogram_time_stretch(sample_spectrogram):
    """
    Test if the spectrogram is augmented with time stretch when specified in the configuration.
    """
    spectrogram_augmentations = {
        "time_stretch": {
            "prob": 1.0,
            "min_rate": 0.8,
            "max_rate": 1.2,
        }
    }
    audio_augmentor = AudioAugmentor(
        spectrogram_augmentations=spectrogram_augmentations, n_fft=400
    )
    augmented_spectrogram = audio_augmentor.augment_spectrogram(sample_spectrogram)
    assert augmented_spectrogram.shape == sample_spectrogram.shape
    assert augmented_spectrogram is not sample_spectrogram


def test_audioaugmentor_augment_spectrogram_multiple_augmentations(sample_spectrogram):
    """
    Test if the spectrogram is augmented with multiple augmentations when specified in the configuration.
    """
    spectrogram_augmentations = {
        "time_masking": {"prob": 1.0, "time_mask_param": 10},
        "frequency_masking": {"prob": 1.0, "freq_mask_param": 10},
        "time_stretch": {
            "prob": 1.0,
            "min_rate": 0.8,
            "max_rate": 1.2,
        },
    }
    audio_augmentor = AudioAugmentor(
        spectrogram_augmentations=spectrogram_augmentations, n_fft=400
    )
    augmented_spectrogram = audio_augmentor.augment_spectrogram(sample_spectrogram)
    assert augmented_spectrogram.shape == sample_spectrogram.shape
    assert augmented_spectrogram is not sample_spectrogram


def test_audioaugmentor_transform_to_spectrogram_shape(sample_waveform):
    """
    Test if the resulting spectrogram has the correct shape.
    """
    n_fft = 256  # Set the number of FFT points for the test
    hop_length = 128
    audio_augmentor = AudioAugmentor(n_fft=n_fft, hop_length=hop_length)
    spectrogram = audio_augmentor.transform_to_spectrogram(sample_waveform)
    print(sample_waveform.shape)
    expected_shape = (n_fft // 2 + 1, 126)
    assert spectrogram.shape == expected_shape


def test_audioaugmentor_transform_to_spectrogram_increasing_n_fft(sample_waveform):
    """
    Test if increasing the number of FFT points results in more frequency bins in the spectrogram.
    """
    n_fft1 = 256
    n_fft2 = 512
    audio_augmentor1 = AudioAugmentor(n_fft=n_fft1)
    audio_augmentor2 = AudioAugmentor(n_fft=n_fft2)
    spectrogram1 = audio_augmentor1.transform_to_spectrogram(sample_waveform)
    spectrogram2 = audio_augmentor2.transform_to_spectrogram(sample_waveform)
    assert spectrogram1.shape[0] < spectrogram2.shape[0]


def test_audioaugmentor_transform_to_mel_scale_shape(sample_spectrogram):
    """
    Test if the resulting mel spectrogram has the correct shape.
    """
    print(sample_spectrogram.shape)
    n_mels = 64  # Set the number of Mel filter banks for the test
    audio_augmentor = AudioAugmentor(n_mels=n_mels, n_fft=400)
    mel_spectrogram = audio_augmentor.transform_to_mel_scale(sample_spectrogram)
    expected_shape = (1, n_mels, sample_spectrogram.shape[2])
    assert mel_spectrogram.shape == expected_shape


def test_audioaugmentor_transform_to_mel_scale_increasing_n_mels(sample_spectrogram):
    """
    Test if increasing the number of Mel filter banks results in a larger mel spectrogram.
    """
    n_mels1 = 64
    n_mels2 = 128
    audio_augmentor1 = AudioAugmentor(n_mels=n_mels1, n_fft=400)
    audio_augmentor2 = AudioAugmentor(n_mels=n_mels2, n_fft=400)
    mel_spectrogram1 = audio_augmentor1.transform_to_mel_scale(sample_spectrogram)
    mel_spectrogram2 = audio_augmentor2.transform_to_mel_scale(sample_spectrogram)
    assert mel_spectrogram1.shape[1] < mel_spectrogram2.shape[1]
    assert mel_spectrogram1.shape[0] == mel_spectrogram2.shape[0]
    assert mel_spectrogram1.shape[2] == mel_spectrogram2.shape[2]


def test_audioaugmentor_combined_augmentations_waveform(sample_waveform):
    """
    Test whether the combined augmentations function returns a waveform if use_spectrogram=False.
    """
    # Assume use_spectrogram=False for this test
    audio_augmentor = AudioAugmentor(use_spectrogram=False)
    augmented_waveform = audio_augmentor.combined_augmentations(sample_waveform)
    assert augmented_waveform.shape[0] == 1
    assert augmented_waveform.shape[1] == sample_waveform.shape[0]


def test_audioaugmentor_combined_augmentations_spectrogram(sample_waveform):
    """
    Test whether the combined augmentations function returns a spectrogram if use_spectrogram=True.
    """
    # Assume use_spectrogram=True for this test
    audio_augmentor = AudioAugmentor(use_spectrogram=True, n_fft=400, hop_length=200)
    augmented_spectrogram = audio_augmentor.combined_augmentations(sample_waveform)
    assert augmented_spectrogram.shape[0] == 1  # Mono-channel audio
    assert augmented_spectrogram.shape[1] == 201
    assert augmented_spectrogram.shape[2] == 81


def test_audioaugmentor_combined_augmentations_db_scale(sample_waveform):
    """
    Test if the combined augmentations apply the db scale transformation to the spectrogram.
    """
    audio_augmentor = AudioAugmentor(
        use_spectrogram=True, db_scale=True, n_fft=400, hop_length=200
    )
    augmented_spectrogram = audio_augmentor.combined_augmentations(sample_waveform)
    assert torch.min(augmented_spectrogram) < 0


def test_audioaugmentor_combined_augmentations_mel_scale_none(sample_waveform):
    """
    Test if the combined augmentations function behaves correctly when n_mels is None.
    """
    # Assume use_spectrogram=True and n_mels=None for this test
    audio_augmentor = AudioAugmentor(
        use_spectrogram=True, n_fft=400, hop_length=200, n_mels=None
    )
    augmented_spectrogram = audio_augmentor.combined_augmentations(sample_waveform)

    assert augmented_spectrogram.shape == torch.Size([1, 201, 81])  # Mono-channel audio


def test_audioaugmentor_combined_augmentations_mel_scale_not_none(sample_waveform):
    """
    Test if the combined augmentations function behaves correctly when n_mels is not None.
    """
    n_mels = 64  # Set the number of Mel filter banks for the test
    audio_augmentor = AudioAugmentor(
        use_spectrogram=True, n_fft=400, hop_length=200, n_mels=n_mels
    )
    augmented_spectrogram = audio_augmentor.combined_augmentations(sample_waveform)

    assert augmented_spectrogram.shape[0] == 1  # Mono-channel audio
    assert augmented_spectrogram.shape[1] == n_mels
    assert augmented_spectrogram.shape[2] == 81


def test_audioaugmentor_augmentations_spectrogram_augmentations(sample_waveform):
    """
    Test if the combined augmentations behave correctly when spectrogram augmentations are used.
    """
    # Assume use_spectrogram=True and spectrogram_augmentations are specified for this test
    audio_augmentor = AudioAugmentor(
        n_fft=400,
        hop_length=200,
        use_spectrogram=True,
        spectrogram_augmentations={
            "time_masking": {"time_mask_param": 10, "prob": 0.5},
            "frequency_masking": {"freq_mask_param": 5, "prob": 0.5},
            "time_stretch": {
                "min_rate": 0.8,
                "max_rate": 1.2,
                "prob": 0.5,
            },
        },
        n_mels=64,
    )
    augmented_spectrogram = audio_augmentor.combined_augmentations(sample_waveform)

    assert augmented_spectrogram.shape[0] == 1  # Mono-channel audio
    assert augmented_spectrogram.shape[1] == 64  # n_mels
    assert augmented_spectrogram.shape[2] == 81


def test_audioaugmentor_combined_augmentations_waveform_augmentations(sample_waveform):
    """
    Test if the combined augmentations behave correctly when waveform augmentations are used.
    """
    # Assume use_spectrogram=False and waveform_augmentations are specified for this test
    audio_augmentor = AudioAugmentor(
        use_spectrogram=False,
        waveform_augmentations={
            "colored_noise": {
                "prob": 1.0,
                "min_snr_in_db": 3.0,
                "max_snr_in_db": 30.0,
                "min_f_decay": -2.0,
                "max_f_decay": 2.0,
            },
            "background_noise": {
                "prob": 1.0,
                "background_paths": [],
                "min_snr_in_db": 3.0,
                "max_snr_in_db": 30.0,
            },
            "pitch_shift": {
                "prob": 1.0,
                "min_transpose_semitones": -4.0,
                "max_transpose_semitones": 4.0,
            },
            "time_mask": {"prob": 1.0, "min_band_part": 0.0, "max_band_part": 0.5},
            "time_stretch": {"prob": 1.0, "min_rate": 0.8, "max_rate": 1.25},
        },
    )

    # Create a temporary directory and save a sample WAV file for background noise
    with tempfile.TemporaryDirectory() as tmpdir:
        background_noise = torch.randn(1, 16000, dtype=torch.float32)
        background_path = os.path.join(tmpdir, "background.wav")
        torchaudio.save(background_path, background_noise, 16000)
        audio_augmentor.waveform_augmentations["background_noise"][
            "background_paths"
        ].append(background_path)

        augmented_waveform = audio_augmentor.combined_augmentations(sample_waveform)
        assert torch.all(augmented_waveform != sample_waveform)
        assert augmented_waveform.shape[0] == 1
        assert augmented_waveform.shape[1] == sample_waveform.shape[0]


def test_audioaugmentor_combined_augmentations_spectrogram_and_waveform_augmentations(
    sample_waveform,
):
    """
    Test if the combined augmentations behave correctly when both spectrogram and waveform augmentations are used.
    """
    # Assume use_spectrogram=True and spectrogram_augmentations and waveform_augmentations are specified for this test
    audio_augmentor = AudioAugmentor(
        n_fft=400,
        hop_length=200,
        use_spectrogram=True,
        spectrogram_augmentations={
            "time_masking": {"time_mask_param": 10, "prob": 0.5},
            "frequency_masking": {"freq_mask_param": 5, "prob": 0.5},
            "time_stretch": {
                "min_rate": 0.8,
                "max_rate": 1.2,
                "prob": 0.5,
            },
        },
        waveform_augmentations={
            "colored_noise": {
                "prob": 1.0,
                "min_snr_in_db": 3.0,
                "max_snr_in_db": 30.0,
                "min_f_decay": -2.0,
                "max_f_decay": 2.0,
            },
            "background_noise": {
                "prob": 1.0,
                "background_paths": [],
                "min_snr_in_db": 3.0,
                "max_snr_in_db": 30.0,
            },
        },
        n_mels=64,
    )
    # Create a temporary directory and save a sample WAV file for background noise
    with tempfile.TemporaryDirectory() as tmpdir:
        background_noise = torch.randn(1, 16000, dtype=torch.float32)
        background_path = os.path.join(tmpdir, "background.wav")
        torchaudio.save(background_path, background_noise, 16000)
        audio_augmentor.waveform_augmentations["background_noise"][
            "background_paths"
        ].append(background_path)

        augmented_spectrogram = audio_augmentor.combined_augmentations(sample_waveform)
        assert augmented_spectrogram.shape[0] == 1  # Mono-channel audio
        assert augmented_spectrogram.shape[1] == 64  # n_mels
        assert augmented_spectrogram.shape[2] == 81
