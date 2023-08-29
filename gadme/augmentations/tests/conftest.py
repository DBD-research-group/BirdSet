"""
This file can be used to define extra pytest rules and fixtures

We already predefined the fixtures slow and ultra_slow, which can be used to mark specific tests.

Use pytest fixtures to define your own variables, data and files!
"""


import pytest
import torch
from torchaudio import transforms


def pytest_configure(config):
    """
    Add marker
    """
    config.addinivalue_line("markers", "slow: Tests will be slow. ")


def pytest_addoption(parser):
    """
    Add option in the pytest run command (for terminal use)
    """
    parser.addoption(
        "--runslow", action="store_true", default=False, help="run slow tests"
    )


def pytest_collection_modifyitems(config, items):
    """
    Add the ultra slow mark as option
    """
    if config.getoption("--runslow"):
        # --runslow given in cli: do not skip slow tests
        return
    skip_slow = pytest.mark.skip(reason="need --runslow option to run")
    for item in items:
        if "ultra_slow" in item.keywords:
            item.add_marker(skip_slow)


@pytest.fixture
def sample_spectrogram():
    """
    Provide a random sample spectrogram for testing.
    """
    random_tensor = torch.rand(1, 16000)
    transform = transforms.Spectrogram()
    spectrogram = transform(random_tensor)
    return spectrogram


@pytest.fixture
def sample_waveform():
    # Provide a sample waveform for testing
    # Here, using a 1-channel waveform with 16000 samples for simplicity
    return torch.randn(16000, dtype=torch.float32)
