import os
from typing import Tuple

import hydra
import lightning as L
from omegaconf import OmegaConf
import pyrootutils
from torch.utils.data import DataLoader

from gadme import utils

log = utils.get_pylogger(__name__)
# rootutils.setup_root(__file__, indicator=".project-root", pythonpath=True)
root = pyrootutils.setup_root(
    search_from=__file__,
    indicator=[".git"],
    pythonpath=True,
    dotenv=True,
)

_HYDRA_PARAMS = {
    "version_base": None,
    # "config_path": "../configs",
    "config_path": str(root / "configs"),
    "config_name": "main.yaml",
}


def calculate_mean_std_from_dataloader(dataloader: DataLoader) -> Tuple[float, float]:
    """
    Calculates the mean and standard deviation from a PyTorch DataLoader.
    Args:
        dataloader (DataLoader): The DataLoader containing the dataset for calculation.
    Returns:
        Tuple[float, float]: The calculated mean and standard deviation.
    """

    sum_, sum_of_squares, num_elements = 0.0, 0.0, 0

    for batch in dataloader:
        input_values = batch["input_values"]
        sum_ += input_values.sum()
        sum_of_squares += (input_values**2).sum()
        num_elements += input_values.nelement()

    mean = sum_ / num_elements
    std = (sum_of_squares / num_elements - mean**2) ** 0.5

    return mean, std


@utils.register_custom_resolvers(**_HYDRA_PARAMS)
@hydra.main(**_HYDRA_PARAMS)
def calculate_normalization_parameters(cfg):
    log.info("Using config: \n%s", OmegaConf.to_yaml(cfg))

    log.info(f"Dataset path: <{os.path.abspath(cfg.paths.dataset_path)}>")
    os.makedirs(cfg.paths.dataset_path, exist_ok=True)

    log.info(f"Log path: <{os.path.abspath(cfg.paths.log_dir)}>")
    os.makedirs(cfg.paths.log_dir, exist_ok=True)

    log.info(f"Seed everything with <{cfg.seed}>")
    L.seed_everything(cfg.seed)
    # log.info(f"Instantiate logger {[loggers for loggers in cfg['logger']]}")

    # Setup data
    log.info(f"Instantiate datamodule <{cfg.datamodule._target_}>")
    datamodule = hydra.utils.instantiate(cfg.datamodule)

    # Set the normalization in the config file to False, since we do not want normalized data for the mean and
    # standard deviation calculations.
    datamodule.transforms.preprocessing.normalize_waveform = False
    datamodule.transforms.preprocessing.normalize_spectrogram = False

    # Set the augmentations in the config file to None, since we do not want augmentations for the mean and
    # standard deviation calculations.
    datamodule.transforms.waveform_augmentations = []
    datamodule.transforms.spectrogram_augmentations = []

    datamodule.prepare_data()

    datamodule.setup(stage="fit")

    mean, std = calculate_mean_std_from_dataloader(
        dataloader=datamodule.train_dataloader()
    )

    log.info(f"Mean: {mean} | Standard deviation: {std}")


if __name__ == "__main__":
    calculate_normalization_parameters()
