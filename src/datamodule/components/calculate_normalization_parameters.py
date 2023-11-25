import os
import rootutils
import hydra
import lightning as L
from omegaconf import OmegaConf
from src import utils
from src.datamodule.components.normalization import NormalizationWrapper

log = utils.get_pylogger(__name__)
rootutils.setup_root(__file__, indicator=".project-root", pythonpath=True)

@hydra.main(version_base=None, config_path="../../../configs", config_name="main")
def calculate_normalization_parameters(cfg):
    log.info('Using config: \n%s', OmegaConf.to_yaml(cfg))

    log.info(f"Dataset path: <{os.path.abspath(cfg.paths.dataset_path)}>")
    os.makedirs(cfg.paths.dataset_path, exist_ok=True)

    log.info(f"Log path: <{os.path.abspath(cfg.paths.log_dir)}>")
    os.makedirs(cfg.paths.log_dir, exist_ok=True)

    log.info(f"Seed everything with <{cfg.seed}>")
    L.seed_everything(cfg.seed)
    log.info(f"Instantiate logger {[loggers for loggers in cfg['logger']]}")

    # Setup data
    log.info(f"Instantiate datamodule <{cfg.datamodule._target_}>")
    datamodule = hydra.utils.instantiate(cfg.datamodule, _recursive_=False)

    log.info(f"normalize: {datamodule.transforms_config.preprocessing.normalize}")
    log.info(f"wave augmentations: {datamodule.transforms_config.waveform_augmentations}")
    log.info(f"spec augmentations: {datamodule.transforms_config.spectrogram_augmentations}")


    # Set the normalization in the config file to False, since we do not want normalized data for the mean and standard deviation calculations.
    datamodule.transforms_config.preprocessing.normalize = False

    log.info(f"normalize: {datamodule.transforms_config.preprocessing.normalize}")

    # Set the augmentations in the config file to None, since we do not want augmentations for the mean and standard deviation calculations.
    datamodule.transforms_config.waveform_augmentations = None
    datamodule.transforms_config.spectrogram_augmentations = None

    log.info(f"wave augmentations: {datamodule.transforms_config.waveform_augmentations}")
    log.info(f"spec augmentations: {datamodule.transforms_config.spectrogram_augmentations}")


    datamodule.prepare_data()

    datamodule.setup(stage="fit")

    normalizer = NormalizationWrapper(config=datamodule.transforms_config)

    mean, std = normalizer.calculate_mean_std_from_dataloader(dataloader=datamodule.train_dataloader)

    log.info(f"Mean: {mean} | Standard deviation: {std}")

if __name__ == "__main__":
    calculate_normalization_parameters()