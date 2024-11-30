import logging
from pytorch_lightning.utilities import rank_zero_only
from pytorch_lightning import loggers
from typing import Dict
from module_configs import NetworkConfig


def get_pylogger(name=__name__):
    # command line logger
    logger = logging.getLogger(name)

    logging_levels = (
        "debug",
        "info",
        "warning",
        "error",
        "exception",
        "fatal",
        "critical",
    )

    for level in logging_levels:
        setattr(logger, level, rank_zero_only(getattr(logger, level)))

    return logger


class TBLogger(loggers.TensorBoardLogger):
    @rank_zero_only
    def log_hyperparams(
        self, params: Dict[str, any], metrics: Dict[str, any] | None = None
    ):
        if isinstance(params, dict):
            try:
                network : NetworkConfig = params.pop("network")
                params["model_name"] = network.model_name
                params["model_type"] = network.model_type
                params["torch_compile"] = network.torch_compile
                params["sampling_rate"] = network.sample_rate
                params["normalize_waverform"] = network.normalize_waveform
                params["normalize_spectrogram"] = network.normalize_spectrogram
            except KeyError:
                network = params["module"].pop("network")
                network.pop("model")
                params.update(network)
        return super().log_hyperparams(params, metrics)
