import logging
from pytorch_lightning.utilities import rank_zero_only


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
        "critical"
    )

    for level in logging_levels: 
        setattr(logger, level, rank_zero_only(getattr(logger, level)))

    return logger