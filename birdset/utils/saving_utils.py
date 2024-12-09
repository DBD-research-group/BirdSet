import csv
import json
from collections import OrderedDict
from pathlib import Path
from typing import Any, List, Optional, Union

import torch
from pytorch_lightning import LightningModule, Trainer

from birdset.utils import pylogger

log = pylogger.get_pylogger(__name__)


def save_state_dicts(trainer, model, dirname, symbols, exceptions=None):

    mapped_state_dict = process_state_dict(
        model.state_dict(), symbols=symbols, exceptions=exceptions
    )
    path = f"{dirname}/last_state_dict.pth"
    torch.save(mapped_state_dict, path)
    log.info(f"Last ckpt state dict saved to: {path}")

    best_ckpt_path = trainer.checkpoint_callback.best_model_path
    if best_ckpt_path == "":
        log.warning("Best ckpt not found!")
        return
    try:
        best_ckpt_score = trainer.checkpoint_callback.best_model_score
        if best_ckpt_score is not None:
            prefix = str(best_ckpt_score.detach().cpu().item())
            prefix = prefix.replace(".", "_")
        else:
            log.warning("Best ckpt score not found! Use prefix <unknown>!")
            prefix = "unknown"
        # load best model and save it
        model = model.load_from_checkpoint(best_ckpt_path)
        mapped_state_dict = process_state_dict(
            model.state_dict(), symbols=symbols, exceptions=exceptions
        )
        path = f"{dirname}/best_ckpt_{prefix}.pth"
        torch.save(mapped_state_dict, path)
        log.info(f"Best ckpt state dict saved to: {path}")
    except:
        log.info(f"Best Model loading did not work")


def process_state_dict(state_dict, symbols, exceptions):

    new_state_dict = OrderedDict()
    if exceptions:
        if isinstance(exceptions, str):
            exceptions = [exceptions]
    for key, value in state_dict.items():
        is_exception = False
        if exceptions:
            for exception in exceptions:
                if key.startswith(exception):
                    is_exception = True
        if not is_exception:
            new_state_dict[key[symbols:]] = value

    return new_state_dict
