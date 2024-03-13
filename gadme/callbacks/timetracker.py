import time
import lightning as L
import logging 
logger = logging.getLogger(__name__)

class TimeCallback(L.Callback):
    """Callback that measures the training and testing time of a PyTorch Lightning module."""
    def __init__(self):
        self.start: float
        self.num_images: int = 0

    def on_fit_start(self, trainer, pl_module):
        self.start = time.time()
    
    def on_fit_end(self, trainer, pl_module):
        logger.info("Training/Val took %5.2f seconds", (time.time() - self.start))

    def on_test_start(self, trainer, pl_module):
        self.start = time.time()

    def on_test_end(self, trainer, pl_module):
        logger.info("Testing took %5.2f seconds", (time.time() - self.start))

        