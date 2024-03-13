from pytorch_lightning import Callback

class MetricCollector(Callback):
    def __init__(self):
        super().__init__()
        self.metrics_history = []

    def on_epoch_end(self, trainer, pl_module):
        # Copy the current epoch's metrics
        current_metrics = trainer.callback_metrics.copy()
        # Append the copied metrics to the history list
        self.metrics_history.append(current_metrics)