import torch
import torchmetrics
from torchmetrics.classification.average_precision import MultilabelAveragePrecision

class cmAP(MultilabelAveragePrecision):
    # implementation from:
    # https://github.com/google-research/perch/blob/b71e1d65989c2b359a473b0f0476c08c1fbce528/chirp/models/metrics.py#L39C10-L39C10
    # sample_threshold: Only classes with at least this many samples will be used
    #   in the calculation of the final metric. By default this is 1, which means
    #   that classes without any positive examples will be ignored.
    def __init__(
            self, 
            num_labels: int, 
            sample_threshold: int = 5,
            average: str = "macro",
            thresholds=None,
            **kwargs):
        
        super().__init__(
            num_labels=num_labels,
            average=average,
            thresholds=thresholds,
            **kwargs)
        
        self.sample_threshold = sample_threshold

    def __call__(self, logits, labels, **kwargs):
        class_ap = super().__call__(logits, labels, **kwargs)
        mask = labels.sum(axis=0) > self.sample_threshold
        cmap = class_ap.where(mask, torch.nan)
        cmap = cmap.nanmean()
        return cmap


class pcmAP(MultilabelAveragePrecision):
    # https://www.kaggle.com/competitions/birdclef-2023/overview/evaluation
    def __init__(
            self, 
            num_labels: int, 
            padding_factor: int = 5,
            average: str = "macro",
            thresholds=None,
            **kwargs):
        
        super().__init__(
            num_labels=num_labels,
            average=average,
            thresholds=thresholds,
            **kwargs)
        
        self.padding_factor = padding_factor

    def __call__(self, logits, targets, **kwargs):
        ones = torch.ones(self.padding_factor, logits.shape[1]) # solve cuda!
        logits = torch.cat((logits, ones), dim=0)
        targets = torch.cat((targets, ones.int()), dim=0)
        cmap5 = super().__call__(logits, targets, **kwargs)
        return cmap5

class T1Accuracy(torchmetrics.Metric):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.add_state("correct", default=torch.tensor(0), dist_reduce_fx="sum")
        self.add_state("total", default=torch.tensor(0), dist_reduce_fx="sum")

    def update(self, preds, targets):
        top_pred_indices = preds.argmax(dim=1)
        # search if top_pred_indices = 1 in targets
        target_top_pred = targets[torch.arange(targets.size(0)), top_pred_indices]
        # sum up the correct predictions
        self.correct += torch.sum(target_top_pred)
        self.total += targets.size(0)

    def compute(self):
        return self.correct.float() / self.total

