import torch
from torchmetrics.classification.average_precision import MultilabelAveragePrecision

class cMAP(MultilabelAveragePrecision):
    # implementation from:
    # https://github.com/google-research/perch/blob/b71e1d65989c2b359a473b0f0476c08c1fbce528/chirp/models/metrics.py#L39C10-L39C10
    # sample_threshold: Only classes with at least this many samples will be used
    #   in the calculation of the final metric. By default this is 1, which means
    #   that classes without any positive examples will be ignored.
    def __init__(self, num_labels: int, sample_threshold: int = 0, **kwargs):
        super().__init__(
            num_labels=num_labels,
            average="macro",
            thresholds=None,
            **kwargs)
        
        self.sample_threshold = sample_threshold

    def __call__(self, logits, labels, **kwargs):
        class_ap = super().__call__(logits, labels, **kwargs)
        mask = labels.sum(axis=0) > self.sample_threshold
        cmap = class_ap.where(mask, torch.nan)
        cmap = cmap.nanmean()
        return cmap


class cMAP5(MultilabelAveragePrecision):
    # https://www.kaggle.com/competitions/birdclef-2023/overview/evaluation
    def __init__(self, num_labels: int, padding_factor: int = 5, **kwargs):
        super().__init__(
            num_labels=num_labels,
            average="macro",
            thresholds=None,
            **kwargs)
        self.padding_factor = padding_factor

    def __call__(self, logits, targets, **kwargs):
        ones = torch.ones(self.padding_factor, logits.shape[1]).to("cuda") # solve!
        logits = torch.cat((logits, ones), dim=0)
        targets = torch.cat((targets, ones.int()), dim=0)
        cmap5 = super().__call__(logits, targets, **kwargs)
        return cmap5