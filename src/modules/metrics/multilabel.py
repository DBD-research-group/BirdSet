import torch
import torchmetrics
from torchmetrics.classification.average_precision import MultilabelAveragePrecision
from torchmetrics import Metric
from torchmetrics.utilities.data import dim_zero_cat

class cmAP5(Metric):
    def __init__(
            self,
            num_labels: int,
            sample_threshold: int,
            thresholds=None,
            dist_sync_on_step=False
        ):
        super().__init__(dist_sync_on_step=dist_sync_on_step)

        self.num_labels = num_labels
        self.sample_threshold = sample_threshold
        self.thresholds = thresholds

        self.multilabel_ap = MultilabelAveragePrecision(
            average='macro',
            num_labels=self.num_labels,
            thresholds=self.thresholds
        )

        # State variable to accumulate predictions and labels across batches
        self.add_state("accumulated_predictions", default=[], dist_reduce_fx="cat")
        self.add_state("accumulated_labels", default=[], dist_reduce_fx="cat")

    def update(self, logits: torch.Tensor, labels: torch.Tensor):
        # Accumulate predictions and labels
        self.accumulated_predictions.append(logits)
        self.accumulated_labels.append(labels)

    def compute(self) -> torch.Tensor:
        # Ensure that accumulated variables are lists
        if not isinstance(self.accumulated_predictions, list):
            self.accumulated_predictions = [self.accumulated_predictions]
        if not isinstance(self.accumulated_labels, list):
            self.accumulated_labels = [self.accumulated_labels]

        # Concatenate accumulated predictions and labels along the batch dimension
        all_predictions = torch.cat(self.accumulated_predictions, dim=0)
        all_labels = torch.cat(self.accumulated_labels, dim=0)

        # self.accumulated_predictions.clear()
        # self.accumulated_labels.clear()

        # Calculate class-wise AP
        class_aps = self.multilabel_ap(all_predictions, all_labels)

        if self.sample_threshold > 1:
            mask = all_labels.sum(axis=0) >= self.sample_threshold
            class_aps = torch.where(mask, class_aps, torch.nan)

        # Compute macro AP by taking the mean of class-wise APs, ignoring NaNs
        macro_cmap = torch.nanmean(class_aps)
        return macro_cmap

    # def reset(self):
    #     # Reset accumulated predictions and labels
    #     self.accumulated_predictions = []
    #     self.accumulated_labels = []

class cmAP(MultilabelAveragePrecision):
    def __init__(
            self,
            num_labels,
            thresholds=None
        ):
        super().__init__(
            num_labels=num_labels,
            average="macro",
            thresholds=thresholds
        )

    def __call__(self, logits, labels):
        macro_cmap = super().__call__(logits, labels)
        return macro_cmap   

class mAP(MultilabelAveragePrecision):
    
    def __init__(
            self,
            num_labels,
            thresholds=None
        ):
        super().__init__(
            num_labels=num_labels,
            average="micro",
            thresholds=thresholds
        )

    def __call__(self, logits, labels):
        micro_cmap = super().__call__(logits, labels)
        return micro_cmap

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
        pcmap = super().__call__(logits, targets, **kwargs)
        return pcmap


# class TopKAccuracy(torchmetrics.Metric):
#     def __init__(self, topk=1, **kwargs):
#         super().__init__(**kwargs)
#         self.topk = topk
#         self.add_state("correct", default=torch.tensor(0), dist_reduce_fx="sum")
#         self.add_state("total", default=torch.tensor(0), dist_reduce_fx="sum")
#     def update(self, preds, targets):
#         # Get the top-k predictions
#         _, topk_pred_indices = preds.topk(self.topk, dim=1, largest=True, sorted=True)
#         targets = targets.to(topk_pred_indices.device)
        
#         # Convert one-hot encoded targets to class indices
#         target_indices = targets.argmax(dim=1)
        
#         # Compare each of the top-k indices with the target index
#         correct = topk_pred_indices.eq(target_indices.unsqueeze(1)).any(dim=1)

#         # Update correct and total
#         self.correct += correct.sum()
#         self.total += targets.size(0)

#     def compute(self):
#         return self.correct.float() / self.total
    
# class TopKAccuracy(torchmetrics.Metric):
#     def __init__(self, topk=1, **kwargs):
#         super().__init__(**kwargs)
#         self.topk = topk
#         self.add_state("correct", default=torch.tensor(0), dist_reduce_fx="sum")
#         self.add_state("total", default=torch.tensor(0), dist_reduce_fx="sum")
        
#     def update(self, preds, targets):
#         # top-k predictions
#         _, topk_pred_indices = preds.topk(self.topk, dim=1, largest=True, sorted=True)
#         targets = targets.to(topk_pred_indices.device)
        
#         #expand targets to match the shape of topk_pred_indices for broadcasting
#         expanded_targets = targets.unsqueeze(1).expand(-1, self.topk, -1)
        
#         #check if any of the top-k predictions match the true labels
#         correct = expanded_targets.gather(2, topk_pred_indices.unsqueeze(-1)).any(dim=1)
        
#         self.correct += correct.sum()
#         self.total += targets.size(0)

#     def compute(self):
#         return self.correct.float() / self.total



class TopKAccuracy(torchmetrics.Metric):
    def __init__(self, topk=1, include_nocalls=False, threshold=0.5, **kwargs):
        super().__init__(**kwargs)
        self.topk = topk
        self.include_nocalls = include_nocalls
        self.threshold = threshold
        self.add_state("correct", default=torch.tensor(0), dist_reduce_fx="sum")
        self.add_state("total", default=torch.tensor(0), dist_reduce_fx="sum")
        
    def update(self, preds, targets):
        # Get the top-k predictions
        _, topk_pred_indices = preds.topk(self.topk, dim=1, largest=True, sorted=True)
        targets = targets.to(preds.device)
        no_call_targets = targets.sum(dim=1) == 0

        #consider no_call instances (a threshold is needed here!)
        if self.include_nocalls:
            #check if top-k predictions for all-negative instances are less than threshold 
            no_positive_predictions = preds.topk(self.topk, dim=1, largest=True).values < self.threshold
            correct_all_negative = (no_call_targets & no_positive_predictions.all(dim=1))

        else:
            #no_calls are removed, set to 0
            correct_all_negative = torch.tensor(0).to(targets.device)

        #convert one-hot encoded targets to class indices for positive cases
        expanded_targets = targets.unsqueeze(1).expand(-1, self.topk, -1)
        correct_positive = expanded_targets.gather(2, topk_pred_indices.unsqueeze(-1)).any(dim=1)
        
        #update correct and total, excluding all-negative instances if specified
        self.correct += correct_positive.sum() + correct_all_negative.sum()
        if not self.include_nocalls:
            self.total += targets.size(0) - no_call_targets.sum()
        else:
            self.total += targets.size(0)

    def compute(self):
        return self.correct.float() / self.total