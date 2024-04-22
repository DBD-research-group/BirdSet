import torch 
import warnings
import torchmetrics
from torchmetrics.classification import MulticlassAUROC, MulticlassCalibrationError, MulticlassAveragePrecision

class MulticlassMetricsConfig:
    """
    A class for configuring multiclass metrics used during model training and evaluation.

    Attributes:
        main_metric (Metric): The main metric used for model training.
        val_metric_best (Metric): The metric used for model validation.
        add_metrics (MetricCollection): A collection of additional metrics used during model training.
        eval_complete (MetricCollection): A collection of metrics used during model evaluation.
    """

    def __init__(
        self,
        num_labels: int = 21,
    ):
        """
        Initializes the MetricsConfig class.

        Args:
            num_labels (int): The number of labels in the dataset. Defaults to 21 as in the HSN dataset.
        """
        self.main_metric: torchmetrics.Metric = torchmetrics.Accuracy(
            task="multiclass",
            num_classes=num_labels,
        )
        self.val_metric_best: torchmetrics.Metric =torchmetrics.MaxMetric()
     
        self.add_metrics: torchmetrics.MetricCollection = torchmetrics.MetricCollection({
            'F1': torchmetrics.F1Score(
                task="multiclass",
                num_classes=num_labels,
            ),
            'MulticlassAUROC': MulticlassAUROC(
                num_classes=num_labels,
                average='macro',
                thresholds=None
            ),
            'ECE': MulticlassCalibrationError(
                num_classes=num_labels,
                n_bins=10,
                norm='l1'
            ),
            'MulticlassAUPR': MulticlassAveragePrecision(
                num_classes=num_labels,
                average='macro',
                thresholds=None
            )
        })
        self.eval_complete: torchmetrics.MetricCollection = torchmetrics.MetricCollection({
            'acc': torchmetrics.Accuracy(
            task="multiclass",
            num_classes=num_labels,
        )
        })

class MulticlassMetricsConfig:
    """
    A class for configuring multiclass metrics used during model training and evaluation.

    Attributes:
        main_metric (Metric): The main metric used for model training.
        val_metric_best (Metric): The metric used for model validation.
        add_metrics (MetricCollection): A collection of additional metrics used during model training.
        eval_complete (MetricCollection): A collection of metrics used during model evaluation.
    """

    def __init__(
        self,
        num_labels: int = 21,
    ):
        """
        Initializes the MetricsConfig class.

        Args:
            num_labels (int): The number of labels in the dataset. Defaults to 21 as in the HSN dataset.
        """
        self.main_metric: torchmetrics.Metric = torchmetrics.Accuracy(
            task="multiclass",
            num_classes=num_labels,
        )
        self.val_metric_best: torchmetrics.Metric =torchmetrics.MaxMetric()
     
        self.add_metrics: torchmetrics.MetricCollection = torchmetrics.MetricCollection({
            'F1': torchmetrics.F1Score(
                task="multiclass",
                num_classes=num_labels,
            ),
        })
        self.eval_complete: torchmetrics.MetricCollection = torchmetrics.MetricCollection({
            'acc': torchmetrics.Accuracy(
            task="multiclass",
            num_classes=num_labels,
        )
        })

class BalancedAccuracy(torchmetrics.Metric):
    def __init__(self, adjusted=False, num_classes=None):
        super().__init__(dist_sync_on_step=False)
        self.adjusted = adjusted
        self.num_classes = num_classes
        self.add_state("conf_matrix", default=torch.zeros(num_classes, num_classes), dist_reduce_fx="sum")

    def update(self, y_pred: torch.Tensor, y_true: torch.Tensor):
        preds = torch.argmax(y_pred, dim=1)
        self.conf_matrix += torch.histc(
            self.num_classes * preds.float() + y_true.float(),
            bins=self.num_classes**2,
            min=0,
            max=self.num_classes**2-1
        ).view(self.num_classes, self.num_classes)

    def compute(self):
        with torch.no_grad():
            per_class_recall = torch.diag(self.conf_matrix) / self.conf_matrix.sum(dim=1)
            per_class_recall = per_class_recall[~torch.isnan(per_class_recall)]
            score = torch.mean(per_class_recall)

            if self.adjusted:
                if self.num_classes is None:
                    warnings.warn("Number of classes should be specified for adjusted score.")
                    return score
                chance = 1 / self.num_classes
                score -= chance
                score /= 1 - chance

            return score     

class BalancedAccuracyTop5(torchmetrics.Metric):
    def __init__(self, adjusted=False, num_classes=None):
        super().__init__(dist_sync_on_step=False)
        self.adjusted = adjusted
        self.num_classes = num_classes
        self.add_state("conf_matrix", default=torch.zeros(num_classes, num_classes), dist_reduce_fx="sum")
        self.add_state("top5_correct", default=torch.tensor(0), dist_reduce_fx="sum")
        self.add_state("total", default=torch.tensor(0), dist_reduce_fx="sum")

    def update(self, y_pred: torch.Tensor, y_true: torch.Tensor):
        # Update for balanced accuracy
        preds = torch.argmax(y_pred, dim=1)
        self.conf_matrix += torch.histc(
            self.num_classes * preds.float() + y_true.float(),
            bins=self.num_classes**2,
            min=0,
            max=self.num_classes**2-1
        ).view(self.num_classes, self.num_classes)

        # Update for top-5 accuracy
        _, top5_preds = y_pred.topk(5, dim=1)
        correct = top5_preds.eq(y_true.view(-1, 1).expand_as(top5_preds))
        self.top5_correct += correct.sum()
        self.total += y_true.numel()

    def compute(self):
        with torch.no_grad():
            # Balanced accuracy computation
            per_class_recall = torch.diag(self.conf_matrix) / self.conf_matrix.sum(dim=1)
            per_class_recall = per_class_recall[~torch.isnan(per_class_recall)]
            balanced_accuracy = torch.mean(per_class_recall)

            if self.adjusted:
                if self.num_classes is None:
                    warnings.warn("Number of classes should be specified for adjusted score.")
                    return balanced_accuracy
                chance = 1 / self.num_classes
                balanced_accuracy -= chance
                balanced_accuracy /= 1 - chance

            # Top-5 accuracy computation
            top5_accuracy = self.top5_correct.float() / self.total

            return top5_accuracy
