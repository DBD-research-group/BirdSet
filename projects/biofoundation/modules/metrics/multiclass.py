import torch
import warnings
import torchmetrics


class MulticlassMetricsConfig:
    """
    A class for configuring multiclass metrics used during model training and evaluation.

    Attributes:
        main_metric (Metric): The main metric used for model training.
        val_metric_best (Metric): The metric used for model validation.
        add_metrics (MetricCollection): A collection of additional metrics used during model training.
        eval_complete (MetricCollection): A collection of metrics used during model evaluation.
    """

    def __init__(self, num_labels: int = 21):
        """
        Initializes the MetricsConfig class.

        Args:
            num_labels (int): The number of labels in the dataset. Defaults to 21 as in the HSN dataset.
        """
        self.main_metric: torchmetrics.Metric = torchmetrics.Accuracy(
            task="multiclass",
            num_classes=num_labels,
        )
        self.val_metric_best: torchmetrics.Metric = torchmetrics.MaxMetric()

        self.add_metrics: torchmetrics.MetricCollection = torchmetrics.MetricCollection(
            {
                "F1": torchmetrics.F1Score(
                    task="multiclass",
                    num_classes=num_labels,
                ),
            }
        )
        self.eval_complete: torchmetrics.MetricCollection = (
            torchmetrics.MetricCollection(
                {
                    "acc": torchmetrics.Accuracy(
                        task="multiclass",
                        num_classes=num_labels,
                    )
                }
            )
        )

class EmbeddingMetricsConfig(MulticlassMetricsConfig):
    """
    A class for embedding metrics used during model training and evaluation.

    Attributes:
        main_metric (Metric): The main metric used for model training.
        val_metric_best (Metric): The metric used for model validation.
        add_metrics (MetricCollection): A collection of additional metrics used during model training.
        eval_complete (MetricCollection): A collection of metrics used during model evaluation.
    """

    def __init__(self, num_labels: int = 21):
        """
        Initializes the MetricsConfig class.

        Args:
            num_labels (int): The number of labels in the dataset. Defaults to 21 as in the HSN dataset.
        """
        self.main_metric: torchmetrics.Metric = torchmetrics.Accuracy(
            task="multiclass", num_classes=num_labels, top_k=1
        )
        self.val_metric_best: torchmetrics.Metric = torchmetrics.MaxMetric()

        self.add_metrics: torchmetrics.MetricCollection = torchmetrics.MetricCollection(
            {
                "T1Accuracy": torchmetrics.Accuracy(
                    task="multiclass",
                    num_classes=num_labels,
                    # average='macro',
                    top_k=1,
                ),
                "T3Accuracy": torchmetrics.Accuracy(
                    task="multiclass",
                    num_classes=num_labels,
                    # average='macro',
                    top_k=3,
                ),
                "AUROC": torchmetrics.AUROC(
                    task="multiclass",
                    num_classes=num_labels,
                    average="macro",
                ),
                "F1": torchmetrics.F1Score(
                    task="multiclass",
                    num_classes=num_labels,
                ),
            }
        )
        self.eval_complete: torchmetrics.MetricCollection = (
            torchmetrics.MetricCollection(
                {  # Only used in multilabel module
                    "acc": torchmetrics.Accuracy(
                        task="multiclass", num_classes=num_labels, top_k=1
                    )
                }
            )
        )


class AUROCMetricWrapper(torchmetrics.AUROC):
    def update(self, preds: torch.Tensor, target: torch.Tensor) -> None:
        # Convert one-hot encoded target labels to class labels
        class_labels = torch.argmax(target, dim=1)
        # Call the original update method with class labels
        super().update(preds, class_labels)