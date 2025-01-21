import torch
import warnings
import torchmetrics

class BalancedAccuracy(torchmetrics.Metric):
    def __init__(self, adjusted=False, num_classes=None):
        super().__init__(dist_sync_on_step=False)
        self.adjusted = adjusted
        self.num_classes = num_classes
        self.add_state(
            "conf_matrix",
            default=torch.zeros(num_classes, num_classes),
            dist_reduce_fx="sum",
        )

    def update(self, y_pred: torch.Tensor, y_true: torch.Tensor):
        preds = torch.argmax(y_pred, dim=1)
        self.conf_matrix += torch.histc(
            self.num_classes * preds.float() + y_true.float(),
            bins=self.num_classes**2,
            min=0,
            max=self.num_classes**2 - 1,
        ).view(self.num_classes, self.num_classes)

    def compute(self):
        with torch.no_grad():
            per_class_recall = torch.diag(self.conf_matrix) / self.conf_matrix.sum(
                dim=1
            )
            per_class_recall = per_class_recall[~torch.isnan(per_class_recall)]
            score = torch.mean(per_class_recall)

            if self.adjusted:
                if self.num_classes is None:
                    warnings.warn(
                        "Number of classes should be specified for adjusted score."
                    )
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
        self.add_state(
            "conf_matrix",
            default=torch.zeros(num_classes, num_classes),
            dist_reduce_fx="sum",
        )
        self.add_state("top5_correct", default=torch.tensor(0), dist_reduce_fx="sum")
        self.add_state("total", default=torch.tensor(0), dist_reduce_fx="sum")

    def update(self, y_pred: torch.Tensor, y_true: torch.Tensor):
        # Update for balanced accuracy
        preds = torch.argmax(y_pred, dim=1)
        self.conf_matrix += torch.histc(
            self.num_classes * preds.float() + y_true.float(),
            bins=self.num_classes**2,
            min=0,
            max=self.num_classes**2 - 1,
        ).view(self.num_classes, self.num_classes)

        # Update for top-5 accuracy
        _, top5_preds = y_pred.topk(5, dim=1)
        correct = top5_preds.eq(y_true.view(-1, 1).expand_as(top5_preds))
        self.top5_correct += correct.sum()
        self.total += y_true.numel()

    def compute(self):
        with torch.no_grad():
            # Balanced accuracy computation
            per_class_recall = torch.diag(self.conf_matrix) / self.conf_matrix.sum(
                dim=1
            )
            per_class_recall = per_class_recall[~torch.isnan(per_class_recall)]
            balanced_accuracy = torch.mean(per_class_recall)

            if self.adjusted:
                if self.num_classes is None:
                    warnings.warn(
                        "Number of classes should be specified for adjusted score."
                    )
                    return balanced_accuracy
                chance = 1 / self.num_classes
                balanced_accuracy -= chance
                balanced_accuracy /= 1 - chance

            # Top-5 accuracy computation
            top5_accuracy = self.top5_correct.float() / self.total

            return top5_accuracy
