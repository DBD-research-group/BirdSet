import torch
from .base_module import BaseModule

class MultilabelModule(BaseModule):
    def __init__(
            self,
            network, 
            output_activation,
            loss,
            optimizer,
            lr_scheduler,
            metrics, 
            logging_params,
            num_epochs,
            len_trainset,
            task
    ):
        super().__init__(
            network = network,
            output_activation=output_activation,
            loss=loss,
            optimizer=optimizer,
            lr_scheduler=lr_scheduler,
            metrics=metrics,
            logging_params=logging_params,
            num_epochs=num_epochs,
            len_trainset=len_trainset,
            task=task
        )

    def test_step(self, batch, batch_idx):
        test_loss, preds, targets = self.model_step(batch, batch_idx)

        #save targets and predictions for test_epoch_end
        self.test_targets.append(targets.detach().cpu())
        self.test_preds.append(preds.detach().cpu())

        self.log(
            f"loss/test",
            test_loss, 
            on_step=False,
            on_epoch=True,
            prog_bar=True
        )

        self.test_metric(preds, targets.int())
        self.log(
            f"{self.test_metric.__class__.__name__}/test",
            self.test_metric,
            **self.logging_params,
        )

        self.test_add_metrics(preds, targets.int())
        self.log_dict(self.test_add_metrics, **self.logging_params)

        return {"loss": test_loss, "preds": preds, "targets": targets}
    
    def on_test_epoch_end(self):
        test_targets = torch.cat(self.test_targets).int()
        test_preds = torch.cat(self.test_preds)
        self.test_complete_metrics(test_preds, test_targets)

        log_dict = {}

        # Rename cmap to cmap5!
        for metric_name, metric in self.test_complete_metrics.named_children():
            # Check for padding_factor attribute
            if hasattr(metric, 'padding_factor') and metric.padding_factor == 5:
                modified_name = 'cmAP5'
            else:
                modified_name = metric_name
            log_dict[f"{modified_name}/test_complete"] = metric

        self.log_dict(log_dict, **self.logging_params)

