import torch
from .base_module import BaseModule
import wandb 

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
            batch_size,
            task,
            class_weights_loss,
            label_counts,
            prediction_table,
            num_gpus
    ):
        self.prediction_table = prediction_table

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
            task=task,
            class_weights_loss=class_weights_loss,
            label_counts=label_counts,
            batch_size=batch_size,
            num_gpus=num_gpus
        )

    def test_step(self, batch, batch_idx):
        test_loss, preds, targets = self.model_step(batch, batch_idx)

        #save targets and predictions for test_epoch_end
        self.test_targets.append(targets.detach().cpu())
        self.test_preds.append(preds.detach().cpu())

        self.log(
            f"test/{self.loss.__class__.__name__}",
            test_loss, 
            on_step=False,
            on_epoch=True,
            prog_bar=True
        )

        self.test_metric(preds, targets.int())
        self.log(
            f"test/{self.test_metric.__class__.__name__}",
            self.test_metric,
            **self.logging_params,
        )

        self.test_add_metrics(preds, targets.int())
        self.log_dict(self.test_add_metrics, **self.logging_params)

        return {"loss": test_loss, "preds": preds, "targets": targets}
    

    
    def on_test_epoch_end(self):
        test_targets = torch.cat(self.test_targets).int()
        test_preds = torch.cat(self.test_preds)
        #self.test_complete_metrics(test_preds, test_targets)

        log_dict = {}

        # Rename cmap to cmap5!
        for metric_name, metric in self.test_complete_metrics.named_children():
            value = metric(test_preds, test_targets)
            log_dict[f"test/{metric_name}"] = value

        self.log_dict(log_dict, **self.logging_params)

        if self.prediction_table:
            self._wandb_prediction_table(test_preds, test_targets)


    def _wandb_prediction_table(self, preds, targets):
        top5_values_preds, top5_indices_preds = preds.topk(dim=1, k=5, sorted=True)
        
        top5_values_preds = top5_values_preds.cpu().typesafe_numpy()
        top5_indices_preds = top5_indices_preds.cpu().typesafe_numpy()

        indices_targets= (targets == 1).nonzero(as_tuple=False)
        multilabel_list = [[] for _ in range(targets.size(0))]        
        for idx in indices_targets:
            row, col = idx.tolist()
            multilabel_list[row].append(col)

        # Convert the indices to strings
        top5_indices_preds_str = [', '.join(map(str, indices)) for indices in top5_indices_preds]
        multilabel_list_str = [', '.join(map(str, indices)) for indices in multilabel_list]

        # Prepare the data for the wandb table
        columns = ["Predictions", "Targets"]
        data = [[pred_str, tgt_str] for pred_str, tgt_str in zip(top5_indices_preds_str, multilabel_list_str)]
        table = wandb.Table(data=data, columns=columns)

        # Log the table to wandb
        wandb.log({"Top 5 Predictions vs Targets": table})





