import hydra
from omegaconf import DictConfig
from torchmetrics import Metric, MetricCollection


def load_metrics(metrics_cfg: DictConfig):

    main_metric = hydra.utils.instantiate(metrics_cfg.main)
    val_metric_best = hydra.utils.instantiate(metrics_cfg.val_best)

    additional_metrics = {}
    if metrics_cfg.get("additional"):
        for name, metric in metrics_cfg.additional.items():
            additional_metrics[name] = hydra.utils.instantiate(metric)
    
    eval_metrics_complete = {}
    if metrics_cfg.get("eval_complete"):
        for name, metric in metrics_cfg.eval_complete.items():
            eval_metrics_complete[name] = hydra.utils.instantiate(metric)
    
    metrics = {
        "main_metric": main_metric, 
        "val_metric_best": val_metric_best,
        "add_metrics": MetricCollection(additional_metrics),
        "eval_complete": MetricCollection(eval_metrics_complete)
    }

    return metrics