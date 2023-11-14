import hydra
from omegaconf import DictConfig
from torchmetrics import Metric, MetricCollection


def load_metrics(metrics_cfg: DictConfig):

    main_metric = hydra.utils.instantiate(metrics_cfg.main)
    val_metric_best = hydra.utils.instantiate(metrics_cfg.val_best)

    additional_metrics = []
    if metrics_cfg.get("additional"):
        for _, metric in metrics_cfg.additional.items():
            additional_metrics.append(hydra.utils.instantiate(metric))
    
    metrics = {
        "main_metric": main_metric, 
        "val_metric_best": val_metric_best,
        "add_metrics": MetricCollection(additional_metrics)
    }

    return metrics