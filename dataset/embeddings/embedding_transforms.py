from src.datamodule.components.transforms import BaseTransforms


class EmbeddingTransforms(BaseTransforms):
    def __init__(self, other: BaseTransforms = BaseTransforms()) -> None:
        super().__init__(other.task, other.sampling_rate, other.max_length, other.event_decoder, other.feature_extractor)
    
    def set_task(self, task):
        self.task = task
    
    def transform_labels(self, batch):
        return batch["labels"]