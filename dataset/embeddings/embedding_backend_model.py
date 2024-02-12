from chirp.inference import models
from torch import from_numpy as torch_from_numpy

class EmbedModel():
    def __init__(self, model_name, num_classes, embed_dim, instance) -> None:
        self.model_name = model_name
        self.num_classes = num_classes
        self.sample_rate = embed_dim
        self.instance = instance
    
    def __call__(self, input_values, **kwds: models.Any):
        inference = self.instance(input_values)
        embeddings = inference.embeddings
        embeddings = self.transform_embeddings(embeddings)
        return torch_from_numpy(embeddings)
    
    def transform_embeddings(self, embeddings):
        return embeddings

class DownloadEmbedModel(EmbedModel):
    def __init__(self, model_key, model_name, num_classes, embed_dim, config) -> None:
        model_class = models.model_class_map()[model_key]
        wrapper = model_class.from_config(config)
        instance = wrapper.batch_embed
        super().__init__(model_name, num_classes, embed_dim, instance)
        

class AverageEmbedModel(EmbedModel):
    def __init__(self, instance, model_name, num_classes, embed_dim) -> None:
        super().__init__(model_name, num_classes, embed_dim, instance)
    
    def transform_embeddings(self, embeddings):
        embeddings = embeddings.mean(axis=1)
        return super().transform_embeddings(embeddings)

class Perch(DownloadEmbedModel):
    def __init__(self, model_key, model_name, num_classes, embed_dim, config) -> None:
        super().__init__(model_key, model_name, num_classes, embed_dim, config)

class BirdNet(DownloadEmbedModel):
    def __init__(self, model_key, model_name, num_classes, embed_dim, config) -> None:
        super().__init__(model_key, model_name, num_classes, embed_dim, config)

class Yamnet(AverageEmbedModel):
    def __init__(self, model_key, model_name, num_classes, embed_dim, config) -> None:
        model_class = models.model_class_map()[model_key]
        instance = model_class.yamnet()
        super().__init__(instance, model_name, num_classes, embed_dim)

class VGGish(AverageEmbedModel):
    def __init__(self, model_key, model_name, num_classes, embed_dim, config) -> None:
        model_class = models.model_class_map()[model_key]
        instance = model_class.vggish()
        super().__init__(instance, model_name, num_classes, embed_dim)