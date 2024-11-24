from birdset.datamodule.components.transforms import BirdSetTransformsWrapper
from birdset.datamodule.embedding_datamodule import (
    EmbeddingDataModule,
    EmbeddingModuleConfig,
)
from birdset.datamodule.beans_datamodule import BEANSDataModule
from birdset.configs import DatasetConfig, LoadersConfig
from birdset.utils import pylogger

log = pylogger.get_pylogger(__name__)

detection_sets = [
    "beans_dcase",
    "beans_enabirds",
    "beans_hiceas",
    "beans_rfcx",
    "beans_gibbons",
]


class BEANSEmbeddingDataModule(EmbeddingDataModule, BEANSDataModule):
    def __init__(
        self,
        dataset: DatasetConfig = DatasetConfig(),
        loaders: LoadersConfig = LoadersConfig(),
        transforms: BirdSetTransformsWrapper = BirdSetTransformsWrapper(),
        k_samples: int = 0,
        val_batches: int = None,  # Should val set be created
        test_ratio: float = 0.5,  # Ratio of test set if val set is also created
        low_train: bool = False,  # If low train set is used
        embedding_model: EmbeddingModuleConfig = EmbeddingModuleConfig(),
        average: bool = True,
        gpu_to_use: int = 0,
    ):
        """
        DataModule for using BEANS and extracting embeddings.

        Args:
            dataset (DatasetConfig, optional): The config of the dataset to use. Defaults to DatasetConfig().
            loaders (LoadersConfig, optional): Loader config. Defaults to LoadersConfig().
            transforms (BirdSetTransformsWrapper, optional): uses EmbeddingsTransform so no Embeddings are cut off. Defaults to EmbeddingTransforms().
            k_samples (int, optional): The amount of samples per class that should be used. Defaults to 0 where the predefined sets are used.
            val_batches (int, optional): If a validation set should be used or not. Defaults to None which means that the normal validation split or amount is used.
            test_ratio (float, optional): Ratio of test set if val set is also created. Defaults to 0.5.
            low_train (bool, optional): If low train set is used (Exists in BEANS for example). Defaults to False.
            embedding_model (EmbeddingModuleConfig, optional): Model for extracting the embeddings. Defaults to EmbeddingModuleConfig().
            average (bool, optional): If embeddings should be averaged if the audio clip is too long. Defaults to True.
            gpu_to_use (int, optional): Which GPU should be used for extracting the embeddings. Defaults to 0.
        """
        BEANSDataModule.__init__(
            self, dataset=dataset, loaders=loaders, transforms=transforms
        )
        EmbeddingDataModule.__init__(
            self,
            dataset=dataset,
            loaders=loaders,
            transforms=transforms,
            k_samples=k_samples,
            val_batches=val_batches,
            test_ratio=test_ratio,
            low_train=low_train,
            embedding_model=embedding_model,
            average=average,
            gpu_to_use=gpu_to_use,
        )
