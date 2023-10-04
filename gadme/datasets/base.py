from datasets import load_dataset
import lightning as L

class BaseGADME(L.LightningDataModule):
    def __init__(self, dataset_name, dataset_path):
        super().__init__()
        self.dataset_name = dataset_name
        self.dataset_path = dataset_path

        print("> Loading data set...")
        self._download_datasets()
    
    def prepare_data(self):
        print("> Loading data set...")
        load_dataset("gadme_v1_1", self.dataset_name, cache_dir=self.dataset_path)
    
    def setup(self, stage):
        self.dataset = load_dataset("gadme_v1_1", self.dataset_name)
        self.dataset = self.dataset.cast_column(
            column="audio",
            feature=Audio(
                sampling_rate=32_000,
                mono=True,
                decode=True
        )





