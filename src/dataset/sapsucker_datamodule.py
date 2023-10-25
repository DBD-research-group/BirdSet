from .base_datamodule import BaseDataModule

class SapsuckerWoods(BaseDataModule):
    def __init__(
            self,
            data_dir="/data",
            dataset_name="sapsucker",
            feature_extractor_name=None,
            hf_path="DBD-research-group/gadme_v1_1",
            hf_name="NA_subset500",
            seed=1, 
            train_batch_size=12, 
            eval_batch_size=48, 
            val_split=0.2,
            column_list=["input_values", "bird_codes"], 
            transforms=None,
            n_classes=500,
            num_workers=1
    ):
        super().__init__(
            data_dir,
            dataset_name,
            feature_extractor_name,
            hf_path,
            hf_name,
            seed, 
            train_batch_size, 
            eval_batch_size, 
            val_split,
            column_list=column_list, 
            transforms=transforms,
            num_workers=num_workers
        )
        self.n_classes = n_classes

    @property
    def num_classes(self):
        return 500

