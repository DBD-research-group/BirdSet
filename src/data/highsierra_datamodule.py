from .hf_datamodule import HFDataModule

class HighSierra(HFDataModule):
    def __init__(
            self,
            data_dir,
            dataset_name,
            feature_extractor_name,
            hf_path,
            hf_name,
            seed, 
            train_batch_size, 
            eval_batch_size, 
            val_split,
            column_list=None, 
            transforms=None
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
            transforms=transforms
        )

    @property
    def num_classes(self):
        return 21

