from .base_datamodule import BaseDataModule

class SapsuckerWoods(BaseDataModule):
    def __init__(
            self, 
            data_dir,
            dataset_name, 
            feature_extractor_name, 
            dataset_loading,
            seed,
            train_batch_size,
            eval_batch_size,
            val_split,
            column_list,
            transforms=None
    ):
        super().__init__(
              data_dir,
              dataset_name, 
              feature_extractor_name,
              dataset_loading,
              seed,
              train_batch_size,
              eval_batch_size,
              val_split,
              column_list=column_list,
              transforms=transforms
        )

    @property
    def num_classes(self):
        return 500
    




