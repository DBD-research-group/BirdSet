from .hf_datamodule import HFDataModule

from .hf_datamodule import HFDataModule

class ESC50(HFDataModule):
    def __init__(
            self,
            data_dir,
            dataset_name,
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
        return 50

    def _create_splits(self, dataset):
        split_1 = dataset["train"].train_test_split(self.val_split, shuffle=True, seed=self.seed)
        split_2 = split_1["test"].train_test_split(0.5, shuffle=False, seed=self.seed)
        train_dataset = split_2["train"]
        val_dataset = split_2["train"]
        test_dataset = split_2["test"]
        return train_dataset, val_dataset, test_dataset
