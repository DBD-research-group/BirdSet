from .hf_datamodule import HFDataModule

from .hf_datamodule import HFDataModule

class ESC50(HFDataModule):
    def __init__(
            self,
            data_dir = "data/",
            dataset_name = "esc50",
            feature_extractor_name = None,
            hf_path = "ashraq/esc50",
            hf_name = None,
            seed = 12345, 
            train_batch_size = 64, 
            eval_batch_size = 64, 
            val_split = 0.2,
            column_list= ["input_values", "target"], 
            transforms=None,
            n_classes = 50,
            num_workers = 0
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
        self.n_classes = 50

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
