from .base_datamodule import BaseDataModule

def SapsuckerWoods(BaseDataModule):
    def __init__(self, dataset_path, val_split, seed):
        super().__init__("SapsuckerWoods", dataset_path, val_split, seed)

    @property
    def num_classes(self):
        return 500
    

    
