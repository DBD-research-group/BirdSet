from .build_dataset import BaseGADME

def SapsuckerWoods(BaseGADME):
    def __init__(self, dataset_path, val_split, seed):
        super().__init__("sapsucker_woods", dataset_path, val_split, seed)

    @property
    def num_classes(self):
        return 10
    
