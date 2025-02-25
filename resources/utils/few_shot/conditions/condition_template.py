from datasets import Dataset

class ConditionTemplate:

    def __call__(self, dataset: Dataset, idx:int , **kwds) -> bool:
        return True