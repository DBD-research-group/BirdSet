from datasets import Dataset

class BaseCondition:

    def __call__(self, dataset: Dataset, idx:int , **kwds) -> bool:
        return True