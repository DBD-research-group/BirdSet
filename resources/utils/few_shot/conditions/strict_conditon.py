from datasets import Dataset
import soundfile as sf
from resources.utils.few_shot.conditions.base_condition import BaseCondition

class StrictCondition(BaseCondition):

    def __call__(self, dataset: Dataset, idx: int, **kwds):
        """
        This condition only allows files that up to 5s long so that no event detection has to occur when sampling.
        """
        file_info = sf.info(dataset[idx]["filepath"])
        if file_info.duration <= 5:
            return True