from datasets import Dataset
import soundfile as sf
from resources.utils.few_shot.conditions.base_condition import BaseCondition

class LenientCondition(BaseCondition):

    def __call__(self, dataset: Dataset, idx: int, **kwds):
        """
        This condition allows files up to 10s but only if one bird occurence is in the file.
        """
        file_info = sf.info(dataset[idx]["filepath"])
        if file_info.duration <= 10 and (not dataset[idx]["ebird_code_secondary"]):
            return True