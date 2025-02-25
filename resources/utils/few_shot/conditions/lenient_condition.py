from datasets import Dataset
import soundfile as sf
from resources.utils.few_shot.conditions.condition_template import ConditionTemplate

class LenientCondition(ConditionTemplate):

    def __call__(self, dataset: Dataset, idx: int, **kwds):
        """
        This condition allows files up to 10s but only if one bird occurence is in the file.
        """
        file_info = sf.info(dataset[idx]["filepath"])
        if file_info.duration <= 10 and (not dataset[idx]["ebird_code_secondary"]) and len(dataset[idx]["ebird_code_multilabel"]) == 1:
            return True