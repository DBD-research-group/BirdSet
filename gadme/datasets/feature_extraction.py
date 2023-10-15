from transformers import BatchFeature
from transformers import SequenceFeatureExtractor
from transformers.utils import PaddingStrategy
import numpy as np 

#we could incorporate some kind of event detector in the customfeatureextractor
#TODO: feature extractor has to be model dependent
class CustomFeatureExtractor(SequenceFeatureExtractor):
    model_input_names = ["input_values", "attention_mask"]

    def __init__(
        self,
        feature_size=1,
        sampling_rate=32_000,
        padding_value=0.0,
        return_attention_mask=True,
        do_normalize=True,
        **kwargs
    ):
        # initialize sequencefeatureextractor
        super().__init__(feature_size=feature_size, sampling_rate=sampling_rate, padding_value=padding_value)
        self.return_attention_mask = return_attention_mask
        self.do_normalize = do_normalize
    
    @staticmethod
    def normalize(
        input_values,
        attention_mask,
        padding_value=0.0
    ):
        if attention_mask is not None:
            attention_mask = np.array(attention_mask, np.int32)
            normed_input_values = []

            for vector, length in zip(input_values, attention_mask.sum(-1)):
                normed_slice = (vector - vector[:length].mean()) / np.sqrt(vector[:length].var() + 1e-7)
                if length < normed_slice.shape[0]:
                    normed_slice[length:] = padding_value

                normed_input_values.append(normed_slice)
        else:
            normed_input_values = [(x-x.mean()) / np.sqrt(x.var() + 1e-7) for x in input_values]
        return normed_input_values
    
    def __call__(
        self, 
        raw_audio,
        padding = False, 
        max_length = None,
        truncation = False, 
        return_tensors = None,
        sampling_rate = None,
        return_attention_mask = None,
        **kwargs
    ) -> BatchFeature:
        
        # control/check sampling rate 
        if self.sampling_rate is not None: 
            if sampling_rate != self.sampling_rate:
                raise ValueError(
                    f"The model corresponding to this feature extractor: {self} was trained using a sampling rate of"
                    f"{self.sampling_rate}. Make sure that the provided `raw_audio`input was sampled with"
                    f"{self.sampling_rate} and not {sampling_rate}."
                )
        else:
            print( "It is strongly recommended to pass the ``sampling_rate`` argument to this function. \
                    Failing to do so can result in silent errors that might be hard to debug.")
        # check batch input
        is_batched_numpy = isinstance(raw_audio, np.ndarray) and len(raw_audio.shape) > 1
        is_batched = is_batched_numpy or (
            isinstance(raw_audio, (list, tuple)) and (isinstance(raw_audio[0], (np.ndarray, tuple, list)))
        )

        if not is_batched:
            raw_audio = [raw_audio]

        encoded_inputs = BatchFeature({"input_values": raw_audio})

        padded_inputs = self.pad(
            encoded_inputs,
            padding=padding,
            max_length=max_length,
            return_attention_mask=return_attention_mask,
            truncation=truncation
        )

        #convert into format 
        input_values = padded_inputs["input_values"]
        if not isinstance(input_values[0], np.ndarray):
            padded_inputs["input_values"] = [np.asarray(array, dtype=np.float32) for array in input_values]
        elif (
            not isinstance(input_values, np.ndarray)
            and isinstance(input_values[0], np.ndarray)
            and input_values[0].dtype is np.dtype(np.float64)
        ):
            padded_inputs["input_values"] = [array.astype(np.float32) for array in input_values]
        elif isinstance(input_values, np.ndarray) and input_values.dtype is np.dtype(np.float64):
            padded_inputs["input_values"] = input_values.astype(np.float32)
        # return_to_tensors comes from: transformers/src/transformers/feature_extraction_utils.py
        if return_tensors is not None:
            padded_inputs = padded_inputs.convert_to_tensors(return_tensors)
        
        attention_mask = padded_inputs.get("attention_mask")
        if attention_mask is not None: 
            padded_inputs["attention_mask"] = [np.asarray(array, dtype=np.int32) for array in attention_mask]
        
        if self.do_normalize:
            attention_mask = (
                attention_mask
                if self._get_padding_strategies(padding, max_length=max_length) is not PaddingStrategy.DO_NOT_PAD
                else None
            )
            padded_inputs["input_values"] = self.normalize(
                padded_inputs["input_values"], attention_mask=attention_mask, padding_value=self.padding_value
            )
  
        return padded_inputs 