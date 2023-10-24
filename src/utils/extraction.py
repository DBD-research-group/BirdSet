from dataclasses import dataclass
from typing import Any
from transformers import BatchFeature
from transformers import SequenceFeatureExtractor
from transformers.utils import PaddingStrategy
import numpy as np 

# we could incorporate some kind of event detector in the customfeatureextractor

class CustomFeatureExtractor(SequenceFeatureExtractor):
    model_input_names = ["input_values", "attention_mask"]

    def __init__(
        self,
        feature_size=1,
        sampling_rate=32_000,
        padding_value=0.0,
        return_attention_mask=False,
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

@dataclass
class CustomCollatorWithPadding():

    feature_extractor: Any
    padding: bool = True
    truncation: bool = True
    max_length: int = None
    return_tensors: str = "pt"
    preprocessed: bool=False

    def __call__(self, batch):

        # preprocessed means that the .map function was applied
        # here, the feature extractor is only used for padding
        if self.preprocessed:
            batch = self.feature_extractor.pad(
                batch,
                padding=self.padding,
                max_length=self.max_length,
                truncation=self.truncation,
                return_tensors=self.return_tensors,
                return_attention_mask = None
            )

        # here, we first have to format the input and then pad it
        # note that everything regarding resampling is not implemented here    
        else:
            audio_arrays = [x["audio"]["array"] for x in batch]
            labels = [x["primary"] for x in batch]

            # batch feature is just a dictionary
            encoded_inputs = BatchFeature({"input_values": audio_arrays})
            batch = {**encoded_inputs, "labels": labels}

            batch = self.feature_extractor.pad(
                batch,
                padding=self.padding,
                max_length=self.max_length,
                truncation=self.truncation,
                return_tensors=self.return_tensors,
                return_attention_mask = None
            )

        if "label" in batch: 
            batch["labels"] = batch["label"]
            del batch["label"]

        if "target" in batch:
            batch["labels"] = batch["target"]
            del batch["target"]
        
        if "primary" in batch:
            batch["labels"] = batch["primary"]
            del batch["primary"]

        
        return batch