from dataclasses import dataclass
from typing import Any
from transformers import BatchFeature

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