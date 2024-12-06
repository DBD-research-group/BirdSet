import torch


class AudioPreprocessor:
    def __init__(self, feature_extractor, n_classes, window_length):
        self.feature_extractor = feature_extractor
        self.window_length = window_length
        self.n_classes = n_classes

    def preprocess_train(self, batch):
        audio_arrays = [x["array"] for x in batch["audio"]]

        inputs = self.feature_extractor(
            audio_arrays,
            sampling_rate=self.feature_extractor.sampling_rate,
            padding=True,
            max_length=self.feature_extractor.sampling_rate * self.window_length,
            truncation=True,
            return_tensors="pt",
        )
        return inputs

    def preprocess_multilabel(self, batch):
        audio_arrays = [x["array"] for x in batch["audio"]]
        label_list = [y for y in batch["ebird_code_multilabel"]]

        output_dict = self.feature_extractor(
            audio_arrays,
            sampling_rate=self.feature_extractor.sampling_rate,
            padding=True,
            max_length=self.feature_extractor.sampling_rate * self.window_length,
            truncation=True,
            return_tensors="pt",
        )
        labels = self._classes_one_hot(label_list)
        output_dict["labels"] = labels
        return output_dict

    def _classes_one_hot(self, class_indices):
        class_one_hot_matrix = torch.zeros(
            (len(class_indices), self.n_classes), dtype=torch.float
        )
        for class_idx, idx in enumerate(class_indices):
            class_one_hot_matrix[class_idx, idx] = 1
        return class_one_hot_matrix
