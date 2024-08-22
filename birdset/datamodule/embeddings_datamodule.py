from birdset.datamodule.components.transforms import BirdSetTransformsWrapper
from birdset.datamodule.base_datamodule import BaseDataModuleHF
from birdset.configs import NetworkConfig, DatasetConfig, LoadersConfig
from datasets import DatasetDict, Dataset, concatenate_datasets
from collections import defaultdict
from tabulate import tabulate
from birdset.utils import pylogger
from tqdm import tqdm
from dataclasses import dataclass
from typing import Union
from birdset.modules.models.embedding_abstract import EmbeddingModel
import torch
import torchaudio
import os

log = pylogger.get_pylogger(__name__)

@dataclass
class EmbeddingModuleConfig(NetworkConfig):
    """
    A dataclass that makes sure the model inherits from EmbeddingClassifier.

    """
    model: Union[EmbeddingModel, torch.nn.Module] = None # Model for extracting the embeddings

class EmbeddingDataModule(BaseDataModuleHF):
    def __init__(
            self,
            dataset: DatasetConfig = DatasetConfig(),
            loaders: LoadersConfig = LoadersConfig(),
            transforms: BirdSetTransformsWrapper = BirdSetTransformsWrapper(),
            mapper: None = None,
            k_samples: int = 0,
            test_ratio: float = 1,
            embedding_model: EmbeddingModuleConfig = EmbeddingModuleConfig(),
            average: bool = True
    ):
        super().__init__(
            dataset=dataset,
            loaders=loaders,
            transforms=transforms
        )
        self.k_samples = k_samples
        self.test_ratio = test_ratio
        self.average = average
        self.id_to_label = defaultdict(str)
        self.embedding_model_name = embedding_model.model_name
        self.embedding_model = embedding_model.model
        self.sampling_rate = embedding_model.sampling_rate
        self.max_length = embedding_model.length
        print(f"Using embedding model:{embedding_model.model_name} (Sampling Rate:{self.sampling_rate}, Window Size:{self.max_length})")
        

    def _ksamples(self, dataset):
        """
        Use k_samples > 0 if you want control over amount of samples per class. The rest is used for validation and testing.
        """
        if self.k_samples > 0:
            print(f">> Selecting {self.k_samples} Samples per Class this may take a bit...")
            merged_data = concatenate_datasets([dataset['train'], dataset['valid'], dataset['test']])

            # Shuffle the merged data
            merged_data.shuffle() #? Check if this is affected by the public seed
            
            # Create a dictionary to store the selected samples per class
            selected_samples = defaultdict(list)
            train_count = defaultdict(int)
            testval_count = defaultdict(int)
            rest_samples = []
            # Iterate over the merged data and select the desired number of samples per class
            for sample in tqdm(merged_data, total=len(merged_data), desc="Selecting samples"):
                label = sample['labels']
                if len(selected_samples[label]) < self.k_samples:
                    selected_samples[label].append(sample)
                    train_count[label] += 1
                else:
                    rest_samples.append(sample)
                    testval_count[label] += 1    

            
            # Create and print table to show class distribution
            headers = ["Class", "#Train-Samples", "#Test,Valid-Samples"]
            rows = []
            
            for class_id in selected_samples.keys():
                rows.append([self.id_to_label[class_id], train_count[class_id], testval_count[class_id]])
            
            print(tabulate(rows, headers, tablefmt="rounded_grid"))
            
            # Flatten the selected samples into a single list
            selected_samples = [sample for samples in selected_samples.values() for sample in samples]

            # Split the selected samples into training, validation, and testing sets

            if self.test_ratio == 1:
                train_data = selected_samples
                test_data = rest_samples
                val_data = Dataset.from_dict({})
            
            else:    
                num_samples = len(rest_samples)
                num_test_samples = int(self.test_ratio * num_samples)

                train_data = selected_samples
                test_data = rest_samples[:num_test_samples]
                val_data = rest_samples[num_test_samples:]
                val_data = Dataset.from_dict({key: [sample[key] for sample in val_data] for key in val_data[0]})
            
            train_data = Dataset.from_dict({key: [sample[key] for sample in train_data] for key in train_data[0]})
            test_data = Dataset.from_dict({key: [sample[key] for sample in test_data] for key in test_data[0]})

            # Combine into a DatasetDict
            dataset = DatasetDict({
                'train': train_data,
                'valid': val_data,
                'test': test_data
            })
            
        return dataset

    def _compute_embeddings(self, dataset):
        """
        Compute Embeddings for the entire dataset and store them in a new DatasetDict.
        """
        # Define save path
        self.disk_save_path = os.path.join(
            self.dataset_config.data_dir,
            f"{self.dataset_config.dataset_name}_processed_{self.dataset_config.seed}_{self.embedding_model_name}_{self.k_samples}",
        )
        
        # Check if embeddings have to be extracted
        if not os.path.exists(self.disk_save_path):
            
            # Create a new DatasetDict to store the embeddings
            embeddings_dataset = DatasetDict()
            
            #! For some reason the .map() from Datasets caused errors
            # Iterate over each split in the dataset
            for split in dataset.keys():
                print(f">> Extracting Embeddings for {split} Split")
                # Get the current split data
                split_data = dataset[split]

                # Create a list to store the embeddings for the current split
                embeddings = []

                # Iterate over each sample in the split
                for sample in tqdm(split_data, total=len(split_data), desc="Extracting Embeddings"):
                    # Get the embedding for the audio sample
                    embedding = self._get_embedding(sample['audio'])
                    
                    # Add the embedding to the list
                    sample['audio']['array'] = embedding.squeeze(0)
                    embeddings.append(sample)

                # Convert the list of embeddings to a tensor
                #embeddings = torch.stack(embeddings)

                # Create a new Dataset with the embeddings
                if len(embeddings) > 0:
                    embeddings_dataset[split] =  Dataset.from_dict({key: [sample[key] for sample in embeddings] for key in embeddings[0]})
        else:
            # Create empty datasetdict
            embeddings_dataset = DatasetDict({
                'train': Dataset.from_dict({}),
                'test': Dataset.from_dict({}),
                'valid': Dataset.from_dict({})
            })

        return embeddings_dataset
        

    def _get_embedding(self, audio):
        # Get waveform and sampling rate
        waveform = torch.tensor(audio['array'], dtype=torch.float32)
        dataset_sampling_rate = audio['sampling_rate']
        # Resample audio
        audio = self._resample_audio(waveform, dataset_sampling_rate)
        
        # Check if waveform is longer than 100s and cut of rest (Otherwise it takes too long)
        '''if waveform.shape[0] > 30 * self.sampling_rate:
            print("Cutting off rest of waveform:", waveform.shape[0]/self.sampling_rate)
            # Chop off the rest of the waveform
            waveform = waveform[:30 * self.sampling_rate]'''
        
        # Zero-padding
        audio = self._zero_pad(waveform)

        # Check if audio is too long 
        if waveform.shape[0] > self.max_length * self.sampling_rate:
            if self.average:
                return self._frame_and_average(waveform) 
            else:
                audio = audio[:self.max_length * self.sampling_rate]
                return self.embedding_model.get_embeddings(audio.view(1, 1, -1))[0] 
        else:
            return self.embedding_model.get_embeddings(audio.view(1, 1, -1))[0] # To just use embeddings not logits

    # Resample function
    def _resample_audio(self, audio, orig_sr):
        resampler = torchaudio.transforms.Resample(orig_freq=orig_sr, new_freq=self.sampling_rate)
        return resampler(audio)

    # Zero-padding function
    def _zero_pad(self, audio):
        desired_num_samples = self.max_length * self.sampling_rate 
        current_num_samples = audio.shape[0]
        padding = desired_num_samples - current_num_samples
        if padding > 0:
            #print('padding')
            pad_left = padding // 2
            pad_right = padding - pad_left
            audio = torch.nn.functional.pad(audio, (pad_left, pad_right))
        return audio

    # Average multiple embeddings function
    def _frame_and_average(self, audio):
        # Frame the audio
        frame_size = self.max_length * self.sampling_rate
        hop_size = self.max_length * self.sampling_rate
        frames = audio.unfold(0, frame_size, hop_size)
        
        # Generate embeddings for each frame
        l = []
        for frame in frames:
            embedding = self.embedding_model.get_embeddings(frame.view(1, 1, -1)) 
            l.append(embedding[0]) # To just use embeddings not logits
        
        embeddings = torch.stack(tuple(l))
        
        # Average the embeddings
        averaged_embedding = embeddings.mean(dim=0)
        
        return averaged_embedding        
    
    def _save_dataset_to_disk(self, dataset: Dataset | DatasetDict):
        """
        Saves the dataset to disk.

        This method sets the format of the dataset to numpy, prepares the path where the dataset will be saved, and saves
        the dataset to disk. If the dataset already exists on disk, it does not save the dataset again.

        Args:
            dataset (datasets.DatasetDict): The dataset to be saved. The dataset should be a Hugging Face `datasets.DatasetDict` object.

        Returns:
            None
        """
        
        #! Due to randomness somewhere in model,resampling,... the fingerprint isn't the same
        # For now now we add k_samples, model_name  (Potentially k_Samples could also be picked from extracted embeddings already but this is easier and it doesn't take that long)
        #! Manual deleting may be needed if you want to recompute the embeddings
        dataset.set_format("np")
        self.disk_save_path = os.path.join(
            self.dataset_config.data_dir,
            f"{self.dataset_config.dataset_name}_processed_{self.dataset_config.seed}_{self.embedding_model_name}_{self.k_samples}",
        )

        if os.path.exists(self.disk_save_path):
            log.info(f"Train fingerprint found in {self.disk_save_path}, saving to disk is skipped")
        else:
            log.info(f"Saving to disk: {self.disk_save_path}")
            dataset.save_to_disk(self.disk_save_path)