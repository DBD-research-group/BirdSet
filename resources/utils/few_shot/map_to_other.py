from datasets import load_from_disk, DatasetDict
from pathlib import Path
import json
import os

def map_dataset_to_other(
    few_shot_dataset: DatasetDict,
    embedded_dataset_path: str,
    pre_filter_cache_file: Path=Path(),
    save_dir: str="") -> DatasetDict:

    def get_filename(path: str):
        return path.split("/")[-1]

    print("Loading Embedded Dataset")
    embedded_dataset = load_from_disk(embedded_dataset_path)
    few_shot_samples = few_shot_dataset["train"]
    indeces = [-1] * len(few_shot_samples)

    samples_per_file = {}
    loaded = False
    if not isinstance(pre_filter_cache_file, Path):
        pre_filter_cache_file = Path(pre_filter_cache_file)
    if pre_filter_cache_file != Path():
        try:
            with open(pre_filter_cache_file, 'r') as file:
                samples_per_file = json.load(file)
                print("Loaded Pre-Filter cache file")
                loaded = True
        except Exception as e:
            print(f"Pre-Filter cache does not exist yet or couldn't be loaded: {e}")

    if not samples_per_file:
        print("Pre-Filtering Embedded Data")
        samples_per_file = {}
        for sample in embedded_dataset["train"]:
            file = get_filename(sample["filepath"])
            if file not in samples_per_file.keys():
                samples_per_file[file] = []
            samples_per_file[file].append(sample["index"])

    if (pre_filter_cache_file != Path()) and not loaded:
        try:
            directory = pre_filter_cache_file.parent
            if not directory.exists():
                os.makedirs(directory)
            with open(pre_filter_cache_file, 'w') as file:
                json.dump(samples_per_file, file)
                print(f"Saved pre-filter cache to {str(pre_filter_cache_file)}")
        except Exception as e:
            print(f"Pre-Filter cache couldn't be saved: {e}")
    
    print("Matching Samples")
    for i, few_shot_sample in enumerate(few_shot_samples):
        fitting_samples = samples_per_file[get_filename(few_shot_sample["filepath"])]
        for sample_idx in fitting_samples:
            embedded_sample = embedded_dataset["train"][sample_idx] 
            if (round(embedded_sample["detected_events"][0], 2) == round(few_shot_sample["detected_events"][0], 2)
                and round(embedded_sample["detected_events"][1], 2) == round(few_shot_sample["detected_events"][1], 2)):
                indeces[i] = sample_idx
                break
        if indeces[i] == -1:
            print(f"No matching event could be found for sample {i}: {few_shot_sample}")
            return
                
    indexed_set = few_shot_samples.add_column("audio", indeces)

    print("Adding Samples")
    mapped_samples = indexed_set.map(lambda batch: {"audio": embedded_dataset["train"].select(batch["audio"])["audio"]}, batched=True)
    mapped_samples.set_format("pt", columns=["audio"], output_all_columns=True)
    mapped_dataset = DatasetDict({"train": mapped_samples, "test": few_shot_dataset["test"]})

    if save_dir:
        print("Saving Dataset")
        mapped_dataset.save_to_disk(save_dir)

    return  mapped_dataset
    