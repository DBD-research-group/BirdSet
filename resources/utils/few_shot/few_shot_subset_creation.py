from datasets import DatasetDict, Dataset, concatenate_datasets, Sequence, Value
import random
from birdset.datamodule.components.event_mapping import XCEventMapping
from resources.utils.few_shot.conditions.base_condition import BaseCondition
from resources.utils.few_shot.conditions.strict_conditon import StrictCondition

def create_few_shot_subset(dataset: DatasetDict, few_shot: int=5, data_selection_condition: BaseCondition=StrictCondition(), fill_up: bool=False, random_seed: int=None) -> DatasetDict:
    """
    This method creates a subset of the given datasets train split with at max `few_shot` samples per label in the dataset split.
    The samples are chosen based on the given condition. If there are more than `few_shot` samples for a label `few_shot`
    random samples are chosen. If exactly `few_shot` samples per label are wanted, `fill_up` should be set to `True`.
    After the samples that pass the condition are added to the subset, this will randomly fill up the unfullfilled labels
    with their respective samples from the given dataset split without regard for the condition. 
    
    Args:
        dataset (DatasetDict): A Huggingface "datasets.DatasetDict" object. A few-shot subset will be created for the `train` split.
        few_shot (int): The number of samples each label can have. Default is 5.
        data_selection_condition (ConditionTemplate): A condition that defines which recordings should be included in the few-shot subset.
        fill_up (bool): If True, labels for which not enough samples can be extracted with the given condition will be supplemented with
          random samples from the dataset. Default is False.
        random_seed (int): The seed with which the random sampler is seeded. If None, no seeding is applied. Default is None.
    Returns:
        DatasetDict: A Huggingface `datasets.DatasetDict` object where the test split is return as it was given and the train
        split is replaced with the few-shot subset of the given train split.
    """
    if random_seed != None:
        print(f"Set random seed to {random_seed}.")
        random.seed(random_seed)
    train_split = dataset["train"]

    print("Applying condition to training data.")
    satisfying_recording_indeces = []
    for i in range(len(train_split)):
        if data_selection_condition(train_split, i):
            satisfying_recording_indeces.append(i)

    print("Mapping satisfying recordings.")
    all_labels = set(train_split["ebird_code"])
    primary_samples_per_label, leftover_samples_per_label = _map_recordings_to_samples(
        train_split,
        all_labels,
        satisfying_recording_indeces
    )

    print("Selecting samples for subset.")
    selected_samples = []
    unfullfilled_labels = {}
    for label, samples in primary_samples_per_label.items():
        num_primary_samples = len(samples)
        num_leftover_samples = len(leftover_samples_per_label[label])
        if (num_primary_samples + num_leftover_samples) < few_shot:
            selected_samples.extend(samples)
            selected_samples.extend(leftover_samples_per_label[label])
            unfullfilled_labels[label] = few_shot - (num_primary_samples + num_leftover_samples)
        elif num_primary_samples < few_shot:
            selected_samples.extend(samples)
            selected_samples.extend(random.sample(leftover_samples_per_label[label], k=(few_shot - num_primary_samples)))
        else:
            selected_samples.extend(random.sample(samples, few_shot))

    if fill_up:
        print("Filling up labels.")
        unused_recordings = set(range(len(train_split))).difference(satisfying_recording_indeces)
        unused_primary, unused_leftover = _map_recordings_to_samples(
            train_split,
            all_labels,
            unused_recordings
        )

        fill_up_samples = []
        for label, count in unfullfilled_labels.items():
            num_primary_samples = len(unused_primary[label])
            num_leftover_samples = len(unused_leftover[label])
            if num_primary_samples < count:
                fill_up_samples.extend(unused_primary[label])
                # if there are not enough samples in the dataset the min() has to be taken to avoid errors. 
                fill_up_samples.extend(random.sample(unused_leftover[label], k=min((count - num_primary_samples), num_leftover_samples)))
            else:
                fill_up_samples.extend(random.sample(unused_primary[label], count))
        selected_samples.extend(fill_up_samples)

    return DatasetDict({"train": Dataset.from_list(selected_samples), "test": dataset["test_5s"]})
    

def _map_recordings_to_samples(train_split: Dataset, all_labels: set, recording_indeces: list):
    """
    This method uses the XCEventMapping to extract samples from the recordings. It also splits
    the extracted samples into primary and leftover. Every recording has exaclty one primary sample,
    which is chosen randomly. All samples that are not a primary sample for a recording are saved as
    leftover samples.
    """
    mapper = XCEventMapping()
    primary_samples_per_label = {label: [] for label in all_labels}
    leftover_samples_per_label = {label: [] for label in all_labels}
    for idx in recording_indeces:
        mapped_batch = mapper({key: [value] for key, value in train_split[idx].items()})
        # in cases where a recording produces multiple samples, choose one as the main sample 
        # to prioritise the selection of samples from differing recordings.
        num_samples = len(mapped_batch["filepath"])
        primary_sample = random.choice(range(num_samples))
        for i in range(num_samples):
            sample = {key: mapped_batch[key][i] for key in mapped_batch.keys()}
            if i == primary_sample:
                primary_samples_per_label[sample["ebird_code"]].append(sample)
            else:
                leftover_samples_per_label[sample["ebird_code"]].append(sample)
    return primary_samples_per_label, leftover_samples_per_label

# for testing purposes
if __name__ == "__main__":
    from datasets import load_dataset
    from resources.utils.few_shot.conditions.lenient_condition import LenientCondition
    
    dataset = load_dataset(
        path="DBD-research-group/BirdSet",
        name="HSN",
        cache_dir=f"/home/rantjuschin/data_birdset/HSN",
        trust_remote_code=True
    )
    subset_one = create_few_shot_subset(dataset, data_selection_condition=StrictCondition(), fill_up=True)
    #subset_two = create_few_shot_subset(dataset, data_selection_condition=StrictCondition(), fill_up=True)
    print(subset_one)
    #print(subset_two)

