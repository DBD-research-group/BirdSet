from datasets import Dataset, concatenate_datasets
import random
from birdset.datamodule.components.event_mapping import XCEventMapping
from resources.utils.few_shot.conditions.base_condition import BaseCondition
from resources.utils.few_shot.conditions.strict_conditon import StrictCondition

def create_few_shot_subset(dataset: Dataset, few_shot: int=5, data_selection_condition: BaseCondition=StrictCondition(), fill_up: bool=False, random_seed: int=None) -> Dataset:
    """
    This method creates a subset of the given dataset split with at max `few_shot` samples per label in the dataset split.
    The samples are chosen based on the given condition. If there are more than `few_shot` samples for a label `few_shot`
    random samples are chosen. If exactly `few_shot` samples per label are wanted, `fill_up` should be set to `True`.
    After the samples that pass the condition are added to the subset, this will randomly fill up the unfullfilled labels
    with their respective samples from the given dataset split without regard for the condition. 
    
    Args:
        dataset (Dataset): A Huggingface "datasets.Dataset" object. Specificly the split for which a few-shot subset should be created.
        few_shot (int): The number of samples each label can have. Default is 5.
        data_selection_condition (ConditionTemplate): A condition that defines which recordings should be included in the few-shot subset.
        fill_up (bool): If True, labels for which not enough samples can be extracted with the given condition will be supplemented with
          random samples from the dataset. Default is False.
        random_seed (int): The seed with which the random sampler is seeded. If None, no seeding is applied. Default is None.
    Returns:
        Dataset: The few-shot subsetted split.
    """
    if random_seed != None:
        random.seed(random_seed)

    indeces_with_condition = []
    for i in range(len(dataset)):
        if data_selection_condition(dataset, i):
            indeces_with_condition.append(i)

    # some samples have multiple detected events even with base conditions, therefore a mapping is needed
    dataset_with_condition = dataset.select(indeces_with_condition).map(XCEventMapping(), batched=True, batch_size=2)
    all_labels = set(dataset_with_condition["ebird_code"])
    indeces_per_label = {label: [] for label in all_labels}
    for i in range(len(dataset_with_condition)):
       indeces_per_label[dataset_with_condition[i]["ebird_code"]].append(i)

    sampled_indeces = []
    unfullfilled_labels = {}
    for label, indeces in indeces_per_label.items():
        if len(indeces) < few_shot:
            sampled_indeces.extend(indeces)
            unfullfilled_labels[label] = few_shot - len(indeces)
        else:
            sampled_indeces.extend(random.sample(indeces, few_shot))
    few_shot_dataset = dataset_with_condition.select(sampled_indeces)

    if fill_up:
        unused_samples = set(range(len(dataset))).difference(indeces_with_condition)
        unused_data = dataset.select(unused_samples).map(XCEventMapping(), batched=True, batch_size=2)

        indeces_per_label = {label: [] for label in all_labels}
        for i in range(len(unused_data)):
            indeces_per_label[unused_data[i]["ebird_code"]].append(i)

        fill_up_indeces = []
        for label, count in unfullfilled_labels.items():
            # if there are not enough samples in the dataset the min() has to be taken to avoid errors
            fill_up_indeces.extend(random.sample(indeces_per_label[label], min(count, len(indeces_per_label[label]))))
        fill_up_data = unused_data.select(fill_up_indeces)

        few_shot_dataset = concatenate_datasets([few_shot_dataset, fill_up_data])

    return few_shot_dataset
    

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
    train_set = dataset["train"]
    subset_one = create_few_shot_subset(train_set, data_selection_condition=StrictCondition())
    subset_two = create_few_shot_subset(train_set, data_selection_condition=StrictCondition(), fill_up=True)
    print(subset_one)
    print(subset_two)

