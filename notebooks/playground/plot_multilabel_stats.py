import matplotlib.pyplot as plt
from datasets import DatasetDict


def get_unique_multilabel_labels(multilabel_labels: list) -> list:
    """
    Generates a list of unique labels in the given label list
    """
    unique_labels = []
    for sample_labels in multilabel_labels:
        for label in sample_labels:
            if label not in unique_labels:
                unique_labels.append(label)
    return unique_labels


def count_multilabel_occurences(multilabel_labels: list) -> dict:
    """
    Generates a dict with labels as keys and the occurence count of the labels as values.
    """
    label_occurences = {}
    for sample_codes in multilabel_labels:
        for code in sample_codes:
            label_occurences[code] = label_occurences.get(code, 0) + 1
    return label_occurences


def get_multilabel_stats(multilabel_labels: list) -> tuple[list, dict]:
    """
    Generates a list of unique labels in the given label list and counts their occurences
    """
    unique_labels = []
    label_occurences = {}
    for sample_labels in multilabel_labels:
        for label in sample_labels:
            if label not in unique_labels:
                unique_labels.append(label)
            label_occurences[label] = label_occurences.get(label, 0) + 1
    return unique_labels, label_occurences


def show_multilabel_stats(dataset_dict: DatasetDict, figwidth: int = 10):
    """
    Plot unique labels and their occurences as absolute values.
    Plots both the train and test set of the given dataset dict.
    """
    train_set = dataset_dict["train"]
    test_set = dataset_dict["test"]

    unique_train_labels, train_label_occurences = get_multilabel_stats(
        train_set["ebird_code_multilabel"]
    )
    unique_test_labels, test_label_occurences = get_multilabel_stats(
        test_set["ebird_code_multilabel"]
    )

    unique_labels_in_dataset = set(unique_train_labels).union(set(unique_test_labels))

    for label in unique_labels_in_dataset:
        # fill missing labels with occurence count of 0
        train_label_occurences[label] = train_label_occurences.get(label, 0)
        test_label_occurences[label] = test_label_occurences.get(label, 0)

    print(f"Unique labels in dataset: {len(unique_labels_in_dataset)}")
    print(
        f"Unique labels in train set vs. test set:  {len(unique_train_labels)} vs. {len(unique_test_labels)}"
    )

    label_occurences = sorted(train_label_occurences.items())
    labels, train_label_counts = zip(*label_occurences)
    label_occurences = sorted(test_label_occurences.items())
    labels, test_label_counts = zip(*label_occurences)

    fig, ax = plt.subplots()
    fig.set_figwidth(figwidth)
    width = 0.8

    ax.set_xticks(labels, labels=labels)

    ax.bar(labels, train_label_counts, label="train_set", width=width)
    ax.bar(
        labels,
        test_label_counts,
        label="test_set",
        width=width,
        bottom=train_label_counts,
    )
    ax.set_xlabel("Bird Code")
    ax.set_ylabel("Occurence count")
    ax.legend()
    plt.show()


def show_multilabel_percentages(dataset_dict: DatasetDict, figwidth: int = 10):
    """
    Plot unique labels and their occurences as percentage based values.
    Plots both the train and test set of the given dataset dict.
    """
    train_set = dataset_dict["train"]
    test_set = dataset_dict["test"]

    unique_train_labels, train_label_occurences = get_multilabel_stats(
        train_set["ebird_code_multilabel"]
    )
    unique_test_labels, test_label_occurences = get_multilabel_stats(
        test_set["ebird_code_multilabel"]
    )

    unique_labels_in_dataset = set(unique_train_labels).union(set(unique_test_labels))

    for label in unique_labels_in_dataset:
        # fill missing labels with occurence count of 0
        train_label_occurences[label] = train_label_occurences.get(label, 0)
        test_label_occurences[label] = test_label_occurences.get(label, 0)

    for label in unique_labels_in_dataset:
        occurence_count_sum = (
            train_label_occurences[label] + test_label_occurences[label]
        )
        train_label_occurences[label] = (
            train_label_occurences[label] / occurence_count_sum
        )
        test_label_occurences[label] = (
            test_label_occurences[label] / occurence_count_sum
        )

    label_occurences = sorted(train_label_occurences.items())
    labels, train_label_counts = zip(*label_occurences)
    label_occurences = sorted(test_label_occurences.items())
    labels, test_label_counts = zip(*label_occurences)

    fig, ax = plt.subplots()
    fig.set_figwidth(figwidth)
    width = 0.8

    ax.set_xticks(labels, labels=labels)

    ax.bar(labels, train_label_counts, label="train_set", width=width)
    ax.bar(
        labels,
        test_label_counts,
        label="test_set",
        width=width,
        bottom=train_label_counts,
    )
    ax.set_xlabel("Bird Code")
    ax.set_ylabel("Percentage per set")
    ax.legend()
    plt.show()
