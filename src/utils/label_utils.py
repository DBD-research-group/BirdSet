from typing import Dict
import json 


def get_label_to_class_mapping_from_metadata(
    file_path: str, task: str
) -> Dict[int, str]:
    """
    Reads a JSON file and extracts the mapping of labels to eBird codes.

    The function expects the JSON structure to be in a specific format, where the mapping
    is a list of names located under the keys 'features' -> 'labels' -> 'names'.
    The index in the list corresponds to the label, and the value at that index is the eBird code.

    Args:
    - file_path (str): The path to the JSON file containing the label to eBird code mapping.
    - task (str): The type of task for which to get the mapping. Expected values are "multiclass" or "multilabel".

    Returns:
    - Dict[int, str]: A dictionary where each key is a label (integer) and the corresponding value is the eBird code.

    Raises:
    - FileNotFoundError: If the file at `file_path` does not exist.
    - json.JSONDecodeError: If the file is not a valid JSON.
    - KeyError: If the expected keys ('features', 'labels', 'names') are not found in the JSON structure.
    """

    # Open the file and read the JSON data
    with open(file_path, "r") as file:
        dataset_info = json.load(file)

        # Extract the list of eBird codes from the loaded JSON structure.
        # Note: This assumes a specific structure of the JSON data.
        # If the structure is different, this line will raise a KeyError.
        if task == "multiclass":
            ebird_codes_list = dataset_info["features"]["labels"]["names"]
        elif task == "multilabel":
            ebird_codes_list = dataset_info["features"]["labels"]["feature"]["names"]
        else:
            # If the task is not recognized (not multiclass or multilabel), raise an error.
            raise NotImplementedError(
                f"Only the multiclass and multilabel tasks are implemented, not task {task}."
            )

    # Create a dictionary mapping each label (index) to the corresponding eBird code.
    mapping = dict(enumerate(ebird_codes_list))

    return mapping