from huggingface_hub import login
import torch
from transformers import (
    AutoModelForAudioClassification,
    ConvNextForImageClassification,
    EfficientNetForImageClassification,
)


# TODO fill out
hf_write_access_token: str = ""  # a hf write access token
cache_dir: str | None = (
    None  # cache dir from paths.yaml, else model will be downloaded into hf default cache dir
)
state_dict_path: str = "path/to/checkpoint.ckpt"
hf_repo_name: str = (
    "{model name and version}-Birdset-{hf_name}"  # format: "{model name and version}-Birdset-{hf_name}"
)
hf_base_model: str = "facebook/wav2vec2-base"
# num_classes: int = 9736  # XCL
num_classes = 411  # XCM
num_channels = 1  # Required for vision models only.


def main():
    login(token=hf_write_access_token)
    if not torch.cuda.is_available():
        state_dict = torch.load(state_dict_path, map_location=torch.device("cpu"))[
            "state_dict"
        ]
    else:
        state_dict = torch.load(state_dict_path)["state_dict"]

    adjusted_state_dict = {}
    for key, value in state_dict.items():
        # Handle 'model.model.' prefix
        new_key = key.replace("model.model.", "")

        # Handle 'model._orig_mod.model.' prefix
        new_key = new_key.replace("model._orig_mod.model.", "")

        # Assign the adjusted key
        adjusted_state_dict[new_key] = value

    # Determine which model to use based on hf_base_model
    if "efficientnet" in hf_base_model.lower():
        model = EfficientNetForImageClassification.from_pretrained(
            hf_base_model,
            num_labels=num_classes,
            num_channels=num_channels,
            cache_dir=cache_dir,
            state_dict=adjusted_state_dict,
            ignore_mismatched_sizes=True,
        )
    elif "convnext" in hf_base_model.lower():
        model = ConvNextForImageClassification.from_pretrained(
            hf_base_model,
            num_labels=num_classes,
            num_channels=num_channels,
            cache_dir=cache_dir,
            state_dict=adjusted_state_dict,
            ignore_mismatched_sizes=True,
        )
    else:
        model = AutoModelForAudioClassification.from_pretrained(
            hf_base_model,
            cache_dir=cache_dir,
            state_dict=adjusted_state_dict,
            num_labels=num_classes,
        )

    model.push_to_hub(f"DBD-research-group/{hf_repo_name}")


if __name__ == "__main__":
    main()
