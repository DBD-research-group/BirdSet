from transformers import AutoModelForAudioClassification
from huggingface_hub import login
import torch

# TODO fill out
hf_write_access_token: str = ""  # a hf write access token
cache_dir: str | None = None  # cache dir from paths.yaml, else model will be downloaded into hf default cache dir
state_dict_path: str = "path/to/checkpoint.ckpt"
hf_repo_name: str = "{model name and version}-Birdset-{dataset_name}"  # format: "{model name and version}-Birdset-{dataset_name}"
hf_base_model: str = "facebook/wav2vec2-base"
#num_classes: int = 9736  # XCL
num_classes = 411  # XCM


def main():
    login(token=hf_write_access_token)
    if not torch.cuda.is_available():
        state_dict = torch.load(state_dict_path, map_location=torch.device('cpu'))["state_dict"]
    else:
        state_dict = torch.load(state_dict_path)["state_dict"]
    state_dict = {key.replace('model.model.', ''): weight for key, weight in state_dict.items()}

    model = AutoModelForAudioClassification.from_pretrained(
        hf_base_model,
        cache_dir=cache_dir,
        state_dict=state_dict,
        num_labels=num_classes)

    model.push_to_hub(f"DBD-research-group/{hf_repo_name}")

if __name__ == "__main__":
    main()
