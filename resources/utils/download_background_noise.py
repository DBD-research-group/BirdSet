from huggingface_hub import hf_hub_download
import tarfile
import os


def main():

    repo_id = "DBD-research-group/BirdSet"  # Replace with the repository ID
    filenames = ["dcase18_shard_0001.tar.gz",
                 "dcase18_shard_0002.tar.gz"]  # Replace with the path to the file in the repository
    subfolder = "data/dcase18"
    revision = "data"
    local_dir = "/workspace/data_birdset/dcase18"  # Replace with the local directory where the file will be downloaded
    repo_type = "dataset"
    # Download the file
    for filename in filenames:
        file_path = hf_hub_download(
            repo_id=repo_id,
            filename=filename,
            subfolder=subfolder,
            local_dir=local_dir,
            revision=revision,
            repo_type=repo_type,
            local_dir_use_symlinks=False,
            force_download=True)

    os.chdir("/workspace/data_birdset/dcase18/data/dcase18/")

    # The directory where you want to extract the files
    output_directory = "/workspace/data_birdset/background_noise/"

    # Create the output directory if it doesn't exist
    os.makedirs(output_directory, exist_ok=True)

    # Iterate through the files and extract each one
    for filename in filenames:
        with tarfile.open(filename, "r:gz") as tar:
            tar.extractall(path=output_directory)
        os.remove(filename)

    print("Extraction complete. No-Call samples ready.")


if __name__ == "__main__":
    main()
