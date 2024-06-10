import subprocess
import pyrootutils

root = pyrootutils.setup_root(
    search_from=__file__,
    indicator=[".git"],
    pythonpath=True,
    dotenv=True,
)


def prepare_background_noise():
    """downloads, extracts, and afterwords removes files for background nose files form HF"""
    import hydra
    import tarfile
    import os

    with hydra.initialize(version_base="1.3", config_path="../configs/"):
        cfg = hydra.compose(config_name="train.yaml",
                            return_hydra_config=True,
                            overrides=["experiment=local/DT_example.yaml"])
    bg_path = cfg.paths.background_path

    os.makedirs(bg_path, exist_ok=True)

    if bg_path and os.listdir(bg_path):
        from huggingface_hub import hf_hub_download

        filenames = ["dcase18_shard_0001.tar.gz", "dcase18_shard_0002.tar.gz"]

        for filename in filenames:
            _ = hf_hub_download(
                repo_id="DBD-research-group/BirdSet",
                filename=filename,
                subfolder="data/dcase18",
                local_dir=bg_path,
                revision="data",
                repo_type="dataset",
                local_dir_use_symlinks=False,
                force_download=True)

        for filename in filenames:
            filepath = os.path.join(bg_path, filename)
            with tarfile.open(filepath, "r:gz") as tar:
                tar.extractall(path=bg_path)
            os.remove(filepath)

        print("Downloaded, extracted, and deleted tar files.")
        return

    print("background_path in config was not empty, skipping noise prepare.")


def main():
    prepare_background_noise()
    subprocess.run("python " + str(root / "birdset/train.py") + " experiment=\"local/DT_example.yaml\"", shell=True)    


if __name__ == "__main__":
    main()
