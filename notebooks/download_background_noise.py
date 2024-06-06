from huggingface_hub import hf_hub_download
import hydra
from pydub import AudioSegment
import tarfile
import os
import pyrootutils
from birdset import utils

log = utils.get_pylogger(__name__)

root = pyrootutils.setup_root(
    search_from=__file__,
    indicator=[".git"],
    pythonpath=True,
    dotenv=True,
)

def ogg2wav(filepath, output_dir):
    audio = AudioSegment.from_ogg(filepath)
    # get filename without extension
    filename = os.path.splitext(os.path.basename(filepath))[0]
    # save wav file
    audio.export(os.path.join(output_dir, filename + ".wav"), format="wav")


_HYDRA_PARAMS = {
    "version_base":None,
    "config_path": str(root / "configs"),
    "config_name": "background_data_download.yaml"
}

@hydra.main(**_HYDRA_PARAMS)
def main(cfg):
    repo_id = "DBD-research-group/BirdSet" 
    filenames = ["dcase18_shard_0001.tar.gz", "dcase18_shard_0002.tar.gz", "dcase18_shard_0003.tar.gz"]
    subfolder = "dcase18"
    revision = "data"
    repo_type = "dataset"

    download_dir = cfg.paths.background_path
    unpacking_dir = download_dir + "/dcase18_unpacked/"
    output_dir = download_dir + "/dcase18_wav"

    log.info("Downloading Files")

    # Download the files
    for filename in filenames:
        hf_hub_download(
            repo_id=repo_id,
            filename=filename,
            subfolder=subfolder,
            local_dir=download_dir,
            revision=revision,
            repo_type=repo_type)
        
    log.info("Files Downloaded")
    log.info("Extracting Files")
        
    if not os.path.exists(unpacking_dir):
        os.makedirs(unpacking_dir)

    # Iterate through the files and extract each one
    for filename in filenames:
        with tarfile.open(os.path.join(download_dir, subfolder + '/', filename), "r:gz") as tar:
            tar.extractall(path=unpacking_dir)

    log.info("Files Extracted")
    log.info("Converting .ogg Files to .wav")

    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    # Iterate through the files and convert each one
    for dirpath, dirs, files in os.walk(unpacking_dir):
        for file in files:
            if file.endswith('.ogg') and not os.path.exists(output_dir + f"/{file[:-3]}" + "wav"):
                ogg2wav(os.path.join(dirpath, file), output_dir)
    
    log.info("Done Converting")


if __name__ == "__main__":
    main()
 