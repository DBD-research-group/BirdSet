import json
import subprocess
import pyrootutils
from pathlib import Path
from birdset import utils

root = pyrootutils.setup_root(
    search_from=__file__,
    indicator=[".git"],
    pythonpath=True,
    dotenv=True,
)

def main():
    subprocess.run("python " + str(root / "birdset/train.py") + " experiment=\"local/MT_example_train.yaml\"", shell=True)

    log_path = str(root / "logs/train/runs/XCM/efficientnet")
    validation_log_path = str(root / "logs/train/runs/POW/efficientnet")
    latest_log = max(Path(log_path).glob("*"), key=lambda file: file.stat().st_ctime)
    checkpoints = Path(str(latest_log / "callback_checkpoints")).glob("*.ckpt")

    best_checkpoint = ("", float("inf"))
    for checkpoint in checkpoints:
        subprocess.run("python " + str(root / "birdset/eval.py") + f" experiment=\"local/MT_example_valid.yaml\" ckpt_path={checkpoint}", shell=True)

        latest_log = max(Path(validation_log_path).glob("*"), key=lambda file: file.stat().st_ctime)
        with open(str(latest_log / "finalmetrics.json")) as json_data:
            metrics = json.load(json_data)
        loss = metrics[0]['value']
        if loss < best_checkpoint[1]:
            best_checkpoint = (checkpoint, loss)
    print(f"The best checkpoint is {best_checkpoint[0]}")



if __name__ == "__main__":
    main()