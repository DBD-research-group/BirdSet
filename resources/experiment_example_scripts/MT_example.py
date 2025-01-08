import json
import subprocess
import hydra
import pyrootutils
from pathlib import Path

root = pyrootutils.setup_root(
    search_from=__file__,
    indicator=[".git"],
    pythonpath=True,
    dotenv=True,
)


_HYDRA_PARAMS = {
    "version_base": None,
    "config_path": str(root / "configs/experiment/local"),
    "config_name": "MT_example.yaml",
}


@hydra.main(**_HYDRA_PARAMS)
def main(cfg):
    subprocess.run(
        "python "
        + str(root / "birdset/train.py")
        + f" experiment={cfg.train_cfg_path}",
        shell=True,
    )

    log_path = str(root / "logs/train/runs/XCM/efficientnet")
    validation_log_path = str(root / "logs/train/runs/POW/efficientnet")
    latest_log = max(Path(log_path).glob("*"), key=lambda file: file.stat().st_ctime)
    checkpoints = Path(str(latest_log / "callback_checkpoints")).glob("*.ckpt")

    best_checkpoint = ("", float("inf"))
    for checkpoint in checkpoints:
        subprocess.run(
            "python "
            + str(root / "birdset/eval.py")
            + f" experiment={cfg.valid_cfg_path} ckpt_path={checkpoint}",
            shell=True,
        )

        latest_log = max(
            Path(validation_log_path).glob("*"), key=lambda file: file.stat().st_ctime
        )
        with open(str(latest_log / "finalmetrics.json")) as json_data:
            metrics = json.load(json_data)
        loss = metrics[0]["value"]
        if loss < best_checkpoint[1]:
            best_checkpoint = (checkpoint, loss)
    print(f"The best checkpoint is {best_checkpoint[0]}")


if __name__ == "__main__":
    main()
