import subprocess
import pyrootutils

root = pyrootutils.setup_root(
    search_from=__file__,
    indicator=[".git"],
    pythonpath=True,
    dotenv=True,
)


def main():
    subprocess.run(
        "python "
        + str(root / "birdset/train.py")
        + ' experiment="local/DT_example.yaml"',
        shell=True,
    )


if __name__ == "__main__":
    main()
