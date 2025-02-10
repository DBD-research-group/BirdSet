import pytest
from tests.utils import generate_commands, run_experiment_test

DATASET_EXPERIMENTS = [
    "birdset_neurips24/HSN/DT/convnext.yaml",
    "birdset_neurips24/NBP/DT/convnext.yaml",
    "birdset_neurips24/NES/DT/convnext.yaml",
    "birdset_neurips24/PER/DT/convnext.yaml",
    "birdset_neurips24/POW/DT/convnext.yaml",
    "birdset_neurips24/SNE/DT/convnext.yaml",
    "birdset_neurips24/SSW/DT/convnext.yaml",
    "birdset_neurips24/UHH/DT/convnext.yaml",
    # to big for everyday testing:
    # "birdset_neurips24/XCL/convnext.yaml",
    # "birdset_neurips24/XCM/convnext.yaml",
]

commands = generate_commands(DATASET_EXPERIMENTS)


@pytest.mark.parametrize(
    "experiment,command",
    commands.items(),
    ids=["/".join(exp.split("/")[-3:]) for exp in commands.keys()],
)
def test_dataset_experiment(experiment, command):
    run_experiment_test(experiment, command)
