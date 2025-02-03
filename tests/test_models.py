import pytest
from tests.utils import generate_commands, run_experiment_test

MODEL_EXPERIMENTS = [
    "birdset_neurips24/HSN/DT/ast.yaml",
    "birdset_neurips24/HSN/DT/convnext.yaml",
    "birdset_neurips24/HSN/DT/eat.yaml",
    "birdset_neurips24/HSN/DT/efficientnet.yaml",
    "birdset_neurips24/HSN/DT/wav2vec2.yaml",
]

commands = generate_commands(MODEL_EXPERIMENTS)


@pytest.mark.parametrize(
    "experiment,command",
    commands.items(),
    ids=["/".join(exp.split("/")[-3:]) for exp in commands.keys()],
)
def test_model_experiment(experiment, command):
    run_experiment_test(experiment, command)
