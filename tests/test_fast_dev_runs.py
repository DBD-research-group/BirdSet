import subprocess
import pytest
from datetime import datetime
import os


def generate_commands():
    """
    Generates a dictionary of commands for running training experiments.
    Returns:
        dict: A dictionary where the keys are experiment configuration file paths
              and the values are lists representing the command to run the experiment.
    """

    base_command = ["python", "birdset/train.py", "trainer.fast_dev_run=True"]
    experiments = [
        "birdset_neurips24/HSN/DT/convnext.yaml",
        "birdset_neurips24/HSN/DT/eat.yaml",
    ]
    return {
        exp: base_command[:2] + [f"experiment={exp}"] + base_command[2:]
        for exp in experiments
    }


commands = generate_commands()
print(commands)


@pytest.mark.parametrize(
    "experiment,command",
    commands.items(),
    ids=["/".join(exp.split("/")[-3:]) for exp in commands.keys()],
)
def test_experiment_command(experiment, command):
    """
    Test the execution of training commands for different experiments.

    Parameters:
    - experiment: The experiment configuration file path (used to construct the command).
    - command: The complete command to be executed, including the experiment parameter.

    The test ensures:
    1. The command runs successfully without errors.
    2. The return code is zero, indicating successful execution.

    If the command fails:
    - `pytest.fail` is called with detailed error output for debugging.
    """
    try:
        result = subprocess.run(command, check=True, capture_output=True, text=True)
        assert result.returncode == 0, f"Non-zero return code for {experiment}"
    except subprocess.CalledProcessError as e:
        pytest.fail(f"{experiment} failed with error: {e.stderr}")
    except Exception as e:
        pytest.fail(f"{experiment} encountered an unexpected error: {e}")
