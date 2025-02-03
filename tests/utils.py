import subprocess
import pytest


def generate_commands(experiments):
    """
    Generates a dictionary of commands for running training experiments.
    Args:
        experiments (list): List of experiment configuration file paths.

    Returns:
        dict: A dictionary where keys are experiment paths and values are commands.
    """
    base_command = ["python", "birdset/train.py", "trainer.fast_dev_run=True"]
    return {
        exp: base_command[:2] + [f"experiment={exp}"] + base_command[2:]
        for exp in experiments
    }


def run_experiment_test(experiment, command):
    """
    Executes an experiment command and asserts success.

    Args:
        experiment (str): Experiment file path.
        command (list): Command to execute.

    Raises:
        pytest.fail if the command fails.
    """
    try:
        result = subprocess.run(command, check=True, capture_output=True, text=True)
        assert result.returncode == 0, f"Non-zero return code for {experiment}"
    except subprocess.CalledProcessError as e:
        pytest.fail(f"{experiment} failed with error: {e.stderr}")
    except Exception as e:
        pytest.fail(f"{experiment} encountered an unexpected error: {e}")
