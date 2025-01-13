import subprocess
import pytest

result_log = []


def test_train_script():
    """
    Tests the training script for multiple experiments using the fast_dev_run option.
    This function constructs and executes a series of commands to run the training script
    with different experiment configurations. It logs the results of each run, indicating
    whether the run passed or failed.
    The function performs the following steps:
    1. Defines a command pattern for running the training script.
    2. Specifies a dictionary of experiments and their corresponding configuration files.
    3. Constructs the commands for each experiment by replacing the placeholder in the command pattern.
    4. Executes each command and captures the output.
    5. Logs the results of each run, including any errors encountered.
    Returns:
        tuple: A tuple containing two dictionaries:
            - passed_result_log (dict): A dictionary where keys are experiment names and values are the stdout of successful runs.
            - failed_result_log (dict): A dictionary where keys are experiment names and values are the stderr of failed runs.
    Raises:
        pytest.fail: If any of the commands return a non-zero exit code, indicating a failure in the training script.
    """

    passed_result_log = {}
    failed_result_log = {}
    all_passed = True

    command_pattern = [
        "python",
        "birdset/train.py",
        "experiment=X" "trainer.fast_dev_run=True",
    ]

    experiment_dict = {
        "HSN/DT/ConvNext": "birdset_neurips24/HSN/DT/eat.yaml",
        "HSN/DT/EAT": "birdset_neurips24/HSN/DT/eat.yaml",
    }

    commands = {}

    for experiment, file in experiment_dict.items():
        command = command_pattern.copy()
        command[2] = f"experiment={file}"
        commands[experiment] = command

    for experiment, command in commands.items():
        try:
            result = subprocess.run(command, check=True, capture_output=True, text=True)
            assert result.returncode == 0
        except subprocess.CalledProcessError as e:
            failed_result_log[experiment] = e.stderr
            pytest.fail(f"{experiment} failed with error: {e.stderr}")

        passed_result_log[experiment] = result.stdout

    return passed_result_log, failed_result_log
