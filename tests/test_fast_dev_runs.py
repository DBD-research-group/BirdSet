import subprocess
import pytest
from datetime import datetime
import os

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

    passed_result_log = []
    failed_result_log = {}

    command_pattern = [
        "python",
        "birdset/train.py",
        "experiment='X'",
        "trainer.fast_dev_run=True",
    ]

    experiment_dict = {
        "HSN/DT/ConvNext": "birdset_neurips24/HSN/DT/convnext.yaml",
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

        passed_result_log.append(experiment)

    return passed_result_log, failed_result_log


def write_result_log(passed_log: dict, failed_log: dict, directory: str = "tests/logs"):
    def write_result_log(passed_log: dict, failed_log: dict, directory: str = "tests"):
        """
        Writes the results of test runs to a log file.

        Parameters:
        passed_log (dict): A dictionary containing the experiments that passed and their results.
        failed_log (dict): A dictionary containing the experiments that failed and their results.
        directory (str): The directory where the log file will be saved. Defaults to "tests".

        The log file will be named with the format "test_fast_dev_runs_YYYYMMDD_HHMMSS.txt"
        where YYYYMMDD_HHMMSS is the current timestamp.
        """

    os.makedirs(directory, exist_ok=True)

    current_time = datetime.now()
    timestamp = current_time.strftime("%Y%m%d_%H%M%S")
    file_name = f"test_fast_dev_runs_{timestamp}.txt"

    with open(f"{directory}/{file_name}", "w") as f:
        if failed_log:
            f.write("\nFailed runs:\n")
            for experiment, result in failed_log.items():
                f.write(f"{experiment}: {result}\n")
        else:
            f.write("All runs passed\n")

        if passed_log:
            f.write("Passed runs:\n")
            for experiment, result in passed_log.items():
                f.write(f"{experiment}: {result}\n")
