import subprocess
import pytest


def test_train_script():
    """
    Test the training script with fast development run enabled.
    This function runs the training script using a subprocess with the
    `trainer.fast_dev_run=True` argument to quickly check for any issues
    in the training pipeline. It asserts that the script completes
    successfully without errors.
    Raises:
        pytest.fail: If the training script fails to run successfully,
        this exception is raised with the error message from the subprocess.
    """

    command = [
        "python",
        "birdset/train.py",
        'experiment="local/DT_example.yaml"',
        "trainer.fast_dev_run=True",
    ]
    try:
        result = subprocess.run(command, check=True, capture_output=True, text=True)
        assert result.returncode == 0
    except subprocess.CalledProcessError as e:
        pytest.fail(f"Command failed with error: {e.stderr}")
