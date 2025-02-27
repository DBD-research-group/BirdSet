from tests.test_models import MODEL_EXPERIMENTS
from tests.test_datasets import DATASET_EXPERIMENTS
from tests.utils import generate_commands


def pytest_addoption(parser):
    parser.addoption(
        "--devices",
        action="store",
        default="0",
        help="Specify devices as a comma-separated list of integers (e.g., 0,1,2)",
    )
    parser.addoption(
        "--workers",
        action="store",
        default="1",
        help="Specify number of cpu workers as an integers (e.g., 4)",
    )


def pytest_generate_tests(metafunc):
    if "experiment" in metafunc.fixturenames and "command" in metafunc.fixturenames:
        devices = metafunc.config.getoption("devices")
        workers = metafunc.config.getoption("workers")

        if metafunc.function.__name__ == "test_dataset_experiment":
            experiments = DATASET_EXPERIMENTS
        elif metafunc.function.__name__ == "test_model_experiment":
            experiments = MODEL_EXPERIMENTS

        commands = generate_commands(experiments, devices, workers)
        metafunc.parametrize(
            "experiment,command",
            commands.items(),
            ids=["/".join(exp.split("/")[-3:]) for exp in commands.keys()],
        )
