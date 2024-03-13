# GADME


## Setup

### Devcontainer

You can use the [devcontainer](https://code.visualstudio.com/docs/devcontainers/containers) configured as as git submodule:
```bash
git submodule update --init --recursive
```

### Install dependencies

Either with [conda](https://docs.conda.io/en/latest/) and [pip](https://pip.pypa.io/en/stable/).
```
conda create -n gadme python=3.10
pip install -e .
```

Or [poetry](https://python-poetry.org/).
```
mv pyproject.poetry pyproject.toml
poetry install
poetry shell
```

## Log in to Huggingface

Our datasets are shared via HuggingFace Datasets in our [HuggingFace GADME repository](https://huggingface.co/datasets/DBD-research-group/gadme_v1). Huggingface is a central hub for sharing and utilizing datasets and models, particularly beneficial for machine learning and data science projects. For accessing private datasets hosted on HuggingFace, you need to be authenticated. Here's how you can log in to HuggingFace:

1. **Install HuggingFace CLI**: If you haven't already, you need to install the HuggingFace CLI (Command Line Interface). This tool enables you to interact with HuggingFace services directly from your terminal. You can install it using pip:

   ```bash
   pip install huggingface_hub
   ```

2. **Login via CLI**: Once the HuggingFace CLI is installed, you can log in to your HuggingFace account directly from your terminal. This step is essential for accessing private datasets or contributing to the HuggingFace community. Use the following command:

   ```bash
   huggingface-cli login
   ```

   After executing this command, you'll be prompted to enter your HuggingFace credentials ([User Access Token](https://huggingface.co/docs/hub/security-tokens)). Once authenticated, your credentials will be saved locally, allowing seamless access to HuggingFace resources.


## Logging
Logs will be written to [Weights&Biases](https://wandb.ai/) by default.

## Background noise
To enhance model performance we mix in additional background noise from downloaded from the [DCASE18](https://dcase.community/challenge2018/index). To download the files and convert them to the correct format, run the notebook 'download_background_noise.ipynb' in the 'notebooks' folder.

## Run experiments

Our experiments are defined in the `configs/experiment` folder. To run an experiment, use the following command:

```
python gadme/main.py experiment=EXPERIMENT_NAME
```



## Project structure

This repository is inspired by the [Yet Another Lightning Hydra Template](https://github.com/gorodnitskiy/yet-another-lightning-hydra-template).

```
├── configs                     <- Hydra configuration files
│   ├── callbacks               <- Callbacks configs
│   ├── datamodule              <- Datamodule configs
│   ├── debug                   <- Debugging configs
│   ├── experiment              <- Experiment configs
│   ├── extras                  <- Extra utilities configs
│   ├── hydra                   <- Hydra settings configs
│   ├── logger                  <- Logger configs
│   ├── module                  <- Module configs
│   ├── paths                   <- Project paths configs
│   ├── trainer                 <- Trainer configs
│   ├── transformations         <- Transformations / augmentation configs
│   |
│   ├── main.yaml               <- Main config
│
├── data_gadme                  <- Project data
├── dataset                     <- Code to build the GADME dataset
├── notebooks                   <- Jupyter notebooks.
│
├── gadme                         <- Source code
│   ├── augmentations           <- Augmentations
│   ├── callbacks               <- Additional callbacks
│   ├── datamodules             <- Lightning datamodules
│   ├── modules                 <- Lightning modules
│   ├── utils                   <- Utility scripts
│   │
│   ├── main.py                 <- Run experiments
│
├── .gitignore                  <- List of files ignored by git
├── pyproject.toml              <- Poetry project file
├── requirements.txt            <- File for installing python dependencies
├── requirements-dev.txt        <- File for installing python dev dependencies
├── setup.py                    <- File for installing project as a package
└── README.md
```

# Data pipeline

Our datasets are shared via HuggingFace Datasets in our [GADME repository](https://huggingface.co/datasets/DBD-research-group/gadme_v1).
First log in to HuggingFace with:
```bash
huggingface-cli login
```

For a detailed guide to using the GADME data pipeline and its many configuration options, see our comprehensive [GADME Data Pipeline Tutorial](notebooks/tutorials/gadme-pipeline_tutorial.ipynb).

## Datamodule

The datamodules are defined in `gadme/datamodule` and configurations are stored under `configs/datamodule`.
`base_datamodule` is the main class that can be inherited for specific datasets. It is responsible for preparing the data in the function `prepare_data` and loading the data in the function `setup`. `prepare_data` downloads the dataset, applies preprocessing, creates validation splits and saves the data to disk. `setup` initiates the dataloaders and configures data transformations.

The following steps are performed in `prepare_data`:

1. Data is downloaded from HuggingFace Datasets `_load_data`
2. Data gets preprocessed with `_preprocess_data`
3. Data is split into train validation and test sets with `_create_splits`
4. Length of the dataset gets saved to access later
5. Data is saved to disk with `_save_dataset_to_disk`

The following steps are performed in `setup`:
1. Data is loaded from disk with `_get_dataset` in which the transforms are applied

## Transformations

Data transformations are referred to data transformations that are applied to the data during training. They include e.g. augmentations. The transformations are added to the huggingface dataset with [`set_transform`](https://huggingface.co/docs/datasets/main/en/package_reference/main_classes#datasets.Dataset.set_transform).




