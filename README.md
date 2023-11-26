# GADME


## Setup

### Install dependencies

Eather with [conda](https://docs.conda.io/en/latest/) and [pip](https://pip.pypa.io/en/stable/).
```
conda create -n gadme python=3.9
pip install -e .
```

Or [poetry](https://python-poetry.org/).
```
mv pyproject.raphael pyproject.toml
poetry install
poetry shell
```

## Logging
Logs will be written to [Weights&Biases](https://wandb.ai/) by default. 

## Run experiments

Our experiments are defined in the `configs/experiment` folder. To run an experiment, use the following command:

```
python src/main.py experiment=EXPERIMENT_NAME
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
├── src                         <- Source code
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

## Data pipeline

Our datasets are shared via HuggingFace Datasets in our [GADME repository](https://huggingface.co/datasets/DBD-research-group/gadme_v1).
First log in to HuggingFace with:
```bash
huggingface-cli login
```

### Datamodule

The datamodules are defined in `src/datamodule` and configurations are stored under `configs/datamodule`.
`base_datamodule` is the main class that can be inherited for specific datasets. It is responsible for preparing the data in the function `prepare_data` and loading the data in the function `setup`. `prepare_data` downloads the dataset, applies preprocessing, creates validation splits and saves the data to disk. `setup` initiates the dataloaders and configures data transformations.

### Preprocessing

Preprocessing is referred to data transformations that can be applied to the data before training and are therefore done only once. The `feature_extractor` is responsible for this.

### Transformations

Data transformations are referred to data transformations that are applied to the data during training. They include e.g. augmentations. The transformations are added to the huggingface dataset with [`set_transform`](https://huggingface.co/docs/datasets/main/en/package_reference/main_classes#datasets.Dataset.set_transform).




