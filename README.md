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