
[![python](https://img.shields.io/badge/-Python_3.10-blue?logo=python&logoColor=white)](https://github.com/pre-commit/pre-commit)
<a href="https://pytorch.org/get-started/locally/"><img alt="PyTorch" src="https://img.shields.io/badge/PyTorch-ee4c2c?logo=pytorch&logoColor=white"></a>
<a href="https://pytorchlightning.ai/"><img alt="Lightning" src="https://img.shields.io/badge/-Lightning-792ee5?logo=pytorchlightning&logoColor=white"></a>
<a href="https://hydra.cc/"><img alt="Config: Hydra" src="https://img.shields.io/badge/Config-Hydra-89b8cd"></a>
<a href="https://github.com/DBD-research-group/BirdSet"><img alt="GitHub: github.com/DBD-research-group/BirdSet " src="https://img.shields.io/badge/-BirdSet-017F2F?style=flat&logo=github&labelColor=gray"></a>
[![arXiv](https://img.shields.io/badge/arXiv-1234.56789-b31b1b.svg)](https://arxiv.org/abs/2403.10380)

![logo](https://github.com/DBD-research-group/BirdSet/blob/main/resources/perch/birdsetsymbol.png)

Deep learning models have emerged as a powerful tool in avian bioacoustics to assess environmental health. To maximize the potential of cost-effective and minimal-invasive passive acoustic monitoring (PAM), models must analyze bird vocalizations across a wide range of species and environmental conditions. However, data fragmentation challenges a evaluation of generalization performance. Therefore, we introduce the $\texttt{BirdSet}$ dataset, comprising approximately 520,000 global bird recordings for training and over 400 hours PAM recordings for testing.
## User Installation

The simplest way to install $\texttt{BirdSet}$ is to clone this repository and install it as an editable package using [conda](https://docs.conda.io/en/latest/) and [pip](https://pip.pypa.io/en/stable/):
```
conda create -n birdset python=3.10
pip install -e .
```

You can also use the [devcontainer](https://code.visualstudio.com/docs/devcontainers/containers) configured as as git submodule:
```bash
git submodule update --init --recursive
```

Or [poetry](https://python-poetry.org/).
```
poetry install
poetry shell
```

## Reproduce Neurips2024 Baselines

First, you have to download the background noise files for augmentations

``` bash
python resources/utils/download_background_noise.py
```

We provide all experiment YAML files used to generate our results in the path `birdset/configs/experiment/birdset_neurips24`. For each dataset, we specify the parameters for all training scenario: `DT`, `MT`, and `LT`

### Dedicated Training (DT)

The experiments for `DT` with the dedicated subset can be easily run with a single line: 

``` bash
python birdset/train.py experiment="birdset_neurips24/DT/$Model"
```

### Medium Training (MT) and Large Training (LT)
Experiments for training scenarios `MT` and `LT` are harder to reproduce since they require more extensive training times. 
Additionally, the datasets are quite large (90GB for XCM and 480GB for XCL). Therefore, we provide the best model checkpoints via Hugging Face in the experiment files to avoid the need for retraining. These checkpoints can be executed by running the evaluation script, which will automatically download the model and perform inference on the test datasets:

``` bash
python birdset/eval.py experiment="birdset_neurips24/$EXPERIMENT_PATH"
```

As the model EAT is not implemented in Hugging Face transformer (yet), the checkpoints are available to download from the tracked experiments on [Weights and Biases LT_XCL_eat](https://wandb.ai/deepbirddetect/birdset/runs/pretrain_eat_3_2024-05-17_075334/files?nw=nwuserraphaelschwinger).

If you want to start the large-scale trainings and download the big training datasets, you can also employ the `XCM` and `XCL` trainings via the experiment YAML files. 

``` bash
python birdset/train.py experiment="birdset_neurips24/$EXPERIMENT_PATH"
```
After training, the best model checkpoint is saved based on the validation loss and can then be used for inference:

``` bash
python birdset/eval.py experiment="birdset_neurips24/$EXPERIMENT_PATH" module.model.network.local_checkpoint="$CHECKPOINT_PATH"
```


## Example

<!-- ## Log in to Huggingface

Our datasets are shared via HuggingFace Datasets in our [HuggingFace BirdSet repository](https://huggingface.co/datasets/DBD-research-group/birdset_v1). Huggingface is a central hub for sharing and utilizing datasets and models, particularly beneficial for machine learning and data science projects. For accessing private datasets hosted on HuggingFace, you need to be authenticated. Here's how you can log in to HuggingFace:

1. **Install HuggingFace CLI**: If you haven't already, you need to install the HuggingFace CLI (Command Line Interface). This tool enables you to interact with HuggingFace services directly from your terminal. You can install it using pip:

   ```bash
   pip install huggingface_hub
   ```

2. **Login via CLI**: Once the HuggingFace CLI is installed, you can log in to your HuggingFace account directly from your terminal. This step is essential for accessing private datasets or contributing to the HuggingFace community. Use the following command:

   ```bash
   huggingface-cli login
   ```

   After executing this command, you'll be prompted to enter your HuggingFace credentials ([User Access Token](https://huggingface.co/docs/hub/security-tokens)). Once authenticated, your credentials will be saved locally, allowing seamless access to HuggingFace resources. -->
[Tutorial Notebook](https://github.com/DBD-research-group/BirdSet/blob/main/notebooks/tutorials/birdset-pipeline_tutorial.ipynb)
## Prepare Data

```python
from birdset.datamodule.base_datamodule import DatasetConfig
from birdset.datamodule.birdset_datamodule import BirdSetDataModule

# initiate the data module
dm = BirdSetDataModule(
    dataset= DatasetConfig(
        data_dir='data_birdset/HSN', # specify your data directory!
        dataset_name='HSN',
        hf_path='DBD-research-group/BirdSet',
        hf_name='HSN',
        n_classes=21,
        n_workers=3,
        val_split=0.2,
        task="multilabel",
        classlimit=500,
        eventlimit=5,
        sampling_rate=32000,
    ),
)

# prepare the data (download dataset, ...)
dm.prepare_data()

# setup the dataloaders
dm.setup(stage="fit")

# get the dataloaders
train_loader = dm.train_dataloader()
```

## Prepare Model and Start Training

```python
from lightning import Trainer
min_epochs = 1
max_epochs = 5
trainer = Trainer(min_epochs=min_epochs, max_epochs=max_epochs, accelerator="gpu", devices=1)

from birdset.modules.multilabel_module import MultilabelModule
model = MultilabelModule(
    len_trainset=dm.len_trainset,
    task=dm.task,
    batch_size=dm.train_batch_size,
    num_epochs=max_epochs)

trainer.fit(model, dm)
```
<!---
## Results (AUROC)
| <sub>Title</sub> | <sub>Notes</sub> |<sub>PER</sub> | <sub>NES</sub> | <sub>UHH</sub> | <sub>HSN</sub> | <sub>NBP</sub> | <sub>POW</sub> | <sub>SSW</sub> | <sub>SNE</sub>  | <sub>Overall</sub> | <sub>Code</sub> |
| :----| :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| <sub>**BirdSet: A Multi-Task Benchmark For Classification In Avian Bioacoustics**</sub> | | | | | | | |
| <sub>**BIRB: A Generalization Benchmark for Information Retrieval in Bioacoustics**</sub> | | | | | | | |  | | | |-->
## Logging
Logs will be written to [Weights&Biases](https://wandb.ai/) by default.

## Background noise
To enhance model performance we mix in additional background noise from downloaded from the [DCASE18](https://dcase.community/challenge2018/index). To download the files and convert them to the correct format, run the notebook 'download_background_noise.ipynb' in the 'notebooks' folder.

## Run experiments

Our experiments are defined in the `configs/experiment` folder. To run an experiment, use the following command in the directory of the repository:

``` bash
python birdset/train.py experiment="EXPERIMENT_PATH"
```
Replace `EXPERIMENT_PATH` with the path to the disired experiment YAML config originating from the `experiment` directory. For example, here's a command for training an EfficientNet on HSN: 

``` bash
python birdset/train.py experiment="local/HSN/efficientnet.yaml"
```

# Data pipeline

Our datasets are shared via HuggingFace Datasets in our [BirdSet repository](https://huggingface.co/datasets/DBD-research-group/birdset_v1).
First log in to HuggingFace with:
```bash
huggingface-cli login
```

For a detailed guide to using the BirdSet data pipeline and its many configuration options, see our comprehensive [BirdSet Data Pipeline Tutorial](notebooks/tutorials/birdset-pipeline_tutorial.ipynb).

## Datamodule

The datamodules are defined in `birdset/datamodule` and configurations are stored under `configs/datamodule`.
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




