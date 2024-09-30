# $\texttt{BirdSet}$ - : A Large-Scale Dataset for Audio Classification in Avian Bioacoustics
[![python](https://img.shields.io/badge/-Python_3.10-blue?logo=python&logoColor=white)](https://github.com/pre-commit/pre-commit)
<a href="https://huggingface.co/"><img alt="Hugging Face" src="https://img.shields.io/badge/HuggingFace-ffcc00?logo=huggingface&logoColor=white"></a>
<a href="https://pytorch.org/get-started/locally/"><img alt="PyTorch" src="https://img.shields.io/badge/PyTorch-ee4c2c?logo=pytorch&logoColor=white"></a>
<a href="https://www.pytorchlightning.ai/"><img alt="PyTorch Lightning" src="https://img.shields.io/badge/PyTorch_Lightning-792ee5?logo=pytorch-lightning&logoColor=white"></a>
<a href="https://hydra.cc/"><img alt="Config: Hydra" src="https://img.shields.io/badge/Config-Hydra-89b8cd"></a>

Deep learning models have emerged as a powerful tool in avian bioacoustics to assess environmental health. To maximize the potential of cost-effective and minimal-invasive passive acoustic monitoring (PAM), models must analyze bird vocalizations across a wide range of species and environmental conditions. However, data fragmentation challenges a evaluation of generalization performance. Therefore, we introduce the $\texttt{BirdSet}$ dataset, comprising approximately 520,000 global bird recordings for training and over 400 hours PAM recordings for testing in a multi-label classification setting.

Our **datasets** are shared via Hugging Face 🤗 Datasets in our [BirdSet repository](https://huggingface.co/datasets/anonymous-birder/BirdSet). Our accompanying code package includes modules for further data preparation, model training, and evaluation.

<div align="center">
  
|                            | **Task**                                     | **Description** | **# Train Recordings** | **# Test\_5s Segments** | **Pielou’s evenness J** | **# Species**   |
|----------------------------|----------------------------------------------|-----------------|-----------|--------------|-------|----------|
| **Large Train**                  | [XCL](https://xeno-canto.org/)               | Complete Xeno-Canto snapshot with focals for large (pre-) training.                | 528,434   | -            | -     | 9,734    |
|                            | [XCM](https://xeno-canto.org/)               | Smaller subset of XCL only containing focals of bird species available in test datasets.                 | 89,798    | -            | -     | 409      |
| **Auxiliary**              | [POW](https://zenodo.org/records/4656848)    | Powdermill Nature soundscape validation dataset and class-dedicated focal training subset of XCL.     | 14,911    | 4,560        | 0.66  | 48       |
|                            | [VOX](https://zenodo.org/records/1208080)    | BirdVox-DCASE soundscape background dataset without bird vocalizations for augmentations.              | 20,331    | -            | -     | -        |
| **Test & Dedicated Train** | [PER](https://zenodo.org/records/7079124) | Amazon Basin soundscape test dataset and class-dedicated focal training subset of XCL.                 | 16,802    | 15,120       | 0.78  | 132      |
|                            | [NES](https://zenodo.org/records/7525349)    |  Columbia Costa Rica soundscape test dataset and class-dedicated focal training subset of XCL.               | 16,117    | 24,480       | 0.76  | 89       |
|                            | [UHH](https://zenodo.org/records/7078499)    |  Hawaiian Islands soundscape test dataset and class-dedicated focal training subset of XCL.               | 3,626     | 36,637       | 0.64  | 25       |
|                            | [HSN](https://zenodo.org/records/7525805)    |  High Sierras Nevada soundscape test dataset and class-dedicated focal training subset of XCL.               | 5,460     | 12,000       | 0.54  | 21       |
|                            | [NBP](https://link-to-birddb)                |  NIPS4BPlus test dataset and class-dedicated focal training subset of XCL.               | 24,327    | 563          | 0.92  | 51       |
|                            | [SSW](https://zenodo.org/records/7018484)    |  Sapsucker Woods soundscape test dataset and class-dedicated focal training subset of XCL.               | 28,403    | 205,200      | 0.77  | 81       |
|                            | [SNE](https://zenodo.org/records/7050014)    |  Sierre Nevada soundscape test dataset and class-dedicated focal training subset of XCL.               | 19,390    | 23,756       | 0.70  | 56       |

</div>

## User Installation 🐣

The simplest way to install $\texttt{BirdSet}$ is to clone this repository and install it as an editable package using [conda](https://docs.conda.io/en/latest/) and [pip](https://pip.pypa.io/en/stable/):
```
conda create -n birdset python=3.10
pip install -e .
```

<!-- 
You can also use the [devcontainer](https://code.visualstudio.com/docs/devcontainers/containers) configured as as git submodule:
```bash
git submodule update --init --recursive
```

Or [poetry](https://python-poetry.org/).
```
poetry install
poetry shell
```
-->

## Examples 🐤

We offer an in-depth tutorial notebook on how to use this repository. In the following, we provide simple code snippets:

### Manual Data Preparation

You can manually download the datasets from Hugging Face. We offer a uniform metadata format but also provide flexibility on how to prepare the data (e.g. you can manually decide which events to filter from the training data). The dataset dictionary comes with: 

- `train`: Focal instance with variable lengths. Possible `detected_events` and corresponding event clusters are provided.  
- `test_5s`: Processed test datasets where each soundscape instance corresponds to a 5-second clip with a `ebird_code_multilabel` format.  
- `test`: Unprocessed test datasets where each soundscape instance points to the full soundscape recording and the correspoding `ebird_code` with ground truth `start_time` and `end_time`.

 
```python
from datasets import load_dataset, dataset, Audio

# download the dataset 
dataset = load_dataset("anonymous-birder/BirdSet","HSN")

# set HF decoder (decodes the complete file!)
dataset = dataset.cast_column("audio", Audio(sampling_rate=32_000)

```
### BirdSet: Data Preparation :bird:

This code snippet utilizes the datamodule for an example dataset $\texttt{HSN}$. 

>**prepare_data**
>- downloads the data (or loads from cache)
>- preprocesses the data (event_mapping/sampling, one-hot encodes classes, create splits)
>- saves dataset to disk

>**setup**
>- sets up and loads the dataset for training and evaluating
>- adds `set_transforms` that transforms on-the-fly (decoding, augmentation etc.)
  
```python
from birdset.datamodule.base_datamodule import DatasetConfig
from birdset.datamodule.birdset_datamodule import BirdSetDataModule
from datasets import load_from_disk

# initiate the data module
dm = BirdSetDataModule(
    dataset= DatasetConfig(
        data_dir='data_birdset/HSN', # specify your data directory!
        hf_path='anonymous-birder/BirdSet',
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

# prepare the data
dm.prepare_data()

# manually load the complete prepared dataset (without any transforms). you have to cast the column with audio for decoding
ds = load_from_disk(dm.disk_save_path)

# OR setup the datasets with BirdSet ("test" for testdata)
dm.setup(stage="fit")

# audio is now decoded when a sample is called
train_ds = dm.train_dataset
val_ds = dm.val_dataset

# get the dataloaders
train_loader = dm.train_dataloader()
```

More details are available in the `datamodule_configs.py`and the tutorial notebook. 

### BirdSet: Prepare Model and Start Training  :bird:

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
## Reproduce ICLR2024 Baselines 🚧

> This repository is still under active development. You can access the ICLR 24 code at the tag  `iclr2024`
> ```bash
> git checkout ICLR2024 

First, you have to download the background noise files for augmentations

```bash
python resources/utils/download_background_noise.py
```

We provide all experiment YAML files used to generate our results in the path `birdset/configs/experiment/birdset_iclr24`. For each dataset, we specify the parameters for all training scenario: `DT`, `MT`, and `LT`

### Dedicated Training (DT)

The experiments for `DT` with the dedicated subset can be easily run with a single line: 

``` bash
python birdset/train.py experiment="birdset_iclr24/$Dataset/DT/$Model"
```

### Medium Training (MT) and Large Training (LT)
Experiments for training scenarios `MT` and `LT` are harder to reproduce since they require more extensive training times. 
Additionally, the datasets are quite large (90GB for XCM and 480GB for XCL). Therefore, we provide the best model checkpoints via Hugging Face in the experiment files to avoid the need for retraining. These checkpoints can be executed by running the evaluation script, which will automatically download the model and perform inference on the test datasets:

``` bash
python birdset/eval.py experiment="birdset_iclr24/$EXPERIMENT_PATH"
```

If you want to start the large-scale trainings and download the big training datasets, you can also employ the `XCM` and `XCL` trainings via the experiment YAML files. 

``` bash
python birdset/train.py experiment="birdset_icrl24/$EXPERIMENT_PATH"
```
After training, the best model checkpoint is saved based on the validation loss and can then be used for inference:

``` bash
python birdset/eval.py experiment="birdset_iclr24/$EXPERIMENT_PATH" module.model.network.local_checkpoint="$CHECKPOINT_PATH"
```

**Disclaimer on results:** The results obtained using the `eval.py` script may differ from those reported in the paper. This discrepancy is because only the "best" model checkpoint was uploaded to Hugging Face, whereas the paper’s results were averaged over three different random seeds for a more robust evaluation.

<!---
## Results (AUROC)
| <sub>Title</sub> | <sub>Notes</sub> |<sub>PER</sub> | <sub>NES</sub> | <sub>UHH</sub> | <sub>HSN</sub> | <sub>NBP</sub> | <sub>POW</sub> | <sub>SSW</sub> | <sub>SNE</sub>  | <sub>Overall</sub> | <sub>Code</sub> |
| :----| :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| <sub>**BirdSet: A Multi-Task Benchmark For Classification In Avian Bioacoustics**</sub> | | | | | | | |
| <sub>**BIRB: A Generalization Benchmark for Information Retrieval in Bioacoustics**</sub> | | | | | | | |  | | | |
## Logging
Logs will be written to [Weights&Biases](https://wandb.ai/) by default.
-->
## Background noise
To enhance model performance we mix in additional background noise from downloaded from the [DCASE18](https://dcase.community/challenge2018/index). To download the files and convert them to the correct format, run the notebook 'download_background_noise.ipynb' in the 'notebooks' folder.

## Run experiments

Our experiments are defined in the `configs/experiment` folder. To run an experiment, use the following command in the directory of the repository:

``` bash
python birdset/train.py experiment="EXPERIMENT_PATH"
```

Replace `EXPERIMENT_PATH` with the path to the experiment YAML config originating from the `experiment` directory. Here's a command for training an EfficientNet on HSN: 

``` bash
python birdset/train.py experiment="local/HSN/efficientnet.yaml"
```

<!-- 
# Data pipeline

Our datasets are shared via Hugging Face Datasets in our [BirdSet repository](https://huggingface.co/datasets/anonymous-birder/BirdSet).
First log in to Hugging Face with:
```bash
huggingface-cli login
```

For a detailed guide to using the BirdSet data pipeline and its many configuration options, see our comprehensive [BirdSet Data Pipeline Tutorial](notebooks/tutorials/birdset-pipeline_tutorial.ipynb).

## Datamodule

The datamodules are defined in `birdset/datamodule` and configurations are stored under `configs/datamodule`.
`base_datamodule` is the main class that can be inherited for specific datasets. It is responsible for preparing the data in the function `prepare_data` and loading the data in the function `setup`. `prepare_data` downloads the dataset, applies preprocessing, creates validation splits and saves the data to disk. `setup` initiates the dataloaders and configures data transformations.

The following steps are performed in `prepare_data`:

1. Data is downloaded from Hugging Face Datasets `_load_data`
2. Data gets preprocessed with `_preprocess_data`
3. Data is split into train validation and test sets with `_create_splits`
4. Length of the dataset gets saved to access later
5. Data is saved to disk with `_save_dataset_to_disk`

The following steps are performed in `setup`:
1. Data is loaded from disk with `_get_dataset` in which the transforms are applied

## Transformations

Data transformations are referred to data transformations that are applied to the data during training. They include e.g. augmentations. The transformations are added to the Hugging Face dataset with [`set_transform`](https://huggingface.co/docs/datasets/main/en/package_reference/main_classes#datasets.Dataset.set_transform).

-->



