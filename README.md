# BioFoundation

[![python](https://img.shields.io/badge/-Python_3.10-blue?logo=python&logoColor=white)](https://github.com/pre-commit/pre-commit)
<a href="https://pytorch.org/get-started/locally/"><img alt="PyTorch" src="https://img.shields.io/badge/PyTorch-ee4c2c?logo=pytorch&logoColor=white"></a>
<a href="https://pytorchlightning.ai/"><img alt="Lightning" src="https://img.shields.io/badge/-Lightning-792ee5?logo=pytorchlightning&logoColor=white"></a>
<a href="https://hydra.cc/"><img alt="Config: Hydra" src="https://img.shields.io/badge/Config-Hydra-89b8cd"></a>


Deep learning models have emerged as a powerful tool in avian bioacoustics to assess environmental health. To maximize the potential of cost-effective and minimal-invasive passive acoustic monitoring (PAM), models must analyze bird vocalizations across a wide range of species and environmental conditions. However, data fragmentation challenges a evaluation of generalization performance. Therefore, we introduce the $\texttt{BirdSet}$ dataset, comprising approximately 520,000 global bird recordings for training and over 400 hours PAM recordings for testing.
## User Installation

### Devcontainer

You can use the [devcontainer](https://code.visualstudio.com/docs/devcontainers/containers) configured as as git submodule:
```bash
git submodule update --init --recursive
```

### Install dependencies

With [poetry](https://python-poetry.org/).
```
poetry install
poetry shell
```
# Experiments

## BEANS

### Running Linear Probing Experiments on BEANS
Foundation Models are tested on the Benchmark of Animal Sounds (BEANS) which we host on [Huggingface](https://huggingface.co/collections/DBD-research-group/beans-datasets-6611bd670cd7eb7b0bfc614e) and we focus on the classification datasets (watkins bats, cbi, dogs & humbugdb). Using the [beans.sh](scripts/beans.sh) script you can specify one or multiple experiment Paths to execute linear probing on all the BEANS datasets:

`$./projects/biofoundation/scripts/run_beans_embeddings_experiments.sh embedding/BEANS/perch [additional experiments]`

Currently the available embedding experiments are:
- [Perch](configs/experiment/biofoundation/embedding/BEANS/perch.yaml)
- [BirdNET](configs/experiment/biofoundation/embedding/BEANS/birdnet.yaml)
- [Hubert](configs/experiment/biofoundation/embedding/BEANS/hubert.yaml)
- [Wav2Vec2](configs/experiment/biofoundation/embedding/BEANS/wav2vec2.yaml)
- [AudioMAE](configs/experiment/biofoundation/embedding/BEANS/audiomae.yaml)
- [AVES](configs/experiment/local/embedding/BEANS/aves.yaml)
- [BEATS](configs/experiment/local/embedding/BEANS/beats.yaml)
- [BioLingual](configs/experiment/local/embedding/BEANS/biolingual.yaml)
- [ConvNeXt](configs/experiment/local/embedding/BEANS/convnext.yaml)
- [EAT](configs/experiment/local/embedding/BEANS/eat.yaml)
- [SSAST](configs/experiment/local/embedding/BEANS/ssast.yaml)


They all inherit from the base configuration [embedding_config.yaml](configs/experiment/biofoundation/embedding/BEANS/embedding_config.yaml) where most changes for extracting Embeddings are set.
To execute an experiment on a specific dataset you have to change the following lines in the experiment file:
```yaml
datamodule:
  dataset:
    dataset_name: beans_watkins # Change 
    hf_path: DBD-research-group/beans_watkins # Change 
    hf_name: default
    n_classes: 31 # Change 
```

|dataset_name|n_classes|
|------------|---------|
|beans_watkins|31|
|beans_bats|10|
|beans_cbi|264|
|beans_dogs|10|
|beans_humbugdb|14|

Regarding the embedding extraction multiple things can be configured by changing the params of the [embeddings_datamodule.py](birdset/datamodule/embeddings_datamodule.py) for example through the experiment config:

```yaml
defaults:
  # Inherit from default embedding config  
  - biofoundation/embedding/BEANS/embedding_config.yaml 
  # Use Hubert for embedding extraction 
  - override /datamodule/embedding_model: ../../module/network/hubert.yaml

datamodule:
    # If >0 only X samples per class are used for training; The rest is used for validation and testing
    k_samples: 0 
    # If a validation set should be used: Use null to use val set and 0 for no validation at all
    val_batch: null
    # (If 0 and k_samples > 0 then all remaining samples land in test set; If k_samples = 0 val and test split in BEANS are combined in the test set)

    # Test/Validation_ratio if k_samples > 0
    test_ratio: 0.5 
    # BEANS provides a low_train split which can be used instead of the default train split
    low_train: False 
    # If embeddings should be averaged or if just the first seconds should be used
    average: True 
```

The classifier can also be changed and right now [this](birdset/modules/models/linear_classifier.py) is used.

## Running Finetuning Experiments

The same models can also be finetuned and the experiments can be found in the respective [folder](configs/experiment/local/finetune/BEANS/) (except Perch). For finetuning a much lower learning rate is recommended and the [finetune_module](birdset/modules/finetune_module.py) is used. 

Compared to linear probing embeddings can't be computed beforehand which is why the computation times are considerably longer. To reduce these a bit, a hybrid method can be used that first applies linear probing and then a few epochs of finetuning. The results are usually better than linear probing but worse than finetuning. ATM the embeddings are not computed beforehand for the linear probing phase but the hybrid approach is still faster.

For this the [hybrid_module](birdset/modules/hybrid_module.py) is used and the experiments can be found in the hybrid [folder](configs/experiment/local/hybrid/BEANS/)

## Results
The results [folder](results) contains plots and plot-code that gives insights on the different performance between <span style="color: blue;">linear probing (blue)</span>, <span style="color: orange;">finetuning (orange)</span> and the <span style="color: green;">hybrid</span>(green) method.

![averaged_results](results/model_radar.png)

As a reference the embedding results can be used for future work:
![embedding_results](results/probing_hm.png)

## BirdSet

On the BirdSet benchmark we run three different experiments:
- Fine-tuning: The pretrained model is fine-tuned on the BirdSet dataset, similar to the dedicated training (DT) approach of the BirdSet paper.
- Linear Probing: The pretrained model is used as a fixed feature extractor and a linear classifier is trained on top of the extracted features.
- Few-shot: A small subset (k) samples per class are used for training, ether from the training set or from a split of the test data to evaluate the impact of the covariate shift.

### Running Fine-tuning Experiments on BirdSet

```bash
python birdset/train.py experiment=biofoundation/birdset/finetuning/{model_name}
```

#### Results

Results on HSN:

| Model | cmAP | AUROC | Wandb |
|-------| -------| ---- | ---- |
| BEATs| **0.44** | **0.87** | [BEATs_HSN#1_2024-11-22_135915](https://wandb.ai/deepbirddetect/BioFoundation/runs/beats_finetune_HSN_1_2024-11-22_135915) |
| BioLingual| 0.33 | 0.79 | [biolingual_HSN#1_2024-11-29_110143](https://wandb.ai/deepbirddetect/BioFoundation/runs/biolingual_finetune_BirdSet_HSN_1_2024-11-29_110143) |
| ConvNext| 0.41 | 0.84 | [convnext_HSN#1_2024-11-29_130206](https://wandb.ai/deepbirddetect/BioFoundation/runs/convnext_finetune_BirdSet_HSN_1_2024-11-29_130206) |
| EAT| ? | ? | ? |
| AVES| 0.19 | 0.68 | [aves_HSN#1_2024-11-28_123701](https://wandb.ai/deepbirddetect/BioFoundation/runs/aves_finetune_BirdSet_HSN_1_2024-11-28_123701) (ES: Only 7 epochs)|
| AST| 0.21 | 0.70 | [ast_HSN#1_2024-11-28_163020](https://wandb.ai/deepbirddetect/BioFoundation/runs/ast_finetune_BirdSet_HSN_1_2024-11-28_163020) |
| AudioMAE| ? | ? | ? |
| ConvNext_BS| ? | ? | ? |
| HUBERT| ? | ? | ? |
| SSAST| ? | ? | ? |
| Wav2Vec2| ? | ? | ? |
| BirdNET| ? | ? | ? |
### Running Linear Probing Experiments on BirdSet

```bash
python birdset/train.py experiment=biofoundation/birdset/linearprobing/{model_name}
```

Results on HSN:

| Model | cmAP | AUROC | Wandb |
|-------| -------| ---- | ---- |
| BEATS | 0.11 | **0.73** | [BEATs_HSN#1_2024-11-25_155526](https://wandb.ai/deepbirddetect/BioFoundation/runs/beats_linearprobing_BirdSet_HSN_1_2024-11-25_155526) |
| Perch | **0.22** | 0.66 | [perch_HSN#1_2024-11-25_175223](https://wandb.ai/deepbirddetect/BioFoundation/runs/perch_linearprobing_BirdSet_HSN_1_2024-11-25_175223) |
| BioLingual| 0.12 | 0.75 | [biolingual_HSN#1_2024-11-29_111328](https://wandb.ai/deepbirddetect/BioFoundation/runs/biolingual_linearprobing_BirdSet_HSN_1_2024-11-29_111328) # Episode 26 |
| ConvNext| 0.03 | 0.52 | [convnext_HSN#1_2024-11-29_131024](https://wandb.ai/deepbirddetect/BioFoundation/runs/convnext_linearprobing_BirdSet_HSN_1_2024-11-29_131024) # Episode 00  |
| EAT| ? | ? | ? |
| AVES| 0.04 | 0.63 | [aves_HSN#1_2024-11-28_112422](https://wandb.ai/deepbirddetect/BioFoundation/runs/aves_linearprobing_BirdSet_HSN_1_2024-11-28_112422) |
| AST| 0.03 | 0.52 | [ast_HSN#1_2024-11-28_143827](https://wandb.ai/deepbirddetect/BioFoundation/runs/ast_linearprobing_BirdSet_HSN_1_2024-11-28_143827) |
| AudioMAE| ? | ? | ? |
| ConvNext_BS| 0.05 | 0.57 | [convnext_HSN#1_2024-11-29_131825](https://wandb.ai/deepbirddetect/BioFoundation/runs/convnext_linearprobing_BirdSet_HSN_1_2024-11-29_131825) # Episode 00 |
| HUBERT| ? | ? | ? | 
| SSAST| ? | ? | ? |
| Wav2Vec2| ? | ? | ? |
| BirdNET| ? | ? | ? |

### Running FewShot Experiments on BirdSet

```bash
python birdset/train.py experiment=biofoundation/birdset/fewshot/{model_name}
```

Results on HSN with 32 samples per class:


| Model | cmAP | AUROC | Wandb |
|-------| -------| ---- | ---- |
| BEATS | 0.10 | **0.66** | [BEATs_HSN#3_2024-11-25_160815](https://wandb.ai/deepbirddetect/BioFoundation/runs/beats_fewshot_BirdSet_HSN_3_2024-11-25_160815) |
| Perch | **0.14** | 0.65 | [perch_HSN#1_2024-11-25_180458](https://wandb.ai/deepbirddetect/BioFoundation/runs/perch_fewshot_BirdSet_HSN_1_2024-11-25_180458) |
| BioLingual| ? | ? | ? |
| ConvNext| 0.03 | 0.48 | [convnext_HSN#1_2024-11-29_125505](https://wandb.ai/deepbirddetect/BioFoundation/runs/convnext_fewshot_BirdSet_HSN_1_2024-11-29_125505) |
| EAT| ? | ? | ? |
| AVES| 0.04 | 0.53 | [aves_HSN#1_2024-11-28_134553](https://wandb.ai/deepbirddetect/BioFoundation/runs/aves_fewshot_BirdSet_HSN_1_2024-11-28_134553) |
| AST| 0.03 | 0.53 | [ast_HSN#1_2024-11-28_163304](https://wandb.ai/deepbirddetect/BioFoundation/runs/ast_fewshot_BirdSet_HSN_1_2024-11-28_163304) |
| AudioMAE| ? | ? | ? |
| ConvNext_BS| 0.04 | 0.50 | [convnext_HSN#1_2024-11-29_130609](https://wandb.ai/deepbirddetect/BioFoundation/runs/convnext_fewshot_BirdSet_HSN_1_2024-11-29_130609) |
| HUBERT| ? | ? | ? |
| SSAST| ? | ? | ? |
| Wav2Vec2| ? | ? | ? |
| BirdNET| ? | ? | ? |

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
python bridset/train.py experiment="local/HSN/efficientnet.yaml"
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