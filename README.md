# BioFoundation

# $\texttt{BirdSet}$ - A Dataset for Audio Classification in Avian Bioacoustics 🤗
[![python](https://img.shields.io/badge/-Python_3.10-blue?logo=python&logoColor=white)](https://github.com/pre-commit/pre-commit)
<a href="https://huggingface.co/"><img alt="Hugging Face" src="https://img.shields.io/badge/HuggingFace-ffcc00?logo=huggingface&logoColor=white"></a>
<a href="https://pytorch.org/get-started/locally/"><img alt="PyTorch" src="https://img.shields.io/badge/PyTorch-ee4c2c?logo=pytorch&logoColor=white"></a>
<a href="https://www.pytorchlightning.ai/"><img alt="PyTorch Lightning" src="https://img.shields.io/badge/PyTorch_Lightning-792ee5?logo=pytorch-lightning&logoColor=white"></a>
<a href="https://hydra.cc/"><img alt="Config: Hydra" src="https://img.shields.io/badge/Config-Hydra-89b8cd"></a>


  <img src="https://github.com/DBD-research-group/BirdSet/blob/main/resources/graphical_abstract.png" alt="logo", width=950>
</div>

<br>

**TL;DR**
> - Explore our **datasets** shared on Hugging Face 🤗 in the [BirdSet repository](https://huggingface.co/datasets/DBD-research-group/BirdSet).
> - This accompanying **code** provides comprehensive support tool for data preparation, model training, and evaluation. 
> - Participate in our Hugging Face [leaderboard](https://huggingface.co/spaces/DBD-research-group/BirdSet-Leaderboard) by submitting new results and comparing performance across models.
> - Access our pre-trained [model checkpoints](https://huggingface.co/collections/DBD-research-group/birdset-dataset-and-models-665ef710a28cbe70dfaa028a) on Hugging Face, ready to fine-tune or evaluate for various tasks.
> - A Q&A section is included at the end of this README. If you have further questions or encounter any issues, please raise an issue. 
<br>

<div align="center">
  
|                            | **Task**                                     | **Description** | **# Train Recordings** | **# Test\_5s Segments** | **Pielou’s evenness J** | **# Species**   |
|----------------------------|----------------------------------------------|-----------------|-----------|--------------|-------|----------|
| **Large Train**                  | [XCL](https://xeno-canto.org/)               | Complete Xeno-Canto snapshot with focals for large (pre-) training.                | 528,434   | -            | -     | 9,734    |
|                            | [XCM](https://xeno-canto.org/)               | Smaller subset of XCL only containing focals of bird species available in test datasets.                 | 89,798    | -            | -     | 409      |
| **Auxiliary**              | [POW](https://zenodo.org/records/4656848)    | Powdermill Nature soundscape validation dataset and class-dedicated focal training subset of XCL.     | 14,911    | 4,560        | 0.66  | 48       |
|                            | [VOX](https://zenodo.org/records/1208080)    | BirdVox-DCASE soundscape background dataset without bird vocalizations.              | 20,331    | -            | -     | -        |
| **Test & Dedicated Train** | [PER](https://zenodo.org/records/7079124) | Amazon Basin soundscape test dataset and class-dedicated focal training subset.                 | 16,802    | 15,120       | 0.78  | 132      |
|        Train Subsets XCL!                    | [NES](https://zenodo.org/records/7525349)    |  Columbia Costa Rica soundscape test dataset and class-dedicated focal training subset.               | 16,117    | 24,480       | 0.76  | 89       |
|                            | [UHH](https://zenodo.org/records/7078499)    |  Hawaiian Islands soundscape test dataset and class-dedicated focal training subset.               | 3,626     | 36,637       | 0.64  | 25       |
|                            | [HSN](https://zenodo.org/records/7525805)    |  High Sierras Nevada soundscape test dataset and class-dedicated focal training subset.               | 5,460     | 12,000       | 0.54  | 21       |
|                            | [NBP](https://link-to-birddb)                |  NIPS4BPlus test dataset and class-dedicated focal training subset.               | 24,327    | 563          | 0.92  | 51       |
|                            | [SSW](https://zenodo.org/records/7018484)    |  Sapsucker Woods soundscape test dataset and class-dedicated focal training.               | 28,403    | 205,200      | 0.77  | 81       |
|                            | [SNE](https://zenodo.org/records/7050014)    |  Sierre Nevada soundscape test dataset and class-dedicated focal training subset.               | 19,390    | 23,756       | 0.70  | 56       |

</div>

## User Installation 🐣

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
./projects/biofoundation/train.sh experiment=birdset/finetuning/{model_name}
```

#### Results

Results on HSN:

| Model | cmAP | AUROC | Wandb |
|-------| -------| ---- | ---- |
| BEATs| 0.44 | 0.87 | [BEATs_HSN#1_2024-11-22_135915](https://wandb.ai/deepbirddetect/BioFoundation/runs/beats_finetune_HSN_1_2024-11-22_135915) |
| BioLingual| 0.33 | 0.79 | [biolingual_HSN#1_2024-11-29_110143](https://wandb.ai/deepbirddetect/BioFoundation/runs/biolingual_finetune_BirdSet_HSN_1_2024-11-29_110143) |
| ConvNext| 0.41 | 0.84 | [convnext_HSN#1_2024-11-29_130206](https://wandb.ai/deepbirddetect/BioFoundation/runs/convnext_finetune_BirdSet_HSN_1_2024-11-29_130206) |
| EAT| 0.22 | 0.69 | [eat_HSN#1_2024-12-01_174320](https://wandb.ai/deepbirddetect/BioFoundation/runs/eat_finetune_BirdSet_HSN_1_2024-12-01_174320) |
| AVES| 0.19 | 0.68 | [aves_HSN#1_2024-11-28_123701](https://wandb.ai/deepbirddetect/BioFoundation/runs/aves_finetune_BirdSet_HSN_1_2024-11-28_123701) (ES: Only 7 epochs)|
| AST| 0.21 | 0.70 | [ast_HSN#1_2024-11-28_163020](https://wandb.ai/deepbirddetect/BioFoundation/runs/ast_finetune_BirdSet_HSN_1_2024-11-28_163020) |
| AudioMAE| 0.34 | 0.83 |  [audio_mae_HSN#1_2024-11-29_162609](https://wandb.ai/deepbirddetect/BioFoundation/runs/audiomae_finetuning_BirdSet_HSN_1_2024-11-29_162609) |
| ConvNext_BS| **0.51** | **0.88** | [convnext_bs_HSN#1_2024-11-29_154136](https://wandb.ai/deepbirddetect/BioFoundation/runs/convnext_bs_finetune_BirdSet_HSN_1_2024-11-29_154136) |
| HUBERT| 0.32 | 0.80 | [hubert_HSN#1_2024-11-29_142052](https://wandb.ai/deepbirddetect/BioFoundation/runs/hubert_finetune_BirdSet_HSN_1_2024-11-29_142052) |
| SSAST| 0.19 | 0.68 | [ssast_HSN#1_2024-12-03_181308](https://wandb.ai/deepbirddetect/BioFoundation/runs/ssast_finetune_BirdSet_HSN_1_2024-12-03_181308) |
| EAT_SSL| 0.10 | 0.61 | [eat_ssl_HSN#1_2024-12-01_181129](https://wandb.ai/deepbirddetect/BioFoundation/runs/eat_ssl_finetune_BirdSet_HSN_1_2024-12-01_181129) |
| Wav2Vec2| ? | ? | ? |
| BirdNET| ? | ? | ? |

### Running Linear Probing Experiments on BirdSet

```bash
./projects/biofoundation/train.sh experiment=birdset/linearprobing/{model_name}
```

Results on HSN:

| Model | cmAP | AUROC | Wandb |
|-------| -------| ---- | ---- |
| BEATS | 0.11 | **0.73** | [BEATs_HSN#1_2024-11-25_155526](https://wandb.ai/deepbirddetect/BioFoundation/runs/beats_linearprobing_BirdSet_HSN_1_2024-11-25_155526) |
| Perch | **0.22** | 0.66 | [perch_HSN#1_2024-11-25_175223](https://wandb.ai/deepbirddetect/BioFoundation/runs/perch_linearprobing_BirdSet_HSN_1_2024-11-25_175223) |
| BioLingual| 0.12 | 0.75 | [biolingual_HSN#1_2024-11-29_111328](https://wandb.ai/deepbirddetect/BioFoundation/runs/biolingual_linearprobing_BirdSet_HSN_1_2024-11-29_111328) # Episode 26 |
| ConvNext| 0.03 | 0.52 | [convnext_HSN#1_2024-11-29_131024](https://wandb.ai/deepbirddetect/BioFoundation/runs/convnext_linearprobing_BirdSet_HSN_1_2024-11-29_131024) # Episode 00  |
| EAT| 0.20 | 0.62 | [eat_HSN#1_2024-12-01_170452](https://wandb.ai/deepbirddetect/BioFoundation/runs/eat_linearprobing_BirdSet_HSN_1_2024-12-01_170452) |
| AVES| 0.04 | 0.63 | [aves_HSN#1_2024-11-28_112422](https://wandb.ai/deepbirddetect/BioFoundation/runs/aves_linearprobing_BirdSet_HSN_1_2024-11-28_112422) |
| AST| 0.03 | 0.52 | [ast_HSN#1_2024-11-28_143827](https://wandb.ai/deepbirddetect/BioFoundation/runs/ast_linearprobing_BirdSet_HSN_1_2024-11-28_143827) |
| AudioMAE| ? | ? | ? |
| ConvNext_BS| 0.07 | 0.61 | [convnext_bs_HSN#1_2024-11-29_174232](https://wandb.ai/deepbirddetect/BioFoundation/runs/convnext_bs_linearprobing_BirdSet_HSN_1_2024-11-29_174232) |
| HUBERT| 0.07 | 0.57 | [hubert_HSN#1_2024-11-29_142350]( https://wandb.ai/deepbirddetect/BioFoundation/runs/hubert_linearprobing_BirdSet_HSN_1_2024-11-29_142350) |
| SSAST| 0.03 | 0.45 | [ssast_HSN#1_2024-12-01_171656](https://wandb.ai/deepbirddetect/BioFoundation/runs/ssast_linearprobing_BirdSet_HSN_1_2024-12-01_171656) |
| EAT_SSL| 0.21 | 0.79 | [eat_ssl_HSN#1_2024-12-01_181122](https://wandb.ai/deepbirddetect/BioFoundation/runs/eat_ssl_linearprobing_BirdSet_HSN_1_2024-12-01_181122) |
| Wav2Vec2| 0.03 | 0.45 | [wav2vec2_HSN#1_2024-11-29_171616](https://wandb.ai/deepbirddetect/BioFoundation/runs/wav2vec2_linearprobing_BirdSet_HSN_1_2024-11-29_171616)  |
| BirdNET| ? | ? | ? |

### Running FewShot Experiments on BirdSet

```bash
./projects/biofoundation/train.sh experiment=birdset/fewshot/{model_name}
```

Results on HSN with 32 samples per class:


| Model | cmAP | AUROC | Wandb |
|-------| -------| ---- | ---- |
| BEATS | 0.10 | **0.66** | [BEATs_HSN#3_2024-11-25_160815](https://wandb.ai/deepbirddetect/BioFoundation/runs/beats_fewshot_BirdSet_HSN_3_2024-11-25_160815) |
| Perch | 0.14 | 0.65 | [perch_HSN#1_2024-11-25_180458](https://wandb.ai/deepbirddetect/BioFoundation/runs/perch_fewshot_BirdSet_HSN_1_2024-11-25_180458) |
| BioLingual| 0.07 | 0.49 | [biolingual_HSN#1_2024-11-29_171140](https://wandb.ai/deepbirddetect/BioFoundation/runs/biolingual_fewshot_BirdSet_HSN_1_2024-11-29_171140) |
| ConvNext| 0.03 | 0.48 | [convnext_HSN#1_2024-11-29_125505](https://wandb.ai/deepbirddetect/BioFoundation/runs/convnext_fewshot_BirdSet_HSN_1_2024-11-29_125505) |
| EAT| **0.17** | 0.63 | [eat_HSN#1_2024-12-01_170656](https://wandb.ai/deepbirddetect/BioFoundation/runs/eat_fewshot_BirdSet_HSN_1_2024-12-01_170656) |
| AVES| 0.04 | 0.53 | [aves_HSN#1_2024-11-28_134553](https://wandb.ai/deepbirddetect/BioFoundation/runs/aves_fewshot_BirdSet_HSN_1_2024-11-28_134553) |
| AST| 0.03 | 0.53 | [ast_HSN#1_2024-11-28_163304](https://wandb.ai/deepbirddetect/BioFoundation/runs/ast_fewshot_BirdSet_HSN_1_2024-11-28_163304) |
| AudioMAE| 0.03 | 0.47 |[audio_mae_HSN#1_2024-11-29_152352](https://wandb.ai/deepbirddetect/BioFoundation/runs/audiomae_fewshot_BirdSet_HSN_1_2024-11-29_1523) |
| ConvNext_BS| 0.04 | 0.50 | [convnext_HSN#1_2024-11-29_130609](https://wandb.ai/deepbirddetect/BioFoundation/runs/convnext_fewshot_BirdSet_HSN_1_2024-11-29_130609) |
| HUBERT| 0.05 | 0.53 | [hubert_HSN#1_2024-11-29_141537](https://wandb.ai/deepbirddetect/BioFoundation/runs/hubert_fewshot_BirdSet_HSN_1_2024-11-29_141537) |
| SSAST| 0.032 | 0.46 | [ssast_HSN#1_2024-12-01_174950](https://wandb.ai/deepbirddetect/BioFoundation/runs/ssast_fewshot_BirdSet_HSN_1_2024-12-01_174950) |
| EAT_SSL| 0.02 | 0.30 | [eat_ssl_HSN#1_2024-12-01_180844](https://wandb.ai/deepbirddetect/BioFoundation/runs/eat_ssl_fewshot_BirdSet_HSN_1_2024-12-01_180844) |
| Wav2Vec2| 0.03 | 0.47 | [wav2vec2_HSN#1_2024-11-29_172107](https://wandb.ai/deepbirddetect/BioFoundation/runs/wav2vec2_fewshot_BirdSet_HSN_1_2024-11-29_172107) |
| BirdNET| ? | ? | ? |

## BEANS

On the BEANS benchmark we also run the three different experiments but in a multiclass scenario:

### Running Fine-tuning Experiments on BEANS

```bash
./projects/biofoundation/train.sh experiment=beans/finetuning/{model_name}
```

#### Results

Results on Watkins:

| Model | T1 | AUROC | Wandb |
|-------| -------| ---- | ---- |
| BEATs| 0.91 | 0.99 | [BEATs_finetune_BEANS_beans_watkins_1_2025-01-02_142041](https://wandb.ai/deepbirddetect/BioFoundation/runs/BEATs_finetune_BEANS_beans_watkins_1_2025-01-02_142041) |
| BioLingual|0.88 | 0.98 | [biolingual_finetune_beans_watkins_1_2025-01-04_164323](https://wandb.ai/deepbirddetect/BioFoundation/runs/biolingual_finetune_beans_watkins_1_2025-01-04_164323) |
| ConvNext| 0.89 | 0.99 | [convnext_finetune_beans_watkins_1_2025-01-04_173811](https://wandb.ai/deepbirddetect/BioFoundation/runs/convnext_finetune_beans_watkins_1_2025-01-04_173811) |
| EAT| 0.76 | 0.98 | [eat_finetune_beans_watkins_1_2025-01-06_122645](https://wandb.ai/deepbirddetect/BioFoundation/runs/eat_finetune_beans_watkins_1_2025-01-06_122645) |
| AVES| 0.76 | 0.98 | [aves_finetune_beans_watkins_1_2025-01-06_123523](https://wandb.ai/deepbirddetect/BioFoundation/runs/aves_finetune_beans_watkins_1_2025-01-06_123523) |
| AST|0.73 | 0.98 | [ast_finetune_beans_watkins_1_2025-01-09_122523](https://wandb.ai/deepbirddetect/BioFoundation/runs/ast_finetune_beans_watkins_1_2025-01-09_122523) |
| AudioMAE| 0.73 | 0.98 | [audio_mae_finetune_beans_watkins_1_2025-01-06_135838](https://wandb.ai/deepbirddetect/BioFoundation/runs/audio_mae_finetune_beans_watkins_1_2025-01-06_135838) |
| ConvNext_BS| 0.89 | 0.99 | [convnext_bs_finetune_beans_watkins_1_2025-01-06_120053](https://wandb.ai/deepbirddetect/BioFoundation/runs/convnext_bs_finetune_beans_watkins_1_2025-01-06_120053) |
| HUBERT| 0.85 | 0.99 | [hubert_finetune_beans_watkins_1_2025-01-09_140320](https://wandb.ai/deepbirddetect/BioFoundation/runs/hubert_finetune_beans_watkins_1_2025-01-09_140320) |
| SSAST| 0.79 | 0.98 | [ssast_finetune_beans_watkins_1_2025-01-09_142902](https://wandb.ai/deepbirddetect/BioFoundation/runs/ssast_finetune_beans_watkins_1_2025-01-09_142902) |
| EAT_SSL| 0.84 | 0.99 | [eat_ssl_finetune_beans_watkins_1_2025-01-04_161731](https://wandb.ai/deepbirddetect/BioFoundation/runs/eat_ssl_finetune_beans_watkins_1_2025-01-04_161731) |
| Wav2Vec2| 0.81 | 0.98 | [wav2vec2_finetune_beans_watkins_1_2025-01-09_145847](https://wandb.ai/deepbirddetect/BioFoundation/runs/wav2vec2_finetune_beans_watkins_1_2025-01-09_145847) |

### Running Linear Probing Experiments on BEANS

```bash
./projects/biofoundation/train.sh experiment=beans/linearprobing/{model_name}
```

Results on HSN:

| Model | T1 | AUROC | Wandb |
|-------| -------| ---- | ---- |
| BEATs| 0.86 | 0.99 | [BEATs_default#1_2025-01-02_144748](https://wandb.ai/deepbirddetect/BioFoundation/runs/BEATs_linearprobing_BEANS_beans_watkins_1_2025-01-02_144748) |
| BioLingual|? | ? | ? |
| Perch | ? | ? | ? |
| ConvNext| ? | ? | ? |
| EAT| ? | ? | ? |
| AVES| ? | ? | ? |
| AST|? | ? | ? |
| AudioMAE| ? | ? | ? |
| ConvNext_BS| ? | ? | ? |
| HUBERT| ? | ? | ? |
| SSAST| ? | ? | ? |
| EAT_SSL| ? | ? | ? |
| Wav2Vec2| ? | ? | ? |
| BirdNET| ? | ? | ? |

### Running FewShot Experiments on BEANS

```bash
./projects/biofoundation/train.sh experiment=beans/fewshot/{model_name}
```

Results on HSN with 32 samples per class:


| Model | T1 | AUROC | Wandb |
|-------| -------| ---- | ---- |
| BEATs| ? | ? | ? |
| BioLingual|? | ? | ? |
| Perch | ? | ? | ? |
| ConvNext| ? | ? | ? |
| EAT| ? | ? | ? |
| AVES| ? | ? | ? |
| AST|? | ? | ? |
| AudioMAE| ? | ? | ? |
| ConvNext_BS| ? | ? | ? |
| HUBERT| ? | ? | ? |
| SSAST| ? | ? | ? |
| EAT_SSL| ? | ? | ? |
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
| <sub>**BIRB: A Generalization Benchmark for Information Retrieval in Bioacoustics**</sub> | | | | | | | |  | | | |
## Logging
Logs will be written to [Weights&Biases](https://wandb.ai/) by default.
-->
## Background noise
To enhance model performance we mix in additional background noise from downloaded from the [DCASE18](https://dcase.community/challenge2018/index). To download the files and convert them to the correct format, run the notebook 'download_background_noise.ipynb' in the 'notebooks' folder.


## Reproduce Baselines

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
Additionally, the datasets are quite large (90GB for XCM and 480GB for XCL). Therefore, we provide the best model checkpoints via Hugging Face in the experiment files to avoid the need for retraining.
These checkpoints can be executed by running the evaluation script, which will automatically download the model and perform inference on the test datasets:

``` bash
python birdset/eval.py experiment="birdset_neurips24/$EXPERIMENT_PATH"
```

If you want to start the large-scale trainings and download the big training datasets, you can also employ the `XCM` and `XCL` trainings via the experiment YAML files. 

``` bash
python birdset/train.py experiment="birdset_neurips24/$EXPERIMENT_PATH"
```
After training, the best model checkpoint is saved based on the validation loss and can then be used for inference:

``` bash
python birdset/eval.py experiment="birdset_neurips24/$EXPERIMENT_PATH" module.model.network.local_checkpoint="$CHECKPOINT_PATH"
```




## Run experiments

Our experiments are defined in the `configs/experiment` folder. To run an experiment, use the following command in the directory of the repository:

``` bash
python birdset/train.py experiment="EXPERIMENT_PATH"
```

Replace `EXPERIMENT_PATH` with the path to the experiment YAML config originating from the `experiment` directory. Here's a command for training an EfficientNet on HSN: 

``` bash
python birdset/train.py experiment="local/HSN/efficientnet.yaml"
```


## Q&A

#### **How to access the label names in the datasets?**
The class names are available in the Hugging Face datasets (with the [ClassLabel Feature](https://huggingface.co/docs/datasets/v3.1.0/en/package_reference/main_classes#datasets.ClassLabel))

```python
from datasets import load_dataset

dataset = load_dataset(
    "DBD-research-group/BirdSet", 
    "HSN", 
    cache_dir="the directory you used", 
    num_proc=1, 
    #revision="629b54c06874b6d2fa886e1c0d73146c975612d0" <-- if your cache directory is correct and a new download is starting,
    #you can use this revision (we added some metadata ~2 days ago which forces a redownload). if not, ignore this
)

dataset["train"].features["ebird_code"]
```
This should be the output: 
```
ClassLabel(names=['gcrfin', 'whcspa', 'amepip', 'sposan', 'rocwre', 'brebla', 'daejun', 'foxspa', ...], id=None)
```
These ebird codes should correspond to the respective columns in the label matrix. 
You could also `ds.features["label"].int2str(0)`

Additionally you can find JSON files containing `id2label` and `label2id` dictionaries for each dataset under the [resources/ebird_codes](https://github.com/DBD-research-group/BirdSet/tree/main/resources/ebird_codes) directory in the git repository.

[Issue](https://github.com/DBD-research-group/BirdSet/issues/280)

-------
#### **How to access the label names of the pre-trained models?**
The class list of pre-trained models corresponds to the datasets they were trained on (same indices). To get the class list, you can visit this [link on HF](https://huggingface.co/datasets/DBD-research-group/BirdSet/blob/main/classes.py) or use the following code example:

```python

import datasets 

dataset_meta = datasets.load_dataset_builder("dbd-research-group/BirdSet", "XCL")
dataset_meta.info.features["ebird_code"]
```

We have also added class information to the models on HF. You can find them in the config of the respective models. To access the model config you can refer to the following code snippet:

```python

from transformers import ConvNextForImageClassification

# load model
model = ConvNextForImageClassification.from_pretrained("DBD-research-group/ConvNeXT-Base-BirdSet-XCL")

# access label dicts
model.config.id2label # or model.config.label2id depending on what you need

```

`id2label` and `label2id` are dictionaries so to access a specific element you can do this:

```python

model.config.id2label[0]

```

In the case of XCL this should output `ostric2`.

**Please note:** Changing the last layer in any way (e.g. changing the output layer to 21 classes to fine-tune on HSN) will remove or invalidate that label information from the configs. In that case you will need to get that information differently. In case of BirdSet datasets you can look under [resources/ebird_codes](https://github.com/DBD-research-group/BirdSet/tree/main/resources/ebird_codes). The json files in that directory contain `label2id` and `id2label` dicts for every dataset.

-------
#### **Why are the datasets larger than expected? (should only apply to downloads before 05-12-2024! fixed)**

Currently, our HF builder script extracts all zipped files to ensure clear file paths while retaining the original zipped files. This results in increased storage requirements.

_Example_:  
For the largest dataset, `XCL`, the zipped files are approximately 480GB. However, due to the extraction process, you’ll need around 990GB of available disk space. After the extraction, the zipped files will account for roughly 510GB.  

*Quick Workaround*:  
After extraction, you can delete unnecessary files by running in `XCL/downloads/`
```bash
find . -mindepth 1 -maxdepth 1 ! -name 'extracted' -exec rm -rfv {} +
```
**This issue is fixed, more information: see Q below.**

------
#### **Hugging Face downloads the dataset again even though I already downloaded it**
We made a samll update fixing [Issue 267: Data download size descrepancy](https://github.com/DBD-research-group/BirdSet/issues/267) on **05-12-2024**:
- **This only works for datasets<3.0.0!**
- TL;DR: During the extraction process, unnecessary archives are now removed immediately. This reduces the required disk space by *half*, now aligning it with the table below.
- If you downloaded the data between this and last update and don't want to redownload yet, you can use the following `revision=b0c14a03571a7d73d56b12c4b1db81952c4f7e64`:
```python
from datasets import load_dataset
ds = load_dataset("DBD-research-group/BirdSet", "HSN", trust_remote_code=True, revision="b0c14a03571a7d73d56b12c4b1db81952c4f7e64")
```

We made a small update to the metadata on **27-11-2024**: 

- Additional bird taxonomy metadata, including "Genus," "Species Group," and "Order," is provided using the 2021 eBird taxonomy, consistent with the taxonomy used for the 'ebird_code' data. These metadata fields follow the same format and encoding as 'ebird_code' and 'ebird_code_multilabel'. Further explanation can be found on our Hugging Face [BirdSet repository](https://huggingface.co/datasets/DBD-research-group/BirdSet).

- If you don't require the additional taxonomy and prefer to **avoid re-downloading all files**, you can specify the previous revision directly in load_dataset as follows:

```python
from datasets import load_dataset
ds = load_dataset("DBD-research-group/BirdSet", "HSN", trust_remote_code=True, revision="629b54c06874b6d2fa886e1c0d73146c975612d0")
```


## Citation

```
@misc{rauch2024birdset,
      title={BirdSet: A Large-Scale Dataset for Audio Classification in Avian Bioacoustics}, 
      author={Lukas Rauch and Raphael Schwinger and Moritz Wirth and René Heinrich and Denis Huseljic and Marek Herde and Jonas Lange and Stefan Kahl and Bernhard Sick and Sven Tomforde and Christoph Scholz},
      year={2024},
      eprint={2403.10380},
      archivePrefix={arXiv},
      primaryClass={cs.SD},
      url={https://arxiv.org/abs/2403.10380}, 
}
```

<!-- 
# Data pipeline

Our datasets are shared via Hugging Face Datasets in our [BirdSet repository](https://huggingface.co/datasets/DBD-research-group/BirdSet).
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

Data transformations are referred to data transformations that are applied to the data during training. They include e.g. augmentations. The transformations are added to the huggingface dataset with [`set_transform`](https://huggingface.co/docs/datasets/main/en/package_reference/main_classes#datasets.Dataset.set_transform).
