<div align="center">
  <img src="https://github.com/DBD-research-group/BirdSet/blob/main/resources/perch/birdsetsymbol.png" alt="logo" width="100">
</div>

# $\texttt{BirdSet}$ - A Dataset for Audio Classification in Avian Bioacoustics 🤗
[![python](https://img.shields.io/badge/-Python_3.10-blue?logo=python&logoColor=white)](https://github.com/pre-commit/pre-commit)
<a href="https://huggingface.co/"><img alt="Hugging Face" src="https://img.shields.io/badge/HuggingFace-ffcc00?logo=huggingface&logoColor=white"></a>
<a href="https://pytorch.org/get-started/locally/"><img alt="PyTorch" src="https://img.shields.io/badge/PyTorch-ee4c2c?logo=pytorch&logoColor=white"></a>
<a href="https://www.pytorchlightning.ai/"><img alt="PyTorch Lightning" src="https://img.shields.io/badge/PyTorch_Lightning-792ee5?logo=pytorch-lightning&logoColor=white"></a>
<a href="https://hydra.cc/"><img alt="Config: Hydra" src="https://img.shields.io/badge/Config-Hydra-89b8cd"></a>
[![arXiv paper](https://img.shields.io/badge/arXiv-1234.56789-b31b1b.svg)](https://arxiv.org/abs/2403.10380)


Deep learning (DL) has greatly advanced audio classification, yet the field is limited by the scarcity of large-scale benchmark datasets that have propelled progress in other domains. While AudioSet aims to bridge this gap as a universal-domain dataset, its restricted accessibility and lack of diverse real-world evaluation use cases challenge its role as the only resource. Additionally, to maximize the potential of cost-effective and minimal-invasive passive acoustic monitoring (PAM), models must analyze bird vocalizations across a wide range of species and environmental conditions. Therefore, we introduce $\texttt{BirdSet}$, a large-scale benchmark dataset for audio classification focusing on avian bioacoustics. $\texttt{BirdSet}$ surpasses AudioSet with over 6,800 recording hours ($\uparrow\!17\%$) from nearly 10,000 classes ($\uparrow\!18\times$) for training and more than 400 hours ($\uparrow\!7\times$) across eight strongly labeled evaluation datasets. It serves as a versatile resource for use cases such as multi-label classification, covariate shift or self-supervised learning.

<br>
<div align="center">
  <img src="https://github.com/DBD-research-group/BirdSet/blob/main/resources/graphical_abstract.png" alt="logo", width=950>
</div>

<br>

**TL;DR**
> - Explore our **datasets** shared on Hugging Face 🤗 in the [BirdSet repository](https://huggingface.co/datasets/DBD-research-group/BirdSet).
> - This accompanying **code** provides comprehensive support tool for data preparation, model training, and evaluation. 
> - Participate in our Hugging Face [leaderboard](https://huggingface.co/spaces/DBD-research-group/BirdSet-Leaderboard) by submitting new results and comparing performance across models.
> - Access our pre-trained [model checkpoints](https://huggingface.co/collections/DBD-research-group/birdset-dataset-and-models-665ef710a28cbe70dfaa028a) on Hugging Face, ready to fine-tune or evaluate for various tasks.

<br>

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

## Update (2024.11.27) 
- Additional bird taxonomy metadata, including "Genus," "Species Group," and "Order," is provided using the 2021 eBird taxonomy, consistent with the taxonomy used for the 'ebird_code' data. These metadata fields follow the same format and encoding as 'ebird_code' and 'ebird_code_multilabel'. Further explanation can be found on our Hugging Face [BirdSet repository](https://huggingface.co/datasets/DBD-research-group/BirdSet).

- If you don't require the additional taxonomy at the moment and prefer to **avoid re-downloading all files**, you can specify the previous revision directly in load_dataset as follows:

```python
from datasets import load_dataset
ds = load_dataset("DBD-research-group/BirdSet", "HSN", trust_remote_code=True, revision="629b54c06874b6d2fa886e1c0d73146c975612d0")
```


## User Installation 🐣

The simplest way to install $\texttt{BirdSet}$ is to clone this repository and install it as an editable package using [conda](https://docs.conda.io/en/latest/) and [pip](https://pip.pypa.io/en/stable/):
```
conda create -n birdset python=3.10
pip install -e .
```
or editable in your own repository: 
```
pip install -e git+https://github.com/DBD-research-group/BirdSet.git#egg=birdset
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

We offer an in-depth [tutorial notebook](https://github.com/DBD-research-group/BirdSet/blob/main/notebooks/tutorials/birdset-pipeline_tutorial.ipynb) on how to use this repository. In the following, we provide simple code snippets:

### Manual Data Preparation

You can manually download the datasets from Hugging Face. We offer a uniform metadata format but also provide flexibility on how to prepare the data (e.g. you can manually decide which events to filter from the training data). The dataset dictionary comes with: 

- `train`: Focal instance with variable lengths. Possible `detected_events` and corresponding event clusters are provided.  
- `test_5s`: Processed test datasets where each soundscape instance corresponds to a 5-second clip with a `ebird_code_multilabel` format.  
- `test`: Unprocessed test datasets where each soundscape instance points to the full soundscape recording and the correspoding `ebird_code` with ground truth `start_time` and `end_time`.

 
```python
from datasets import load_dataset, Audio

# download the dataset 
dataset = load_dataset("DBD-research-group/BirdSet","HSN")

# set HF decoder (decodes the complete file!)
dataset = dataset.cast_column("audio", Audio(sampling_rate=32_000))

```

The `audio` column natively contains only file paths. While automatic decoding via HF can be enabled (as shown above), decoding the entire audio files can introduce computational redundancies. This is because we provide flexible event decoding with varying file lengths that are often much longer than the targeted 5 seconds. To optimize, consider using a custom decoding scheme (e.g., with soundfile/BirdSet) or preprocessing the dataset with `.map` to include only the relevant audio segments.

### BirdSet: Data Preparation :bird:

This code snippet utilizes the datamodule for an example dataset $\texttt{HSN}$. 

**prepare_data**
>- downloads the data (or loads from cache)
>- preprocesses the data
>    - event_mapping (extract n events from each sample. this could expand the training dataset and provides event timestamps for each sample)
>    - one-hot encoding (classses for multi-label)
>    - create splits
>- saves dataset to disk (path can be accessed with `dm.disk_save_path` and loaded with `datasets.load_from_disk`)
  
```python
from birdset.configs.datamodule_configs import DatasetConfig, LoadersConfig
from birdset.datamodule.components.transforms import BirdSetTransformsWrapper
from birdset.datamodule.birdset_datamodule import BirdSetDataModule
from datasets import load_from_disk

# initiate the data module
dm = BirdSetDataModule(
    dataset= DatasetConfig(
        data_dir='data_birdset/HSN', # specify your data directory!
        hf_path='DBD-research-group/BirdSet',
        hf_name='HSN',
        n_workers=3,
        val_split=0.2,
        task="multilabel",
        classlimit=500, #limit of samples per class 
        eventlimit=5, #limit of events that are extracted for each sample
        sampling_rate=32000,
    ),
    loaders=LoadersConfig(), # only utilized in setup; default settings
    transforms=BirdSetTransformsWrapper() # set_transform in setup; default settings to spectrogram
)

# prepare the data
dm.prepare_data()

# manually load the complete prepared dataset (without any transforms). you have to cast the column with audio for decoding
ds = load_from_disk(dm.disk_save_path)
```

The dataset is now split into training, validation, and test sets, with each sample corresponding to a unique event in a sound file. A sample output from the training set looks like this:

```python
{
 'filepath': 'filepath.ogg',
 'labels': array([0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0]),
 'detected_events': array([1.216, 3.76 ], dtype=float32), # only in train. begin and end of event within the file 
 'start_time': nan, # only in test, segment start and segment end within the soundfile
 'end_time': nan
}
```

You can now create a custom loading script. For instance: 

```python
def load_audio(sample, min_len, max_len, sampling_rate):
    path = sample["filepath"]
    
    if sample["detected_events"] is not None:
        start = sample["detected_events"][0]
        end = sample["detected_events"][1]
        event_duration = end - start
        
        if event_duration < min_len:
            extension = (min_len - event_duration) / 2
            
            # try to extend equally 
            new_start = max(0, start - extension)
            new_end = min(total_duration, end + extension)
            
            if new_start == 0:
                new_end = min(total_duration, new_end + (start - new_start))
            elif new_end == total_duration:
                new_start = max(0, new_start - (new_end - end))
            
            start, end = new_start, new_end

        if end - start > max_len:
            # if longer than max_len
            end = min(start + max_len, total_duration)
            if end - start > max_len:
                end = start + max_len
    else:
        start = sample["start_time"]
        end = sample["end_time"]

    file_info = sf.info(path)
    sr = file_info.samplerate
    total_duration = file_info.duration

    start, end = int(start * sr), int(end * sr)
    audio, sr = sf.read(path, start=start, stop=end)

    if audio.ndim != 1:
        audio = audio.swapaxes(1, 0)
        audio = librosa.to_mono(audio)
    if sr != sampling_rate:
        audio = librosa.resample(audio, orig_sr=sr, target_sr=sampling_rate)
        sr = sampling_rate
    return audio, sr

audio_train, _ = load_audio(ds["train"][11], min_len=5, max_len=5, sampling_rate=32_000) # loads a 5 second clip around the detected event
audio_test, _ = load_audio(ds["test"][30], min_len=5, max_len=5, sampling_rate=32_000) # loads a 5 second test segment
```

or utilize the BirdSet `set_transform` with built-in event decoding etc.: 

**setup**
>- sets up and loads the dataset for training and evaluating
>- adds `set_transforms` that transforms on-the-fly (decoding, spectrogram conversion, augmentation etc.)

```python
# OR setup the datasets with BirdSet ("test" for testdata)
# this includes the set_transform with processing/specrogram conversion etc. 
dm.setup(stage="fit")

# audio is now decoded when a sample is called
train_ds = dm.train_dataset
val_ds = dm.val_dataset

# get the dataloaders
train_loader = dm.train_dataloader()
```

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
<!-- 
## Reproduce Neurips2024 Baselines 🚧

> This repository is still under active development. You can access the NeurIPS 24 code at the tag  `neurips2024`
> ```bash
> git checkout neurips2024 

First, you have to download the background noise files for augmentations

```bash
python resources/utils/download_background_noise.py
```

We provide all experiment YAML files used to generate our results in the path `birdset/configs/experiment/birdset_neurips24`. For each dataset, we specify the parameters for all training scenario: `DT`, `MT`, and `LT`

### Dedicated Training (DT)

The experiments for `DT` with the dedicated subset can be easily run with a single line: 

``` bash
python birdset/train.py experiment="birdset_neurips24/$Dataset/DT/$Model"
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

**Disclaimer on results:** The results obtained using the `eval.py` script may differ from those reported in the paper. This discrepancy is because only the "best" model checkpoint was uploaded to Hugging Face, whereas the paper’s results were averaged over three different random seeds for a more robust evaluation.
-->
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

We have also added class information to the models on HF. You can find them in the config of the respective models.

-------
#### **Why are the datasets larger than expected?**

Currently, our HF builder script extracts all zipped files to ensure clear file paths while retaining the original zipped files. This results in increased storage requirements. Although resolving this issue is complex, we are actively working on a solution and aim to provide updates soon.

_Example_:  
For the largest dataset, `XCL`, the zipped files are approximately 480GB. However, due to the extraction process, you’ll need around 990GB of available disk space. After the extraction, the zipped files will account for roughly 510GB.  

*Quick Workaround*:  
After extraction, you can delete unnecessary files by running in `XCL/downloads/`
```bash
find . -mindepth 1 -maxdepth 1 ! -name 'extracted' -exec rm -rfv {} +
```
While we could omit the unzipping process entirely, this would prevent us from maintaining accurate file paths and relying on soundfile to read directly from the zipped files.

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

Data transformations are referred to data transformations that are applied to the data during training. They include e.g. augmentations. The transformations are added to the Hugging Face dataset with [`set_transform`](https://huggingface.co/docs/datasets/main/en/package_reference/main_classes#datasets.Dataset.set_transform).

-->



