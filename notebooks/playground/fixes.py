# %%
from birdset.datamodule import DatasetConfig
from birdset.datamodule.birdset_datamodule import BirdSetDataModule

# %%
# initiate the data module
dm = BirdSetDataModule(
    dataset=DatasetConfig(
        data_dir="../../data_birdset/HSN",
        hf_path="DBD-research-group/BirdSet",
        hf_name="HSN",
        val_split=0.01,
        task="multilabel",
        classlimit=500,
        eventlimit=5,
        sampling_rate=32_000,
    ),
)

# %%
from birdset.datamodule.esc50_datamodule import ESC50DataModule

dm = ESC50DataModule(
    dataset=DatasetConfig(
        data_dir="../../data_birdset",
        hf_path="ashraq/esc50",
        hf_name="esc50",
        seed=1,
        n_workers=1,
        val_split=0.2,
        task="multiclass",
        subset=None,
        sampling_rate=44100,
    ),
)
# %%
dm.prepare_data()
# %%
dm.disk_save_path
# %%
from datasets import load_from_disk

ds = load_from_disk(dm.disk_save_path)

# %%
from datasets import Audio

ds["test"][0]

ds = ds.cast_column("filepath", Audio(sampling_rate=32_000))

# %%

ds["train"][0]
# %%

dm.setup(stage="fit")

# %%
dm.train_dataset
# %%
dm.train_dataset[0]

# %%


# %%
from datasets import load_dataset

test = load_dataset(path="ashraq/esc50", name=None, split="train")
