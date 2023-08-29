import pandas as pd
from datasets import Dataset, Audio
from sklearn.model_selection import train_test_split


def train_test_val_split(metadata: pd.DataFrame, min_occ: int = 20):
    sizes = metadata.groupby("primary").size()
    sizes = sizes[sizes >= min_occ]
    sel = metadata[metadata["primary"].isin(sizes.index)]

    train, test = train_test_split(sel, test_size=0.2, stratify=sel["primary"])
    test, val = train_test_split(test, test_size=0.5, stratify=test["primary"])
    return train, test, val


def create_hg_dataset(metadata: pd.DataFrame, min_occ: int = 1) -> Dataset:
    sizes = metadata.groupby("primary").size()
    sizes = sizes[sizes >= min_occ]
    sel = metadata[metadata["primary"].isin(sizes.index)]

    ds = Dataset.from_pandas(sel)
    ds = ds.class_encode_column("primary")
    #ds = ds.cast_column("file_name", Audio(sampling_rate=16_000))
    return ds

