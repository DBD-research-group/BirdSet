#%%
from datasets import load_dataset

if __name__ == "__main__":    
    ds_name = "columbia_costa_rica"
    print(f"start {ds_name}")
    ds = load_dataset(
        "dbd-research-group/gadme_v1", 
        ds_name,
        cache_dir="~/projects/GADME/data_gadme",
        num_proc=3
    )
    print("finished")
#%%
from datasets import Audio

ds = ds.cast_column(
    column="audio",
    feature=Audio(
        sampling_rate=32_000,
        mono=True,
        decode=False
    )
)
#%%
ds["train"][0]
# %%

from huggingface_hub import hf_hub_download

meta = hf_hub_download(
    repo_id="dbd-research-group/gadme_v1",
    filename="test.parquet",
    revision="data",
    subfolder="data/amazon_basin",
    repo_type="dataset"
)
#%%
import pandas as pd 


pd.read_parquet(meta)