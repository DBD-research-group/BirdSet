#%%
from datasets import load_dataset

if __name__ == "__main__":    
    ds_name = "high_sierras"
    print(f"start {ds_name}")
    ds = load_dataset(
        "dbd-research-group/gadme_v1", 
        ds_name,
        cache_dir="~/projects/GADME/data_gadme",
        num_proc=3,
        download_mode='force_redownload'
    )
    print("finished")
#%%
ds["train"]["peaks"]
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
import pandas as pd 

folder = "data/xenocanto"
meta0 = hf_hub_download(
    repo_id="dbd-research-group/gadme_v1",
    filename="train.parquet",
    revision="data",
    subfolder=folder,
    repo_type="dataset"
)

meta1 = hf_hub_download(
    repo_id="dbd-research-group/gadme_v1",
    filename="test.parquet",
    revision="data",
    subfolder=folder,
    repo_type="dataset"
)

# meta2 = hf_hub_download(
#     repo_id="dbd-research-group/gadme_v1",
#     filename="test_5s.parquet",
#     revision="data",
#     subfolder=folder,
#     repo_type="dataset"
# )
print(pd.read_parquet(meta0)["ebird_code"].nunique())
print(pd.read_parquet(meta1)["ebird_code"].nunique())
#print(pd.read_parquet(meta2)["ebird_code"].nunique())

#%%
from huggingface_hub import hf_hub_download
import pandas as pd 

folder = "data/amazon_basin"
meta_test = hf_hub_download(
    repo_id="dbd-research-group/gadme_v1",
    filename="train.parquet",
    revision="data",
    subfolder=folder,
    repo_type="dataset"
)
#%%
meta_test = pd.read_parquet(meta_test)
#%%
meta_test
#%%
col_index = meta_test.columns.get_loc("ebird_code") + 1
meta_test.insert(col_index, "ebird_code_multiclass", meta_test["ebird_code"])
meta_test["ebird_code"] = None
meta_test.insert(col_index+2, "ebird_code_secondary", None)
#%%

meta_test
#%%
df = pd.read_parquet(meta0)
#%%
len(df["ebird_code"].unique())
#%%
df.head()
#%%
from datasets import load_dataset
ds = load_dataset("dbd-research-group/gadme_v1", "iit")

#%%
import pandas as pd 


pd.read_parquet(meta)