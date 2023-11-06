from datasets import load_dataset

names = ["hawaiian_islands", "nips", "powdermill_nature", "sapsucker_woods"]



for name in names: 
    try:

        load_dataset(
            name=name,
            path="DBD-research-group/gadme_v1",
            cache_dir="/home/lukas/projects/GADME/data_gadme",
            num_proc=5
        )
    except:
        print(f"An error occured with {name}") 



#%%
from datasets import load_dataset 

ds = load_dataset(
        name="columbia_costa_rica",
        path="DBD-research-group/gadme_v1",
        cache_dir="/home/lukas/projects/GADME/data_gadme",
        num_proc=5
    )


#%%

ds = load_dataset(
        name="amazon_basin",
        path="DBD-research-group/gadme_v1",
        cache_dir="/home/lukas/projects/GADME/data_gadme",
        num_proc=5
    )

#%%
ds["train"][0]["audio"]
#%%

df_train = ds["train"].to_pandas()
df_test = ds["test"].to_pandas()
#%%
ds["train"].features
#%%

df_train.head(5)
#%%
df_test.head(5)