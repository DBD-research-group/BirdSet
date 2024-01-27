#%%
from datasets import load_dataset
from datasets import Audio 
import librosa
import pandas as pd 

ds = load_dataset(
        name="amazon_basin",
        path="DBD-research-group/gadme_v1",
        cache_dir="/home/lukas/projects/GADME/data_gadme",
        num_proc=5
    )


ds = ds.cast_column(
    column="audio",
    feature=Audio(
        sampling_rate=32_000,
        mono=True,
        decode=False
    )
)

ds_scape = ds["test"]
df_scape = ds_scape.to_pandas()


df_scape.head()
#%%

ds["test"].features
#%%
def generate_frames(duration, step=5):
    return [(i, i+step) for i in range(0, duration, step)]

def soundscape_generator(df):
    expanded_rows = []

    path_durations = {filepath: librosa.get_duration(path=filepath) for filepath in df_scape['filepath'].unique()}
    for file_path, duration in path_durations.items():
        frames = generate_frames(int(duration))
        filtered_df = df[df["filepath"] == file_path]

        for start, end in frames: 
            overlapping = filtered_df[(filtered_df['start_time'] < end) & (filtered_df['end_time'] > start)]
            if overlapping.empty:
                # If there is no overlapping, create a template row with default values
                template_row = filtered_df.iloc[0].copy()
                template_row['start_time'] = start
                template_row['end_time'] = end
                template_row['ebird_code'] = [-1]
                expanded_rows.append(template_row)
            else:
                # If there is overlapping, create a row for each unique ebird_code
                # We'll use the first row as a template and update necessary fields
                template_row = filtered_df.iloc[0].copy()
                template_row['start_time'] = start
                template_row['end_time'] = end
                template_row['ebird_code'] = overlapping['ebird_code'].unique().tolist()
                expanded_rows.append(template_row)
    
    expanded_df = pd.DataFrame(expanded_rows).reset_index(drop=True)
    return expanded_df
            
exp_df = soundscape_generator(df_scape)

#%%

exp_df.head(3)

#%%

df_exploded = exp_df.explode('ebird_code')#%%
df_exploded

#%%
df_dummies = pd.get_dummies(df_exploded['ebird_code'])
#%%
df_dummies
#%%
df_grouped = df_dummies.groupby(df_dummies.index).sum()
#%%
df_grouped
#%%
df_final = exp_df.drop('ebird_code', axis=1).join(df_grouped)
#%%
df_final

#%%
LABEL_COLUMNS = df_final.columns.tolist()[12:]

#%%
LABEL_COLUMNS
#%%
data_row = df_final.iloc[0]

#%%
labels = data_row[LABEL_COLUMNS]
#%%
labels
#%%
df_exploded["ebird_code"].nunique()
#%%
df_new= exp_df.join(exploded_df)
# %%
len(exp_df[exp_df["start_time"]==0])
#%%
filtered_df= df_scape[(df_scape['start_time'] >= 700) & (df_scape["filepath"]=="/home/lukas/projects/GADME/data_gadme/downloads/extracted/cff8b1273408353c7165fc9b8da624bdbad585ddd6fc7bb7aab245cc2f543629/data/zenodo/PER/PER_001_S01_20190116_100007Z.ogg")]

filtered_df.head(10)
#%%
filtered_df["filepath"].nunique()
#%%
exp_df[
    (exp_df["start_time"] == 845) &
    (exp_df["filepath"] == "/home/lukas/projects/GADME/data_gadme/downloads/extracted/cff8b1273408353c7165fc9b8da624bdbad585ddd6fc7bb7aab245cc2f543629/data/zenodo/PER/PER_001_S01_20190116_100007Z.ogg")
]["ebird_code"]
#%%
filtered_df.loc[filtered_df["filepath"]=="/home/lukas/projects/GADME/data_gadme/downloads/extracted/cff8b1273408353c7165fc9b8da624bdbad585ddd6fc7bb7aab245cc2f543629/data/zenodo/PER/PER_001_S01_20190116_100007Z.ogg"]

#%%
df_scape[(df_scape["ebird_code"]==18) & (df_scape["filepath"]=="/home/lukas/projects/GADME/data_gadme/downloads/extracted/cff8b1273408353c7165fc9b8da624bdbad585ddd6fc7bb7aab245cc2f543629/data/zenodo/PER/PER_001_S01_20190116_100007Z.ogg")]
#%%
filtered_df.head(5)
#%%
exp_df.loc[100]

#%%
exp_df.shape
#%%
df_scape.shape

#%%
import torch

def indices_to_one_hot(class_indices, num_classes):
    """Convert an iterable of indices to a one-hot encoded tensor."""
    one_hot = torch.zeros(num_classes)
    one_hot[class_indices] = 1
    return one_hot

# Example usage:
class_indices = [10, 11, 29]
num_classes = 30
one_hot_vector = indices_to_one_hot(class_indices, num_classes)
#%%
one_hot_vector
#%%

exp_df

#%%

df_scape["filepath"].nunique()

#%%
720*21

#%%

from datasets import Dataset

def soundscape_map_function(file_path, path_durations, step=5):
    duration = int(path_durations[file_path])
    frames = generate_frames(duration, step=step)
    expanded_rows = []

    # Filter the dataset for the current file_path
    filtered_dataset = ds_scape.filter(lambda example: example['filepath'] == file_path)

    for start, end in frames:
        # Find overlapping events for the current frame
        overlapping = filtered_dataset.filter(lambda example: example['start_time'] < end and example['end_time'] > start)

        if len(overlapping) == 0:
            # If there is no overlapping, create a template row with default values
            template_row = filtered_dataset[0]  # assuming the dataset has at least one row
            template_row['start_time'] = start
            template_row['end_time'] = end
            template_row['ebird_code'] = [-1]
            expanded_rows.append(template_row)
        else:
            # If there is overlapping, create a row for each unique ebird_code
            unique_codes = set()
            for example in overlapping:
                unique_codes.update([example['ebird_code']])
            
            template_row = filtered_dataset[0]  # assuming the dataset has at least one row
            template_row['start_time'] = start
            template_row['end_time'] = end
            template_row['ebird_code'] = list(unique_codes)
            expanded_rows.append(template_row)
    
    return expanded_rows

# Use this function for each unique filepath
unique_filepaths = dataset.unique('filepath')
path_durations = {filepath: librosa.get_duration(filename=filepath) for filepath in unique_filepaths}
all_expanded_rows = []

for file_path in unique_filepaths:
    expanded_rows = soundscape_map_function(file_path, path_durations)
    all_expanded_rows.extend(expanded_rows)

# Create a new dataset from the expanded rows
expanded_dataset = Dataset.from_pandas(pd.DataFrame(all_expanded_rows))