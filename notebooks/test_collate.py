#%%
from datasets import Dataset


from datasets import Dataset
import pandas as pd
import torch

# Creating a dummy dataset
data = {'text': ['sample1', 'sample2', 'sample3', 'sample4'],
        'label': [0, 1, 0, 1]}
df = pd.DataFrame(data)
dummy_dataset = Dataset.from_pandas(df)
#%%

def custom_batch_transform(batch):
    print("TRANSFORM")
    # Dummy transformation: append "_transformed" to text and convert label to tensor
    batch['text'] = [text + '_transformed' for text in batch['text']]
    batch['label'] = torch.tensor(batch['label'])
    return batch

dummy_dataset.set_transform(custom_batch_transform)
#%%

dummy_dataset.format
#%
dummy_dataset.reset_format()
#%%

from torch.utils.data import DataLoader

def custom_collate_fn(batch):
    # Custom collate: Aggregate texts and labels
    print("COLLATE")
    texts = [item['text'] for item in batch]
    labels = torch.stack([item['label'] for item in batch])
    return {'text': texts, 'label': labels}
#%%
# Creating DataLoader with the custom collate function
data_loader = DataLoader(dummy_dataset, batch_size=2, collate_fn=custom_collate_fn)

#%%

for batch in data_loader:
    print(batch)

# transform before collate! 