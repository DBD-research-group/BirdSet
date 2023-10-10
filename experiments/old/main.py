from torch.utils.data import DataLoader
from transformers import Wav2Vec2FeatureExtractor, Wav2Vec2ForSequenceClassification
import lightning as pl
import torch
from torch.nn import functional as F
from lightning.pytorch import Trainer, seed_everything
import pandas as pd
from data.utils import create_hg_dataset
import librosa
import time
SEED = 0


class LitWav2Vec2(pl.LightningModule):
    def __init__(self):
        super().__init__()
        self.metadata = pd.read_csv("data/xeno-canto/na_metadata.csv", index_col="id")

        sizes = self.metadata.groupby("primary").size()
        sizes = sizes[sizes >= 20]

        self.model = Wav2Vec2ForSequenceClassification.from_pretrained("facebook/wav2vec2-base-960h", num_labels=len(sizes))

        self.feature_extractor = Wav2Vec2FeatureExtractor.from_pretrained("facebook/wav2vec2-base-960h")
        self.hg_dataset = None

        self.train_ds = None
        self.val_ds = None
        self.create_ds()

        self.t = 0

    def create_ds(self):
        self.hg_dataset = create_hg_dataset(self.metadata, min_occ=20)

        ds = self.hg_dataset.train_test_split(test_size=0.2, stratify_by_column="primary", seed=SEED)
        self.train_ds = ds["train"]
        self.val_ds = ds["test"]

    def forward(self, x):
        return self.model(x)

    def training_step(self, batch, batch_idx):
        x = torch.stack([i["wave"] for i in batch])
        outputs = self(x)
        target = torch.tensor([i["primary"] for i in batch], device=self.device)
        loss = F.cross_entropy(outputs["logits"], target)
        return loss

    def preprocess_function(self, batch):
        start = time.time()
        try:
            for x in batch:
                x["wave"], x["sr"] = librosa.load(x["file_name"], sr=16_000)
        except:
            print("-"*50)
            print(batch)
            print("-"*50)
            exit()

        audio_arrays = [x["wave"] for x in batch]
        # limited to first 20 seconds of audio clip, some audios are too long to be processed in model
        inputs = self.feature_extractor(audio_arrays,
                                        sampling_rate=16_000,
                                        padding=True, return_tensors="pt",
                                        max_length=16_000 * 20, truncation=True)
        end = time.time()
        self.t += end - start
        for i in range(inputs["input_values"].size(0)):
            batch[i]["wave"] = inputs["input_values"][i]
        return batch

    def train_dataloader(self):
        dataloader = DataLoader(self.train_ds, batch_size=8, collate_fn=self.preprocess_function, num_workers=8)
        return dataloader

    def val_dataloader(self):
        df = DataLoader(self.val_ds, batch_size=8, collate_fn=self.preprocess_function, num_workers=8)
        return df

    def validation_step(self, batch, batch_idx):
        x = torch.stack([i["wave"] for i in batch])
        outputs = self(x)
        target = torch.tensor([i["primary"] for i in batch], device=self.device)
        loss = F.cross_entropy(outputs["logits"], target)
        return loss

    def configure_optimizers(self):
        return torch.optim.AdamW(self.parameters(), lr=0.002)


def main():
    model = LitWav2Vec2()
    trainer = Trainer(max_epochs=3, deterministic=False, log_every_n_steps=None, fast_dev_run=False)
    seed_everything(SEED)
    trainer.fit(model)

    print("-"*30)
    print("time on preprocessing function:", model.t)
    print("-"*30)


if __name__ == "__main__":
    main()
