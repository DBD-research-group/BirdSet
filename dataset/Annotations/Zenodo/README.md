---
license: cc-by-4.0
dataset_info:
  features:
  - name: ID
    dtype: string
  - name: Filepath
    dtype: string
  - name: Start Time (s)
    dtype: string
  - name: End Time (s)
    dtype: string
  - name: Low Freq (Hz)
    dtype: string
  - name: High Freq (Hz)
    dtype: string
  - name: Species eBird Code
    dtype: string
  - name: Call Type
    dtype: 'null'
  - name: Sex
    dtype: 'null'
  - name: Latitude
    dtype: float64
  - name: Longitude
    dtype: float64
  - name: Uncertainty
    dtype: 'null'
  - name: Microphone
    dtype: string
  - name: License
    dtype: string
  - name: Source
    dtype: string
  - name: BirdNet Training Data
    dtype: bool
  - name: audio
    dtype: audio
  splits:
  - name: train
    num_bytes: 156232275362.216
    num_examples: 10976
  download_size: 265537350
  dataset_size: 156232275362.216
---

# Dataset Card for Zenodo Dataset

## Dataset Description

- **Homepage:** https://zenodo.org/record/7525805

### Dataset Summary

This dataset contains annotated soundscape data from Sequoia and Kings Canyon National Parks (California, USA). The Dataset contains 10976
samples accross 22 different species.

### Citation Information

@dataset{mary_clapp_2023_7525805,
  author       = {Mary Clapp and
                  Stefan Kahl and
                  Erik Meyer and
                  Megan McKenna and
                  Holger Klinck and
                  Gail Patricelli},
  title        = {{A collection of fully-annotated soundscape 
                   recordings from the southern Sierra Nevada
                   mountain range}},
  month        = jan,
  year         = 2023,
  publisher    = {Zenodo},
  version      = 1,
  doi          = {10.5281/zenodo.7525805},
  url          = {https://doi.org/10.5281/zenodo.7525805}
}g/10.5281/zenodo.7525805
