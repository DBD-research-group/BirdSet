import os
from pathlib import Path
import json
import pandas as pd



EBIRD_TAXONOMY = "./eBird_taxonomy_codes_2021E.json"
BIRDNET_LABELS = "./BirdNET_GLOBAL_6K_V2.4_Labels.txt"
CSV_COLUMN_NAME = "ebird2021"
CSV_PATH = "./birdnet_V2.4_labels_ebird.csv"

def readLines(path: str):
    return Path(path).read_text(encoding="utf-8").splitlines() if path else []

def main():
    abs_ebird = os.path.abspath(EBIRD_TAXONOMY)
    abs_birdnet = os.path.abspath(BIRDNET_LABELS)
    
    birdnet_labels = readLines(abs_birdnet)
    
    f = open(abs_ebird)
    ebird_to_label = json.load(f)
    f.close()
    
    label_to_ebird = {v: k for k, v in ebird_to_label.items()}
    
    birdnet_ebirds = []
    
    for label in birdnet_labels:
        if label in label_to_ebird.keys():
            ebird = label_to_ebird[label]
            birdnet_ebirds.append(ebird)
        else:
            print(f"Error with label {label}")
    
    dct = {CSV_COLUMN_NAME: birdnet_ebirds}
    
    ds = pd.DataFrame(dct)
    print(ds)
    
    ds.to_csv(CSV_PATH, index=False)
    

    

if __name__=="__main__":
    main()