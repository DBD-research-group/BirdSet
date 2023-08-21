import ratelimiter
import pandas as pd
import itertools
from tqdm import tqdm
import concurrent.futures
import os
import requests
import pickle
from lightning.pytorch import seed_everything

pbar = None

seed_everything(0, workers=True)

session = requests.Session()
session.mount(
    'http://',
    requests.adapters.HTTPAdapter(
        max_retries=requests.adapters.Retry(total=5, backoff_factor=1)
    ),
)

@ratelimiter.RateLimiter(max_calls=8, period=1)
def download_recordings(data):
    idx, row = data
    url = row["file"]
    os.makedirs(f"data/xeno-canto/{row['continent']}/{row['Scientific name']}", exist_ok=True)

    doc = session.get(url=url)
    pbar.update()

    if doc.status_code != 200:
        print("Failed:", idx, "With code:", doc.status_code)
        return idx
    with open(f"data/xeno-canto/{row['continent']}/{row['Scientific name']}/XC{idx}.{row['file-name'].split('.')[-1]}", 'wb') as f:
        f.write(doc.content)
    return


def main():
    global pbar
    df = pd.read_csv("./data/xeno-canto/to_download.csv", index_col="id")
    occ = df.groupby(["Scientific name"]).size()
    df = df[(df["q"] == "A") | (df["q"] == "B") | (df["q"] == "C")]
    occ = occ[occ.values >= 15]
    df = df[df["Scientific name"].apply(lambda x: x in occ)]

    sel = pd.concat([df[df["q"] == "A"], df[df["q"] == "B"].sample(frac=0.60), df[df["q"] == "C"].sample(frac=0.60)])
    sel.to_csv("./data/xeno-canto/downloaded_ABC.csv")
    if True:
        files = []
        for d in os.listdir("./data/xeno-canto/North America"):
            if not d.endswith(".csv"):
                files += os.listdir(f"./data/xeno-canto/North America/{d}")
        files = pd.Series(files)
        files = files.apply(lambda x: int(x[2:].split(".")[0]))
    print("done:", len(files))
    sel = sel[~sel.index.isin(files.values)]

    pbar = tqdm(total=len(sel))

    with concurrent.futures.ThreadPoolExecutor(max_workers=8) as executor:
            species_recordings = executor.map(download_recordings, sel.iterrows())
            #species_recordings = tqdm(species_recordings, total=len(df), desc='Downloading recordings')

            recordings = list(itertools.chain.from_iterable(species_recordings))
    print("\n")
    print(recordings)
    with open("missed.pkl", 'wb') as f:
        pickle.dump(recordings, f)

    pbar.close()

if __name__ == "__main__":
    main()
