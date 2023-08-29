import argparse
import pandas as pd
from datasets import Dataset, Audio
import time 
from huggingface_hub import login

def main(metadata, hf_name, decode, token):
    metadata = pd.read_csv(metadata, index_col='id')
    #metadata['file_name'] = metadata['file_name'].str.replace(r'North America/.*?/', 'NA_subset1k/', regex=True)
    ds = Dataset.from_pandas(metadata)
    ds = ds.add_column("file", ds["file_name"])
    ds = ds.rename_column("file_name", "audio")
    ds = ds.cast_column(
        column="audio",
        feature=Audio(
            sampling_rate=32_000,
            mono=True,
            decode=decode
        )
    )
    ds = ds.class_encode_column('primary')
    print(f"decode = {decode}")
    try:
        ds_split = ds.train_test_split(test_size=0.2, stratify_by_column='primary')
    except Exception as e: 
        print(e)
        ds_split = ds.train_test_split(test_size=0.2)

    login(token=token)
    
    start_time = time.time()
    ds_split.push_to_hub(
        repo_id=f"DBD-research-group/{hf_name}",
        private=True,
        embed_external_files=True
    )
    end_time = time.time() - start_time
    print(f"Uploading took {end_time/60}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    parser.add_argument("metadata", type=str)
    parser.add_argument("hf_name", type=str)
    parser.add_argument("token", type=str)
    #parser.add_argument("--decode", type=bool, default=True)
    parser.add_argument('--decode', default=True, action=argparse.BooleanOptionalAction)

    args = parser.parse_args()

    main(
        metadata=args.metadata,
        hf_name=args.hf_name,
        decode=args.decode,
        token=args.token
    )



