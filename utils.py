import os
import urllib.request
import zipfile
import pandas as pd
from pathlib import Path
import tiktoken

def download_and_prepare_data(data_dir="data"):
    data_path = Path(data_dir)
    data_path.mkdir(parents=True, exist_ok=True)
    
    zip_path = data_path / "sms_spam_collection.zip"
    extracted_path = data_path / "SMSSpamCollection"
    train_csv = data_path / "train.csv"
    
    # Download
    if not zip_path.exists() and not extracted_path.exists():
        url = "https://archive.ics.uci.edu/static/public/228/sms+spam+collection.zip"
        urllib.request.urlretrieve(url, zip_path)
        with zipfile.ZipFile(zip_path, "r") as zip_ref:
            zip_ref.extractall(data_path)
    
    if not train_csv.exists():
        # Load and process
        df = pd.read_csv(extracted_path, sep="\t", header=None, names=["Label", "Text"])
        
        # Create balanced dataset
        # 1. Map labels
        df["Label"] = df["Label"].map({"ham": 0, "spam": 1})
        
        # 2. Random split
        shuffled_df = df.sample(frac=1, random_state=123).reset_index(drop=True)
        train_idx = int(0.7 * len(shuffled_df))
        val_idx = int(0.8 * len(shuffled_df))
        
        train_df = shuffled_df.iloc[:train_idx]
        val_df = shuffled_df.iloc[train_idx:val_idx]
        test_df = shuffled_df.iloc[val_idx:]
        
        train_df.to_csv(data_path / "train.csv", index=False)
        val_df.to_csv(data_path / "validation.csv", index=False)
        test_df.to_csv(data_path / "test.csv", index=False)
        
    return data_path

def get_tokenizer():
    return tiktoken.get_encoding("gpt2")
