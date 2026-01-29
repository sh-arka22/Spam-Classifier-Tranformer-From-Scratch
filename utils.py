import os
import pandas as pd
from pathlib import Path

def create_balanced_dataset(df):
    """Create a balanced dataset with equal number of spam and ham messages."""
    # Count the instances of "spam"
    num_spam = df[df["Label"] == "spam"].shape[0]

    # Randomly sample "ham" instances to match the number of "spam" instances
    ham_subset = df[df["Label"] == "ham"].sample(num_spam, random_state=123)

    # Combine ham "subset" with "spam"
    balanced_df = pd.concat([ham_subset, df[df["Label"] == "spam"]])

    return balanced_df

def random_split(df, train_frac, validation_frac):
    """Split dataframe into train, validation, and test sets."""
    # Shuffle the entire DataFrame
    df = df.sample(frac=1, random_state=123).reset_index(drop=True)

    # Calculate split indices
    train_end = int(len(df) * train_frac)
    validation_end = train_end + int(len(df) * validation_frac)

    # Split the DataFrame
    train_df = df[:train_end]
    validation_df = df[train_end:validation_end]
    test_df = df[validation_end:]

    return train_df, validation_df, test_df


def download_and_prepare_data(data_dir="data"):
    """
    Prepare the spam dataset from local files.
    Uses the existing dataset at /Users/arkaj/Desktop/LLM-Scratch/sms+spam+collection/
    """
    local_data_file = Path("/Users/arkaj/Desktop/LLM-Scratch/sms+spam+collection/SMSSpamCollection")

    if not local_data_file.exists():
        raise FileNotFoundError(
            f"Dataset not found at {local_data_file}. "
            "Please ensure the sms+spam+collection folder exists."
        )

    # Create output data directory
    data_path = Path(data_dir)
    data_path.mkdir(parents=True, exist_ok=True)

    # Check if already processed
    if (data_path / "train.csv").exists():
        print("Data already prepared. Skipping preparation.")
        return data_path

    # Load data (tab-separated, no header)
    print(f"Loading data from {local_data_file}")
    df = pd.read_csv(local_data_file, sep="\t", header=None, names=["Label", "Text"])
    print(f"Total samples: {len(df)}")
    print(f"Spam: {(df['Label'] == 'spam').sum()}, Ham: {(df['Label'] == 'ham').sum()}")


    # Create balanced dataset
    balanced_df = create_balanced_dataset(df)
    print(f"Balanced samples: {len(balanced_df)}")

    # Convert labels to binary (ham=0, spam=1)
    balanced_df["Label"] = balanced_df["Label"].map({"ham": 0, "spam": 1})


    # Split data (70% train, 10% validation, 20% test)
    train_df, validation_df, test_df = random_split(balanced_df, 0.7, 0.1)

    # Save to CSV files
    train_df.to_csv(data_path / "train.csv", index=None)
    validation_df.to_csv(data_path / "validation.csv", index=None)
    test_df.to_csv(data_path / "test.csv", index=None)

    print(f"Train size: {len(train_df)}")
    print(f"Validation size: {len(validation_df)}")
    print(f"Test size: {len(test_df)}")

    return data_path


def get_tokenizer():
    """Initialize and return the tiktoken tokenizer for GPT-2."""
    import tiktoken
    tokenizer = tiktoken.get_encoding("gpt2")
    return tokenizer
    