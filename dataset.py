"""
SpamDataset and DataLoader utilities for spam classification.
"""

import torch
from torch.utils.data import Dataset, DataLoader
import pandas as pd


class SpamDataset(Dataset):
    def __init__(self, csv_file, tokenizer, max_length=None, pad_token_id=50256):
        self.data = pd.read_csv(csv_file)

        # Pre-tokenize texts
        self.encoded_texts = [
            tokenizer.encode(text) for text in self.data["Text"]
        ]

        if max_length is None:
            self.max_length = self._longest_encoded_length()
        else:
            self.max_length = max_length
            
            # Truncate sequences if they are longer than max_length
            self.encoded_texts = [
                encoded_text[:self.max_length]
                for encoded_text in self.encoded_texts
            ]

        # Pad sequences to the longest sequence
        self.encoded_texts = [
            encoded_text + [pad_token_id] * (self.max_length - len(encoded_text))
            for encoded_text in self.encoded_texts
        ]

    def __getitem__(self, index):
        encoded = self.encoded_texts[index]
        label = self.data.iloc[index]["Label"]
        return (
            torch.tensor(encoded, dtype=torch.long),
            torch.tensor(label, dtype=torch.long)
        )

    def __len__(self):
        return len(self.data)

    def _longest_encoded_length(self):
        max_length = 0
        for encoded_text in self.encoded_texts:
            encoded_length = len(encoded_text)
            if encoded_length > max_length:
                max_length = encoded_length
        return max_length


def create_dataloaders(train_csv, val_csv, test_csv, tokenizer, batch_size=8, num_workers=0):
    """Create train, validation, and test dataloaders."""
    
    # Create training dataset first to get max_length
    train_dataset = SpamDataset(
        csv_file=train_csv,
        max_length=None,
        tokenizer=tokenizer
    )
    print(f"Max length: {train_dataset.max_length}")

    # Create validation and test datasets with same max_length
    val_dataset = SpamDataset(
        csv_file=val_csv,
        max_length=train_dataset.max_length,
        tokenizer=tokenizer
    )
    test_dataset = SpamDataset(
        csv_file=test_csv,
        max_length=train_dataset.max_length,
        tokenizer=tokenizer
    )

    # Create dataloaders
    torch.manual_seed(123)

    train_loader = DataLoader(
        dataset=train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        drop_last=True,
    )

    val_loader = DataLoader(
        dataset=val_dataset,
        batch_size=batch_size,
        num_workers=num_workers,
        drop_last=False,
    )

    test_loader = DataLoader(
        dataset=test_dataset,
        batch_size=batch_size,
        num_workers=num_workers,
        drop_last=False,
    )

    return train_loader, val_loader, test_loader, train_dataset.max_length
