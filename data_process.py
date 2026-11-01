import pandas as pd

from sklearn.model_selection import train_test_split
from torch.utils.data import Dataset, DataLoader
from transformers import AutoTokenizer
import torch

DATA_URL = (
    "https://raw.githubusercontent.com/LTSACH/AILearningHub/"
    "main/datasets/bbcnews/data/bbc-news.csv"
)

TOKENIZER_NAME = "bert-base-uncased"
TEXT_COL = "text"
LABEL_COL = "label"

# Fixed label mapping so class IDs stay consistent across runs.
LABEL2ID = {
    "business": 0,
    "entertainment": 1,
    "politics": 2,
    "sport": 3,
    "tech": 4,
}

ID2LABEL = {v: k for k, v in LABEL2ID.items()}

class BBCNewsDataset(Dataset):
    def __init__(self, df):
        self.texts = df[TEXT_COL].tolist()
        self.labels = [
            LABEL2ID[label] for label in df[LABEL_COL]
        ]

    def __len__(self):
        return len(self.texts)

    def __getitem__(self, index):
        return self.texts[index], self.labels[index]

def make_collate_fn(tokenizer, max_length):
    """Tokenize a batch and dynamically pad its sequences."""

    def collate_fn(batch):
        texts, labels = zip(*batch)

        encoded = tokenizer(
            list(texts),
            padding=True,
            truncation=True,
            max_length=max_length,
            return_tensors="pt",
        )

        encoded["labels"] = torch.tensor(
            labels, dtype=torch.long
        )

        return encoded

    return collate_fn

def get_loaders(
    batch_size=16,
    eval_batch_size=32,
    seed=42,
    tokenizer_name=TOKENIZER_NAME,
    max_length=512,
):
    # ---- 1. Load and validate dataset ----
    
    df = pd.read_csv(DATA_URL)

    required_columns = {TEXT_COL, LABEL_COL}
    missing = required_columns - set(df.columns)

    if missing:
        raise ValueError(f"Missing required columns: {missing}")

    if df[[TEXT_COL, LABEL_COL]].isna().any().any():
        raise ValueError("Dataset contains missing text or labels.")

    unknown_labels = set(df[LABEL_COL].unique()) - set(LABEL2ID)

    if unknown_labels:
        raise ValueError(f"Unexpected labels: {unknown_labels}")

    # Identical texts should not have conflicting labels.
    label_counts = df.groupby(TEXT_COL)[LABEL_COL].nunique()

    if (label_counts > 1).any():
        raise ValueError(
            "Identical texts have conflicting labels."
        )

    # ---- 2. Remove duplicate articles ----

    df = df.drop_duplicates(
        subset=TEXT_COL,
        keep="first",
    ).reset_index(drop=True)

    # ---- 3. Stratified 70/15/15 split ----

    train_df, holdout_df = train_test_split(
        df,
        test_size=0.30,
        random_state=seed,
        stratify=df[LABEL_COL],
    )

    valid_df, test_df = train_test_split(
        holdout_df,
        test_size=0.50,
        random_state=seed,
        stratify=holdout_df[LABEL_COL],
    )

    print(f"Total unique articles: {len(df)}")
    print(f"Train: {len(train_df)}")
    print(f"Validation: {len(valid_df)}")
    print(f"Test: {len(test_df)}")

    # ---- 4. Tokenizer and datasets ----
    
    tokenizer = AutoTokenizer.from_pretrained(tokenizer_name)

    train_ds = BBCNewsDataset(train_df)
    valid_ds = BBCNewsDataset(valid_df)
    test_ds = BBCNewsDataset(test_df)

    collate_fn = make_collate_fn(tokenizer, max_length)

    # ---- 5. DataLoaders ----

    train_loader = DataLoader(
        train_ds,
        batch_size=batch_size,
        shuffle=True,
        collate_fn=collate_fn,
    )

    valid_loader = DataLoader(
        valid_ds,
        batch_size=eval_batch_size,
        shuffle=False,
        collate_fn=collate_fn,
    )

    test_loader = DataLoader(
        test_ds,
        batch_size=eval_batch_size,
        shuffle=False,
        collate_fn=collate_fn,
    )

    train_eval_loader = DataLoader(
        train_ds,
        batch_size=eval_batch_size,
        shuffle=False,
        collate_fn=collate_fn,
    )

    return (
        train_loader,
        valid_loader,
        test_loader,
        train_eval_loader,
    )