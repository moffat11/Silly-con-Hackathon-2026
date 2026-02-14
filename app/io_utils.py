from __future__ import annotations
from typing import List
import re
import pandas as pd


def split_thread(thread_text: str) -> List[str]:
    """
    Convert pasted conversation text into a list of cleaned messages.

    Supports:
    - One message per line
    - Messages separated by blank lines
    - Mixed formats

    Returns:
        List[str]: cleaned messages
    """
    if not thread_text:
        return []

    text = thread_text.strip()

    # Prefer blank-line separation if present
    if "\n\n" in text:
        blocks = [b.strip() for b in text.split("\n\n")]
        messages = [
            re.sub(r"\s+", " ", block.replace("\n", " ")).strip()
            for block in blocks
            if block.strip()
        ]
    else:
        lines = [line.strip() for line in text.split("\n")]
        messages = [
            re.sub(r"\s+", " ", line)
            for line in lines
            if line
        ]

    # Remove extremely short garbage lines
    cleaned = [m for m in messages if len(m) > 2]

    return cleaned


def load_thread_from_csv(path: str, text_column: str = "text") -> List[str]:
    """
    Load thread messages from a CSV file.

    Args:
        path: path to CSV
        text_column: column containing message text

    Returns:
        List[str]: cleaned messages
    """
    df = pd.read_csv(path)

    if text_column not in df.columns:
        raise ValueError(
            f"Column '{text_column}' not found. Available columns: {list(df.columns)}"
        )

    messages = (
        df[text_column]
        .fillna("")
        .astype(str)
        .str.strip()
        .tolist()
    )

    return [m for m in messages if len(m) > 2]