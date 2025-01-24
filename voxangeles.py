from pathlib import Path

import numpy as np
import pandas as pd
from textgrids import TextGrid
from tqdm import tqdm
import argparse
from datasets import Dataset
import os


dataset_path = Path("data/voxangeles")
textgrid_path = Path("data/voxangeles/data/audited_aligned")
output_path = Path("data/voxangeles")


rows = []
# each file is one word
for p in tqdm(textgrid_path.glob("*/*.TextGrid")):
    grid = TextGrid(p)

    word_key = "word" if "word" in grid else ("words")
    word = ""
    for interval in grid[word_key]:
        if interval.text:
            word = interval.text

    phone_key = "phone" if "phone" in grid else ("phones" if "phones" in grid else "Narrow")
    for phone in grid[phone_key]:
        if phone.text:
            rows.append({
                'file': p.stem,
                'lang': p.stem.split('-')[0],
                'word': word,
                "phone": phone.text,
                "start": phone.xmin,
                "finish": phone.xmax,
                # "path": str((dataset_path / p.relative_to(p.parents[3]).with_suffix(".wav")).absolute()),
            })
df = pd.DataFrame(rows)
Dataset.from_pandas(df).push_to_hub("kalbin/VoxAngeles_phones", split="test", token=os.environ["HF_TOKEN"])
