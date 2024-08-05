import os
import random
from itertools import product

import pandas as pd
from pathlib import Path

from utils import rows_to_dataset, validate_dataset


questions = [
    "Given these two sentences, which sentence is grammatically correct?",
    "Between these two sentences, which one is grammatically correct?",
    "Out of these two sentences, which one is grammatically accurate?",
    "Considering these two sentences, which is grammatically proper?",
    "From the two sentences provided, which is correct grammatically?",
    "Given the two sentences, which one is grammatically right?",
    "Of these two sentences, which is grammatically accurate?",
    "Between the two sentences given, which is grammatically accurate?",
    "Which of these two sentences is grammatically correct?",
    "Looking at these two sentences, which is correct in terms of grammar?",
    "Out of the two sentences, which one is correct grammatically?",
]
formats = [
    "Respond with 1 or 2.",
    "Choose 1 or 2.",
    "Answer with 1 or 2.",
    "Indicate by selecting 1 or 2.",
    "Respond by choosing 1 or 2.",
    "Reply with 1 or 2.",
    "Respond with either 1 or 2.",
    "Indicate with 1 or 2.",
]
instructions = [f"{q} {f}" for q, f in product(questions, formats)]



def _sample_df(df):
    voice_list = df.voice.unique()
    indices = []

    # Choose 8 sentence pairs per each subtype
    random.seed(42)
    for subtype in sorted(df.subtype.unique()):
        pair_ids = random.sample(range(1, 101), k=8)

        # Make speaker distribution uniform
        for voice_id, pair_id in enumerate(pair_ids):
            indices += df[
                (df.voice == voice_list[voice_id % len(voice_list)])
                & (df.id == pair_id)
                & (df.subtype == subtype)
            ].index.to_list()

    return df[df.index.isin(indices)].copy()


def _determine_inversed_order(df):
    indices = []
    random.seed(42)
    df["inversed_order"] = False
    for subtype in sorted(df.subtype.unique()):
        for voice in sorted(df.voice.unique()):
            pair_ids = df[(df.subtype == subtype) & (df.voice == voice)].id.unique()
            pair_ids = random.sample(sorted(pair_ids), k=1)
            indices += df[
                (df.subtype == subtype)
                & (df.voice == voice)
                & (df.id.isin(pair_ids))
            ].index.tolist()
    df.loc[indices, "inversed_order"] = True


if __name__ == "__main__":
    root_path = Path("zrc/datasets/sLM21/syntactic/dev").absolute()
    gold_path = root_path / "gold.csv"
    df = pd.read_csv(gold_path)

    assert df.subtype.nunique() == 63
    assert df.voice.nunique() == 4
    df = _sample_df(df)
    _determine_inversed_order(df)


    rows = {"audio": [], "audio2": [], "instruction": [], "label": []}
    for subtype in sorted(df.subtype.unique()):
        for _, _df in df[df.subtype == subtype].groupby("id"):
            assert len(_df) == 2

            wrong_row = _df[_df.correct == 0].iloc[0]
            correct_row = _df[_df.correct == 1].iloc[0]

            if wrong_row.inversed_order:
                rows["label"].append("1")
                rows["audio"].append(str(root_path / f"{correct_row.filename}.wav"))
                rows["audio2"].append(str(root_path / f"{wrong_row.filename}.wav"))
            else:
                rows["label"].append("2")
                rows["audio"].append(str(root_path / f"{wrong_row.filename}.wav"))
                rows["audio2"].append(str(root_path / f"{correct_row.filename}.wav"))

    random.seed(42)
    rows["instruction"] = [instructions[i] for i in random.choices(range(len(instructions)), k=len(rows["label"]))]

    ds = rows_to_dataset(rows)
    validate_dataset(ds)
    ds.push_to_hub(repo_id="DynamicSuperb/SentenceGrammarAcceptability_sBLIMP", split="test", token=os.environ["HF_TOKEN"])
