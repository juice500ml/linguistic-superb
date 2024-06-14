import os
import random

import pandas as pd
from pathlib import Path

from utils import rows_to_dataset, validate_dataset


instructions = [
    "Given these two sentences, which sentence is grammatically correct? Respond with 1 or 2.",
    "Between these two sentences, which one is grammatically correct? Choose 1 or 2.",
    "Out of these two sentences, which one is grammatically accurate? Answer with 1 or 2.",
    "Considering these two sentences, which is grammatically proper? Indicate by selecting 1 or 2.",
    "From the two sentences provided, which is correct grammatically? Reply with 1 or 2.",
    "Given the two sentences, which one is grammatically right? Respond by choosing 1 or 2.",
    "Of these two sentences, which is grammatically correct? Reply with 1 or 2.",
    "Between the two sentences given, which is grammatically accurate? Answer with 1 or 2.",
    "Which of these two sentences is grammatically correct? Respond with either 1 or 2.",
    "Looking at these two sentences, which is correct in terms of grammar? Choose 1 or 2.",
    "Out of the two sentences, which one is correct grammatically? Indicate with 1 or 2.",
]


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


    rows = {"audio1": [], "audio2": [], "instruction": [], "label": []}
    for subtype in sorted(df.subtype.unique()):
        for _, _df in df[df.subtype == subtype].groupby("id"):
            assert len(_df) == 2

            wrong_row = _df[_df.correct == 0].iloc[0]
            correct_row = _df[_df.correct == 1].iloc[0]

            if wrong_row.inversed_order:
                rows["label"].append("1")
                rows["audio1"].append(str(root_path / f"{correct_row.filename}.wav"))
                rows["audio2"].append(str(root_path / f"{wrong_row.filename}.wav"))
            else:
                rows["label"].append("2")
                rows["audio1"].append(str(root_path / f"{wrong_row.filename}.wav"))
                rows["audio2"].append(str(root_path / f"{correct_row.filename}.wav"))

    random.seed(42)
    rows["instruction"] = [instructions[i] for i in random.choices(range(11), k=len(rows["label"]))]

    ds = rows_to_dataset(rows)
    validate_dataset(ds)
    ds.push_to_hub(repo_id="DynamicSuperb/SentenceGrammarAcceptability_sBLIMP", split="test", token=os.environ["HF_TOKEN"])
