import os
import random

import pandas as pd
from pathlib import Path

from utils import rows_to_dataset, validate_dataset


instructions = [
    "Given these two utterances, which sentence sounds more prosodically natural? Respond with 1 or 2.",
    "Between these two utterances, which one has a more natural prosody? Answer with 1 or 2.",
    "Considering these two utterances, which one sounds more natural in terms of prosody? Choose 1 or 2.",
    "From these two utterances, which one sounds more prosodically natural? Indicate with 1 or 2.",
    "Of these two utterances, which one sounds more natural in prosody? Answer with 1 or 2.",
    "Given these two utterances, which one has a more natural prosody? Reply with 1 or 2.",
    "Which of these two utterances more prosodically natural? Choose 1 or 2.",
    "Considering these two utterances, which one sounds more prosodically natural? Answer with 1 or 2.",
    "Between these two utterances, which one sounds more natural in terms of prosody? Answer with 1 or 2.",
    "Which of these two utterances sounds more natural in terms of prosody? Reply with 1 or 2.",
    "Considering these two utterances, which sounds more natural in prosody? Answer 1 or 2.",
]


def _determine_inversed_order(df):
    df["inversed_order"] = False
    indices = df.id.unique()
    random.seed(42)
    random.shuffle(indices)
    df.loc[df.id.isin(indices[:len(indices) // 2]), "inversed_order"] = True


if __name__ == "__main__":
    root_path = Path("zrc/datasets/prosaudit/english/dev").absolute()
    gold_path = root_path / "gold.csv"
    df = pd.read_csv(gold_path)
    _determine_inversed_order(df)

    rows = {"audio1": [], "audio2": [], "instruction": [], "label": []}
    for _, _df in df.groupby("id"):
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
    ds.push_to_hub(repo_id="DynamicSuperb/ProsodyNaturalness_ProsAudit", split="test", token=os.environ["HF_TOKEN"])
