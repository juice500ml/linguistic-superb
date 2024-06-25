import os
import random
from itertools import product

import pandas as pd
from pathlib import Path

from utils import rows_to_dataset, validate_dataset


questions = [
    "Given these two utterances, which sentence sounds more prosodically natural?",
    "Between these two utterances, which one has a more natural prosody?",
    "Considering these two utterances, which one sounds more natural in terms of prosody?",
    "From these two utterances, which one sounds more prosodically natural?"
    "Of these two utterances, which one sounds more natural in prosody?",
    "Given these two utterances, which one has a more natural prosody?",
    "Which of these two utterances more prosodically natural?",
    "Considering these two utterances, which one sounds more prosodically natural?",
    "Between these two utterances, which one sounds more natural in terms of prosody?",
    "Which of these two utterances sounds more natural in terms of prosody?",
    "Considering these two utterances, which sounds more natural in prosody?",
    "Given these two utterances, which sentence sounds more natural?",
    "Considering these two utterances, which one sounds more natural?",
    "From these two utterances, which one sounds more natural?"
    "Of these two utterances, which one sounds more natural?",
    "Which of these two utterances more natural?",
    "Considering these two utterances, which one sounds more natural?",
    "Between these two utterances, which one sounds more natural?",
    "Which of these two utterances sounds more natural?",
    "Considering these two utterances, which sounds more natural?",
]
formats = [
    "Respond with 1 or 2.",
    "Answer with 1 or 2.",
    "Choose 1 or 2.",
    "Indicate with 1 or 2.",
    "Reply with 1 or 2.",
    "Answer 1 or 2.",
]
instructions = [f"{q} {f}" for q, f in product(questions, formats)]


def _determine_inversed_order(df):
    df["inversed_order"] = False
    indices = df.id.unique()
    random.seed(42)
    random.shuffle(indices)
    df.loc[df.id.isin(indices[:len(indices) // 2]), "inversed_order"] = True


if __name__ == "__main__":
    root_path = Path("zrc/datasets/prosaudit/english/dev").absolute()
    gold_path = root_path / "gold.csv"
    gold_df = pd.read_csv(gold_path)

    for task_type in ("protosyntax", "lexical"):
        df = gold_df[gold_df.type == task_type].copy()
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
        rows["instruction"] = [instructions[i] for i in random.choices(range(len(instructions)), k=len(rows["label"]))]

        ds = rows_to_dataset(rows)
        validate_dataset(ds)
        ds.push_to_hub(repo_id=f"DynamicSuperb/ProsodyNaturalness_ProsAudit-{task_type.capitalize()}", split="test", token=os.environ["HF_TOKEN"])
