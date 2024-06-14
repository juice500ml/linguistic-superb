import os
import random

import pandas as pd
from pathlib import Path

from utils import rows_to_dataset, validate_dataset


# adapted from sentence_grammar_acceptability.py

instructions = [
    "Given these two utterances (utterance 1 and utterance 2), which is not an actual word in the English language? Respond with 1 or 2."
    "Based on the two audio files (1 and 2), spot which word is not a word in English. The answer should be 1 or 2.",
    "Please determine whether word 1 or word 2 is not an English word given the two audio files. Respond with 1 or 2.",
    "Determine if word 1 or word 2 is a fake English word given the two audio files (1 and 2). Write your answer as either 1 or 2.",
    "Examine if word 1 or word 2 is not an English word given the two audio files (1 and 2)? Write your answer as either 1 or 2.",
    "Listen to the two audio clips (1 and 2) and judge which of the two is not an English word. Write your answer as either 1 or 2.",
    "Decide which of the two words is not an actual English word in these two audio clips (1 and 2). The answer could be 1 or 2.",
    "From the two audio inputs (1 and 2), ascertain if word 1 or word 2 is a fake English word. The answer is either 1 or 2.",
    "Judge whether word 1 or word 2 is not a word in the English language. Respond with 1 or 2.",
    "Please determine if word 1 or word 2 is not part of the English lexicon. The answer is either 1 or 2.",
    "Decide whether word 1 or word 2 is the fake English word in the provided audio clips (1 and 2). The answer is 1 or 2.",
    "Based on these two audio clips (1 and 2), judge whether word 1 or word 2 is not a word in English. Write your answer as 1 or 2.",
    "Given the two audio files (1 and 2), determine whether word 1 or word 2 lacks. The answer could be 1 or 2",
    "Using the two audio files (1 and 2), determine whether word 1 is the fake word or if word 2 is. The answer should be 1 or 2.",
    "Please identify the fake English word among the two given clips (1 and 2). The answer is either 1 or 2.",
    "Assess which of the two words is a fake English word, based on the two audio files (1 and 2). Respond with 1 or 2.",
    "There is a fake word among us. Listen to the two audio files (1 and 2) and decide if word 1 or word 2 is a fake English word. Your answer should be 1 or 2.",
    "Using the two audio recordings (1 and 2), judge whether word 1 or word 2 is not an English word. Your answer should be either 1 or 2.",
    "From the two audio clips (1 and 2), decide whether word 1 or word 2 is a fake English word. Write 1 or 2 as your answer.",
    "Using the two audio recordings (1 and 2), decide if word 1 or word 2 is not a real English word. Your answer should be either 1 or 2.",
    "With the provided audio files (1 and 2), decide which of the two words is a fake English word. Indicate your answer as either 1 or 2."
]

def _sample_df(df):
    voice_list = df.voice.unique() # 4 voices
    indices = []

    # Choose 32 word pairs for each word length (8 per speaker)
    # the frequencies of lengths: Counter({6: 20060, 7: 16712, 5: 13016, 8: 11900, 9: 7600, 10: 3976, 4: 2796, 11: 2376, 12: 1036, 13: 412, 14: 96, 3: 20})
        # divide by 2 since each member of word pairs double counted
    random.seed(42)
    for length in sorted(df.length.unique()):
        if len(df[df.length == length]) < 2 * 32:
            continue

        # Make speaker distribution uniform
        for voice in sorted(df.voice.unique()):
            pair_ids = df[(df.length == length) & (df.voice == voice)].id.unique()
            # sample 8 word pairs per speaker
            pair_ids = random.sample(sorted(pair_ids), k=8)
            for pair_id in pair_ids:
                # for each pair id, there will be 2 entries
                # the nonce word may not have the same word length (and similar phoneme freqs) as the real one
                entries = (df[
                    (df.voice == voice)
                    & (df.id == pair_id)
                ]).index.to_list()
                assert len(entries) == 2
                indices += entries

    return df[df.index.isin(indices)].copy()


if __name__ == "__main__":
    # full path: zrc/datasets/sLM21/lexical/dev
    root_path = Path("lexical/dev").absolute()
    gold_path = root_path / "gold.csv"
    df = pd.read_csv(gold_path)

    df = _sample_df(df)

    from collections import Counter
    print(Counter(df.length))
    print(Counter(df.voice))
