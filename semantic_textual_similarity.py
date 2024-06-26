import os
import numpy as np

from datasets import load_dataset, Audio

from utils import validate_dataset, rows_to_dataset


questions = [ # first pair is more similar
    "Consider two pairs of utterances, where the first pair consists of the first and the second utterance, and the second pair consists of the third and the fourth utterance. Is the first pair more similar in meaning than the second pair? Answer yes or no.",
    "Given two pairs of utterances, the first pair being the first and second utterance, and the second pair being the third and fourth utterance. Does the first pair share more similarity in meaning than the second pair? Please indicate yes or no.",
    "Evaluate two pairs of utterances: the first pair includes the first and second utterance, and the second pair includes the third and fourth utterance. Is the first pair more alike in meaning than the second pair? Answer with yes or no.",
    "Compare the first and second utterance as a pair and the third and fourth utterance as another pair. Is the first pair more closely related in meaning than the second pair? Provide a yes or no response.",
    "For the utterance pairs, first and second, and third and fourth, determine if the first pair has greater similarity in meaning compared to the second pair. Reply with yes or no.",
    "Are the utterances in the first pair, first and second, more alike in meaning than those in the second pair, third and fourth? State yes or no.",
    "Assess the meaning similarity between the first and second utterances and between the third and fourth utterances. Is the first pair more alike in meaning? Indicate yes or no.",
    "Given two pairs of utterances: the first pair, first and second, and the second pair, third and fourth, decide if the first pair has a higher similarity in meaning. Answer yes or no.",
    "Between the first pair, first and second utterances, and the second pair, third and fourth utterances, does the first pair show more similarity in meaning? Respond with yes or no.",
    "Is the meaning similarity between the first and second utterances greater than the similarity between the third and fourth utterances? Please answer yes or no.",
    "Consider the first pair, first and second utterances, and the second pair, third and fourth utterances. Is the first pair more closely related in meaning? Reply yes or no.",
    "Compare the first and second utterances as one pair, and the third and fourth utterances as another pair. Does the first pair share more similarity in meaning? Indicate with yes or no.",
    "In terms of meaning similarity, are the first and second utterances more alike than the third and fourth utterances? Provide your answer: yes or no.",
    "Look at two pairs of utterances: the first, first and second, and the second, third and fourth. Is the first pair more closely related in meaning? State your response: yes or no.",
    "Are the first and second utterances more alike in meaning compared to the third and fourth utterances? Please reply with yes or no.",
    "For the pairs of utterances, first pair: first and second, second pair: third and fourth, is the similarity in meaning higher in the first pair? Answer with a yes or no."
]
inversed_questions = [ # second pair is more similar
    "Consider two pairs of utterances, where the first pair consists of the first and the second utterance, and the second pair consists of the third and the fourth utterance. Is the first pair less similar in meaning than the second pair? Answer yes or no.",
    "Given two pairs of utterances, the first pair being the first and second utterance, and the second pair being the third and fourth utterance. Does the first pair share less similarity in meaning than the second pair? Please indicate yes or no.",
    "Evaluate two pairs of utterances: the first pair includes the first and second utterance, and the second pair includes the third and fourth utterance. Is the first pair less alike in meaning than the second pair? Answer with yes or no.",
    "Compare the first and second utterance as a pair and the third and fourth utterance as another pair. Is the second pair more closely related in meaning than the first pair? Provide a yes or no response.",
    "For the utterance pairs, first and second, and third and fourth, determine if the first pair has lesser similarity in meaning compared to the second pair. Reply with yes or no.",
    "Are the utterances in the first pair, first and second, less alike in meaning than those in the second pair, third and fourth? State yes or no.",
    "Assess the meaning similarity between the first and second utterances and between the third and fourth utterances. Is the first pair less alike in meaning compared to the second pair? Indicate yes or no.",
    "Given two pairs of utterances: the first pair, first and second, and the second pair, third and fourth, decide if the first pair has a lower similarity in meaning. Answer yes or no.",
    "Between the first pair, first and second utterances, and the second pair, third and fourth utterances, does the first pair show less similarity in meaning? Respond with yes or no.",
    "Is the meaning similarity between the first and second utterances lesser than the similarity between the third and fourth utterances? Please answer yes or no.",
    "Consider the first pair, first and second utterances, and the second pair, third and fourth utterances. Is the first pair less related in meaning compared to the second pair? Reply yes or no.",
    "Compare the first and second utterances as one pair, and the third and fourth utterances as another pair. Does the first pair share less similarity in meaning? Indicate with yes or no.",
    "In terms of meaning similarity, are the first and second utterances less alike than the third and fourth utterances? Provide your answer: yes or no.",
    "Look at two pairs of utterances: the first, first and second, and the second, third and fourth. Is the second pair more closely related in meaning compared to the first pair? State your response: yes or no.",
    "Are the first and second utterances less alike in meaning compared to the third and fourth utterances? Please reply with yes or no.",
    "For the pairs of utterances, first pair: first and second, second pair: third and fourth, is the similarity in meaning lower in the first pair? Answer with a yes or no."
]


if __name__ == "__main__":
    ds = load_dataset("juice500/spoken_sts", cache_dir="datasets_cache")

    similarities = np.array(ds["test"]["similarity"])
    speakers = np.array(ds["test"]["speaker_id"])
    indices = np.arange(len(similarities))
    unique_ids = [
        f"{t}_{st}_{pid}"
        for (t, st, pid) in zip(ds["test"]["task"], ds["test"]["subtask"], ds["test"]["pair_id"])
    ]

    total_quadruplets = 250

    l_pair, r_pair, uids = [], [], []
    np.random.seed(42)
    while len(l_pair) < total_quadruplets:
        l_index = np.random.randint(len(indices))
        r_index = np.random.randint(len(indices))
        if (l_index not in (l_pair + r_pair)) \
            and (r_index not in (l_pair + r_pair)) \
            and (abs(similarities[l_index] - similarities[r_index]) >= 1.0) \
            and (speakers[l_index] == speakers[r_index]) \
            and (unique_ids[l_index] not in uids):
            l_pair.append(l_index)
            r_pair.append(r_index)
            uids.append(unique_ids[l_index])
            uids.append(unique_ids[r_index])

    is_flipped_list = [False] * (total_quadruplets//2) + [True] * (total_quadruplets - (total_quadruplets//2))
    np.random.seed(42)
    np.random.shuffle(is_flipped_list)

    rows = {"instruction": [], "label": [], "file1": [], "file2": [], "file3": [], "file4": []}
    for i, (l_index, r_index, is_flipped) in enumerate(zip(l_pair, r_pair, is_flipped_list)):
        rows["instruction"].append(inversed_questions[i % len(inversed_questions)] if is_flipped else questions[i % len(questions)])
        rows["label"].append("yes" if (
            ((ds["test"]["similarity"][l_index] > ds["test"]["similarity"][r_index]) and (not is_flipped))
            or ((ds["test"]["similarity"][l_index] < ds["test"]["similarity"][r_index]) and (is_flipped))
        ) else "no")
        rows["file1"].append(f"{ds['test']['subtask'][l_index]}_{ds['test']['pair_id'][l_index]}_0_human-speaker-{ds['test']['speaker_id'][l_index]}.wav")
        rows["file2"].append(f"{ds['test']['subtask'][l_index]}_{ds['test']['pair_id'][l_index]}_1_human-speaker-{ds['test']['speaker_id'][l_index]}.wav")
        rows["file3"].append(f"{ds['test']['subtask'][r_index]}_{ds['test']['pair_id'][r_index]}_0_human-speaker-{ds['test']['speaker_id'][r_index]}.wav")
        rows["file4"].append(f"{ds['test']['subtask'][r_index]}_{ds['test']['pair_id'][r_index]}_1_human-speaker-{ds['test']['speaker_id'][r_index]}.wav")

    new_ds = rows_to_dataset(rows)
    new_ds = new_ds.add_column("audio1", ds["test"].select(l_pair)["audio_a"]).cast_column("audio1", Audio(sampling_rate=16000))
    new_ds = new_ds.add_column("audio2", ds["test"].select(l_pair)["audio_b"]).cast_column("audio2", Audio(sampling_rate=16000))
    new_ds = new_ds.add_column("audio3", ds["test"].select(r_pair)["audio_a"]).cast_column("audio3", Audio(sampling_rate=16000))
    new_ds = new_ds.add_column("audio4", ds["test"].select(r_pair)["audio_b"]).cast_column("audio4", Audio(sampling_rate=16000))

    # Validate & Push
    validate_dataset(new_ds)
    new_ds.push_to_hub(repo_id="DynamicSuperb/SemanticTextualSimilarity_SpokenSTS", split="test", token=os.environ["HF_TOKEN"])
