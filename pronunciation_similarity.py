import os
import random
from collections import defaultdict

from datasets import Dataset, load_dataset, Audio
from utils import validate_dataset
from panphon.distance import Distance
from tqdm import tqdm


# (normalized) feature edit distance threshold
SIM_THRESH = 0.3

instructions = [
    "Based on the three audio files (A, B, X), determine whether word X is closer in pronunciation to word A or word B. The answer could be A or B.",
    "Please determine whether word X is closer in pronunciation to word A or word B given the three audio files (A, B, X). Respond with A or B.",
    "Determine if word X is more similar in pronunciation to word A or word B given the three audio files (A, B, X). Write your answer as either A or B.",
    "Examine if word X is more similar in pronunciation to word A or word B given the three audio files (A, B, X)? Write your answer as either A or B.",
    "Listen to the three audio clips (A, B, X) and judge whether word X is closer in pronunciation to word A or word B. Write your answer as either A or B.",
    "Decide if word X is more similar in pronunciation to word A or word B in these three audio clips (A, B, X). The answer could be A or B.",
    "From the three audio inputs (A, B, X), ascertain if word X is more similar in pronunciation to word A or word B. The answer is A or B.",
    "Judge whether word X is closer in pronunciation to word A or word B. Respond with A or B.",
    "Please determine if word X sounds more similar to word A or word B in the audio clips provided (A, B, X). The answer is A or B.",
    "Decide whether word X sounds more similar to word A or word B in the provided audio clips (A, B, X). this audio sample. The answer is A or B.",
    "Based on these three audio clips (A, B, X), judge whether word X sounds more similar to word A or word B. Write your answer as A or B.",
    "Given the three audio files (A, B, X), determine whether word X is closer in pronunciation to word A or word B. The answer could be A or B",
    "Using the three audio files (A, B, X), determine if word X is pronounced more similarly to word A or word B. The answer should be A or B.",
    "Please determine if word X sounds more like word A or word B in the provided audio clips (A, B, X). The answer is A or B.",
    "Assess whether word X is closer in pronunciation to word A or word B, based on the three audio files (A, B, X). Respond with A or B.",
    "Listen to the three audio files (A, B, X) and decide if word X is pronounced more similarly to word A or word B. Your answer should be A or B.",
    "Using the three audio recordings (A, B, X), judge whether word X is closer in pronunciation to word A or word B. Your answer should be either A or B.",
    "From the three audio clips (A, B, X), decide whether word X is pronounced more similarly to word A or word B. Write A or B as your answer.",
    "Using the three audio recordings (A, B, X), decide if word X sounds more like word A or word B. Your answer should be either A or B.",
    "With the provided audio files (A, B, X), decide if word X is closer in pronunciation to word A or word B. Indicate your answer as either A or B."
]


def generate_triplets(filenames):
    """
    Generate ABX (ordered) triplets within each language
    """
    # example filename: nan-004-028
    lang_to_file = defaultdict(list)
    for filename in filenames:
        lang = filename.split('-')[0]
        lang_to_file[lang].append(filename)

    # since all possible triplets takes too long, just generate triplets
        # where each element is used once
    triplets = []
    for lang, files in lang_to_file.items():
        random.shuffle(files)

        for i in range(0, len(files) - 2, 3):
            triplets.append((files[i], files[i + 1], files[i + 2]))

    return triplets


if __name__ == "__main__":
    ds = load_dataset(
        "speech31/voxangeles",
        cache_dir="datasets_cache",
        revision="refs/convert/parquet",
    )

    # multilingual pronunciation similarity
        # A,B,X from the same language
        # but we cover multiple languages

    random.seed(15213)

    # Filter out samples longer than 2 seconds
    ds = ds.filter(lambda sample: len(sample["audio"]["array"]) / sample["audio"]["sampling_rate"] <= 2)

    # ensure no transcriptions are nan
    ds = ds.filter(lambda sample: sample["word"] is not None)

    # get indices
    file_to_index = defaultdict(str)
    for i, ex in enumerate(ds['test']):
        file_to_index[ex["file"]] = i

    # columns: file1, audio1; file2, audio2; file3, audio3
    rows = defaultdict(list)
    for (file1, file2, file3) in tqdm(generate_triplets(ds["test"]["file"])):
        rows["file1"].append(file1)
        file1_ex = ds["test"][file_to_index[file1]]
        rows["audio1"].append(file1_ex["audio"])
        # the phones for the word, e.g. "kaʊ"
        rows["word1"].append(file1_ex["word"])

        rows["file2"].append(file2)
        file2_ex = ds["test"][file_to_index[file2]]
        rows["audio2"].append(file2_ex["audio"])
        rows["word2"].append(file2_ex["word"])

        rows["file3"].append(file3)
        file3_ex = ds["test"][file_to_index[file3]]
        rows["audio3"].append(file3_ex["audio"])
        rows["word3"].append(file3_ex["word"])
    ds = Dataset.from_dict(rows)

    dist = Distance()
    def generate_label(sample):
        # use feature edit distance to quantify pronunciation similarity
        A, B, X = sample["word1"], sample["word2"], sample["word3"]
        sample["dist_A_X"] = dist.feature_edit_distance(A, X)
        sample["dist_B_X"] = dist.feature_edit_distance(B, X)
        sample["label"] = "A" \
            if sample["dist_A_X"] < sample["dist_B_X"] \
            else "B"
        return sample

    ds = ds.map(generate_label, num_proc=32)

    # only keep examples where the pronunciation similarity is unambiguous
    ds = ds.filter(lambda sample: abs(sample["dist_A_X"] - sample["dist_B_X"]) > SIM_THRESH)
