import os
import random
from collections import defaultdict

from datasets import Dataset, load_dataset, Audio
from utils import validate_dataset
from panphon.distance import Distance
from tqdm import tqdm


# (normalized) feature edit distance threshold
CLOSE_LOWER_BOUND = 0.2
CLOSE_UPPER_BOUND = 0.4
FAR_LOWER_BOUND = 0.6
FAR_UPPER_BOUND = 0.8


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


def generate_triplets(filenames, words, dist):
    """
    Generate ABX (ordered) triplets of filenames within each language
    """
    # example filename: nan-004-028
    # example word: tsiʔ
    lang_to_words = defaultdict(list)
    word_to_file = defaultdict(str)
    for i, filename in enumerate(filenames):
        lang = filename.split('-')[0]
        word = words[i]
        lang_to_words[lang].append(word)
        word_to_file[word] = filename

    triplets = []
    for lang, words in tqdm(lang_to_words.items()):
        # compute feature edit distance between every other word
        fed = defaultdict(float)
        pairs = set()
        for i, w1 in enumerate(words):
            for j, w2 in enumerate(words):
                if w1 and w2 and i != j:
                    fed[(w1, w2)] = dist.feature_edit_distance(w1, w2)
                    pairs.add((w1, w2))

        # find close word pairs (CLOSE_LOWER_BOUND <= dist <= CLOSE_UPPER_BOUND)
        close_pairs = [(w1, w2) for (w1, w2) in pairs \
            if CLOSE_LOWER_BOUND <= fed[(w1, w2)] <= CLOSE_UPPER_BOUND ]
        # find far word pairs (FAR_LOWER_BOUND <= dist <= FAR_UPPER_BOUND)
        far_pairs = [(w1, w2) for (w1, w2) in pairs \
            if FAR_LOWER_BOUND <= fed[(w1, w2)] <= FAR_UPPER_BOUND ]

        # then create (A, B, X) triplets
            # (A, X) = close_pair then find all (B, X) in far_pair
        for (A, X) in close_pairs:
            for (B, _) in [(w1, w2) for (w1, w2) in far_pairs if X == w2]:
                file_A = word_to_file[A]
                file_B = word_to_file[B]
                file_X = word_to_file[X]
                triplets.append((file_A, file_B, file_X))

        # does not duplicate cases where A and B are swapped

    return triplets
    # 127,743 triplets


if __name__ == "__main__":
    ds = load_dataset(
        "speech31/voxangeles",
        cache_dir="datasets_cache",
        revision="refs/convert/parquet",
    )
    new_ds = ds["test"]

    # multilingual pronunciation similarity
        # A,B,X from the same language
        # but we cover multiple languages

    random.seed(15213)

    # Filter out samples longer than 2 seconds
    new_ds = new_ds.filter(lambda sample: len(sample["audio"]["array"]) / sample["audio"]["sampling_rate"] <= 2)

    # ensure no transcriptions are nan
    new_ds = new_ds.filter(lambda sample: sample["word"] is not None)

    # get indices
    file_to_index = defaultdict(str)
    for i, ex in enumerate(new_ds):
        file_to_index[ex["file"]] = i

    # columns: file1, audio1; file2, audio2; file3, audio3
    rows = defaultdict(list)

    # aim for 1000 examples (approx 1 hour)
    triplets = generate_triplets(new_ds["file"], new_ds["word"], Distance())
    triplets = random.sample(triplets, 1000)

    # randomly shuffle A and B so the answer is not always A
    answer_is_B = set(random.sample(range(1000), 500))
    triplets = [(B_f, A_f, X_f) if i in answer_is_B else (A_f, B_f, X_f) for i, (A_f, B_f, X_f) in enumerate(triplets)]

    for i, (file1, file2, file3) in tqdm(enumerate(triplets)):
        rows["label"].append("B" if i in answer_is_B else "A")

        rows["file1"].append(file1)
        file1_ex = new_ds[file_to_index[file1]]
        rows["audio1"].append(file1_ex["audio"])
        # the phones for the word, e.g. "kaʊ"
        rows["word1"].append(file1_ex["word"])

        rows["file2"].append(file2)
        file2_ex = new_ds[file_to_index[file2]]
        rows["audio2"].append(file2_ex["audio"])
        rows["word2"].append(file2_ex["word"])

        rows["file3"].append(file3)
        file3_ex = new_ds[file_to_index[file3]]
        rows["audio3"].append(file3_ex["audio"])
        rows["word3"].append(file3_ex["word"])
    new_ds = Dataset.from_dict(rows)

    # Reformatting
    def _map(sample, index):
        return {
            "audio1": sample["audio1"],
            "file1": sample["file1"],
            "audio2": sample["audio2"],
            "file2": sample["file2"],
            "audio3": sample["audio3"],
            "file3": sample["file3"],
            "instruction": instructions[index % len(instructions)],
            "label": sample["label"],
        }
    new_ds = new_ds.map(_map, with_indices=True, remove_columns=new_ds.column_names)
    new_ds = new_ds.cast_column("audio1", Audio(sampling_rate=16_000))
    new_ds = new_ds.cast_column("audio2", Audio(sampling_rate=16_000))
    new_ds = new_ds.cast_column("audio3", Audio(sampling_rate=16_000))

    # Validate & Push
    validate_dataset(new_ds)
    new_ds.push_to_hub(repo_id="kalbin/MultilingualPronunciationSimilarity_VoxAngeles", split="test", token=os.environ["HF_TOKEN"])
