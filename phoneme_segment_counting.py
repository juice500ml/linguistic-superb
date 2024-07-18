import os
import random
from collections import defaultdict
from datasets import load_dataset, Audio
from utils import validate_dataset

first = [
    'Listen to the audio and count the number of English phonemes present.',
    'Count how many English phonemes you hear in this audio.',
    'Determine and count the number of English phonemes in the provided audio.',
    'Calculate the number of English phonemes in the provided audio.',
    'Identify the number of English phonemes in the provided audio recording.',
    'Identify the number of English phonemes in this audio file.',
    'Identify the total English phonemes in this audio clip.',
    'Count the English phonemes present in this given audio file.',
    'From the audio, ascertain the number of English phonemes in the utterance.',
    'Listen and count the English phonemes in this audio recording.',
    'Please count the number of English phonemes in this audio.',
    'Identify and count the English phonemes in this audio sample.',
    'Count the total number of English phonemes in the given audio sample.',
    'Please determine the number of English phonemes in the audio provided.',
    'Determine the number of English phonemes in the given utterance.',
    'Based on this audio, count the number of English phonemes in the corresponding utterance.',
    'Based on this audio, identify the number of English phonemes.',
    'Calculate the total number of English phonemes in this audio clip.',
    'Based on the audio clip, calculate the number of English phonemes.',
    'Listen to the audio file and count the total English phonemes.'
]

second = [
    ' Use Arabic numbers for your answer.',
    ' Write in digits.',
    ' Use Arabic digits.',
    ' Write your answer in numbers.',
    ' Write in Arabic numerals.',
    ' Please use Arabic numerals.',
    ' Write in Arabic numbers.',
    ' Use numbers for your response.',
    ' Write in Arabic digits.',
    ' Please write the number in Arabic numerals.',
    ' Answer with Arabic numbers.',
    ' Respond using Arabic digits.',
    ' Write your answer with Arabic digits.',
    ' Use digits to write your answer.',
    ' Provide your answer in Arabic numbers.',
    ' Respond in Arabic numerals.',
    ' Use Arabic numerals for your response.',
    ' Please write your answer using numbers.',
    ' Provide the answer in digits.',
    ' Use numbers for your answer.'
]


# Using list comprehension to generate the combined list
instructions = [f + " " + s for f in first for s in second]

if __name__ == "__main__":
    ds = load_dataset(
        "speech31/Librispeech_word",
        cache_dir="datasets_cache",
        revision="refs/convert/parquet",
    )
    new_ds = ds["test"]

    # Create a set of unique filenames
    unique_filenames = set(new_ds['filename'])
    print(len(unique_filenames))

    # Filter the dataset to only include samples with filenames in the set
    new_ds = new_ds.filter(lambda sample: sample['filename'] in unique_filenames)
    print(len(new_ds))

    # Filter out samples with length
    new_ds = new_ds.filter(lambda sample: 1 < sample["phonemeCount"])

    # Filter out samples longer than 2 seconds
    new_ds = new_ds.filter(lambda sample: 0.3 < len(sample["audio"]["array"]) / sample["audio"]["sampling_rate"])

    # Categorize the samples by their lengths
    length_categories = defaultdict(list)
    for i, sample in enumerate(new_ds):
        length_categories[sample["phonemeCount"]].append(i)

    # Randomly select 60% of the samples for each length
    selected_indices = []
    for i, (length, indices) in enumerate(sorted(length_categories.items())):
        num_to_select = int(len(indices) * 0.3)
        random.seed(i)
        selected_indices.extend(random.sample(indices, num_to_select))

    # Create a new dataset with the selected indices
    new_ds = new_ds.select(selected_indices)

    # Reformatting
    def _map(sample, index):
        return {
            "audio": sample["audio"],
            "file": sample["filename"],
            "instruction": instructions[index % len(instructions)],
            "label": sample["phonemeCount"],
        }
    new_ds = new_ds.map(_map, with_indices=True, remove_columns=ds["test"].column_names, num_proc=8)
    new_ds = new_ds.cast_column("audio", Audio(sampling_rate=16_000))

    # Validate & Push
    validate_dataset(new_ds)
    new_ds.push_to_hub(repo_id="speech31/PhonemeSegmentCounting_Librispeech-words", split="test", token=os.environ["HF_TOKEN"])
