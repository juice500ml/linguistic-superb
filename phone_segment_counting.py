import os
import random
from collections import defaultdict

from datasets import load_dataset, Audio
from utils import validate_dataset

first = [
    'Listen to the audio and count the number of phones present.',
    'Count how many phones you hear in this audio.',
    'Determine and count the number of phones in the provided audio.',
    'Calculate the number of phones in the provided audio.',
    'Identify the number of phones in the provided audio recording.',
    'Identify the number of phones in this audio file.',
    'Identify the total phones in this audio clip.',
    'Count the phones present in this given audio file.',
    'From the audio, ascertain the number of phones in the utterance.',
    'Listen and count the phones in this audio recording.',
    'Please count the number of phones in this audio.',
    'Identify and count the phones in this audio sample.',
    'Count the total number of phones in the given audio sample.',
    'Please determine the number of phones in the audio provided.',
    'Determine the number of phones in the given utterance.',
    'Based on this audio, count the number of phones in the corresponding utterance.',
    'Based on this audio, identify the number of phones.',
    'Calculate the total number of phones in this audio clip.',
    'Based on the audio clip, calculate the number of phones.'
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
    ' Write in Arabic digits.'
]

# Using list comprehension to generate the combined list
instructions = [f + " " + s for f in first for s in second]

if __name__ == "__main__":
    ds = load_dataset(
        "speech31/voxangeles_v3",
        cache_dir="datasets_cache",
        revision="refs/convert/parquet",
    )
    new_ds = ds["test"]

    diacritics =  ["'", 'ʰ', 'ʱ', 'ʲ', 'ʷ', 'ʼ', 'ˀ', 'ˁ', 'ː', '˞', 
    'ˠ', 'ˢ', 'ˤ', '˭', '̃', '̄', '̆', '̈', '̘', '̙', '̚', '̜', 
    '̝', '̞', '̟', '̠', '̣', '̤', '̥', '̩', '̪', '̯', '̰', '̱', '̴', '͡', 'ⁿ', 'ᵐ']

    def remove_diacritics(input_str, diacritics):
        for diacritic in diacritics:
            input_str = input_str.replace(diacritic, '')
        return input_str

    def calculate_length(sample):
        phn_without_diacritics = remove_diacritics(sample["phn"], diacritics)
        phones = [phone for phone in list(phn_without_diacritics) if phone.strip()]
        print(sample["phn"], phones, len(phones))
        sample["label"] = len(phones)
        return sample

    new_ds = new_ds.map(calculate_length)

    # Filter out samples with length
    new_ds = new_ds.filter(lambda sample: 1 < sample["label"] <= 10)

    # Filter out samples longer than 2 seconds
    new_ds = new_ds.filter(lambda sample: len(sample["audio"]["array"]) / sample["audio"]["sampling_rate"] <= 2)

    # Categorize the samples by their lengths
    length_categories = defaultdict(list)
    for i, sample in enumerate(new_ds):
        length_categories[sample["label"]].append(i)

    # Randomly select 60% of the samples for each length
    selected_indices = []
    for i, (length, indices) in enumerate(sorted(length_categories.items())):
        num_to_select = int(len(indices) * 0.6)
        random.seed(i)
        selected_indices.extend(random.sample(indices, num_to_select))

    # Create a new dataset with the selected indices
    new_ds = new_ds.select(selected_indices)

    # Reformatting
    def _map(sample, index):
        return {
            "audio": sample["audio"],
            "file": sample["file"],
            "instruction": instructions[index % len(instructions)],
            "label": sample["label"],
        }
    new_ds = new_ds.map(_map, with_indices=True, remove_columns=ds["test"].column_names)
    new_ds = new_ds.cast_column("audio", Audio(sampling_rate=16_000))

    # Validate & Push
    # validate_dataset(new_ds)
    new_ds.push_to_hub(repo_id="DynamicSuperb/PhoneSegmentCounting_VoxAngeles", split="test", token=os.environ["HF_TOKEN"])
