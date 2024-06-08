import os
import numpy as np
import matplotlib.pyplot as plt
from datasets import load_dataset, Audio
from collections import defaultdict
import random
from utils import validate_dataset

instructions = [
    "Based on this audio, count the number of phones in the corresponding utterance. Please write in Arabic numerals.",
    "Please count the number of phones in this audio. Use Arabic numerals for your answer.",
    "Determine the number of phones in the given utterance. Write your answer in Arabic numerals.",
    "Identify the number of phones in this audio file. Use Arabic numerals.",
    "Listen to the audio and count the number of phones present. Write in Arabic numerals.",
    "Calculate the total number of phones in this audio clip. Please use Arabic numerals.",
    "From the audio, ascertain the number of phones in the utterance. Write in Arabic numerals.",
    "Count how many phones you hear in this audio. Use Arabic numerals for your response.",
    "Please determine the number of phones in the audio provided. Write in Arabic numerals.",
    "Identify and count the phones in this audio sample. Use Arabic numerals.",
    "Based on this audio, identify the number of phones. Write in Arabic numerals.",
    "Calculate the number of phones in the provided audio. Use Arabic numerals.",
    "Listen and count the phones in this audio recording. Write your answer in Arabic numerals.",
    "Count the phones present in this given audio file. Use Arabic numerals.",
    "Identify the total phones in this audio clip. Write in Arabic numerals.",
    "Determine and count the number of phones in the provided audio. Use Arabic numerals.",
    "Based on the audio clip, calculate the number of phones. Write in Arabic numerals.",
    "Count the total number of phones in the given audio sample. Use Arabic numerals.",
    "Listen to the audio and determine the number of phones. Write in Arabic numerals.",
    "Identify the number of phones in the provided audio recording. Use Arabic numerals."
]

if __name__ == "__main__":
    ds = load_dataset(
        "speech31/voxangeles",
        # "DynamicSuperb/PhoneSegmentCounting_VoxAngeles",
        cache_dir="datasets_cache",
        # revision="refs/convert/parquet",
        num_proc=32
    )
    new_ds = ds['test']

    def calculate_length(sample):
        sample["label"] = len(sample["phn"].split())
        return sample

    new_ds = new_ds.map(calculate_length, num_proc=32)

    # Filter out samples with length 
    new_ds = new_ds.filter(lambda sample: 1 < sample["label"] <= 8)

    def calculate_length(sample):
        sample["label"] = len(sample["phn"].split())
        return sample

    new_ds = new_ds.map(calculate_length, num_proc=32)

    # Filter out samples with length
    new_ds = new_ds.filter(lambda sample: 1 < sample["label"] <= 10)

    # Categorize the samples by their lengths
    length_categories = defaultdict(list)
    for i, sample in enumerate(new_ds):
        length_categories[sample["label"]].append(i)

    # Randomly select 80% of the samples for each length
    selected_indices = []
    for length, indices in length_categories.items():
        num_to_select = int(len(indices) * 0.6)
        selected_indices.extend(random.sample(indices, num_to_select))

    # Create a new dataset with the selected indices
    new_ds = new_ds.select(selected_indices)

    # # Reformatting
    def _map(sample, index):
        return {
            "audio": sample["audio"]["path"],
            "file": sample["file"],
            "instruction": instructions[index % len(instructions)],
            "label": sample["label"],
        }
    new_ds = new_ds.map(_map, with_indices=True, remove_columns=ds["test"].column_names)
    new_ds = new_ds.cast_column("audio", Audio(sampling_rate=16_000))

    # Validate & Push
    validate_dataset(new_ds)
    new_ds.push_to_hub(repo_id="speech31/PhoneSegmentCounting_VoxAngeles", split="test", token=os.environ["HF_TOKEN"])
