import os
import numpy as np
from datasets import load_dataset, Audio

from utils import validate_dataset


scale_description = "5 being completely equivalent, 4 being equivalent but unimportant details differing, 3 being equivalent but some important details missing or differs, 2 being not equivalent and only sharing some details, 1 being not equivalent but on the same topic, and 0 being on different topics."
instructions = [
    "Score how similar two sentences are to each other according to the following scale: " + scale_description,
    "Rate the similarity of two sentences using this scale: " + scale_description,
    "Assess how alike two sentences are based on the following criteria: " + scale_description,
    "Measure the similarity between two sentences according to this scoring rubric: " + scale_description,
    "Rate how alike two sentences are based on the following scale: " + scale_description,
    "Use the following scale to rate the similarity between two sentences: " + scale_description,
    "Score the likeness of two sentences according to the scale given: " + scale_description,
    "Determine the similarity between two sentences using this scale: " + scale_description,    
    "Judge the similarity between two sentences using this scale: " + scale_description,    
    "Measure the similarity of two sentences with the following scale: " + scale_description,
    "Assess the degree of similarity between two sentences according to this scale: " + scale_description,
]


if __name__ == "__main__":
    ds = load_dataset("juice500/spoken_sts", cache_dir="datasets_cache")

    # Filter 500 samples
    indices = np.arange(len(ds["test"]["similarity"]))
    np.random.RandomState(42).shuffle(indices)
    indices = indices[:500]
    new_ds = ds["test"].filter(lambda _, idx: idx in indices, with_indices=True)

    # Reformatting
    def _map(sample, index):
        return {
            "label": sample["similarity"],
            "audio1": sample["audio_a"],
            "audio2": sample["audio_b"],
            "file1": f"{sample['task']}_{sample['audio_a']['path']}",
            "file2": f"{sample['task']}_{sample['audio_b']['path']}",
            "instruction": instructions[index % len(instructions)],
        }
    new_ds = new_ds.map(_map, with_indices=True, remove_columns=ds["test"].column_names)
    new_ds = new_ds.cast_column("audio1", Audio(sampling_rate=16_000))
    new_ds = new_ds.cast_column("audio2", Audio(sampling_rate=16_000))

    # Validate & Push
    validate_dataset(new_ds)
    new_ds.push_to_hub(repo_id="DynamicSuperb/SemanticTextualSimilarity_SpokenSTS", split="test", token=os.environ["HF_TOKEN"])
