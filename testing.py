from datasets import Dataset, load_dataset, Audio
from utils import validate_dataset
from panphon.distance import Distance
import time
from collections import defaultdict
from tqdm import tqdm
import os


ds = load_dataset(
    "speech31/voxangeles",
    # cache_dir="datasets_cache",
    revision="refs/convert/parquet",
)
ds = ds["test"]
dist = Distance()


file_to_word = defaultdict(str)
words, filenames = ds["word"], ds["file"]
for i, filename in enumerate(filenames):
    lang = filename.split('-')[0]
    word = words[i]
    file_to_word[filename] = word

new_ds = load_dataset(
    "kalbin/MultilingualPronunciationSimilarity_VoxAngeles",
    # cache_dir="datasets_cache",
    revision="refs/convert/parquet",
)
new_ds = new_ds["test"]

# display the FED
def get_fed(sample):
    A, B, X = file_to_word[sample["file1"]], file_to_word[sample["file2"]], file_to_word[sample["file3"]]
    sample["dist_A_X"] = dist.feature_edit_distance(A, X)
    sample["dist_B_X"] = dist.feature_edit_distance(B, X)
    sample["A"] = A
    sample["B"] = B
    sample["X"] = X
    return sample
new_ds = new_ds.map(get_fed)


new_ds.push_to_hub(repo_id="kalbin/MultilingualPronunciationSimilarity_VoxAngeles", split="test", token=os.environ["HF_TOKEN"])


# TODO: then empirically observe the values around the ranges

CLOSE_LOWER_BOUND = 0.2
CLOSE_UPPER_BOUND = 0.4
FAR_LOWER_BOUND = 0.6
FAR_UPPER_BOUND = 0.8