import os
import random
from collections import defaultdict

from datasets import load_dataset, Audio
from utils import validate_dataset
from panphon.distance import Distance


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
