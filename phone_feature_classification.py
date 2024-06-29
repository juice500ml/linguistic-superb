import random
import os

import pandas as pd
from datasets import load_dataset, Dataset, Audio
from panphon import FeatureTable
import panphon.sonority
from panphon.distance import Distance
from ipapy import UNICODE_TO_IPA
from ipapy.ipachar import IPAVowel, IPAConsonant
import soundfile as sf

from utils import validate_dataset

# TODO: pick limited phone set and only pick these phones (with 1 diacritic?)
# TODO: could also narrow the set down when we generate the answer - include the 5 closest phones using FED?
PHONE_SET = ["a", "e", "i", "o", "u"]
PHONE_SET_STR = ", ".join(PHONE_SET[:-1]) + ", or " + PHONE_SET[-1]

# phone classification
phone_classification_instructions = [
    "The audio clip consists of three phones. What is the phone in the middle? The answer could be: ",
    "The audio clip consists of three phones. Identify the phone in the middle. The answer could be: ",
    "The audio clip consists of three phones. Label the phone in the middle. The answer could be: ",
    "The audio clip consists of three phones. Name the phone in the middle. The answer could be: ",
    "The audio clip consists of three phones. Determine the phone in the middle. The answer could be: ",
    "The audio clip consists of three phones. Please identify the phone in the middle. The answer could be: ",
    "The audio clip consists of three phones. Please label the phone in the middle. The answer could be: ",
    "The audio clip consists of three phones. Please name the phone in the middle. The answer could be: ",
    "The audio clip consists of three phones. Please determine the phone in the middle. The answer could be: ",
    "In the given triphone utterance, what is the phone in the middle? Choose from the following: ",
    "In the given triphone utterance, identify the phone in the middle. Choose from the following: ",
    "In the given triphone utterance, label the phone in the middle. Choose from the following: ",
    "In the given triphone utterance, name the phone in the middle. Choose from the following: ",
    "In the given triphone utterance, determine the phone in the middle. Choose from the following: ",
    "In the given triphone utterance, please identify the phone in the middle. Choose from the following: ",
    "In the given triphone utterance, please label the phone in the middle. Choose from the following: ",
    "In the given triphone utterance, please name the phone in the middle. Choose from the following: ",
    "In the given triphone utterance, please determine the phone in the middle. Choose from the following: ",
    "What is the phone in the middle in this triphone audio file? Pick one phone from the following: ",
    "Identify the phone in the middle in this triphone audio file. Pick one phone from the following: ",
    "Label the phone in the middle in this triphone audio file. Pick one phone from the following: ",
    "Name the phone in the middle in this triphone audio file. Pick one phone from the following: ",
    "Determine the phone in the middle in this triphone audio file. Pick one phone from the following: ",
    "Please identify the phone in the middle in this triphone audio file. Pick one phone from the following: ",
    "Please label the phone in the middle in this triphone audio file. Pick one phone from the following: ",
    "Please name the phone in the middle in this triphone audio file. Pick one phone from the following: ",
    "Please determine the phone in the middle in this triphone audio file. Pick one phone from the following: ",
    "Listen to the audio clip, which consists of 3 phones. What is the phone in the middle? The phone could be ",
    "Listen to the audio clip, which consists of 3 phones, and identify the phone in the middle. The phone could be ",
    "Listen to the audio clip, which consists of 3 phones, and label the phone in the middle. The phone could be ",
    "Listen to the audio clip, which consists of 3 phones, and name the phone in the middle. The phone could be ",
    "Listen to the audio clip, which consists of 3 phones, and determine the phone in the middle. The phone could be ",
    "Listen to the audio clip, which consists of 3 phones, and please identify the phone in the middle. The phone could be ",
    "Listen to the audio clip, which consists of 3 phones, and please label the phone in the middle. The phone could be ",
    "Listen to the audio clip, which consists of 3 phones, and please name the phone in the middle. The phone could be ",
    "Listen to the audio clip, which consists of 3 phones, and please determine the phone in the middle. The phone could be ",
    "From the audio, which contains 3 phones, what is the phone in the middle? Choose one phone from the following: ",
    "From the audio, which contains 3 phones, identify the phone in the middle. Choose one phone from the following: ",
    "From the audio, which contains 3 phones, label the phone in the middle. Choose one phone from the following: ",
    "From the audio, which contains 3 phones, name the phone in the middle. Choose one phone from the following: ",
    "From the audio, which contains 3 phones, determine the phone in the middle. Choose one phone from the following: ",
    "From the audio, which contains 3 phones, please identify the phone in the middle. Choose one phone from the following: ",
    "From the audio, which contains 3 phones, please label the phone in the middle. Choose one phone from the following: ",
    "From the audio, which contains 3 phones, please name the phone in the middle. Choose one phone from the following: ",
    "From the audio, which contains 3 phones, please determine the phone in the middle. Choose one phone from the following: ",
    "What phone do you hear in the middle of this 3-phone audio clip? Choose one from the following: ",
    "Identify the phone you hear in the middle of this 3-phone audio clip. Choose one from the following: ",
    "Label the phone you hear in the middle of this 3-phone audio clip. Choose one from the following: ",
    "Name the phone you hear in the middle of this 3-phone audio clip. Choose one from the following: ",
    "Determine the phone you hear in the middle of this 3-phone audio clip. Choose one from the following: ",
    "Please identify the phone you hear in the middle of this 3-phone audio clip. Choose one from the following: ",
    "Please label the phone you hear in the middle of this 3-phone audio clip. Choose one from the following: ",
    "Please name the phone you hear in the middle of this 3-phone audio clip. Choose one from the following: ",
    "Please determine the phone you hear in the middle of this 3-phone audio clip. Choose one from the following: ",
    "Based on this audio clip that consists of 3 phones, what is the phone in the middle? The phone is one of the following: ",
    "Based on this audio clip that consists of 3 phones, identify the phone in the middle. The phone is one of the following: ",
    "Based on this audio clip that consists of 3 phones, label the phone in the middle. The phone is one of the following: ",
    "Based on this audio clip that consists of 3 phones, name the phone in the middle. The phone is one of the following: ",
    "Based on this audio clip that consists of 3 phones, determine the phone in the middle. The phone is one of the following: ",
    "Based on this audio clip that consists of 3 phones, please identify the phone in the middle. The phone is one of the following: ",
    "Based on this audio clip that consists of 3 phones, please label the phone in the middle. The phone is one of the following: ",
    "Based on this audio clip that consists of 3 phones, please name the phone in the middle. The phone is one of the following: ",
    "Based on this audio clip that consists of 3 phones, please determine the phone in the middle. The phone is one of the following: "
]
phone_classification_instructions = [inst + PHONE_SET_STR + "." for inst in phone_classification_instructions]

# manner of articulation
MANNER_SET = ["plosive", "nasal", "trill", "tap/flap", "fricative", "affricate", "approximant", "lateral", "glide", "click", "ejective", "implosive", "vowel"] # IPA https://www.internationalphoneticassociation.org/sites/default/files/IPA_Kiel_2015.pdf
MANNER_SET_STR = ", ".join(MANNER_SET[:-1]) + ", or " + MANNER_SET[-1]
manner_classification_instructions = [
    "The audio clip consists of three phones. What is the manner of articulation of the phone in the middle? The answer could be: ",
    "The audio clip consists of three phones. Identify the manner of articulation of the phone in the middle. The answer could be: ",
    "The audio clip consists of three phones. Label the manner of articulation of the phone in the middle. The answer could be: ",
    "The audio clip consists of three phones. Name the manner of articulation of the phone in the middle. The answer could be: ",
    "The audio clip consists of three phones. Determine the manner of articulation of the phone in the middle. The answer could be: ",
    "The audio clip consists of three phones. Please identify the manner of articulation of the phone in the middle. The answer could be: ",
    "The audio clip consists of three phones. Please label the manner of articulation of the phone in the middle. The answer could be: ",
    "The audio clip consists of three phones. Please name the manner of articulation of the phone in the middle. The answer could be: ",
    "The audio clip consists of three phones. Please determine the manner of articulation of the phone in the middle. The answer could be: ",
    "In the given triphone utterance, what is the manner of articulation of the phone in the middle? Choose from the following: ",
    "In the given triphone utterance, identify the manner of articulation of the phone in the middle. Choose from the following: ",
    "In the given triphone utterance, label the manner of articulation of the phone in the middle. Choose from the following: ",
    "In the given triphone utterance, name the manner of articulation of the phone in the middle. Choose from the following: ",
    "In the given triphone utterance, determine the manner of articulation of the phone in the middle. Choose from the following: ",
    "In the given triphone utterance, please identify the manner of articulation of the phone in the middle. Choose from the following: ",
    "In the given triphone utterance, please label the manner of articulation of the phone in the middle. Choose from the following: ",
    "In the given triphone utterance, please name the manner of articulation of the phone in the middle. Choose from the following: ",
    "In the given triphone utterance, please determine the manner of articulation of the phone in the middle. Choose from the following: ",
    "What is the manner of articulation of the phone in the middle in this triphone audio file? Pick one phone from the following: ",
    "Identify the manner of articulation of the phone in the middle in this triphone audio file. Pick one phone from the following: ",
    "Label the manner of articulation of the phone in the middle in this triphone audio file. Pick one phone from the following: ",
    "Name the manner of articulation of the phone in the middle in this triphone audio file. Pick one phone from the following: ",
    "Determine the manner of articulation of the phone in the middle in this triphone audio file. Pick one phone from the following: ",
    "Please identify the manner of articulation of the phone in the middle in this triphone audio file. Pick one phone from the following: ",
    "Please label the manner of articulation of the phone in the middle in this triphone audio file. Pick one phone from the following: ",
    "Please name the manner of articulation of the phone in the middle in this triphone audio file. Pick one phone from the following: ",
    "Please determine the manner of articulation of the phone in the middle in this triphone audio file. Pick one phone from the following: ",
    "Listen to the audio clip, which consists of 3 phones. What is the manner of articulation of the phone in the middle? The phone could be ",
    "Listen to the audio clip, which consists of 3 phones, and identify the manner of articulation of the phone in the middle. The phone could be ",
    "Listen to the audio clip, which consists of 3 phones, and label the manner of articulation of the phone in the middle. The phone could be ",
    "Listen to the audio clip, which consists of 3 phones, and name the manner of articulation of the phone in the middle. The phone could be ",
    "Listen to the audio clip, which consists of 3 phones, and determine the manner of articulation of the phone in the middle. The phone could be ",
    "Listen to the audio clip, which consists of 3 phones, and please identify the manner of articulation of the phone in the middle. The phone could be ",
    "Listen to the audio clip, which consists of 3 phones, and please label the manner of articulation of the phone in the middle. The phone could be ",
    "Listen to the audio clip, which consists of 3 phones, and please name the manner of articulation of the phone in the middle. The phone could be ",
    "Listen to the audio clip, which consists of 3 phones, and please determine the manner of articulation of the phone in the middle. The phone could be ",
    "From the audio, which contains 3 phones, what is the manner of articulation of the phone in the middle? Choose one phone from the following: ",
    "From the audio, which contains 3 phones, identify the manner of articulation of the phone in the middle. Choose one phone from the following: ",
    "From the audio, which contains 3 phones, label the manner of articulation of the phone in the middle. Choose one phone from the following: ",
    "From the audio, which contains 3 phones, name the manner of articulation of the phone in the middle. Choose one phone from the following: ",
    "From the audio, which contains 3 phones, determine the manner of articulation of the phone in the middle. Choose one phone from the following: ",
    "From the audio, which contains 3 phones, please identify the manner of articulation of the phone in the middle. Choose one phone from the following: ",
    "From the audio, which contains 3 phones, please label the manner of articulation of the phone in the middle. Choose one phone from the following: ",
    "From the audio, which contains 3 phones, please name the manner of articulation of the phone in the middle. Choose one phone from the following: ",
    "From the audio, which contains 3 phones, please determine the manner of articulation of the phone in the middle. Choose one phone from the following: ",
    "What is the manner of articulation of the phone you hear in the middle of this 3-phone audio clip? Choose one from the following: ",
    "Identify the manner of articulation of the phone you hear in the middle of this 3-phone audio clip. Choose one from the following: ",
    "Label the manner of articulation of the phone you hear in the middle of this 3-phone audio clip. Choose one from the following: ",
    "Name the manner of articulation of the phone you hear in the middle of this 3-phone audio clip. Choose one from the following: ",
    "Determine the manner of articulation of the phone you hear in the middle of this 3-phone audio clip. Choose one from the following: ",
    "Please identify the manner of articulation of the phone you hear in the middle of this 3-phone audio clip. Choose one from the following: ",
    "Please label the manner of articulation of the phone you hear in the middle of this 3-phone audio clip. Choose one from the following: ",
    "Please name the manner of articulation of the phone you hear in the middle of this 3-phone audio clip. Choose one from the following: ",
    "Please determine the manner of articulation of the phone you hear in the middle of this 3-phone audio clip. Choose one from the following: ",
    "Based on this audio clip that consists of 3 phones, what is the manner of articulation of the phone in the middle? The phone is one of the following: ",
    "Based on this audio clip that consists of 3 phones, identify the manner of articulation of the phone in the middle. The phone is one of the following: ",
    "Based on this audio clip that consists of 3 phones, label the manner of articulation of the phone in the middle. The phone is one of the following: ",
    "Based on this audio clip that consists of 3 phones, name the manner of articulation of the phone in the middle. The phone is one of the following: ",
    "Based on this audio clip that consists of 3 phones, determine the manner of articulation of the phone in the middle. The phone is one of the following: ",
    "Based on this audio clip that consists of 3 phones, please identify the manner of articulation of the phone in the middle. The phone is one of the following: ",
    "Based on this audio clip that consists of 3 phones, please label the manner of articulation of the phone in the middle. The phone is one of the following: ",
    "Based on this audio clip that consists of 3 phones, please name the manner of articulation of the phone in the middle. The phone is one of the following: ",
    "Based on this audio clip that consists of 3 phones, please determine the manner of articulation of the phone in the middle. The phone is one of the following: "
]
manner_classification_instructions = [inst + MANNER_SET_STR + "." for inst in manner_classification_instructions]

# place of articulation
PLACE_SET = list(set([p.place for p in UNICODE_TO_IPA.values() if isinstance(p, IPAConsonant)]))
PLACE_SET_STR = ", ".join(PLACE_SET[:-1]) + ", or " + PLACE_SET[-1]
place_classification_instructions = [
    "The audio clip consists of three phones. What is the manner of articulation of the phone in the middle? The answer could be: ",
    "The audio clip consists of three phones. Identify the manner of articulation of the phone in the middle. The answer could be: ",
    "The audio clip consists of three phones. Label the manner of articulation of the phone in the middle. The answer could be: ",
    "The audio clip consists of three phones. Name the manner of articulation of the phone in the middle. The answer could be: ",
    "The audio clip consists of three phones. Determine the manner of articulation of the phone in the middle. The answer could be: ",
    "The audio clip consists of three phones. Please identify the manner of articulation of the phone in the middle. The answer could be: ",
    "The audio clip consists of three phones. Please label the manner of articulation of the phone in the middle. The answer could be: ",
    "The audio clip consists of three phones. Please name the manner of articulation of the phone in the middle. The answer could be: ",
    "The audio clip consists of three phones. Please determine the manner of articulation of the phone in the middle. The answer could be: ",
    "In the given triphone utterance, what is the manner of articulation of the phone in the middle? Choose from the following: ",
    "In the given triphone utterance, identify the manner of articulation of the phone in the middle. Choose from the following: ",
    "In the given triphone utterance, label the manner of articulation of the phone in the middle. Choose from the following: ",
    "In the given triphone utterance, name the manner of articulation of the phone in the middle. Choose from the following: ",
    "In the given triphone utterance, determine the manner of articulation of the phone in the middle. Choose from the following: ",
    "In the given triphone utterance, please identify the manner of articulation of the phone in the middle. Choose from the following: ",
    "In the given triphone utterance, please label the manner of articulation of the phone in the middle. Choose from the following: ",
    "In the given triphone utterance, please name the manner of articulation of the phone in the middle. Choose from the following: ",
    "In the given triphone utterance, please determine the manner of articulation of the phone in the middle. Choose from the following: ",
    "What is the manner of articulation of the phone in the middle in this triphone audio file? Pick one phone from the following: ",
    "Identify the manner of articulation of the phone in the middle in this triphone audio file. Pick one phone from the following: ",
    "Label the manner of articulation of the phone in the middle in this triphone audio file. Pick one phone from the following: ",
    "Name the manner of articulation of the phone in the middle in this triphone audio file. Pick one phone from the following: ",
    "Determine the manner of articulation of the phone in the middle in this triphone audio file. Pick one phone from the following: ",
    "Please identify the manner of articulation of the phone in the middle in this triphone audio file. Pick one phone from the following: ",
    "Please label the manner of articulation of the phone in the middle in this triphone audio file. Pick one phone from the following: ",
    "Please name the manner of articulation of the phone in the middle in this triphone audio file. Pick one phone from the following: ",
    "Please determine the manner of articulation of the phone in the middle in this triphone audio file. Pick one phone from the following: ",
    "Listen to the audio clip, which consists of 3 phones. What is the manner of articulation of the phone in the middle? The phone could be ",
    "Listen to the audio clip, which consists of 3 phones, and identify the manner of articulation of the phone in the middle. The phone could be ",
    "Listen to the audio clip, which consists of 3 phones, and label the manner of articulation of the phone in the middle. The phone could be ",
    "Listen to the audio clip, which consists of 3 phones, and name the manner of articulation of the phone in the middle. The phone could be ",
    "Listen to the audio clip, which consists of 3 phones, and determine the manner of articulation of the phone in the middle. The phone could be ",
    "Listen to the audio clip, which consists of 3 phones, and please identify the manner of articulation of the phone in the middle. The phone could be ",
    "Listen to the audio clip, which consists of 3 phones, and please label the manner of articulation of the phone in the middle. The phone could be ",
    "Listen to the audio clip, which consists of 3 phones, and please name the manner of articulation of the phone in the middle. The phone could be ",
    "Listen to the audio clip, which consists of 3 phones, and please determine the manner of articulation of the phone in the middle. The phone could be ",
    "From the audio, which contains 3 phones, what is the manner of articulation of the phone in the middle? Choose one phone from the following: ",
    "From the audio, which contains 3 phones, identify the manner of articulation of the phone in the middle. Choose one phone from the following: ",
    "From the audio, which contains 3 phones, label the manner of articulation of the phone in the middle. Choose one phone from the following: ",
    "From the audio, which contains 3 phones, name the manner of articulation of the phone in the middle. Choose one phone from the following: ",
    "From the audio, which contains 3 phones, determine the manner of articulation of the phone in the middle. Choose one phone from the following: ",
    "From the audio, which contains 3 phones, please identify the manner of articulation of the phone in the middle. Choose one phone from the following: ",
    "From the audio, which contains 3 phones, please label the manner of articulation of the phone in the middle. Choose one phone from the following: ",
    "From the audio, which contains 3 phones, please name the manner of articulation of the phone in the middle. Choose one phone from the following: ",
    "From the audio, which contains 3 phones, please determine the manner of articulation of the phone in the middle. Choose one phone from the following: ",
    "What is the manner of articulation of the phone you hear in the middle of this 3-phone audio clip? Choose one from the following: ",
    "Identify the manner of articulation of the phone you hear in the middle of this 3-phone audio clip. Choose one from the following: ",
    "Label the manner of articulation of the phone you hear in the middle of this 3-phone audio clip. Choose one from the following: ",
    "Name the manner of articulation of the phone you hear in the middle of this 3-phone audio clip. Choose one from the following: ",
    "Determine the manner of articulation of the phone you hear in the middle of this 3-phone audio clip. Choose one from the following: ",
    "Please identify the manner of articulation of the phone you hear in the middle of this 3-phone audio clip. Choose one from the following: ",
    "Please label the manner of articulation of the phone you hear in the middle of this 3-phone audio clip. Choose one from the following: ",
    "Please name the manner of articulation of the phone you hear in the middle of this 3-phone audio clip. Choose one from the following: ",
    "Please determine the manner of articulation of the phone you hear in the middle of this 3-phone audio clip. Choose one from the following: ",
    "Based on this audio clip that consists of 3 phones, what is the manner of articulation of the phone in the middle? The phone is one of the following: ",
    "Based on this audio clip that consists of 3 phones, identify the manner of articulation of the phone in the middle. The phone is one of the following: ",
    "Based on this audio clip that consists of 3 phones, label the manner of articulation of the phone in the middle. The phone is one of the following: ",
    "Based on this audio clip that consists of 3 phones, name the manner of articulation of the phone in the middle. The phone is one of the following: ",
    "Based on this audio clip that consists of 3 phones, determine the manner of articulation of the phone in the middle. The phone is one of the following: ",
    "Based on this audio clip that consists of 3 phones, please identify the manner of articulation of the phone in the middle. The phone is one of the following: ",
    "Based on this audio clip that consists of 3 phones, please label the manner of articulation of the phone in the middle. The phone is one of the following: ",
    "Based on this audio clip that consists of 3 phones, please name the manner of articulation of the phone in the middle. The phone is one of the following: ",
    "Based on this audio clip that consists of 3 phones, please determine the manner of articulation of the phone in the middle. The phone is one of the following: "
]
place_classification_instructions = [inst + PLACE_SET_STR + "." for inst in place_classification_instructions]


HEIGHT_SET = ["close", "mid", "open"]  # IPA https://www.internationalphoneticassociation.org/sites/default/files/IPA_Kiel_2015.pdf
    # group "close-mid", "open-mid" into "mid"
HEIGHT_SET_STR = ", ".join(HEIGHT_SET[:-1]) + ", or " + HEIGHT_SET[-1]
FRONTNESS_SET = ["[+back]", "[-back]"]
    # there is no feature for "mid" (Zsiga, p. 269)
FRONTNESS_SET_STR = ", ".join(FRONTNESS_SET[:-1]) + ", or " + FRONTNESS_SET[-1]
ROUNDEDNESS_SET = ["rounded", "unrounded"]
ROUNDEDNESS_SET_STR = ", ".join(ROUNDEDNESS_SET[:-1]) + ", or " + ROUNDEDNESS_SET[-1]
height_classification_instructions = [
    "The audio clip consists of three phones. What is the height of the vowel in the middle? The answer could be: ",
    "The audio clip consists of three phones. Identify the height of the vowel in the middle. The answer could be: ",
    "The audio clip consists of three phones. Label the height of the vowel in the middle. The answer could be: ",
    "The audio clip consists of three phones. Name the height of the vowel in the middle. The answer could be: ",
    "The audio clip consists of three phones. Determine the height of the vowel in the middle. The answer could be: ",
    "The audio clip consists of three phones. Please identify the height of the vowel in the middle. The answer could be: ",
    "The audio clip consists of three phones. Please label the height of the vowel in the middle. The answer could be: ",
    "The audio clip consists of three phones. Please name the height of the vowel in the middle. The answer could be: ",
    "The audio clip consists of three phones. Please determine the height of the vowel in the middle. The answer could be: ",
    "In the given triphone utterance, what is the height of the vowel in the middle? Choose from the following: ",
    "In the given triphone utterance, identify the height of the vowel in the middle. Choose from the following: ",
    "In the given triphone utterance, label the height of the vowel in the middle. Choose from the following: ",
    "In the given triphone utterance, name the height of the vowel in the middle. Choose from the following: ",
    "In the given triphone utterance, determine the height of the vowel in the middle. Choose from the following: ",
    "In the given triphone utterance, please identify the height of the vowel in the middle. Choose from the following: ",
    "In the given triphone utterance, please label the height of the vowel in the middle. Choose from the following: ",
    "In the given triphone utterance, please name the height of the vowel in the middle. Choose from the following: ",
    "In the given triphone utterance, please determine the height of the vowel in the middle. Choose from the following: ",
    "What is the height of the vowel in the middle in this triphone audio file? Pick one phone from the following: ",
    "Identify the height of the vowel in the middle in this triphone audio file. Pick one phone from the following: ",
    "Label the height of the vowel in the middle in this triphone audio file. Pick one phone from the following: ",
    "Name the height of the vowel in the middle in this triphone audio file. Pick one phone from the following: ",
    "Determine the height of the vowel in the middle in this triphone audio file. Pick one phone from the following: ",
    "Please identify the height of the vowel in the middle in this triphone audio file. Pick one phone from the following: ",
    "Please label the height of the vowel in the middle in this triphone audio file. Pick one phone from the following: ",
    "Please name the height of the vowel in the middle in this triphone audio file. Pick one phone from the following: ",
    "Please determine the height of the vowel in the middle in this triphone audio file. Pick one phone from the following: ",
    "Listen to the audio clip, which consists of 3 phones. What is the height of the vowel in the middle? The phone could be ",
    "Listen to the audio clip, which consists of 3 phones, and identify the height of the vowel in the middle. The phone could be ",
    "Listen to the audio clip, which consists of 3 phones, and label the height of the vowel in the middle. The phone could be ",
    "Listen to the audio clip, which consists of 3 phones, and name the height of the vowel in the middle. The phone could be ",
    "Listen to the audio clip, which consists of 3 phones, and determine the height of the vowel in the middle. The phone could be ",
    "Listen to the audio clip, which consists of 3 phones, and please identify the height of the vowel in the middle. The phone could be ",
    "Listen to the audio clip, which consists of 3 phones, and please label the height of the vowel in the middle. The phone could be ",
    "Listen to the audio clip, which consists of 3 phones, and please name the height of the vowel in the middle. The phone could be ",
    "Listen to the audio clip, which consists of 3 phones, and please determine the height of the vowel in the middle. The phone could be ",
    "From the audio, which contains 3 phones, what is the height of the vowel in the middle? Choose one phone from the following: ",
    "From the audio, which contains 3 phones, identify the height of the vowel in the middle. Choose one phone from the following: ",
    "From the audio, which contains 3 phones, label the height of the vowel in the middle. Choose one phone from the following: ",
    "From the audio, which contains 3 phones, name the height of the vowel in the middle. Choose one phone from the following: ",
    "From the audio, which contains 3 phones, determine the height of the vowel in the middle. Choose one phone from the following: ",
    "From the audio, which contains 3 phones, please identify the height of the vowel in the middle. Choose one phone from the following: ",
    "From the audio, which contains 3 phones, please label the height of the vowel in the middle. Choose one phone from the following: ",
    "From the audio, which contains 3 phones, please name the height of the vowel in the middle. Choose one phone from the following: ",
    "From the audio, which contains 3 phones, please determine the height of the vowel in the middle. Choose one phone from the following: ",
    "What is the height of the vowel you hear in the middle of this 3-phone audio clip? Choose one from the following: ",
    "Identify the height of the vowel you hear in the middle of this 3-phone audio clip. Choose one from the following: ",
    "Label the height of the vowel you hear in the middle of this 3-phone audio clip. Choose one from the following: ",
    "Name the height of the vowel you hear in the middle of this 3-phone audio clip. Choose one from the following: ",
    "Determine the height of the vowel you hear in the middle of this 3-phone audio clip. Choose one from the following: ",
    "Please identify the height of the vowel you hear in the middle of this 3-phone audio clip. Choose one from the following: ",
    "Please label the height of the vowel you hear in the middle of this 3-phone audio clip. Choose one from the following: ",
    "Please name the height of the vowel you hear in the middle of this 3-phone audio clip. Choose one from the following: ",
    "Please determine the height of the vowel you hear in the middle of this 3-phone audio clip. Choose one from the following: ",
    "Based on this audio clip that consists of 3 phones, what is the height of the vowel in the middle? The phone is one of the following: ",
    "Based on this audio clip that consists of 3 phones, identify the height of the vowel in the middle. The phone is one of the following: ",
    "Based on this audio clip that consists of 3 phones, label the height of the vowel in the middle. The phone is one of the following: ",
    "Based on this audio clip that consists of 3 phones, name the height of the vowel in the middle. The phone is one of the following: ",
    "Based on this audio clip that consists of 3 phones, determine the height of the vowel in the middle. The phone is one of the following: ",
    "Based on this audio clip that consists of 3 phones, please identify the height of the vowel in the middle. The phone is one of the following: ",
    "Based on this audio clip that consists of 3 phones, please label the height of the vowel in the middle. The phone is one of the following: ",
    "Based on this audio clip that consists of 3 phones, please name the height of the vowel in the middle. The phone is one of the following: ",
    "Based on this audio clip that consists of 3 phones, please determine the height of the vowel in the middle. The phone is one of the following: "
]
height_classification_instructions = [inst + HEIGHT_SET_STR + "." for inst in height_classification_instructions]

frontness_classification_instructions = [
    "The audio clip consists of three phones. What is the frontness of the vowel in the middle? The answer could be: ",
    "The audio clip consists of three phones. Identify the frontness of the vowel in the middle. The answer could be: ",
    "The audio clip consists of three phones. Label the frontness of the vowel in the middle. The answer could be: ",
    "The audio clip consists of three phones. Name the frontness of the vowel in the middle. The answer could be: ",
    "The audio clip consists of three phones. Determine the frontness of the vowel in the middle. The answer could be: ",
    "The audio clip consists of three phones. Please identify the frontness of the vowel in the middle. The answer could be: ",
    "The audio clip consists of three phones. Please label the frontness of the vowel in the middle. The answer could be: ",
    "The audio clip consists of three phones. Please name the frontness of the vowel in the middle. The answer could be: ",
    "The audio clip consists of three phones. Please determine the frontness of the vowel in the middle. The answer could be: ",
    "In the given triphone utterance, what is the frontness of the vowel in the middle? Choose from the following: ",
    "In the given triphone utterance, identify the frontness of the vowel in the middle. Choose from the following: ",
    "In the given triphone utterance, label the frontness of the vowel in the middle. Choose from the following: ",
    "In the given triphone utterance, name the frontness of the vowel in the middle. Choose from the following: ",
    "In the given triphone utterance, determine the frontness of the vowel in the middle. Choose from the following: ",
    "In the given triphone utterance, please identify the frontness of the vowel in the middle. Choose from the following: ",
    "In the given triphone utterance, please label the height of the vowel in the middle. Choose from the following: ",
    "In the given triphone utterance, please name the frontness of the vowel in the middle. Choose from the following: ",
    "In the given triphone utterance, please determine the frontness of the vowel in the middle. Choose from the following: ",
    "What is the frontness of the vowel in the middle in this triphone audio file? Pick one phone from the following: ",
    "Identify the frontness of the vowel in the middle in this triphone audio file. Pick one phone from the following: ",
    "Label the frontness of the vowel in the middle in this triphone audio file. Pick one phone from the following: ",
    "Name the frontness of the vowel in the middle in this triphone audio file. Pick one phone from the following: ",
    "Determine the frontness of the vowel in the middle in this triphone audio file. Pick one phone from the following: ",
    "Please identify the frontness of the vowel in the middle in this triphone audio file. Pick one phone from the following: ",
    "Please label the frontness of the vowel in the middle in this triphone audio file. Pick one phone from the following: ",
    "Please name the frontness of the vowel in the middle in this triphone audio file. Pick one phone from the following: ",
    "Please determine the frontness of the vowel in the middle in this triphone audio file. Pick one phone from the following: ",
    "Listen to the audio clip, which consists of 3 phones. What is the frontness of the vowel in the middle? The phone could be ",
    "Listen to the audio clip, which consists of 3 phones, and identify the frontness of the vowel in the middle. The phone could be ",
    "Listen to the audio clip, which consists of 3 phones, and label the frontness of the vowel in the middle. The phone could be ",
    "Listen to the audio clip, which consists of 3 phones, and name the frontness of the vowel in the middle. The phone could be ",
    "Listen to the audio clip, which consists of 3 phones, and determine the frontness of the vowel in the middle. The phone could be ",
    "Listen to the audio clip, which consists of 3 phones, and please identify the frontness of the vowel in the middle. The phone could be ",
    "Listen to the audio clip, which consists of 3 phones, and please label the frontness of the vowel in the middle. The phone could be ",
    "Listen to the audio clip, which consists of 3 phones, and please name the frontness of the vowel in the middle. The phone could be ",
    "Listen to the audio clip, which consists of 3 phones, and please determine the frontness of the vowel in the middle. The phone could be ",
    "From the audio, which contains 3 phones, what is the frontness of the vowel in the middle? Choose one phone from the following: ",
    "From the audio, which contains 3 phones, identify the frontness of the vowel in the middle. Choose one phone from the following: ",
    "From the audio, which contains 3 phones, label the frontness of the vowel in the middle. Choose one phone from the following: ",
    "From the audio, which contains 3 phones, name the frontness of the vowel in the middle. Choose one phone from the following: ",
    "From the audio, which contains 3 phones, determine the frontness of the vowel in the middle. Choose one phone from the following: ",
    "From the audio, which contains 3 phones, please identify the frontness of the vowel in the middle. Choose one phone from the following: ",
    "From the audio, which contains 3 phones, please label the frontness of the vowel in the middle. Choose one phone from the following: ",
    "From the audio, which contains 3 phones, please name the frontness of the vowel in the middle. Choose one phone from the following: ",
    "From the audio, which contains 3 phones, please determine the frontness of the vowel in the middle. Choose one phone from the following: ",
    "What is the frontness of the vowel you hear in the middle of this 3-phone audio clip? Choose one from the following: ",
    "Identify the frontness of the vowel you hear in the middle of this 3-phone audio clip. Choose one from the following: ",
    "Label the frontness of the vowel you hear in the middle of this 3-phone audio clip. Choose one from the following: ",
    "Name the frontness of the vowel you hear in the middle of this 3-phone audio clip. Choose one from the following: ",
    "Determine the frontness of the vowel you hear in the middle of this 3-phone audio clip. Choose one from the following: ",
    "Please identify the frontness of the vowel you hear in the middle of this 3-phone audio clip. Choose one from the following: ",
    "Please label the frontness of the vowel you hear in the middle of this 3-phone audio clip. Choose one from the following: ",
    "Please name the frontness of the vowel you hear in the middle of this 3-phone audio clip. Choose one from the following: ",
    "Please determine the frontness of the vowel you hear in the middle of this 3-phone audio clip. Choose one from the following: ",
    "Based on this audio clip that consists of 3 phones, what is the frontness of the vowel in the middle? The phone is one of the following: ",
    "Based on this audio clip that consists of 3 phones, identify the frontness of the vowel in the middle. The phone is one of the following: ",
    "Based on this audio clip that consists of 3 phones, label the frontness of the vowel in the middle. The phone is one of the following: ",
    "Based on this audio clip that consists of 3 phones, name the frontness of the vowel in the middle. The phone is one of the following: ",
    "Based on this audio clip that consists of 3 phones, determine the frontness of the vowel in the middle. The phone is one of the following: ",
    "Based on this audio clip that consists of 3 phones, please identify the frontness of the vowel in the middle. The phone is one of the following: ",
    "Based on this audio clip that consists of 3 phones, please label the frontness of the vowel in the middle. The phone is one of the following: ",
    "Based on this audio clip that consists of 3 phones, please name the frontness of the vowel in the middle. The phone is one of the following: ",
    "Based on this audio clip that consists of 3 phones, please determine the frontness of the vowel in the middle. The phone is one of the following: "
]
frontness_classification_instructions = [inst + FRONTNESS_SET_STR + "." for inst in frontness_classification_instructions]


roundedness_classification_instructions = [
    "The audio clip consists of three phones. What is the roundedness of the vowel in the middle? The answer could be: ",
    "The audio clip consists of three phones. Identify the roundedness of the vowel in the middle. The answer could be: ",
    "The audio clip consists of three phones. Label the roundedness of the vowel in the middle. The answer could be: ",
    "The audio clip consists of three phones. Name the roundedness of the vowel in the middle. The answer could be: ",
    "The audio clip consists of three phones. Determine the roundedness of the vowel in the middle. The answer could be: ",
    "The audio clip consists of three phones. Please identify the roundedness of the vowel in the middle. The answer could be: ",
    "The audio clip consists of three phones. Please label the roundedness of the vowel in the middle. The answer could be: ",
    "The audio clip consists of three phones. Please name the roundedness of the vowel in the middle. The answer could be: ",
    "The audio clip consists of three phones. Please determine the roundedness of the vowel in the middle. The answer could be: ",
    "In the given triphone utterance, what is the roundedness of the vowel in the middle? Choose from the following: ",
    "In the given triphone utterance, identify the roundedness of the vowel in the middle. Choose from the following: ",
    "In the given triphone utterance, label the roundedness of the vowel in the middle. Choose from the following: ",
    "In the given triphone utterance, name the roundedness of the vowel in the middle. Choose from the following: ",
    "In the given triphone utterance, determine the roundedness of the vowel in the middle. Choose from the following: ",
    "In the given triphone utterance, please identify the roundedness of the vowel in the middle. Choose from the following: ",
    "In the given triphone utterance, please label the roundedness of the vowel in the middle. Choose from the following: ",
    "In the given triphone utterance, please name the roundedness of the vowel in the middle. Choose from the following: ",
    "In the given triphone utterance, please determine the roundedness of the vowel in the middle. Choose from the following: ",
    "What is the roundedness of the vowel in the middle in this triphone audio file? Pick one phone from the following: ",
    "Identify the roundedness of the vowel in the middle in this triphone audio file. Pick one phone from the following: ",
    "Label the roundedness of the vowel in the middle in this triphone audio file. Pick one phone from the following: ",
    "Name the roundedness of the vowel in the middle in this triphone audio file. Pick one phone from the following: ",
    "Determine the roundedness of the vowel in the middle in this triphone audio file. Pick one phone from the following: ",
    "Please identify the roundedness of the vowel in the middle in this triphone audio file. Pick one phone from the following: ",
    "Please label the roundedness of the vowel in the middle in this triphone audio file. Pick one phone from the following: ",
    "Please name the roundedness of the vowel in the middle in this triphone audio file. Pick one phone from the following: ",
    "Please determine the roundedness of the vowel in the middle in this triphone audio file. Pick one phone from the following: ",
    "Listen to the audio clip, which consists of 3 phones. What is the roundedness of the vowel in the middle? The phone could be ",
    "Listen to the audio clip, which consists of 3 phones, and identify the roundedness of the vowel in the middle. The phone could be ",
    "Listen to the audio clip, which consists of 3 phones, and label the roundedness of the vowel in the middle. The phone could be ",
    "Listen to the audio clip, which consists of 3 phones, and name the roundedness of the vowel in the middle. The phone could be ",
    "Listen to the audio clip, which consists of 3 phones, and determine the roundedness of the vowel in the middle. The phone could be ",
    "Listen to the audio clip, which consists of 3 phones, and please identify the roundedness of the vowel in the middle. The phone could be ",
    "Listen to the audio clip, which consists of 3 phones, and please label the roundedness of the vowel in the middle. The phone could be ",
    "Listen to the audio clip, which consists of 3 phones, and please name the roundedness of the vowel in the middle. The phone could be ",
    "Listen to the audio clip, which consists of 3 phones, and please determine the roundedness of the vowel in the middle. The phone could be ",
    "From the audio, which contains 3 phones, what is the roundedness of the vowel in the middle? Choose one phone from the following: ",
    "From the audio, which contains 3 phones, identify the roundedness of the vowel in the middle. Choose one phone from the following: ",
    "From the audio, which contains 3 phones, label the roundedness of the vowel in the middle. Choose one phone from the following: ",
    "From the audio, which contains 3 phones, name the roundedness of the vowel in the middle. Choose one phone from the following: ",
    "From the audio, which contains 3 phones, determine the roundedness of the vowel in the middle. Choose one phone from the following: ",
    "From the audio, which contains 3 phones, please identify the roundedness of the vowel in the middle. Choose one phone from the following: ",
    "From the audio, which contains 3 phones, please label the roundedness of the vowel in the middle. Choose one phone from the following: ",
    "From the audio, which contains 3 phones, please name the roundedness of the vowel in the middle. Choose one phone from the following: ",
    "From the audio, which contains 3 phones, please determine the roundedness of the vowel in the middle. Choose one phone from the following: ",
    "What is the roundedness of the vowel you hear in the middle of this 3-phone audio clip? Choose one from the following: ",
    "Identify the roundedness of the vowel you hear in the middle of this 3-phone audio clip. Choose one from the following: ",
    "Label the roundedness of the vowel you hear in the middle of this 3-phone audio clip. Choose one from the following: ",
    "Name the roundedness of the vowel you hear in the middle of this 3-phone audio clip. Choose one from the following: ",
    "Determine the roundedness of the vowel you hear in the middle of this 3-phone audio clip. Choose one from the following: ",
    "Please identify the roundedness of the vowel you hear in the middle of this 3-phone audio clip. Choose one from the following: ",
    "Please label the roundedness of the vowel you hear in the middle of this 3-phone audio clip. Choose one from the following: ",
    "Please name the roundedness of the vowel you hear in the middle of this 3-phone audio clip. Choose one from the following: ",
    "Please determine the roundedness of the vowel you hear in the middle of this 3-phone audio clip. Choose one from the following: ",
    "Based on this audio clip that consists of 3 phones, what is the roundedness of the vowel in the middle? The phone is one of the following: ",
    "Based on this audio clip that consists of 3 phones, identify the roundedness of the vowel in the middle. The phone is one of the following: ",
    "Based on this audio clip that consists of 3 phones, label the roundedness of the vowel in the middle. The phone is one of the following: ",
    "Based on this audio clip that consists of 3 phones, name the roundedness of the vowel in the middle. The phone is one of the following: ",
    "Based on this audio clip that consists of 3 phones, determine the roundedness of the vowel in the middle. The phone is one of the following: ",
    "Based on this audio clip that consists of 3 phones, please identify the roundedness of the vowel in the middle. The phone is one of the following: ",
    "Based on this audio clip that consists of 3 phones, please label the roundedness of the vowel in the middle. The phone is one of the following: ",
    "Based on this audio clip that consists of 3 phones, please name the roundedness of the vowel in the middle. The phone is one of the following: ",
    "Based on this audio clip that consists of 3 phones, please determine the roundedness of the vowel in the middle. The phone is one of the following: "
]
roundedness_classification_instructions = [inst + ROUNDEDNESS_SET_STR + "." for inst in roundedness_classification_instructions]


def manner_of_articulation(phone, ft):
    # using Table 12.2 from Zsiga
    # to verify the manner of articulation, ```wget https://raw.githubusercontent.com/dmort27/panphon/master/panphon/data/ipa_bases.csv'''
    # then df = pd.read_csv('ipa_bases.csv')

    # df[(df['son'] == '-') & (df['cont'] == '-') & (df['delrel'] == '-')]
    PLOSIVE = {
        'son': -1,
        'cont': -1,
        'delrel': -1
    }
    # df[(df['son'] == '-') & (df['cont'] == '+')
    FRICATIVE = {
        'son': -1,
        'cont': 1
    }
    # df[(df['son'] == '-') & (df['cont'] == '-') & (df['delrel'] == '+')]
    AFFRICATE = {
        'son': -1,
        'cont': -1,
        'delrel': 1
    }
    # df[(df['son'] == '+') & (df['cons'] == '+') & (df['nas'] == '+')]
    NASAL = {
        'son': 1,
        'cons': 1,
        'nas': 1
    }
    # df[(df['son'] == '+') & (df['cons'] == '+') & (df['cont'] == '+') & (df['lat'] == '-')]
    #   [h] is an approximant under our classification
    APPROXIMANT = {
        'son': 1,
        'cons': 1,
        'cont': 1,
        'lat': -1, # exclude lateral approximants, which we classify as lateral
    }
    # df[(df['son'] == '+') & (df['cons'] == '+') & (df['cont'] == '+') & (df['lat'] == '+')]
    LATERAL = {
        'son': 1,
        'cons': 1,
        'cont': 1,
        'lat': 1
    }
    # df[(df['cons'] == '-') & (df['syl'] == '-')]; exclude [ʔ] (stop) and [ɻ] (approximant)
    GLIDE = {
        'cons': -1,
        'syl': -1
    }
    # df[(df['cons'] == '-') & (df['syl'] == '+')]
    VOWEL = {
        'cons': -1, # exclude syllabic consonants
        'syl': +1
    }

    if not ft.fts(phone):
        return ""

    if "ʔ" in phone:
        return "plosive"
    elif "ɻ" in phone:
        return "approximant"
    # https://en.wikipedia.org/wiki/Tap_and_flap_consonants#IPA_symbols
    elif any(tap_flap in phone for tap_flap in { "ɾ", "ɺ", "ɽ", "𝼈", "ⱱ" }):
        return "tap/flap"
    # https://en.wikipedia.org/wiki/Trill_consonant
    elif any(trill in phone for trill in { "r", "ʙ", "ʀ", "ʢ", "ʜ" }):
        return "trill"
    # https://en.wikipedia.org/wiki/Click_consonant
    elif any(click in phone for click in { "ʘ", "ǀ", "ǁ", "ǂ", "ǃ", "𝼊" }):
        # could also use [+velaric]
        return "click"
    # https://en.wikipedia.org/wiki/Ejective_consonant
    elif "ʼ" in phone:
        return "ejective"
    # https://en.wikipedia.org/wiki/Implosive_consonant
    elif any(implosive in phone for implosive in { "ɓ", "ɗ", "ᶑ", "ʄ", "ɠ", "ʛ" }):
        return "implosive"

    for features, manner in [(PLOSIVE, "plosive"), (FRICATIVE, "fricative"), (AFFRICATE, "affricate"), (NASAL, "nasal"), \
            (APPROXIMANT, "approximant"), (LATERAL, "lateral"), (GLIDE, "glide"), (VOWEL, "vowel")]:
        if ft.fts(phone).match(features):
            return manner

    print("could not determine manner of articulation for ", phone)
    return ""

def place_of_articulation(consonant, ft, dist):
    if consonant in UNICODE_TO_IPA:
        return UNICODE_TO_IPA[consonant].place

    # find the closest phone in UNICODE_TO_IPA then get its place
    min_fed = 1.0
    closest_phone = ''
    for phone in UNICODE_TO_IPA.keys():
        # not recognized by panphon
        if not ft.ipa_segs(phone):
            continue

        fed = dist.feature_edit_distance(consonant, phone)
        if fed < min_fed:
            min_fed = fed
            closest_phone = phone
    if isinstance(UNICODE_TO_IPA[closest_phone], IPAConsonant):
        print('mapped', consonant, 'to', closest_phone)
        return UNICODE_TO_IPA[closest_phone].place
    return ""

def vowel_frontness(vowel, ft):
    BACK = {
        'back': 1,
    }
    if not ft.fts(vowel):
        print("panphon lacks features for", vowel)
        return ""
    elif ft.fts(vowel).match(BACK):
        # back and central
        return "[+back]"
    else:
        # front
        return "[-back]"

def vowel_height(vowel, ft):
    HIGH = {
        'hi': 1,
        'lo': -1
    }
    MID = {
        'hi': -1,
        'lo': -1
    }
    LOW = {
        'hi': -1,
        'lo': 1
    }
    if not ft.fts(vowel):
        print("panphon lacks features for", vowel)
        return ""
    elif ft.fts(vowel).match(HIGH):
        return "close"
    elif ft.fts(vowel).match(MID):
        return "mid"
    else: # low
        return "open"

def vowel_roundedness(vowel, ft):
    ROUNDED = {
        'round': 1,
    }
    if not ft.fts(vowel):
        print("panphon lacks features for", vowel)
        return ""
    elif ft.fts(vowel).match(ROUNDED):
        return "rounded"
    else:
        return "unrounded"


if __name__ == "__main__":
    ds = load_dataset(
        "kalbin/VoxAngeles_phones",
        cache_dir="datasets_cache",
        revision="refs/convert/parquet",
    )
    ds = ds["test"]
    df = pd.DataFrame(ds)

    random.seed(15213)

    son, ft, dist = panphon.sonority.Sonority(), FeatureTable(), Distance()

    # pick subset of each language's words (1000 / 95)
    WORD_LIMIT = 1000  # approx 1 hour
    NUM_LANGS = len(df["lang"].unique())
    NUM_WORDS = WORD_LIMIT // NUM_LANGS
    def subset_words(lang_df):
        words = lang_df["word"].unique()
        selected_words = set(random.sample(set(words), k=NUM_WORDS + 5))  # without replacement
        return lang_df[lang_df["word"].isin(selected_words)]
    df = df.groupby(['lang']) \
            .apply(subset_words) \
            .reset_index(drop=True)

    # select triphone environment from the phonetic transcription of the word

    # ensure there are at least 3 phones in the word
    df = df.groupby(['lang', 'word']) \
            .filter(lambda group: len(group) >= 3) \
            .reset_index(drop=True)
    # skip duplicate words in the language
    #   if a language has 2+ instances of the same word
    #   then the number of entries in the groupby will be a multiple of the # phones in the word
    def no_duplicate(word_df):
        (_, word) = word_df.name
        return len(ft.ipa_segs(word)) == len(word_df)
    df = df.groupby(['lang', 'word']) \
            .filter(no_duplicate) \
            .reset_index(drop=True)
    def get_triphone(word_df):
        # for each word in the lang, pick 3 consecutive phone entries
        start_pos = random.randint(0, len(word_df) - 3)
        return word_df.iloc[start_pos:start_pos + 3]
    df = df.groupby(['lang', 'word']) \
            .apply(get_triphone) \
            .reset_index(drop=True)

    # exclude diphthongs for now (the formant transitions may reveal anyway)
    VOWEL_SONORITY = 8
    def is_vowel(phone):
        # ft.ipa_segs to convert to normalized decomposed form
        return son.sonority(ft.ipa_segs(phone)[-1]) >= VOWEL_SONORITY
    def no_diphthong(triphone_df):
        assert len(triphone_df) == 3
        # ensure index starts at 0
        triphone_df = triphone_df.reset_index(drop=True)
        x, mid, y = triphone_df.loc[0, 'phone'], triphone_df.loc[1, 'phone'], triphone_df.loc[2, 'phone']
        try:
            return not ((is_vowel(x) and is_vowel(mid)) or (is_vowel(mid) and is_vowel(y)))
        except:
            print(x, mid, y)
            # panphon cannot handle one of the phones, discard for now
            # j a ʡ
            # m æ ʜ
            # ʡ y r
            # ʡ æ ɡʷ
            # ʡ æ ʃ
            # h ӕː t
            # i g i
            return False
    df = df.groupby(['lang', 'word']) \
            .filter(no_diphthong) \
            .reset_index(drop=True)

    # extract timestamps for the triphone environment
        # reduce each group into one
    def extract_timestamps(word_df):
        assert len(word_df) == 3
        word_df = word_df.reset_index(drop=True)

        word_df['start_t'] = word_df.loc[0, 'start']
        word_df['finish_t'] = word_df.loc[2, 'finish']
        word_df['phones'] = word_df.loc[0, 'phone'] + " " + word_df.loc[1, 'phone'] + " " + word_df.loc[2, 'phone']
        return word_df.iloc[0]
    df = df.groupby(['lang', 'word']) \
        .apply(extract_timestamps) \
        .reset_index(drop=True)
    df = df.drop(['start', 'finish', 'phone'], axis=1)

    # Filter out samples longer than 2 seconds
    df = df[(df['finish_t'] - df['start_t']) <= 2]

    # load the audio
        # extract the timestamp from the audio
        # save as new file
    VOXANGELES_PATH = 'data/voxangeles/data/audited_aligned'
    if not os.path.exists(VOXANGELES_PATH):
        raise Exception("Please download VoxAngeles at data/")
    def segment_audio(row):
        full_word_path = f"{VOXANGELES_PATH}/{row['lang']}/{row['file']}.wav"
        # 44100 Hz
        audio, sampling_rate = sf.read(full_word_path)
        start, end = int(row['start_t'] * sampling_rate), int(row['finish_t'] * sampling_rate)
        segmented_path = f"{VOXANGELES_PATH}/{row['lang']}/{row['file']}_{start}_{end}.wav"
        # convert sampling rate later
        assert len(audio[start:end]) > 0, (full_word_path, start, end)
        sf.write(segmented_path, audio[start:end], samplerate=sampling_rate)
        return segmented_path
    df['audio'] = df.apply(segment_audio, axis=1)

    # prepare the answer

    # phone classification - the phone
    phone_df = df.copy()
    # manner of articulation - both vowels, consonants
    manner_df = df.copy()
    # place of articulation - only consonants
    vowels = df.apply(lambda sample: is_vowel(sample['phones'].split(' ')[1]), axis=1)
    place_df = df[~vowels].copy()
    # vowel height, frontness, roundedness - only vowels
    vowel_height_df, vowel_frontness_df, vowel_roundedness_df = df[vowels].copy(), df[vowels].copy(), df[vowels].copy()

    phone_df['label'] = phone_df.apply(lambda row: row['phones'].split(' ')[1], axis=1)

    manner_df['label'] = manner_df.apply(lambda row: manner_of_articulation(row['phones'].split(' ')[1], ft), axis=1)
    manner_df = manner_df[manner_df['label'].str.len() > 0]

    place_df['label'] = place_df.apply(lambda row: place_of_articulation(row['phones'].split(' ')[1], ft, dist), axis=1)
    place_df = place_df[place_df['label'].str.len() > 0]

    vowel_height_df['label'] = vowel_height_df.apply(lambda row: vowel_height(row['phones'].split(' ')[1], ft), axis=1)
    vowel_height_df = vowel_height_df[vowel_height_df['label'].str.len() > 0]

    vowel_frontness_df['label'] = vowel_frontness_df.apply(lambda row: vowel_frontness(row['phones'].split(' ')[1], ft), axis=1)
    vowel_frontness_df = vowel_frontness_df[vowel_frontness_df['label'].str.len() > 0]

    vowel_roundedness_df['label'] = vowel_roundedness_df.apply(lambda row: vowel_roundedness(row['phones'].split(' ')[1], ft), axis=1)
    vowel_roundedness_df = vowel_roundedness_df[vowel_roundedness_df['label'].str.len() > 0]

    for task_name, instructions, dataframe in [("PhoneClassification", phone_classification_instructions, phone_df), \
        ("MannerOfArticulationClassification", manner_classification_instructions, manner_df), \
        ("ConsonantPlaceOfArticulationClassification", place_classification_instructions, place_df), \
        ("VowelFrontnessClassification", frontness_classification_instructions, vowel_frontness_df), \
        ("VowelHeightClassification", height_classification_instructions, vowel_height_df), \
        ("VowelRoundednessClassification", roundedness_classification_instructions, vowel_roundedness_df)]:
        ds = Dataset.from_pandas(dataframe)

        # Reformatting
        def _map(sample, index):
            return {
                "audio": sample["audio"],
                "file": sample["audio"],
                "instruction": instructions[index % len(instructions)],
                "label": sample["label"],
            }
        ds = ds.map(_map, with_indices=True, remove_columns=ds.column_names)
        ds = ds.cast_column("audio", Audio(sampling_rate=16_000))

        # Validate & Push
        validate_dataset(ds)
        ds.push_to_hub(repo_id=f"DynamicSuperb/{task_name}_VoxAngeles", split="test", token=os.environ["HF_TOKEN"])
