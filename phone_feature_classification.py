import random

import pandas as pd
from datasets import load_dataset
from panphon import FeatureTable
import panphon.sonority

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
MANNER_SET = ["plosive", "nasal", "trill", "tap", "fricative", "affricate", "approximant", "lateral", "glide", "vowel"] # IPA https://www.internationalphoneticassociation.org/sites/default/files/IPA_Kiel_2015.pdf
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
# TODO: separate instructions for vowels (frontness and backness)
PLACE_SET = ["bilabial", "labiodental", "dental", "alveolar", "postalveolar", "velar", "uvular", "pharyngeal", "glottal" ]  # IPA https://www.internationalphoneticassociation.org/sites/default/files/IPA_Kiel_2015.pdf
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


if __name__ == "__main__":
    ds = load_dataset(
        "kalbin/VoxAngeles_phones",
        cache_dir="datasets_cache",
        revision="refs/convert/parquet",
    )
    ds = ds["test"]
    df = pd.DataFrame(ds)

    random.seed(15213)

    # pick subset of each language's words (1000 / 95)
    WORD_LIMIT = 1000  # approx 1 hour
    NUM_LANGS = len(df["lang"].unique())
    NUM_WORDS = WORD_LIMIT // NUM_LANGS
    def subset_words(lang_df):
        words = lang_df["word"].unique()
        selected_words = set(random.choices(words, k=NUM_WORDS))
        return lang_df[lang_df["word"].isin(selected_words)]
    df = df.groupby(['lang']) \
            .apply(subset_words) \
            .reset_index(drop=True)

    # select triphone environment from the phonetic transcription of the word

    # ensure there are at least 3 phones in the word
    df = df.groupby(['lang', 'word']) \
            .filter(lambda group: len(group) >= 3) \
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
    son = panphon.sonority.Sonority()
    ft = FeatureTable()
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
