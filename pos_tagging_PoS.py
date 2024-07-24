import os
import numpy as np
from datasets import load_dataset, Audio, Dataset
import stanza
from string import punctuation
from utils import validate_dataset
from collections import defaultdict
import pandas as pd

# Initialize the stanza pipeline to use GPU
stanza.download('en')
nlp = stanza.Pipeline('en', use_gpu=True)

TAG_DESCRIPTION = "The tags include ADJ, ADP, ADV, AUX, CCONJ, DET, INTJ, NOUN, NUM, PART, PRON, PROPN, SCONJ, and VERB."

# First part of the new set of instructions
FIRST_POS = [
    "For each word in the given audio utterance, determine and assign the corresponding Part-of-Speech (POS) tag.",
    "For each word in the given audio clip, identify and assign the appropriate Part-of-Speech (POS) tag.",
    "Listen to the audio and determine the Part-of-Speech (POS) tag for each word in the utterance.",
    "For each word in the audio, identify its Part-of-Speech (POS) tag.",
    "After listening to the audio, assign the correct Part-of-Speech (POS) tag to each word.",
    "Determine the Part-of-Speech (POS) tag for each word in the audio utterance.",
    "Identify and assign the Part-of-Speech (POS) tag for each word in the given audio.",
    "Listen to the audio and determine the Part-of-Speech (POS) tag for each word.",
    "For each word in the audio clip, identify its Part-of-Speech (POS) tag.",
    "Determine and assign the Part-of-Speech (POS) tag for each word in the given audio."
]

FIRST_TRANS = [
    "For each word in the given audio utterance, transcribe the audio and then determine and assign the corresponding Part-of-Speech (POS) tag.",
    "For each word in the given audio clip, transcribe the audio and then identify and assign the appropriate Part-of-Speech (POS) tag.",
    "Listen to the audio, transcribe it, and determine the Part-of-Speech (POS) tag for each word in the utterance.",
    "For each word in the audio, transcribe it and identify its Part-of-Speech (POS) tag.",
    "After listening to the audio and transcribing it, assign the correct Part-of-Speech (POS) tag to each word.",
    "Transcribe the audio utterance and determine the Part-of-Speech (POS) tag for each word.",
    "Identify and assign the Part-of-Speech (POS) tag for each word in the given transcribed audio.",
    "Listen to the audio, transcribe it, and determine the Part-of-Speech (POS) tag for each word.",
    "For each word in the audio clip, transcribe it and identify its Part-of-Speech (POS) tag.",
    "Transcribe the given audio and determine and assign the Part-of-Speech (POS) tag for each word."
]

# Second part of the new set of instructions with examples
SECOND_POS = [
    "As an example, for the audio 'We watched a movie last night, and it was very entertaining', generate PRON VERB DET NOUN ADJ NOUN CCONJ PRON AUX ADV ADJ.",
    "For example, when you listen to an audio 'The quick brown fox jumps over the lazy dog', please generate DET ADJ ADJ NOUN VERB ADP DET ADJ NOUN.",
    "For example, for the audio 'It was a sunny day and the children were playing outside', generate PRON AUX DET ADJ NOUN CCONJ DET NOUN AUX VERB ADV.",
    "For instance, for 'I enjoy reading books in the quiet park every afternoon', produce PRON VERB VERB NOUN ADP DET ADJ NOUN ADV NOUN.",
    "As an example, for 'He is running fast because he is late for the meeting', produce PRON AUX VERB ADV SCONJ PRON AUX ADJ ADP DET NOUN.",
    "For example, for 'They will travel tomorrow to visit their grandparents in the city', generate PRON AUX VERB ADV SCONJ VERB PRON NOUN ADP DET NOUN.",
    "For example, when listening to 'Cats are great pets, and they make wonderful companions', generate NOUN AUX ADJ NOUN CCONJ PRON VERB ADJ NOUN.",
    "For example, when you hear 'Birds are flying high in the sky, enjoying the warm weather', generate NOUN AUX VERB ADV ADP DET NOUN VERB DET ADJ NOUN.",
    "For instance, for the audio 'She sells sea shells by the seashore on a sunny day', generate PRON VERB NOUN NOUN ADP DET NOUN ADP DET ADJ NOUN.",
    "For instance, for 'My friend is coming over to help me with my homework', produce DET NOUN AUX VERB ADV ADP VERB PRON ADP DET NOUN."
]

SECOND_TRANS = [
    "Write the transcribed utterance first, and then the POS tags followed by a '/'. For example, for 'The quick brown fox jumps over the lazy dog', generate the tags in the form of: The quick brown fox jumps over the lazy dog / DET ADJ ADJ NOUN VERB ADP DET ADJ NOUN.",
    "Write the transcription of the utterance first, followed by the POS tags separated by a '/'. For example, for 'She decided to take a walk in the beautiful park despite the rain', format the response as: She decided to take a walk in the beautiful park despite the rain / PRON VERB PART VERB DET NOUN ADP DET ADJ NOUN SCONJ DET NOUN.",
    "Write the transcription first, then the POS tags separated by a '/'. For example, for 'He was excited to receive the prestigious award at the annual ceremony', write: He was excited to receive the prestigious award at the annual ceremony / PRON AUX ADJ PART VERB DET ADJ NOUN ADP DET ADJ NOUN.",
    "Transcribe the utterance first, followed by the POS tags separated by a '/'. For instance, 'Although it was raining, they decided to go for a hike in the mountains' should be formatted as: Although it was raining, they decided to go for a hike in the mountains / SCONJ PRON AUX VERB PRON VERB PART VERB ADP DET NOUN ADP DET NOUN.",
    "Write the transcription first, then the POS tags separated by a '/'. For example, for 'The children enjoyed playing outside even though it was cold and windy', write: The children enjoyed playing outside even though it was cold and windy / DET NOUN VERB VERB ADV SCONJ PRON AUX ADJ CCONJ ADJ.",
    "Transcribe the utterance first, followed by the POS tags separated by a '/'. For example, for 'She was delighted to see her friends after such a long time', write: She was delighted to see her friends after such a long time / PRON AUX ADJ PART VERB PRON NOUN ADP DET ADJ NOUN.",
    "Write the transcription first, then the POS tags separated by a '/'. For instance, for 'They didn't realize how much they had missed the city until they returned', write: They didn't realize how much they had missed the city until they returned / PRON AUX PART VERB ADV ADV PRON AUX VERB DET NOUN SCONJ PRON VERB.",
    "Write the transcription first, then the POS tags separated by a '/'. For example, for 'Despite the long journey, they were happy to reach their destination', write: Despite the long journey, they were happy to reach their destination / ADP DET ADJ NOUN PRON AUX ADJ PART VERB PRON NOUN.",
    "Write the transcription first, followed by the POS tags separated by a '/'. For instance, 'The conference was postponed due to unforeseen circumstances', should be formatted as: The conference was postponed due to unforeseen circumstances / DET NOUN AUX VERB ADP ADJ NOUN.",
    "Write the transcription first, then the POS tags separated by a '/'. For example, for 'The new policy will affect all employees starting next month', write: The new policy will affect all employees starting next month / DET ADJ NOUN AUX VERB DET NOUN VERB ADP NOUN."
]

# Using list comprehension to generate the combined list
INSTRUCTIONS_POS = [f + " " + TAG_DESCRIPTION + " " + s for f in FIRST_POS for s in SECOND_POS]
INSTRUCTIONS_TRANS = [f + " " + TAG_DESCRIPTION + " " + s for f in FIRST_TRANS for s in SECOND_TRANS]

def filter_text(text):
    unwanted_chars = "\";-:"
    if any(char in text for char in unwanted_chars):
        return None
    return text.strip()

def sent_and_tag(text):
    doc = nlp(text)
    sent_and_tags = [
        (
            " ".join([word.text for word in sentence.words if word.text not in punctuation]),
            " ".join([word.upos for word in sentence.words if word.text not in punctuation]),
            [word.upos for word in sentence.words if word.text not in punctuation]
        )
        for sentence in doc.sentences
    ]
    return sent_and_tags

def contains_sym_x(pos_tags):
    return any(tag in ['SYM', 'X'] for tag in pos_tags)

if __name__ == "__main__":
    ds = load_dataset(
        "mythicinfinity/libritts", "clean", split="test.clean",
        cache_dir="/data/user_data/eyeo2",
    )

    # Filter sentences by text length and group by text length
    samples_by_length = defaultdict(list)
    for sample in ds:
        text_length = len(sample['text_normalized'].split())
        if 3 <= text_length <= 15:
            samples_by_length[text_length].append(sample)

    # Determine the minimum number of samples for any text length
    min_samples = min(len(samples) for samples in samples_by_length.values())
    print(f"Minimum number of samples for any text length: {min_samples}")

    # Select the same number of samples for each text length
    balanced_samples = []
    for length, samples in samples_by_length.items():
        balanced_samples.extend(samples[:min_samples])
        print(f"Selected {min_samples} samples for length {length}")

    # Convert list to Dataset
    new_ds = Dataset.from_pandas(pd.DataFrame(balanced_samples))

    def _map(sample, index, instructions, with_transcription=False):
        text = filter_text(sample['text_normalized'])
        if text is None:
            return {
                "audio": sample["audio"],
                "file": os.path.basename(sample["audio"]["path"]),
                "instruction": instructions[index % len(instructions)],
                "label": ""
            }

        sent_and_tags = sent_and_tag(text)
        if any(contains_sym_x(tags) for _, _, tags in sent_and_tags):
            return {
                "audio": sample["audio"],
                "file": os.path.basename(sample["audio"]["path"]),
                "instruction": instructions[index % len(instructions)],
                "label": ""
            }

        filename = os.path.basename(sample["audio"]["path"])  # Extract filename from audio path
        instruction = instructions[index % len(instructions)]
        sentence, tag, _ = sent_and_tags[0]
        label = f"{sentence} / {tag}" if with_transcription else tag

        return {
            "audio": sample["audio"],
            "file": filename,
            "instruction": instruction,
            "label": label
        }

    new_ds_pos = new_ds.map(lambda sample, index: _map(sample, index, INSTRUCTIONS_POS, with_transcription=False), with_indices=True, remove_columns=new_ds.column_names)
    new_ds_pos = new_ds_pos.filter(lambda x: "label" in x and x["label"] != "")
    new_ds_pos = new_ds_pos.cast_column("audio", Audio(sampling_rate=16_000))

    # Validate & Push
    validate_dataset(new_ds_pos)
    new_ds_pos.push_to_hub(repo_id="DynamicSuperb/PoS_Estimation_LibriTTS_PoS", split="test", token=os.environ["HF_TOKEN"])

    new_ds_trans = new_ds.map(lambda sample, index: _map(sample, index, INSTRUCTIONS_TRANS, with_transcription=True), with_indices=True, remove_columns=new_ds.column_names)
    new_ds_trans = new_ds_trans.filter(lambda x: "label" in x and x["label"] != "")
    new_ds_trans = new_ds_trans.cast_column("audio", Audio(sampling_rate=16_000))

    # Validate & Push
    validate_dataset(new_ds_trans)
    new_ds_trans.push_to_hub(repo_id="DynamicSuperb/PoS_Estimation_LibriTTS_PoS_with_transcription", split="test", token=os.environ["HF_TOKEN"])
