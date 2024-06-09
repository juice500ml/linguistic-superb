import os
import numpy as np
from datasets import load_dataset, Audio
import stanza
from string import punctuation
from utils import validate_dataset

# Initialize the stanza pipeline to use CPU
stanza.download('en')
nlp = stanza.Pipeline('en', use_gpu=False)

tag_description = "The tags include ADJ, ADP, ADV, AUX, CCONJ, DET, INTJ, NOUN, NUM, PART, PRON, PROPN, SCONJ, and VERB."
instructions = [
    "For each word in the given audio utterance, determine and assign the corresponding Part-of-Speech (POS) tag. " + tag_description + " Write the transcribed utterance first, and then the POS tags followed by a '/'. For example, for 'The quick brown fox jumps over the lazy dog', generate the tags in the form of (transcription): The quick brown fox jumps over the lazy dog / (POS): DET ADJ ADJ NOUN VERB ADP DET ADJ NOUN.",
    "For each word in the given audio clip, identify and assign the appropriate Part-of-Speech (POS) tag. " + tag_description + " Write the transcription of the utterance first, followed by the POS tags separated by a '/'. For example, for 'She decided to take a walk in the beautiful park despite the rain', format the response as (transcription): She decided to take a walk in the beautiful park despite the rain / (POS): PRON VERB PART VERB DET NOUN ADP DET ADJ NOUN SCONJ DET NOUN.",
    "Listen to the audio and determine the Part-of-Speech (POS) tag for each word in the utterance. " + tag_description + " Write the transcription first, then the POS tags separated by a '/'. For example, for 'He was excited to receive the prestigious award at the annual ceremony', write (transcription): He was excited to receive the prestigious award at the annual ceremony / (POS): PRON AUX ADJ PART VERB DET ADJ NOUN ADP DET ADJ NOUN.",
    "For each word in the audio, identify its Part-of-Speech (POS) tag. " + tag_description + " Transcribe the utterance first, followed by the POS tags separated by a '/'. For instance, 'Although it was raining, they decided to go for a hike in the mountains' should be formatted as (transcription): Although it was raining, they decided to go for a hike in the mountains / (POS): SCONJ PRON AUX VERB PRON VERB PART VERB ADP DET NOUN ADP DET NOUN.",
    "After listening to the audio, assign the correct Part-of-Speech (POS) tag to each word. " + tag_description + " Write the transcription first, then the POS tags separated by a '/'. For example, for 'The children enjoyed playing outside even though it was cold and windy', write (transcription): The children enjoyed playing outside even though it was cold and windy / (POS): DET NOUN VERB VERB ADV SCONJ PRON AUX ADJ CCONJ ADJ.",
    "Determine the Part-of-Speech (POS) tag for each word in the audio utterance. " + tag_description + " Transcribe the utterance first, followed by the POS tags separated by a '/'. For example, for 'She was delighted to see her friends after such a long time', write (transcription): She was delighted to see her friends after such a long time / (POS): PRON AUX ADJ PART VERB PRON NOUN ADP DET ADJ NOUN.",
    "Identify and assign the Part-of-Speech (POS) tag for each word in the given audio. " + tag_description + " Write the transcription first, then the POS tags separated by a '/'. For instance, for 'They didn't realize how much they had missed the city until they returned', write (transcription): They didn't realize how much they had missed the city until they returned / (POS): PRON AUX PART VERB ADV ADV PRON AUX VERB DET NOUN SCONJ PRON VERB.",
    "Listen to the audio and determine the Part-of-Speech (POS) tag for each word. " + tag_description + " Write the transcription first, then the POS tags separated by a '/'. For example, for 'Despite the long journey, they were happy to reach their destination', write (transcription): Despite the long journey, they were happy to reach their destination / (POS): ADP DET ADJ NOUN PRON AUX ADJ PART VERB PRON NOUN.",
    "For each word in the audio clip, identify its Part-of-Speech (POS) tag. " + tag_description + " Write the transcription first, followed by the POS tags separated by a '/'. For instance, 'The conference was postponed due to unforeseen circumstances', should be formatted as (transcription): The conference was postponed due to unforeseen circumstances / (POS): DET NOUN AUX VERB ADP ADJ NOUN.",
    "Determine and assign the Part-of-Speech (POS) tag for each word in the given audio. " + tag_description + " Write the transcription first, then the POS tags separated by a '/'. For example, for 'The new policy will affect all employees starting next month', write (transcription): The new policy will affect all employees starting next month / (POS): DET ADJ NOUN AUX VERB DET NOUN VERB ADP NOUN."
]

def filter_text(text):
    unwanted_chars = set("\"`;-:")
    if any(char in text for char in unwanted_chars):
        return None
    text = text.strip()  # Remove leading and trailing whitespace
    return text

def sent_and_tag(text):
    doc = nlp(text)
    sent_and_tags = []

    for sentence in doc.sentences:
        words_no_punct = [word for word in sentence.words if word.text not in punctuation]
        pos_tags = [word.upos for word in words_no_punct]
        sent = " ".join([word.text for word in words_no_punct])
        tag = " ".join(pos_tags)
        sent_and_tags.append((sent, tag, pos_tags))
    return sent_and_tags

def contains_sym_x(pos_tags):
    return any(tag in ['SYM', 'X'] for tag in pos_tags)

if __name__ == "__main__":
    ds = load_dataset(
        "blabble-io/libritts", "clean", split="test.clean",
        cache_dir="/data1/datasets_cache",
    )
    
    # Filter sentences by audio length
    new_ds = ds.filter(lambda sample: 1 <= len(sample["audio"]["array"]) / sample["audio"]["sampling_rate"] <= 6)
    
    # Filter sentences by text length
    new_ds = new_ds.filter(lambda sample: 3 <= len(sample['text_normalized'].split()) <= 15)

    def _map(sample, index):
        text = filter_text(sample['text_normalized'])
        if text is None:
            return {"audio": sample["audio"], "file": os.path.basename(sample["audio"]["path"]), "instruction": instructions[index % len(instructions)], "label": ""}

        sent_and_tags = sent_and_tag(text)
        if any(contains_sym_x(tags) for _, _, tags in sent_and_tags):
            return {"audio": sample["audio"], "file": os.path.basename(sample["audio"]["path"]), "instruction": instructions[index % len(instructions)], "label": ""}

        sentence, tag, _ = sent_and_tags[0]
        instruction = instructions[index % len(instructions)]
        label = "(transcription): " + sentence + " / (POS): " + tag
        filename = os.path.basename(sample["audio"]["path"])  # Extract filename from audio path

        return {
            "audio": sample["audio"],
            "file": filename,
            "instruction": instruction,
            "label": label
        }

    new_ds = new_ds.map(_map, with_indices=True, remove_columns=new_ds.column_names, num_proc=32)
    new_ds = new_ds.filter(lambda x: "label" in x and x["label"] != "")
    new_ds = new_ds.cast_column("audio", Audio(sampling_rate=16_000))
        
    # Convert to pandas DataFrame and save as CSV
    df = new_ds.to_pandas()
    df.to_csv("pos_tagging.csv", index=False)
    
    # Validate & Push
    validate_dataset(new_ds)
<<<<<<< HEAD
    new_ds.push_to_hub(repo_id="DynamicSuperb/PoS_Estimation_LibriTTS", split="test", token=os.environ["HF_TOKEN"])
=======
    new_ds.push_to_hub(repo_id="DynamicSuperb/PoS_Estimation_LibriTTS", split="test", token="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")
>>>>>>> 2d040ec45a9a97c3a5dcacf41da080071203c61c
