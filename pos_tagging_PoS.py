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
    "For each word in the given audio utterance, determine the corresponding Part-of-Speech (POS) tag. " + tag_description + " For example, when you listen to an audio 'The quick brown fox jumps over the lazy dog', please generate DET ADJ ADJ NOUN VERB ADP DET ADJ NOUN.",
    "Listen to the given audio clip and provide the Part-of-Speech (POS) tag for each word. " + tag_description + " For instance, for the audio 'She sells sea shells by the seashore on a sunny day', generate PRON VERB NOUN NOUN ADP DET NOUN ADP DET ADJ NOUN.",
    "Determine the Part-of-Speech (POS) tag for every word in the audio sample. " + tag_description + " As an example, for 'He is running fast because he is late for the meeting', produce PRON AUX VERB ADV SCONJ PRON AUX ADJ ADP DET NOUN.",
    "Identify the Part-of-Speech (POS) tag for each word in the provided audio. " + tag_description + " For example, for the audio 'It was a sunny day and the children were playing outside', generate PRON AUX DET ADJ NOUN CCONJ DET NOUN AUX VERB ADV.",
    "Assign the appropriate Part-of-Speech (POS) tag to each word in the audio. " + tag_description + " For example, when listening to 'Cats are great pets, and they make wonderful companions', generate NOUN AUX ADJ NOUN CCONJ PRON VERB ADJ NOUN.",
    "Listen to the audio and determine the Part-of-Speech (POS) tag for each word. " + tag_description + " For instance, for 'I enjoy reading books in the quiet park every afternoon', produce PRON VERB VERB NOUN ADP DET ADJ NOUN ADV NOUN.",
    "From the given audio utterance, provide the Part-of-Speech (POS) tag for every word. " + tag_description + " For example, for 'They will travel tomorrow to visit their grandparents in the city', generate PRON AUX VERB ADV SCONJ VERB PRON NOUN ADP DET NOUN.",
    "Determine and list the Part-of-Speech (POS) tag for each word in the audio. " + tag_description + " For instance, for 'My friend is coming over to help me with my homework', produce DET NOUN AUX VERB ADV ADP VERB PRON ADP DET NOUN.",
    "Identify the POS tag for each word in the audio clip. " + tag_description + " For example, when you hear 'Birds are flying high in the sky, enjoying the warm weather', generate NOUN AUX VERB ADV ADP DET NOUN VERB DET ADJ NOUN.",
    "Provide the Part-of-Speech (POS) tag for every word in the given audio. " + tag_description + " As an example, for the audio 'We watched a movie last night, and it was very entertaining', generate PRON VERB DET NOUN ADJ NOUN CCONJ PRON AUX ADV ADJ."
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
        label = tag
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
    new_ds.push_to_hub(repo_id="DynamicSuperb/PoS_Estimation_LibriTTS_PoS", split="test", token=os.environ["HF_TOKEN"])
=======
    new_ds.push_to_hub(repo_id="DynamicSuperb/PoS_Estimation_LibriTTS_PoS", split="test", token="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")
>>>>>>> 2d040ec45a9a97c3a5dcacf41da080071203c61c
