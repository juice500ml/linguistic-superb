import os
from tqdm import tqdm
from praatio import tgio

def extract_intervals(textgrid, tier_name):
    try:
        tier = textgrid.tierDict[tier_name]
        return tier.entryList
    except KeyError:
        return []

def save_intervals(file_path, intervals):
    with open(file_path, 'w', encoding='utf-8') as file:
        for interval in intervals:
            xmin, xmax, text = interval
            if text:
                file.write(f"{xmin:.6f} {xmax:.6f} {text}\n")

def process_textgrid_files(folder_path):
    for filename in tqdm(os.listdir(folder_path)):
        if filename.endswith(".TextGrid"):
            base_name = os.path.splitext(filename)[0]
            textgrid_path = os.path.join(folder_path, filename)

            try:
                textgrid = tgio.openTextgrid(textgrid_path)
            except Exception as e:
                print(f"Failed to read {textgrid_path}: {e}")
                continue

            words_intervals = extract_intervals(textgrid, "words")
            if len(words_intervals) < 1:
                words_intervals = extract_intervals(textgrid, "word")

            phones_intervals = extract_intervals(textgrid, "phones")
            if len(phones_intervals) < 1:
                phones_intervals = extract_intervals(textgrid, "phone")
            if len(phones_intervals) < 1:
                phones_intervals = extract_intervals(textgrid, "Narrow")

            wrd_file_path = os.path.join(folder_path, f"{base_name}.wrd")
            phn_file_path = os.path.join(folder_path, f"{base_name}.phn")

            save_intervals(wrd_file_path, words_intervals)
            save_intervals(phn_file_path, phones_intervals)

folder_path = 'voxangeles_datasets/annotations'  # folder with TextGrid files
process_textgrid_files(folder_path)
