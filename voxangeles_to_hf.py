import os
import csv
from tqdm import tqdm

def read_file_content(file_path):
    with open(file_path, 'r', encoding='utf-8') as file:
        content = file.read().strip()
    return content

def read_lines_in_files(folder_path, extension):
    line_counts = {}
    file_contents = {}
    for filename in tqdm(os.listdir(folder_path)):
        if filename.endswith(extension):
            file_path = os.path.join(folder_path, filename)
            content = read_file_content(file_path)
            file_contents[filename] = content
    return file_contents

def write_metadata_csv(wav_files, wrd_contents, phn_contents, audio_folder_path, output_csv):
    with open(output_csv, 'w', newline='') as csvfile:
        fieldnames = ['file_name', 'file', 'word', 'phn']
        writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
        writer.writeheader()

        for wav_file in sorted(wav_files):
            base_filename = os.path.splitext(wav_file)[0]
            wrd_content = wrd_contents.get(f"{base_filename}.wrd", "")
            phn_content = phn_contents.get(f"{base_filename}.phn", "")
            words = " ".join([" ".join(line.split()[2:]) for line in wrd_content.splitlines() if 3 <= len(line.split())])
            phns = " ".join([line.split()[2] for line in phn_content.splitlines() if len(line.split()) == 3])

            if not words or not phns:
                if not words:
                    print(wrd_content)
                    print(f"Excluding {wav_file} due to missing word content.")
                if not phns:
                    print(f"Excluding {wav_file} due to missing phone content.")
                continue

            writer.writerow({
                'file_name': os.path.join(audio_folder_path, wav_file),
                'file': base_filename,
                'word': words,
                'phn': phns
            })

# Paths to your directories
audio_folder_path = 'voxangeles_datasets/data/test'
annotations_folder_path = 'voxangeles_datasets/annotations'
output_csv = 'metadata.csv'

# Read the wav files, wrd files, and phn files
wav_files = os.listdir(audio_folder_path)
wrd_contents = read_lines_in_files(annotations_folder_path, ".wrd")
phn_contents = read_lines_in_files(annotations_folder_path, ".phn")

# Write the metadata to CSV
write_metadata_csv(wav_files, wrd_contents, phn_contents, audio_folder_path, output_csv)

print(f'Metadata has been written to {output_csv}')
