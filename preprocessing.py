import numpy as np
import librosa
import glob
import os
import zipfile
import tempfile
from tqdm import tqdm  # Progress bar

def detect_leading_silence(sound, silence_threshold=.001, chunk_size=10):
    trim_ms = 0
    max_num = max(sound)
    sound = sound / max_num
    sound = np.array(sound)
    for i in range(len(sound)):
        if sound[trim_ms] < silence_threshold:
            trim_ms += 1
    return trim_ms

def feature_extract():
    sr = 44100
    window_size = 2048
    hop_size = window_size // 2
    data = []

    # Create a temporary directory to extract the ZIP files
    with tempfile.TemporaryDirectory() as temp_dir:
        # Define labels for each instrument based on the ZIP file names
        instrument_labels = {
            'banjo': 1,
            'bass clarinet': 2,
            # 'bassoon': 3,
            'cello': 4,
            # 'clarinet': 5,
            # 'contrabassoon': 6,
            # 'cor anglais': 7,
            # 'double bass': 8,
            # 'flute': 9,
            # 'french horn': 10,
            'guitar': 11,
            # 'mandolin': 12,
            # 'oboe': 13,
            # 'percussion': 14,
            'saxophone': 15,
            # 'trombone': 16,
            # 'trumpet': 17,
            # 'tuba': 18,
            # 'viola': 19,
            'violin': 20
        }

        # Process all ZIP files in the 'all-samples' folder
        zip_files = glob.glob('all-samples/*.zip')

        # Iterate over each ZIP file (each instrument)
        for zip_file in tqdm(zip_files, desc="Processing ZIP files"):
            instrument_name = os.path.splitext(os.path.basename(zip_file))[0]  # Get instrument name
            label = instrument_labels.get(instrument_name.lower(), 0)  # Get label for the instrument

            if label == 0:
                continue  # Skip unknown instruments

            # Extract the ZIP file contents
            with zipfile.ZipFile(zip_file, 'r') as zip_ref:
                zip_ref.extractall(temp_dir)  # Extract to temp directory

            # Find all the extracted .mp3 files
            mp3_files = glob.glob(f'{temp_dir}/**/*.mp3', recursive=True)
            np.random.shuffle(mp3_files)

            # Process each mp3 file to extract features
            for filename in tqdm(mp3_files, desc=f"Extracting features from {instrument_name} files", leave=False):
                try:
                    music, sr = librosa.load(filename, sr=sr)
                except Exception as e:
                    print(f"Error loading {filename}: {e}")
                    continue  # Skip the problematic file

                start_trim = detect_leading_silence(music)
                end_trim = detect_leading_silence(np.flipud(music))

                duration = len(music)
                trimmed_sound = music[start_trim:duration-end_trim]

                # Extract MFCC features
                mfccs = librosa.feature.mfcc(y=trimmed_sound, sr=sr)
                aver = np.mean(mfccs, axis=1)
                feature = aver.reshape(20)  # 20 MFCC features

                data2 = [filename, feature, label]
                data.append(data2)

    return data
