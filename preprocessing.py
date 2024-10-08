import numpy as np
import librosa
import glob
import os
from tqdm import tqdm


def detect_leading_silence(sound, silence_threshold=.001):
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
    data = []
    labels = []
    label_map = {}
    label_counter = 0
    allowed_instruments = ['cel', 'cla', 'flu', 'gac', 'gel', 'org', 'pia', 'sax', 'tru', 'vio', 'voi']

    all_folders = [folder for folder in os.listdir('IRMAS-TrainingData') if folder in allowed_instruments]
    if len(all_folders) == 0:
        print("No folders found in 'IRMAS-TrainingData'. Please check the directory.")
        return [], [], {}
    print(f"Extracting features from {len(all_folders)} instruments...")

    for folder in tqdm(all_folders, desc="Processing Instruments"): #tqdm is used for progress bar
        folder_path = os.path.join('IRMAS-TrainingData', folder)
        if not os.path.isdir(folder_path):
            print(f"'{folder_path}' is not a directory, skipping...")
            continue
        print(f"Processing folder: {folder}")

        if folder not in label_map:
            label_map[folder] = label_counter
            label_counter += 1
        label = label_map[folder]

        for filename in glob.glob(os.path.join(folder_path, '*.wav')):
            print(f"Processing file: {filename}")
            try:
                music, sr = librosa.load(filename, sr=sr)
                start_trim = detect_leading_silence(music)
                end_trim = detect_leading_silence(np.flipud(music))
                trimmed_sound = music[start_trim:len(music) - end_trim]
                mfccs = librosa.feature.mfcc(y=trimmed_sound, sr=sr, n_mfcc=20)
                feature = np.mean(mfccs, axis=1)
                data.append(feature)
                labels.append(label)
            except Exception as e:
                print(f"Error processing file: {filename}")
                print(e)
    print("Feature extraction completed.")
    return np.array(data), np.array(labels), label_map
