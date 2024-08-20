from classify import load_model_and_label_map

def main():
    # Load the model and label map
    model, label_map = load_model_and_label_map()

    # Predict with a new audio file
    test_audio = input("Please provide the path to the audio file for prediction: ")
    predict(test_audio, model, label_map)

# Function to predict the instrument of a given audio file
def predict(audio_filename, model, label_map):
    import librosa
    from preprocessing import detect_leading_silence
    import numpy as np

    sr = 44100
    music, sr = librosa.load(audio_filename, sr=sr)
    start_trim = detect_leading_silence(music)
    end_trim = detect_leading_silence(np.flipud(music))
    trimmed_sound = music[start_trim:len(music)-end_trim]

    mfccs = librosa.feature.mfcc(y=trimmed_sound, sr=sr, n_mfcc=20)
    feature = np.mean(mfccs, axis=1).reshape(1, -1)

    prediction = model.predict(feature)
    predicted_label = np.argmax(prediction)
    instrument = [k for k, v in label_map.items() if v == predicted_label][0]
    print(f"The predicted instrument is: {instrument}")

if __name__ == '__main__':
    main()
