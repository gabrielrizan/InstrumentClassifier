import numpy as np
import librosa
import joblib
import preprocessing

def extract_features_from_file(audio_filename):
    # Load the audio file
    sr = 44100
    music, sr = librosa.load(audio_filename, sr=sr)

    # Trim leading and trailing silence
    start_trim = preprocessing.detect_leading_silence(music)
    end_trim = preprocessing.detect_leading_silence(np.flipud(music))
    duration = len(music)
    trimmed_sound = music[start_trim:duration-end_trim]

    # Extract MFCC features
    mfccs = librosa.feature.mfcc(y=trimmed_sound, sr=sr)
    aver = np.mean(mfccs, axis=1)
    feature = aver.reshape(1, -1)  # Reshape for the model (1 sample, many features)

    return feature

def load_trained_model():
    # Load the previously trained model
    svc = joblib.load('last_svc.model')  # Load only the SVC model
    return svc

def predict_instrument(model, feature):
    # Directly use the SVC model (skip PCA transformation)
    prediction = model.predict(feature)
    return prediction

def main():
    # Ask the user to input an audio file for testing
    audio_file = input("Enter the path to the audio file: ")

    # Extract features from the new audio file
    features = extract_features_from_file(audio_file)

    # Load the trained model
    model = load_trained_model()

    # Predict the instrument
    prediction = predict_instrument(model, features)

    # Output the result
    if prediction == 1:
        print("The instrument is Banjo.")
    elif prediction == 4:
        print("The instrument is Cello.")
    elif prediction == 11:
        print("The instrument is Guitar.")
    elif prediction == 15:
        print("The instrument is Saxophone.")
    elif prediction == 20:
        print("The instrument is Violin.")
    else:
        print("Unknown instrument.")

if __name__ == "__main__":
    main()
