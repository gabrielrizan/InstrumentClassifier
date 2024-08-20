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

    # Extract MFCC features with the same number of MFCCs used during training (50)
    mfccs = librosa.feature.mfcc(y=trimmed_sound, sr=sr, n_mfcc=20)
    aver = np.mean(mfccs, axis=1)
    feature = aver.reshape(1, -1)  # Reshape for the model (1 sample, many features)

    return feature

def load_trained_model():
    # Load the trained model, scaler, and PCA from training
    svc = joblib.load('best_model_svc.model')  # Load the SVC model
    scaler = joblib.load('last_scaler.model')  # Load the scaler used during training
    pca = joblib.load('last_pca.model')  # Load the PCA model used during training
    return svc, scaler, pca

def predict_instrument(model, scaler, pca, feature):
    # Apply the same standardization and PCA transformation as during training
    feature = scaler.transform(feature)  # Standardize features
    feature = pca.transform(feature)  # Apply PCA

    # Predict the instrument
    prediction = model.predict(feature)
    return prediction

def main():
    # Ask the user to input an audio file for testing
    audio_file = input("Enter the path to the audio file: ")

    # Extract features from the new audio file
    features = extract_features_from_file(audio_file)

    # Load the trained model, scaler, and PCA
    model, scaler, pca = load_trained_model()

    # Predict the instrument
    prediction = predict_instrument(model, scaler, pca, features)

    # Output the result
    if prediction == 1:
        print("The instrument is Banjo.")
    elif prediction == 2:
        print("The instrument is Bass Clarinet.")
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
