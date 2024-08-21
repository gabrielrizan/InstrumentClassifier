import numpy as np
from keras import Sequential
from keras.src.saving import load_model
from keras.src.utils import to_categorical
from tensorflow.keras.layers import Dense, Dropout
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns
import joblib
import os


def train_and_save_model(data, labels, label_map):
    num_classes = len(label_map)
    labels = to_categorical(labels, num_classes)
    X_train, X_test, y_train, y_test = train_test_split(data, labels, test_size=0.2, random_state=42)

    # Define model
    model = Sequential()
    model.add(Dense(256, input_shape=(data.shape[1],), activation='relu'))
    model.add(Dropout(0.3))
    model.add(Dense(128, activation='relu'))
    model.add(Dropout(0.3))
    model.add(Dense(num_classes, activation='softmax'))

    # Compile model
    model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

    # Train model
    history = model.fit(X_train, y_train, epochs=300, batch_size=32, validation_data=(X_test, y_test))

    # Save the model and label map
    model.save('instrument_model.h5')
    joblib.dump(label_map, 'label_map.pkl')

    # Plot accuracy and loss
    plot_accuracy_and_loss(history)

    # Generate and display confusion matrix
    generate_confusion_matrix(model, X_test, y_test, label_map)

    # Evaluate model
    test_loss, test_accuracy = model.evaluate(X_test, y_test, verbose=0)
    print(f"\nFinal Test Accuracy: {test_accuracy * 100:.2f}%")

    return model


def plot_accuracy_and_loss(history):
    # Plot accuracy
    plt.figure()
    plt.plot(history.history['accuracy'], label='train accuracy')
    plt.plot(history.history['val_accuracy'], label='val accuracy')
    plt.title('Accuracy over epochs')
    plt.xlabel('Epochs')
    plt.ylabel('Accuracy')
    plt.legend()
    plt.show()

    # Plot loss
    plt.figure()
    plt.plot(history.history['loss'], label='train loss')
    plt.plot(history.history['val_loss'], label='val loss')
    plt.title('Loss over epochs')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.legend()
    plt.show()


def generate_confusion_matrix(model, X_test, y_test, label_map):
    # Predict test data
    predictions = model.predict(X_test)
    predicted_labels = np.argmax(predictions, axis=1)
    true_labels = np.argmax(y_test, axis=1)

    # Use sklearn's confusion matrix to calculate
    conf_matrix = confusion_matrix(true_labels, predicted_labels)

    # Plot confusion matrix using Seaborn heatmap
    sns.heatmap(conf_matrix, annot=True, cmap='Blues',
                xticklabels=label_map.keys(),
                yticklabels=label_map.keys())

    plt.title('Confusion Matrix')
    plt.xlabel('Predicted Labels')
    plt.ylabel('True Labels')
    plt.show()


def load_model_and_label_map():
    # Load model and label map
    model = load_model('instrument_model.h5')
    label_map = joblib.load('label_map.pkl')
    return model, label_map


def main():
    if not model_exists():
        from preprocessing import feature_extract
        print("Starting feature extraction...")
        data, labels, label_map = feature_extract()
        print("Feature extraction complete.")

        print("Training the model...")
        train_and_save_model(data, labels, label_map)
        print("Model training complete.")
    else:
        print("Model already exists, loading the model.")
        model, label_map = load_model_and_label_map()
        test_audio = input("Please provide the path to the audio file for prediction: ")
        predict(test_audio, model, label_map)


def model_exists():  # this function checks if the model exists, and if so it doesn't run the training again
    return os.path.exists('instrument_model.h5') and os.path.exists('label_map.pkl')


def predict(audio_filename, model, label_map):
    import librosa
    from preprocessing import detect_leading_silence
    import numpy as np

    sr = 44100
    music, sr = librosa.load(audio_filename, sr=sr)
    start_trim = detect_leading_silence(music)
    end_trim = detect_leading_silence(np.flipud(music))
    trimmed_sound = music[start_trim:len(music) - end_trim]

    # Extract MFCC features
    mfccs = librosa.feature.mfcc(y=trimmed_sound, sr=sr, n_mfcc=20)
    feature = np.mean(mfccs, axis=1).reshape(1, -1)
    prediction = model.predict(feature)
    predicted_label = np.argmax(prediction)
    instrument = [k for k, v in label_map.items() if v == predicted_label][0]
    print(f"The predicted instrument is: {instrument}")


if __name__ == '__main__':
    main()
