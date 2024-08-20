import numpy as np
from keras import Sequential
from keras.src.saving import load_model
from keras.src.utils import to_categorical
from tensorflow.keras.layers import Dense, Dropout
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
import seaborn as sns
import joblib
import os

# Function to train the model and save it
def train_and_save_model(data, labels, label_map):
    # One-hot encode the labels
    num_classes = len(label_map)
    labels = to_categorical(labels, num_classes)

    # Split the data into training and testing sets
    X_train, X_test, y_train, y_test = train_test_split(data, labels, test_size=0.2, random_state=42)

    # Build a simple neural network model
    model = Sequential()
    model.add(Dense(256, input_shape=(data.shape[1],), activation='relu'))
    model.add(Dropout(0.3))
    model.add(Dense(128, activation='relu'))
    model.add(Dropout(0.3))
    model.add(Dense(num_classes, activation='softmax'))

    # Compile the model
    model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

    # Train the model
    history = model.fit(X_train, y_train, epochs=30, batch_size=32, validation_data=(X_test, y_test))

    # Save the model in the Keras format
    model.save('instrument_model.h5')

    # Save the label map
    joblib.dump(label_map, 'label_map.pkl')

    # Plot accuracy and loss graphs
    plot_accuracy_and_loss(history)

    # Generate confusion matrix
    generate_confusion_matrix(model, X_test, y_test, label_map)

    # Calculate and print final accuracy score
    test_loss, test_accuracy = model.evaluate(X_test, y_test, verbose=0)
    print(f"\nFinal Test Accuracy: {test_accuracy * 100:.2f}%")

    return model

# Function to plot accuracy and loss graphs
def plot_accuracy_and_loss(history):
    # Plot accuracy graph
    plt.figure()
    plt.plot(history.history['accuracy'], label='train accuracy')
    plt.plot(history.history['val_accuracy'], label='val accuracy')
    plt.title('Accuracy over epochs')
    plt.xlabel('Epochs')
    plt.ylabel('Accuracy')
    plt.legend()
    plt.show()

    # Plot loss graph
    plt.figure()
    plt.plot(history.history['loss'], label='train loss')
    plt.plot(history.history['val_loss'], label='val loss')
    plt.title('Loss over epochs')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.legend()
    plt.show()

# Function to generate a confusion matrix
def generate_confusion_matrix(model, X_test, y_test, label_map):
    predictions = model.predict(X_test)
    predicted_labels = np.argmax(predictions, axis=1)
    true_labels = np.argmax(y_test, axis=1)

    confusion_matrix = np.zeros((len(label_map), len(label_map)))

    # Create confusion matrix
    for i in range(len(true_labels)):
        confusion_matrix[true_labels[i], predicted_labels[i]] += 1

    sns.heatmap(confusion_matrix, annot=True, cmap='Blues', xticklabels=label_map.keys(), yticklabels=label_map.keys())
    plt.title('Confusion Matrix')
    plt.xlabel('Predicted Labels')
    plt.ylabel('True Labels')
    plt.show()

# Function to load the saved model and label map
def load_model_and_label_map():
    model = load_model('instrument_model.h5')
    label_map = joblib.load('label_map.pkl')
    return model, label_map

# Main function to train or load the model
def main():
    if not model_exists():
        # Extract the features and labels using your feature extraction function
        from preprocessing import feature_extract  # Assuming you have a preprocessing.py script for feature extraction
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

# Function to check if the model exists
def model_exists():
    return os.path.exists('instrument_model.h5') and os.path.exists('label_map.pkl')

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
