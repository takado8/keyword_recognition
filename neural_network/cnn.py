from data_processing.mfcc import generate_mfcc
import os
import numpy as np
from sklearn.model_selection import train_test_split
from keras.utils import to_categorical
from keras.models import Sequential
from keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout
from sklearn.preprocessing import LabelEncoder


# Navigate your dataset directory and process the data
dataset_path = "path_to_your_dataset"
labels = []
features = []

for label in os.listdir(dataset_path):
    # Go through each subfolder (label) in the dataset directory
    subfolder_path = os.path.join(dataset_path, label)

    for file_name in os.listdir(subfolder_path):
        file_path = os.path.join(subfolder_path, file_name)

        # Extract MFCC features from the audio file
        mfccs = generate_mfcc(file_path, length_seconds=2)

        if mfccs is not None:
            features.append(mfccs)
            labels.append(label)

# Convert into a numpy array
features = np.array(features)
labels = np.array(labels)

# Convert labels to one-hot encoding

le = LabelEncoder()
labels_encoded = to_categorical(le.fit_transform(labels))

# Split the dataset into training and testing sets
mfccs_train, mfccs_test, labels_train, labels_test = train_test_split(features, labels_encoded, test_size=0.2)

# Reshape the data to include the channel dimension
mfccs_train = mfccs_train[..., np.newaxis]  # Adds a channel dimension
mfccs_test = mfccs_test[..., np.newaxis]    # Adds a channel dimension

# CNN Model
model = Sequential()
model.add(Conv2D(filters=32, kernel_size=(3, 3), activation='relu', input_shape=(mfccs_train.shape[1], mfccs_train.shape[2], 1)))
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Conv2D(filters=64, kernel_size=(3, 3), activation='relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Dropout(0.25))
model.add(Flatten())
model.add(Dense(128, activation='relu'))
model.add(Dropout(0.5))
model.add(Dense(labels_train.shape[1], activation='softmax'))  # assuming labels are one-hot encoded

# Compile the model
model.compile(loss='categorical_crossentropy',
              optimizer='adam',
              metrics=['accuracy'])

# Train the model
model.fit(mfccs_train, labels_train, batch_size=32, epochs=10, validation_data=(mfccs_test, labels_test))

# Evaluate the model
score = model.evaluate(mfccs_test, labels_test, verbose=0)
print('Test loss:', score[0])
print('Test accuracy:', score[1])