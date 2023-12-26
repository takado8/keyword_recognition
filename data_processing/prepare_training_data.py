import os
import numpy as np
from sklearn.preprocessing import LabelEncoder
from keras.utils import to_categorical
from data_processing.mfcc import generate_mfcc


def prepare_training_data(dataset_path="../data"):
    labels = []
    features = []

    for label in os.listdir(dataset_path):
        subfolder_path = os.path.join(dataset_path, label)

        for file_name in os.listdir(subfolder_path):
            file_path = os.path.join(subfolder_path, file_name)

            print(f'generating MFCC for {file_path}')
            mfccs = generate_mfcc(file_path, length_seconds=0.5)

            if mfccs is not None:
                features.append(mfccs)
                labels.append(label)

    features = np.array(features)
    labels = np.array(labels)
    le = LabelEncoder()
    labels_encoded = to_categorical(le.fit_transform(labels))
    return features, labels_encoded
