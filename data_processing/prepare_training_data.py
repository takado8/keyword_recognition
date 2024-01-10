import os
import numpy as np
from sklearn.preprocessing import LabelEncoder
from keras.utils import to_categorical
from data_processing.mfcc import generate_mfcc, generate_augmented_mfccs


def prepare_training_data(length_seconds, dataset_path="../data", use_augmentation=False):
    labels = []
    features = []

    for label in os.listdir(dataset_path):
        subfolder_path = os.path.join(dataset_path, label)

        for file_name in os.listdir(subfolder_path):
            file_path = os.path.join(subfolder_path, file_name)

            print(f'generating MFCC for {file_path}')
            if use_augmentation:
                mfccs = generate_augmented_mfccs(file_path, length_seconds)
            else:
                mfccs = [generate_mfcc(file_path, length_seconds=length_seconds)]

            if mfccs is not None:
                for mfcc in mfccs:
                    features.append(mfcc)
                    labels.append(label)

    features = np.array(features)
    labels = np.array(labels)
    le = LabelEncoder()
    labels_encoded = to_categorical(le.fit_transform(labels))
    return features, labels_encoded
