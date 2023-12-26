import numpy as np

from data_processing.mfcc import generate_mfcc
from data_processing.prepare_training_data import prepare_training_data
from sklearn.model_selection import train_test_split


def load_data(dataset_path, test_size=0.2):
    features, labels = prepare_training_data(dataset_path)

    mfccs_train, mfccs_test, labels_train, labels_test = \
        train_test_split(features, labels, test_size=test_size)

    # Reshape the data to include the channel dimension
    x_train = mfccs_train[..., np.newaxis]
    x_test = mfccs_test[..., np.newaxis]
    return x_train, x_test, labels_train, labels_test


def load_input(filepath):
    mfccs = generate_mfcc(filepath, 0.5)

    # Add both the batch dimension and the channel dimension
    x = np.expand_dims(mfccs, axis=0)
    x = np.expand_dims(x, axis=-1)
    return x