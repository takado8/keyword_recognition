import matplotlib.pyplot as plt
import librosa.display
import numpy as np


def crop_or_pad(signal, samples_length):
    # Crop or pad the signal
    if len(signal) > samples_length:
        # Crop the signal if it's longer than the desired length
        signal = signal[:samples_length]
    elif len(signal) < samples_length:
        # Pad with zeros if the signal is shorter than the desired length
        padding = samples_length - len(signal)
        signal = np.pad(signal, (0, padding), 'constant')
    return signal


def generate_mfcc(filepath, length_seconds):
    signal, sample_rate = librosa.load(filepath, sr=None)
    desired_length_in_samples = length_seconds * sample_rate
    signal = crop_or_pad(signal, desired_length_in_samples)
    mfccs = librosa.feature.mfcc(y=signal, sr=sample_rate, n_mfcc=13)
    return mfccs


def pot_MFCC(mfcc, sample_rate):
    plt.figure(figsize=(10, 4))
    librosa.display.specshow(mfcc, x_axis='time', sr=sample_rate)
    plt.colorbar()
    plt.title('MFCC')
    plt.tight_layout()
    plt.show()
