import librosa.display
import numpy as np
import pyaudio
from data_processing.mfcc import crop_or_pad
from keras.models import load_model
import datetime

model = load_model('neural_network/eryk2.h5')

length_seconds = 0.5
target_sample_rate = 44100
batch_size = 5


def stream_recognition():
    desired_length_in_samples = int(length_seconds * target_sample_rate)
    snapshot_count = 5
    shift_ms = 200  # adjust this to the amount of shift you want per snapshot
    shift_samples = int(shift_ms / 1000 * target_sample_rate)

    p = pyaudio.PyAudio()
    stream = p.open(format=pyaudio.paInt16,
        channels=1,
        rate=target_sample_rate,
        input=True,
        frames_per_buffer=1024)
    try:
        buffer = np.zeros(desired_length_in_samples * 2, dtype=np.float32)  # double buffer length
        while True:
            mfccs_batch = []  # To store MFCCs for each snapshot

            # Read a new chunk into the second half of the buffer
            for i in range(0, int(target_sample_rate / 1024 * (shift_ms / 1000))):
                data = stream.read(1024)
                new_data = np.frombuffer(data, dtype=np.int16).astype(np.float32) / 32768.0
                buffer[desired_length_in_samples:desired_length_in_samples + len(new_data)] += new_data

            # Create the snapshots and prepare the data for model prediction
            for i in range(snapshot_count):
                start = shift_samples * i
                end = start + desired_length_in_samples
                audio_data = buffer[start:end]
                audio_data = crop_or_pad(audio_data, desired_length_in_samples)

                mfccs = librosa.feature.mfcc(y=audio_data, sr=target_sample_rate, n_mfcc=13)
                mfccs_batch.append(mfccs)

            # Convert list of mfccs to a numpy array for batch prediction
            x_batch = np.stack(mfccs_batch, axis=0)  # Shape: (snapshot_count, time_steps, num_mfcc)
            x_batch = np.expand_dims(x_batch, axis=-1)  # Add a channel dimension

            # Perform batch prediction
            results = model.predict(x_batch, verbose=1)

            for i, result in enumerate(results):
                prediction = np.argmax(result)
                percent = round(result[prediction] * 100, 2)
                if prediction == 0 and percent < 95:
                    prediction = 1
                label = 'Keyword! <<<<<' if prediction == 0 else 'none'
                print(f'Snapshot {i}: {label} {percent}%')

            # Roll the buffer before reading the next chunk
            buffer[:desired_length_in_samples] = buffer[desired_length_in_samples:]
            buffer[desired_length_in_samples:] = 0

    finally:
        # Close the stream
        stream.stop_stream()
        stream.close()
        p.terminate()


def recognize(frames):
    desired_length_in_samples = int(length_seconds * target_sample_rate)
    frame_size = int(len(frames) / 2)
    frame_shift = int(len(frames) / batch_size)
    # mfccs_batch = []
    for i in range(batch_size):
        # Convert frames to numpy array and normalize to floating-point
        audio_data = np.frombuffer(b''.join(frames[i * frame_shift:i * frame_shift + frame_size]),
            dtype=np.int16).astype(np.float32) / 32768.0
        audio_data = crop_or_pad(audio_data, desired_length_in_samples)
        mfccs = librosa.feature.mfcc(y=audio_data, sr=target_sample_rate, n_mfcc=13)
        # mfccs_batch.append(mfccs)

        x = np.expand_dims(mfccs, axis=0)
        x = np.expand_dims(x, axis=-1)

        result = model.predict(x, verbose=0)[0]
        prediction = np.argmax(result)
        percent = round(result[prediction] * 100, 2)
        if prediction == 0 and percent < 95:
            prediction = 1
        label = 'Keyword! <<<<<' if prediction == 0 else 'none'
        print(f'{label} {percent}%')


def stream_recognition2():
    p = pyaudio.PyAudio()
    stream = p.open(format=pyaudio.paInt16,
                    channels=1,
                    rate=target_sample_rate,
                    input=True,
                    frames_per_buffer=1024)
    try:
        while True:
            # Collect audio data
            frames = []
            print("Recording...")
            for i in range(0, int(target_sample_rate / 1024 * length_seconds * 2)):
                data = stream.read(1024)
                frames.append(data)
            print("Finished recording.")
            recognize(frames)

    finally:
        # Close the stream
        stream.stop_stream()
        stream.close()
        p.terminate()


if __name__ == '__main__':
    stream_recognition2()
