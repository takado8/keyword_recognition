import librosa.display
import numpy as np
import pyaudio
from data_processing.mfcc import crop_or_pad
from keras.models import load_model
import datetime

model = load_model('neural_network/eryk2.h5')


def stream_recognition():
    length_seconds = 0.5
    target_sample_rate = 44100
    desired_length_in_samples = int(length_seconds * target_sample_rate)

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
            # print("Recording...")
            for i in range(0, int(target_sample_rate / 1024 * length_seconds)):
                data = stream.read(1024)
                frames.append(data)
            # print("Finished recording.")

            # Convert frames to numpy array and normalize to floating-point
            audio_data = np.frombuffer(b''.join(frames), dtype=np.int16).astype(np.float32) / 32768.0

            audio_data = crop_or_pad(audio_data, desired_length_in_samples)

            mfccs = librosa.feature.mfcc(y=audio_data, sr=target_sample_rate, n_mfcc=13)
            x = np.expand_dims(mfccs, axis=0)
            x = np.expand_dims(x, axis=-1)

            result = model.predict(x, verbose=0)[0]
            prediction = np.argmax(result)
            percent = round(result[prediction] * 100, 2)
            if prediction == 0 and percent < 95:
                prediction = 1
            label = 'Keyword! <<<<<' if prediction == 0 else 'none'
            print(f'{label} {percent}%')

    finally:
        # Close the stream
        stream.stop_stream()
        stream.close()
        p.terminate()


if __name__ == '__main__':
    stream_recognition()