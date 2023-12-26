import multiprocessing
import threading
import queue
import librosa.display
import numpy as np
import pyaudio
from data_processing.mfcc import crop_or_pad
from keras.models import load_model

length_seconds = 0.5
target_sample_rate = 44100
batch_size = 5
frames_per_buffer = 1024


class StreamRecognition:
    def __init__(self):
        print('loading model...')
        self.model = load_model('neural_network/eryk2.h5')

    def recognize(self, frames):
        desired_length_in_samples = int(length_seconds * target_sample_rate)
        frame_size = int(len(frames) / 2)
        frame_shift = int(len(frames) / batch_size)
        mfccs_batch = []
        for i in range(batch_size):
            # Convert frames to numpy array and normalize to floating-point
            audio_data = np.frombuffer(b''.join(frames[i * frame_shift:i * frame_shift + frame_size]),
                dtype=np.int16).astype(np.float32) / 32768.0
            audio_data = crop_or_pad(audio_data, desired_length_in_samples)
            mfccs = librosa.feature.mfcc(y=audio_data, sr=target_sample_rate, n_mfcc=13)
            mfccs_batch.append(mfccs)

        # Convert list of mfccs to a numpy array for batch prediction
        x_batch = np.stack(mfccs_batch, axis=0)  # Shape: (snapshot_count, time_steps, num_mfcc)
        x_batch = np.expand_dims(x_batch, axis=-1)  # Add a channel dimension

        # Perform batch prediction
        results = self.model.predict(x_batch, verbose=0, batch_size=len(x_batch))

        for i, result in enumerate(results):
            prediction = np.argmax(result)
            percent = round(result[prediction] * 100, 2)
            if prediction == 0 and percent < 95:
                prediction = 1
            if prediction == 0:
                label = 'Keyword! <<<<<' if prediction == 0 else 'none'
                print(f'Snapshot {i}: {label} {percent}%')

    def stream_recognition(self):
        p = pyaudio.PyAudio()
        stream = p.open(format=pyaudio.paInt16,
                        channels=1,
                        rate=target_sample_rate,
                        input=True,
                        frames_per_buffer=1024)
        try:
            p = None
            while True:
                # Collect audio data
                frames = []
                print("Recording...")
                for i in range(0, int(target_sample_rate / 1024 * length_seconds * 2)):
                    data = stream.read(1024)
                    frames.append(data)
                print("Finished recording.")
                self.recognize(frames)

        finally:
            # Close the stream
            stream.stop_stream()
            stream.close()
            p.terminate()

    def start_stream(self, audio_queue):
        # Open the stream in a different thread to prevent blocking
        p = pyaudio.PyAudio()
        stream = p.open(format=pyaudio.paInt16,
            channels=1,
            rate=target_sample_rate,
            input=True,
            frames_per_buffer=frames_per_buffer)
        try:
            while True:
                # Read chunks from the audio stream and put them in the queue
                for i in range(0, int(target_sample_rate / frames_per_buffer * length_seconds * 2)):
                    data = stream.read(frames_per_buffer)
                    audio_queue.put(data)
        finally:
            # Close the stream
            stream.stop_stream()
            stream.close()
            p.terminate()

    def stream_recognition_full_async(self):
        audio_queue = queue.Queue()

        # start the audio stream in a separate thread
        audio_thread = threading.Thread(target=self.start_stream, args=(audio_queue,))
        audio_thread.start()

        try:
            while True:
                # Collect audio data
                frames = []
                while len(frames) < int(target_sample_rate / frames_per_buffer * length_seconds * 2):
                    frames.append(audio_queue.get())  # Take data from the queue

                # Process the audio in a separate thread to make it non-blocking
                threading.Thread(target=self.recognize, args=(frames,)).start()
        except KeyboardInterrupt:
            # User interrupted the process, stop the audio thread
            audio_thread.join()

    def stream_recognition_async(self):
        p = pyaudio.PyAudio()
        stream = p.open(format=pyaudio.paInt16,
            channels=1,
            rate=target_sample_rate,
            input=True,
            frames_per_buffer=frames_per_buffer)

        try:
            while True:
                print("Recording...")
                frames = []
                for i in range(0, int(target_sample_rate / frames_per_buffer * length_seconds * 2)):
                    data = stream.read(frames_per_buffer)
                    frames.append(data)
                print("Finished recording.")

                # Offload processing to another thread
                threading.Thread(target=self.recognize, args=(frames,)).start()

        except KeyboardInterrupt:
            print("Stopping...")
        finally:
            stream.stop_stream()
            stream.close()
            p.terminate()


if __name__ == '__main__':
    sr = StreamRecognition()
    sr.stream_recognition_async()
