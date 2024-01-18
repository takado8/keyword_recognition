import os
import threading
import queue
import librosa.display
import numpy as np
import pyaudio
from data_processing.mfcc import crop_or_pad
from keras.models import load_model

recording_time_multiplier = 2
input_length_seconds = 1
# target_sample_rate = 44100
target_sample_rate = 16000
batch_size = 8
frames_per_buffer = 1024
labels_dir = 'data/30 words'


class StreamRecognition:
    def __init__(self):
        print('loading model...')
        self.model = load_model('neural_network/30words.h5')
        self.labels = {}
        self.assign_labels()
        self.previous_ending_frames = None

    def assign_labels(self):
        i = 0
        for label in os.listdir(labels_dir):
            self.labels[i] = label
            i += 1

    def recognize(self, frames):
        desired_length_in_samples = int(input_length_seconds * target_sample_rate)
        frame_size = int(len(frames) / recording_time_multiplier)
        frame_shift = int(len(frames) / batch_size)
        mfccs_batch = []
        ending_frames = frames[-int(len(frames)/recording_time_multiplier):]
        if self.previous_ending_frames:
            frames = self.previous_ending_frames + frames
            frames_nb = int(batch_size + batch_size / recording_time_multiplier)
        else:
            frames_nb = batch_size

        for i in range(frames_nb):
            # Convert frames to numpy array and normalize to floating-point
            audio_data = np.frombuffer(b''.join(frames[i * frame_shift:i * frame_shift + frame_size]),
                dtype=np.int16).astype(np.float32) / 32768.0
            audio_data = crop_or_pad(audio_data, desired_length_in_samples)

            mfccs = librosa.feature.mfcc(y=audio_data, sr=target_sample_rate, n_mfcc=13)
            mfccs_batch.append(mfccs)
        self.previous_ending_frames = ending_frames
        # Convert list of mfccs to a numpy array for batch prediction
        x_batch = np.stack(mfccs_batch, axis=0)  # Shape: (snapshot_count, time_steps, num_mfcc)
        x_batch = np.expand_dims(x_batch, axis=-1)  # Add a channel dimension
        
        # Perform batch prediction
        results = self.model.predict(x_batch, verbose=0, batch_size=len(x_batch))

        for result in results:
            prediction = np.argmax(result)
            percent = int(round(result[prediction] * 100, 0))
            label = self.labels[prediction]
            print(f'{percent}% {label} ')

    def stream_recognition_async(self):
        p = pyaudio.PyAudio()
        stream = p.open(format=pyaudio.paInt16,
            channels=1,
            rate=target_sample_rate,
            input=True,
            frames_per_buffer=frames_per_buffer)

        try:
            while True:
                frames = []
                # print("Recording...")
                for i in range(0, int(target_sample_rate / frames_per_buffer * input_length_seconds *
                                      recording_time_multiplier)):
                    data = stream.read(frames_per_buffer)
                    frames.append(data)
                # print('done.')
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
