from pydub import AudioSegment
from pydub.silence import detect_nonsilent
import os
from whispers import speech_to_txt
import time


def crop_mp3(input_file_path, start_time_ms, end_time_ms, output_file_path):
    audio = AudioSegment.from_mp3(input_file_path)
    cropped_audio = audio[start_time_ms:end_time_ms]
    cropped_audio.export(output_file_path, format="mp3")


def auto_crop_mp3(input_file_path, output_file_path, chunk=0, silence_thresh=-50, duration=300):
    # Replace 'your_audio_file.wav' with the path to your audio file
    audio = AudioSegment.from_mp3(input_file_path)

    # Define parameters for silence detection
    min_silence_len = 100  # Here you can set the minimum length of silence you are looking for (in milliseconds)
      # Here you can set the threshold for what is considered silence (in dB)
    seek_step = 1          # The step size for interating over the audio for silence detection

    # Detect non-silent chunks in the audio file
    nonsilent_chunks = detect_nonsilent(audio, min_silence_len=min_silence_len,
                                        silence_thresh=silence_thresh, seek_step=seek_step)

    # Assuming the keyword is at the start of the first non-silent chunk
    # You might need to adjust the indices depending on your audio and the expected position of the keyword
    if chunk >= len(nonsilent_chunks):
        return False
    start_time = nonsilent_chunks[chunk][0]
    end_time = start_time + duration
    # end_time = nonsilent_chunks[0][1]
    # print(end_time)
    # Crop the audio
    cropped_audio = audio[start_time:end_time]

    # Save the cropped audio
    cropped_audio.export(output_file_path, format="mp3")
    return True


def auto_crop_directory(input_directory, output_directory):
    temp_results_dir = '../data/temp'
    silence_threshold = -55
    max_chunks = 5
    if not os.path.isdir(output_directory):
        os.mkdir(output_directory)

    for duration in range(300, 900, 200):
        for chunk_nb in range(max_chunks):
            print(f'chunk: {chunk_nb}')
            for i in range(5):
                threshold = silence_threshold + i * 10
                print(f'thresh: {threshold}')
                i=0
                for file in os.listdir(input_directory):
                    i+=1
                    input_path = f"{input_directory}/{file}"
                    output_path = f"{temp_results_dir}/{file}"
                    result = auto_crop_mp3(input_path, output_path,
                        chunk=chunk_nb,
                        silence_thresh=threshold,
                        duration=duration)
                    if result:
                        txt = speech_to_txt(output_path)
                        if txt == 'eryk':
                            print(f"match: {output_path}")
                            os.rename(output_path, f'{output_directory}/{file}')
                            os.remove(input_path)
                            # continue
                    # print(f'{i}/{len(os.listdir(input_directory))}')
                for file in os.listdir(temp_results_dir):
                    while True:
                        try:
                            os.remove(f'{temp_results_dir}/{file}')
                            break
                        except:
                            time.sleep(1)
                            print('waiting to remove file...')


def crop_directory(input_directory, output_directory):
    for file in os.listdir(input_directory):
        auto_crop_mp3(f'{input_directory}/{file}', f'{output_directory}/{file}',
        duration=500)


if __name__ == '__main__':
    # auto_crop_directory('../data/eryk', '../data/eryk_cropped')
    crop_directory('../data/eryk', f'../data/temp')