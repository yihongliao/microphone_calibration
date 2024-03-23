import sounddevice as sd
import soundfile as sf
import numpy as np
import scipy.io.wavfile

def list_output_devices():
    devices = sd.query_devices()
    print("Available output devices:")
    for i, device in enumerate(devices):
        print(f"{i}. {device['name']}")

def play_and_record(wav_file, input_device_id, output_device_id, output_file):

    audio_data, sample_rate = sf.read(wav_file, dtype=np.int16)
    # print(audio_data.shape)

    sd.default.device = input_device_id, output_device_id
    sd.default.dtype = 'int16'

    recorded_audio = sd.playrec(audio_data, samplerate = sample_rate, channels=16)
    sd.wait()

    # Save the recorded audio to a new WAV file
    # print(recorded_audio.shape)
    scipy.io.wavfile.write(output_file, sample_rate, recorded_audio.astype(np.int16))
    print(f"Recording saved to {output_file}")

if __name__ == "__main__":
    list_output_devices()
    wav_file_path = '../signal/white_noise_w_blank.wav'
    input_device_id = 2
    output_device_id = 4 

    # output_file = f'record_pos{1}.wav'
    output_file = '../measurements/test.wav'

    play_and_record(wav_file_path, input_device_id, output_device_id, output_file)
