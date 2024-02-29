import sounddevice as sd
import soundfile as sf
import numpy as np
import scipy.io.wavfile

def list_output_devices():
    devices = sd.query_devices()
    print("Available output devices:")
    for i, device in enumerate(devices):
        print(f"{i}. {device['name']}")

def play(wav_file, output_device_id):

    audio_data, sample_rate = sf.read(wav_file, dtype=np.int16)
    # print(audio_data.shape)

    # sd.default.device = input_device_id, output_device_id
    # sd.default.dtype = 'int16'

    sd.play(audio_data[:, 0], sample_rate)
    sd.wait()


if __name__ == "__main__":
    list_output_devices()
    wav_file_path = 'ambient_w_arm.wav'  # Replace with the path to your WAV file
    output_device_id = 4  # Replace with the ID of the desired output device

    # output_file = f'record_pos{1}.wav'

    play(wav_file_path, output_device_id)
