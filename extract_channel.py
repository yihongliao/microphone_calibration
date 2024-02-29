import sounddevice as sd
import soundfile as sf
import numpy as np
import scipy.io.wavfile


def extract_crop(wav_file, channel, start_time, duration, output_file):

    audio_data, sample_rate = sf.read(wav_file, dtype=np.int16)
    # print(audio_data.shape)

    sd.default.dtype = 'int16'

    print(audio_data.ndim)
    if audio_data.ndim > 1:
        scipy.io.wavfile.write(output_file, sample_rate, audio_data[start_time*sample_rate:(start_time+duration)*sample_rate, channel].astype(np.int16))
    else:
        scipy.io.wavfile.write(output_file, sample_rate, audio_data[start_time*sample_rate:(start_time+duration)*sample_rate].astype(np.int16))
    print(f"Recording saved to {output_file}")

if __name__ == "__main__":
    wav_file_path = "Chirp_Sound2.wav"
    output_file = "Chirp_Sound2_3s.wav"
    channel = 0
    start_time = 1
    duration = 3
    extract_crop(wav_file_path, channel, start_time, duration, output_file)
