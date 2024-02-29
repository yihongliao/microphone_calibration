import numpy as np
import sounddevice as sd
from scipy.io.wavfile import write

fs = 44100  # Sample rate
seconds = 10  # Duration of recording

def list_output_devices():
    devices = sd.query_devices()
    print("Available output devices:")
    for i, device in enumerate(devices):
        print(f"{i}. {device['name']}")

list_output_devices()
sd.default.dtype = 'int16'
myrecording = sd.rec(int(seconds * fs), samplerate=fs, channels=16, device=2, dtype=np.int16)
sd.wait()  # Wait until recording is finished
# print(myrecording)
print('Record finished')
write('../measurements/ambient_w_arm2.wav', fs, myrecording)  # Save as WAV file 
