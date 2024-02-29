import numpy as np
import scipy.io.wavfile as wav

def generate_frequency_sweep(start_freq, end_freq, step_size, step_duration, sample_rate, output_file):
    total_samples = int(step_duration * sample_rate)

    # Generate the frequency sweep
    signal = np.concatenate([
        0.5 * np.sin(2 * np.pi * freq * np.linspace(0, step_duration, total_samples, endpoint=False))
        for freq in range(start_freq, end_freq + 1, step_size)
    ])

    # Normalize the signal
    amplitude = np.iinfo(np.int16).max
    signal = amplitude * (signal / np.max(np.abs(signal)))

    # Add silence
    signal = np.pad(signal, (sample_rate*1), 'constant')
    print(signal)

    # Save the WAV file
    wav.write(output_file, sample_rate, signal.astype(np.int16))

# Parameters
start_frequency = 100  # Hz
end_frequency = 5000  # Hz
frequency_step = 30  # Hz
step_duration = 0.2    # seconds
sampling_rate = 44100  # Hz
output_filename = 'frequency_sweep.wav'

# Generate and save the frequency sweep
generate_frequency_sweep(start_frequency, end_frequency, frequency_step, step_duration, sampling_rate, output_filename)

print(f"WAV file '{output_filename}' generated successfully.")
