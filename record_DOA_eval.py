import sounddevice as sd
import soundfile as sf
import numpy as np
import scipy.io.wavfile
import os
import sys
import time

sys.path.append(os.path.join(os.path.dirname(__file__), 'xArm-Python-SDK'))

from xarm.wrapper import XArmAPI

def hangle_err_warn_changed(item):
    print('ErrorCode: {}, WarnCode: {}'.format(item['error_code'], item['warn_code']))
    # TODO：Do different processing according to the error code

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

    #######################################################
    """
    Input Robot IP Address
    """
    if len(sys.argv) >= 2:
        ip = sys.argv[1]
    else:
        try:
            from configparser import ConfigParser
            parser = ConfigParser()
            parser.read('robot.conf')
            ip = parser.get('xArm', 'ip')
        except:
            ip = input('Please input the xArm ip address:')
            if not ip:
                print('input error, exit')
                sys.exit(1)

    print(f'Find Robot ip: {ip}')

    speed = 20
    arm = XArmAPI(ip, do_not_open=True, is_radian=False)
    arm.register_error_warn_changed_callback(hangle_err_warn_changed)
    arm.connect()

    # enable motion
    arm.motion_enable(enable=True)
    # set mode: position control mode
    arm.set_mode(0)
    # set state: sport state
    arm.set_state(state=0)
    ########################################################

    wav_file_path = '../signal/white_noise_with_0s_silence.wav'  # Replace with the path to your WAV file
    output_file_path = '../measurements/DOA_0708/2/'
    input_device_id = 1
    output_device_id = 5  # Replace with the ID of the desired output device

    ########################################################

    position = arm.get_position()[1]
    print(position)

    for i in range(36):
        arm.set_servo_angle(servo_id=6, angle=i*10, speed=speed, is_radian=False, wait=True)
        print(arm.get_position(), arm.get_position(is_radian=False))
        time.sleep(1)

        output_file = output_file_path + f'record_pos{i}.wav'
        play_and_record(wav_file_path, input_device_id, output_device_id, output_file)

    arm.set_servo_angle(servo_id=6, angle=0, speed=speed, is_radian=False, wait=True)

    arm.disconnect()

    print("finished")