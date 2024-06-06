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

def play_and_record(wav_file, input_device_id, output_device_id, output_file, chs=16):

    audio_data, sample_rate = sf.read(wav_file, dtype=np.int16)
    # print(audio_data.shape)

    sd.default.device = input_device_id, output_device_id
    sd.default.dtype = 'int16'

    recorded_audio = sd.playrec(audio_data, samplerate=sample_rate, channels=chs)
    sd.wait()

    # Save the recorded audio to a new WAV file
    # print(recorded_audio.shape)
    scipy.io.wavfile.write(output_file, sample_rate, recorded_audio.astype(np.int16))
    print(f"Recording saved to {output_file}")

if __name__ == "__main__":

    calib_file_path = '../signal/white_noise_with_10s_silence.wav'  # Replace with the path to your WAV file
    sweep_file_path = '../signal/frequency_sweep.wav'
    output_file_path = '../measurements/test/'
    channels = 16

    ########################################################

    list_output_devices()
    input_device_id = int(input("input_device_id: ")) 
    output_device_id = int(input("output_device_id: "))  
    channels = int(input("channels: "))  

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

    start_pos = [0.0, 319.048248, 277.631134, -180, -90, 90]
    square_step = 126.0
    speed = 10
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

    # go to start position
    arm.set_position(x=start_pos[0], y=start_pos[1], z=start_pos[2], roll=start_pos[3], pitch=start_pos[4], yaw=start_pos[5], speed=speed, is_radian=False, wait=True)
    print(arm.get_position(), arm.get_position(is_radian=False))

    time.sleep(1)
    output_calib_file = output_file_path + f'calib_pos{0}.wav'
    output_sweep_file = output_file_path + f'sweep_pos{0}.wav'
    play_and_record(calib_file_path, input_device_id, output_device_id, output_calib_file, channels)
    time.sleep(0.5)
    play_and_record(sweep_file_path, input_device_id, output_device_id, output_sweep_file, channels)
    
    arm.set_position(x=start_pos[0]+square_step, y=start_pos[1], z=start_pos[2], roll=start_pos[3], pitch=start_pos[4], yaw=start_pos[5], speed=speed, is_radian=False, wait=True)
    print(arm.get_position(), arm.get_position(is_radian=False))

    time.sleep(1)
    output_calib_file = output_file_path + f'calib_pos{1}.wav'
    output_sweep_file = output_file_path + f'sweep_pos{1}.wav'
    play_and_record(calib_file_path, input_device_id, output_device_id, output_calib_file, channels)
    time.sleep(0.5)
    play_and_record(sweep_file_path, input_device_id, output_device_id, output_sweep_file, channels)

    arm.set_position(x=start_pos[0]+square_step, y=start_pos[1], z=start_pos[2]-square_step, roll=start_pos[3], pitch=start_pos[4], yaw=start_pos[5], speed=speed, is_radian=False, wait=True)
    print(arm.get_position(), arm.get_position(is_radian=False))

    time.sleep(1)
    output_calib_file = output_file_path + f'calib_pos{2}.wav'
    output_sweep_file = output_file_path + f'sweep_pos{2}.wav'
    play_and_record(calib_file_path, input_device_id, output_device_id, output_calib_file, channels)
    time.sleep(0.5)
    play_and_record(sweep_file_path, input_device_id, output_device_id, output_sweep_file, channels)

    arm.set_position(x=start_pos[0], y=start_pos[1], z=start_pos[2]-square_step, roll=start_pos[3], pitch=start_pos[4], yaw=start_pos[5], speed=speed, is_radian=False, wait=True)
    print(arm.get_position(), arm.get_position(is_radian=False))

    time.sleep(1)
    output_calib_file = output_file_path + f'calib_pos{3}.wav'
    output_sweep_file = output_file_path + f'sweep_pos{3}.wav'
    play_and_record(calib_file_path, input_device_id, output_device_id, output_calib_file, channels)
    time.sleep(0.5)
    play_and_record(sweep_file_path, input_device_id, output_device_id, output_sweep_file, channels)

    # back to start position
    arm.set_position(x=start_pos[0], y=start_pos[1], z=start_pos[2], roll=start_pos[3], pitch=start_pos[4], yaw=start_pos[5], speed=speed, is_radian=False, wait=True)
    print(arm.get_position(), arm.get_position(is_radian=False))

    arm.disconnect()