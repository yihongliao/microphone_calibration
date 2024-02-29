import os
import sys
import time
sys.path.append(os.path.join(os.path.dirname(__file__), 'xArm-Python-SDK'))

from xarm.wrapper import XArmAPI

def hangle_err_warn_changed(item):
    print('ErrorCode: {}, WarnCode: {}'.format(item['error_code'], item['warn_code']))
    # TODO：Do different processing according to the error code

if __name__=="__main__": 

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
    ########################################################

    start_pos = [-63.0, 289.5, 247.1, -180, -90, 90]
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

    # go to start position
    arm.set_position(x=start_pos[0], y=start_pos[1], z=start_pos[2], roll=start_pos[3], pitch=start_pos[4], yaw=start_pos[5], speed=speed, is_radian=False, wait=True)
    print(arm.get_position(), arm.get_position(is_radian=False))
    
    arm.set_position(x=start_pos[0]+square_step, y=start_pos[1], z=start_pos[2], roll=start_pos[3], pitch=start_pos[4], yaw=start_pos[5], speed=speed, is_radian=False, wait=True)
    print(arm.get_position(), arm.get_position(is_radian=False))

    arm.set_position(x=start_pos[0]+square_step, y=start_pos[1], z=start_pos[2]-square_step, roll=start_pos[3], pitch=start_pos[4], yaw=start_pos[5], speed=speed, is_radian=False, wait=True)
    print(arm.get_position(), arm.get_position(is_radian=False))

    arm.set_position(x=start_pos[0], y=start_pos[1], z=start_pos[2]-square_step, roll=start_pos[3], pitch=start_pos[4], yaw=start_pos[5], speed=speed, is_radian=False, wait=True)
    print(arm.get_position(), arm.get_position(is_radian=False))

    arm.set_position(x=start_pos[0], y=start_pos[1], z=start_pos[2], roll=start_pos[3], pitch=start_pos[4], yaw=start_pos[5], speed=speed, is_radian=False, wait=True)
    print(arm.get_position(), arm.get_position(is_radian=False))

    arm.disconnect()