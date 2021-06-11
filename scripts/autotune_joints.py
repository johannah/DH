import robosuite
from robosuite.robots import SingleArm 
import numpy as np
from IPython import embed
from imageio import mimwrite
from copy import deepcopy
import json

"""
# JRH summary of Ziegler-Nichols method
1) set damping_ratio and kp to zero
2) increase kp slowly until you get "overshoot", (positive ERR on first half of "diffs", negative ERR on second half of "diffs"
3) increase damping ratio to squash the overshoot
4) watch and make sure you aren't "railing" the torque values on the big diffs (30.5 for the first 4 joints on Jaco). If this is happening, you may need to decrease the step size (min_max)
"""

def run_test():
    controller_config = robosuite.load_controller_config(args.config_file)
    active_joint = args.active_joint
    robot_name = args.robot_name
    if 'MIN_MAX_DIFF' in controller_config.keys():
        min_max = controller_config['MIN_MAX_DIFF'][active_joint]
    else:
        min_max = .01
    
    for k,j in controller_config.items():
        print(k,j)
    
    print('TESTING: %s joint: %s with kp:%s d:%s'%(robot_name, active_joint, 
                                                      controller_config['kp'][active_joint], 
                                                      controller_config['damping_ratio'][active_joint]))
    
    frames = []
    for num_action_steps in [1, 5]:
        print('repeat action for %s steps'%num_action_steps)
        print(['>>' for a in range(num_action_steps)])
        for diff in np.linspace(-min_max, min_max, 5):
            env = robosuite.make("Door", robots=robot_name,
                                 has_renderer=False,        
                                 has_offscreen_renderer=True, 
                                 ignore_done=True, 
                                 use_camera_obs=False,
                                 controller_configs=controller_config, 
                                 control_freq=20)
            if diff == 0:
                print('----')
            env.reset()
            active_robot = env.robots[0]
            num_joints = active_robot.controller.control_dim
            action = np.zeros(num_joints+1)
            action[active_joint] = diff
     
            for i in range(num_action_steps):
                prev_pos = deepcopy(env.sim.data.qpos[active_robot._ref_joint_pos_indexes[active_joint]])
                env.step(action)
                print('%d TORQUE: %.02f' %(i, active_robot.torques[active_joint]))
                frames.append(env.sim.render(camera_name="frontview", height=248, width=248)[::-1])
            cur_pos = deepcopy(env.sim.data.qpos[active_robot._ref_joint_pos_indexes[active_joint]])
            target_pos = prev_pos + diff
            error = target_pos-cur_pos
            print('ERR: %.06f DIFF: %.06f '%(error, diff))
            print('PREV: %.06f TARG: %.06f NOW: %.06f'%(prev_pos, target_pos, cur_pos))
            for i in range(args.num_rest_steps):
                env.step(np.zeros_like(action))
                frames.append(env.sim.render(camera_name="frontview", height=248, width=248)[::-1])
    mimwrite(args.movie_file, frames)

if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--robot_name', default='Jaco')
    parser.add_argument('--active_joint', default=3, type=int, help='joint to test')
    parser.add_argument('--config_file', default='', help='path to config file. if not configured, will default to ../confgis/robot_name_joint_position.json')
    parser.add_argument('--num_rest_steps', default=3, type=int)
    parser.add_argument('--movie_file', default='joints.mp4')
    args = parser.parse_args() 
    if args.config_file == '':
        args.config_file = '../configs/%s_joint_position.json'%args.robot_name.lower()
    run_test()
