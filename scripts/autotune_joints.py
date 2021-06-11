import robosuite
from robosuite.robots import SingleArm 
import numpy as np
from IPython import embed
from imageio import mimwrite
from copy import deepcopy
import json

robot_name = "Jaco"
active_joint = 3
num_joints = 7
"""
# JRH summary of Ziegler-Nichols method
1) set damping_ratio and kp to zero
2) increase kp slowly until you get "overshoot", (positive ERR on first half of "diffs", negative ERR on second half of "diffs"
3) increase damping ratio to squash the overshoot
4) watch and make sure you aren't "railing" the torque values on the big diffs (30.5 for the first 4 joints on Jaco). If this is happening, you may need to decrease the step size (min_max)
"""
#controller_config = {
#       "type": "JOINT_POSITION", 
#       "input_max":1, 
#       "input_min":-1, 
#       "damping_ratio":[0.1,.6,.6,.55,.55,.55,.55],
#       "output_max":10,
#       "output_min":-10,
#       "kp":[2500,30,30,25,25,25,25],
#       "impedence_mode":"fixed", 
#       "interpolation":None, 
#       "kp_limits":(0, 1000),
#       }
#
#min_max = [.0004, .04, .06, .08, .08, .04, .04]
controller_config = json.load(open('../configs/%s_joint_position.json'%robot_name.lower()))
min_max = controller_config['MIN_MAX_DIFF']
for k,j in controller_config.items():
    print(k,j)
print('TESTING: %s joint: %s with kp:%s d:%s'%(robot_name, active_joint, 
                                                  controller_config['kp'][active_joint], 
                                                  controller_config['damping_ratio'][active_joint]))

frames = []
for diff in np.linspace(-min_max[active_joint], min_max[active_joint], 7):
    env = robosuite.make("Door", robots=robot_name,
                         has_renderer=False,        
                         has_offscreen_renderer=True, 
                         ignore_done=True, 
                         use_camera_obs=False,
                         controller_configs=controller_config, 
                         control_freq=20)
    env.reset()
    active_robot = env.robots[0]
    action = np.zeros(num_joints+1)
    action[active_joint] = diff
 
    for i in range(1):
        prev_pos = deepcopy(env.sim.data.qpos[active_robot._ref_joint_pos_indexes[active_joint]])
        env.step(action)
        frames.append(env.sim.render(camera_name="frontview", height=248, width=248)[::-1])
    cur_pos = deepcopy(env.sim.data.qpos[active_robot._ref_joint_pos_indexes[active_joint]])
    print('TORQUE: %.02f' %active_robot.torques[active_joint])
    target_pos = prev_pos + diff
    error = target_pos-cur_pos
    print('ERR: %.06f DIFF: %.06f '%(error, diff))
    print('PREV: %.06f TARG: %.06f NOW: %.06f'%(prev_pos, target_pos, cur_pos))
    for i in range(2):
        env.step(np.zeros_like(action))
        frames.append(env.sim.render(camera_name="frontview", height=248, width=248)[::-1])
mimwrite('joints.mp4', frames)

