"""
Play random actions in an environment and render a video that demonstrates segmentation.
"""
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from copy import deepcopy
import argparse
from robosuite.utils import transform_utils
from robosuite.utils.dh_parameters import robot_attributes
from robosuite.wrappers import VisualizationWrapper
import os
import sys
import json
import imageio
import colorsys
import random
from IPython import embed
import numpy as np
np.set_printoptions(suppress=True)
import matplotlib.cm as cm
from PIL import Image
import robosuite as suite
from robosuite.controllers import load_controller_config
# all_joints_move.npz  finger_joints_move.npz  joint_0_full_revolution.npz  tool_orientation.npz
import torch
from utils import build_env, build_model, plot_replay, build_replay_buffer

real_robot_data = ['joint_0_full_revolution', 'all_joints_move',
                               'tool_orientation']#, 'finger_joints_move']

device = 'cuda'

def torch_dh_transform(theta, d, a, alpha, device):
    bs = theta.shape[0]
    T = torch.zeros((bs,4,4), device=device)
    T[:,0,0] = T[:,0,0] +  torch.cos(theta)
    T[:,0,1] = T[:,0,1] + -torch.sin(theta)*torch.cos(alpha)
    T[:,0,2] = T[:,0,2] +  torch.sin(theta)*torch.sin(alpha)
    T[:,0,3] = T[:,0,3] +  a*torch.cos(theta)
    T[:,1,0] = T[:,1,0] +  torch.sin(theta)
    T[:,1,1] = T[:,1,1] +   torch.cos(theta)*torch.cos(alpha)
    T[:,1,2] = T[:,1,2] +   -torch.cos(theta)*torch.sin(alpha)
    T[:,1,3] = T[:,1,3] +  a*torch.sin(theta)
    T[:,2,1] = T[:,2,1] +  torch.sin(alpha)
    T[:,2,2] = T[:,2,2] +   torch.cos(alpha)
    T[:,2,3] = T[:,2,3] +  d
    T[:,3,3] = T[:,3,3] +  1.0
    return T

def get_sim2real_posquat(env):
    sim_eef_pose = deepcopy(env.robots[0].pose_in_base_from_name('gripper0_eef'))
    angle = np.deg2rad(-90)
    direction_axis = [0, 0, 1]
    rotation_matrix = transform_utils.rotation_matrix(angle, direction_axis)

    sim_pose_rotated = np.dot(rotation_matrix, sim_eef_pose)
    sim_eef_pos_rotated = deepcopy(sim_pose_rotated)[:3, 3]
    sim_eef_quat_rotated = deepcopy(transform_utils.mat2quat(sim_pose_rotated))
    return sim_eef_pos_rotated, sim_eef_quat_rotated

def get_real2sim_posquat(pos, quat):
    real_eef_pose = transform_utils.pose2mat((pos,quat))
    angle = np.deg2rad(90)
    direction_axis = [0, 0, 1]
    rotation_matrix = transform_utils.rotation_matrix(angle, direction_axis)

    real_pose_rotated = np.dot(rotation_matrix, real_eef_pose)
    real_eef_pos_rotated = deepcopy(real_pose_rotated)[:3, 3]
    real_eef_quat_rotated = deepcopy(transform_utils.mat2quat(real_pose_rotated))
    return real_eef_pos_rotated, real_eef_quat_rotated

#def build_torch_dh_transform(tdh, dh_index, angles):
#    theta = tdh['DH_theta_sign'][dh_index]*angles+tdh['DH_theta_offset'][dh_index]
#    d = tdh['DH_d'][dh_index]
#    a = tdh['DH_a'][dh_index]
#    alpha = tdh['DH_alpha'][dh_index]
#    return torch_dh_transform(theta, d, a, alpha, device)

def torch_angle2ee(tdh, base_matrix, angles):
    """
        convert joint angle to end effector for reacher for ts,bs,f
    """
    # ts, bs, feat
    ts, fs = angles.shape
    #ee_pred = torch.zeros((ts,4,4)).to(device)
    _T = base_matrix
    for dh_index in range(fs):
         theta = tdh['DH_theta_sign'][dh_index]*angles[:,dh_index]+tdh['DH_theta_offset'][dh_index]
         d = tdh['DH_d'][dh_index]
         a = tdh['DH_a'][dh_index]
         alpha = tdh['DH_alpha'][dh_index]
         _T1 = torch_dh_transform(theta, d, a, alpha, device)
        #_T1 = torch_dh_transform(_a, angles[:,_a])
         _T = torch.matmul(_T, _T1)
    return _T

def run_dh_test(type_test):
    sim_robot_joints = []
    target_robot_joints = []
    dh_robot_joints = []
    sim_robot_eef_sframe = []
    sim_robot_eef_rframe = []
    dh_robot_eef_rframe = []

    # matches rframe
    #base_matrix = torch.FloatTensor([[1,0,0,0],
    #                                 [0,-1,0,0],
    #                                  [0,0,-1,0],
    #                                    [0,0,0,1]]).to(device)

    ## gets x and y correct except orientation
    #base_matrix = torch.FloatTensor([[0,1,0,0],
    #                                 [1,0,0,0],
    #                                 [0,0,1,0],
    #                                 [0,0,0,1]]).to(device)
    # gets x and y correct
    base_matrix = torch.FloatTensor([[0,1,0,0],
                                     [1,0,0,0],
                                     [0,0,-1,0],
                                     [0,0,0,1]]).to(device)



    # Create dict to hold options that will be passed to env creation call
    options = {}

    # Choose environment and add it to options
    robot_name = "Jaco"
    options["env_name"] = "Reach"
    options["robots"] = [robot_name]
    options["robot"] = robot_name


    npdh = robot_attributes[robot_name]
    tdh = {}
    for key, item in npdh.items():
        tdh[key] = torch.FloatTensor(item).to('cuda')
    # Choose camera
    gripper_site = 'gripper0_grip_site'
    camera = "frontview"

    write_path = os.path.join('datasets', 'DH_'+type_test)

    # load data
    # latin1 allows us to load python2
    real_robot = np.load(os.path.join('datasets', type_test + '.npz'), allow_pickle=True, encoding='latin1')
    real_eef_pos = real_robot['eef_pos']
    #real_joint_pos = np.mod(real_robot['joint_pos'], 4*np.pi)
    real_joint_pos = real_robot['joint_pos']

    real_actions = real_robot['actions']
    init_real = real_joint_pos[0][:7]
    # Choose controller

    controller_type = 'JOINT_POSITION'
    control_freq = 5
    controller_file = "%s_%s_%shz.json" %(robot_name.lower(), controller_type.lower(), control_freq)
    controller_fpath = os.path.join(
                os.path.split(suite.__file__)[0], 'controllers', 'config',
                controller_file)
    cfg = {}
    cfg['experiment'] = {}
    cfg['experiment']['env_type'] = 'robosuite'
    cfg['robot'] = {}
    options['controller_config_file'] = controller_fpath
    options['use_camera_obs'] = False
    options['use_object_obs'] = True
    options['control_freq'] = control_freq
    options['horizon'] = 100
    options['camera_names'] = 'frontview'
    options['reward_shaping'] = True
    options['xpos_targets'] = [gripper_site, 'ball_default_site']
    cfg['robot'] = options
    #print('loading controller from', controller_fpath)
    # Load the desired controller
    #options["controller_configs"] = load_controller_config(custom_fpath=controller_fpath)

    n_joints = 7
    n_steps = len(real_actions)

    env = build_env(options, 3,
                      skip_state_keys=[],
                      env_type='robosuite',
                      default_camera='frontview')
    replay_buffer = build_replay_buffer(cfg, env, 10000, (0,0,0), 12)
    # initialize the task
#    env = suite.make(
#        **options,
#        has_renderer=False,
#        has_offscreen_renderer=True,
#        ignore_done=True,
#        use_camera_obs=True,
#        control_freq=control_freq,
#        camera_names=camera,
#        camera_heights=512,
#        camera_widths=512,
#    )

    state, body = env.reset()

    video_writer = imageio.get_writer(write_path + '.mp4', fps=2)
    eef_site_id = env.env.robots[0].eef_site_id

    # Get action limits
    env.env.robots[0].set_robot_joint_positions(init_real[:7])

    sim_joint_pos = env.env.sim.data.qpos[env.env.robots[0]._ref_joint_pos_indexes]
    dh_eef = torch_angle2ee(tdh, base_matrix, torch.FloatTensor(sim_joint_pos)[None,:].to(device))
    for t in range(n_steps-1):
        action = np.deg2rad(real_actions[t-1])
        #action = real_joint_pos[t,:7]-sim_joint_pos

        if len(action) == 7:
            action = np.hstack((action, [0]))

        #obs, reward, done, _ = env.step(action)
        next_state, next_body, reward, done, _ = env.step(action)
        replay_buffer.add(state, body, action, reward, next_state, next_body, done)

        #video_img = obs['%s_image'%camera][::-1]
        #video_writer.append_data(video_img)


        # get simulator position and quaternion in real robot frame
        sim_eef_pose = deepcopy(env.env.robots[0].pose_in_base_from_name('gripper0_eef'))
        sim_eef_pos_sframe = deepcopy(sim_eef_pose)[:3, 3]
        sim_eef_quat_sframe = deepcopy(transform_utils.mat2quat(sim_eef_pose))
        sim_eef_pos_rframe, sim_eef_quat_rframe = get_sim2real_posquat(env.env)

        sim_joint_pos = env.env.sim.data.qpos[env.env.robots[0]._ref_joint_pos_indexes]
        sim_goal_joint_pos = env.env.robots[0].controller.goal_qpos

        torch_joint_pos = torch.FloatTensor(sim_joint_pos)[None,:].to(device)
        dh_eef_pose = torch_angle2ee(tdh, base_matrix, torch_joint_pos)[0].detach().cpu().numpy()
        dh_eef_pos_sframe = deepcopy(dh_eef_pose)[:3, 3]
        dh_eef_quat_sframe = deepcopy(transform_utils.mat2quat(dh_eef_pose))

        sim_robot_eef_sframe.append(deepcopy(np.hstack((sim_eef_pos_sframe, sim_eef_quat_sframe))))
        sim_robot_eef_rframe.append(deepcopy(np.hstack((sim_eef_pos_rframe, sim_eef_quat_rframe))))
        dh_robot_eef_rframe.append(deepcopy(np.hstack((dh_eef_pos_sframe, dh_eef_quat_sframe))))
        sim_robot_joints.append(deepcopy(sim_joint_pos))
        target_robot_joints.append(deepcopy(sim_goal_joint_pos))
        dh_robot_joints.append(deepcopy(sim_goal_joint_pos))


        #real_eef_pos_sframe, real_eef_quat_sframe = env.get_real2sim_posquat(real_eef_pos[t,:3], real_eef_pos[t,3:7])
        #real_robot_eef_rframe.append(real_eef_pos[t])
        #real_robot_eef_sframe.append(deepcopy(np.hstack((real_eef_pos_sframe, real_eef_quat_sframe))))
        #real_robot_joints.append(deepcopy(obs['robot0_joint_pos']))
        state = next_state
        body  = next_body

    plot_replay(env, replay_buffer, write_path)
    f, ax = plt.subplots(7, figsize=(10,20))
    sim_robot_eef_sframe = np.array(sim_robot_eef_sframe)
    sim_robot_eef_rframe = np.array(sim_robot_eef_rframe)
    dh_robot_eef_rframe = np.array(dh_robot_eef_rframe)
    y = np.arange(len(sim_robot_eef_sframe))
    vmin = -np.pi
    vmax = np.pi
    for i in range(7):
        ax[i].scatter(y,  dh_robot_eef_rframe[:,i] , marker='x', s=35, c='g', label='dh_sframe')
        ax[i].scatter(y,  sim_robot_eef_sframe[:,i] , marker='o', s=4, c='b', label='sim_sframe')
        #ax[i].scatter(y,  sim_robot_eef_rframe[:,i] , marker='o', s=4, c='r', label='sim_rframe')
        ax[i].plot(dh_robot_eef_rframe[:, i],  c='g' )
        ax[i].plot(sim_robot_eef_sframe[:, i],  c='b' )
        #ax[i].plot(sim_robot_eef_rframe[:, i],  c='r' )


    for i in range(4, 7):
        ax[i].set_ylim([vmin, vmax])
    ax[0].set_title('x'); ax[1].set_title('y'); ax[2].set_title('z')
    ax[3].set_title('qx'); ax[4].set_title('qy'); ax[5].set_title('qz'); ax[6].set_title('qw')
    ax[0].legend()
    plt.savefig(write_path + '_eef.png')
    plt.close()

    f, ax = plt.subplots(7, figsize=(10,20))
    sim_robot_joints = np.array(sim_robot_joints)
    dh_robot_joints = np.array(dh_robot_joints)
    target_robot_joints = np.array(target_robot_joints)
    vmin = -4*np.pi
    vmax = 4*np.pi
    for i in range(7):
        ax[i].set_title(i)
        if not i:
            ax[i].plot(sim_robot_joints[:,i], c='k', label='sim')
            ax[i].plot(target_robot_joints[:,i], c='b', label='goal')
            ax[i].plot(dh_robot_joints[:,i], c='g', label='dh')
        else:
            ax[i].plot(sim_robot_joints[:,i], c='k')
            ax[i].plot(target_robot_joints[:,i], c='b')
            ax[i].plot(dh_robot_joints[:,i], c='g')
        ax[i].scatter(y, sim_robot_joints[:,i],    s=2, c='k')
        ax[i].scatter(y, target_robot_joints[:,i], s=2, c='c')
        ax[i].scatter(y, dh_robot_joints[:,i], s=2, c='b')

    for x in range(7):
        ax[x].set_ylim([vmin, vmax])
    ax[0].legend()
    plt.savefig(write_path + '_joints.png')
    plt.close()

    video_writer.close()
    print("Video saved to {}".format(write_path))
if __name__ == "__main__":


    parser = argparse.ArgumentParser()

    parser.add_argument("--type-test", type=str, default="all", help="type",
                      choices=real_robot_data)

    args = parser.parse_args()
    if args.type_test == 'all':
        for type_test in real_robot_data:
            run_dh_test(type_test)
    else:
        run_dh_test(args.type_test)

