import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import robosuite
import imageio
import numpy as np
import os
from glob import glob
from copy import deepcopy
import pickle

import json
from imageio import mimwrite

from replay_buffer import ReplayBuffer, compress_frame

from torch.utils.tensorboard import SummaryWriter
import torch
import robosuite.utils.macros as macros
torch.set_num_threads(3)
import TD3_kinematic

from dh_utils import seed_everything, normalize_joints, skip_state_keys, robotDH
from utils import build_replay_buffer, build_env, build_model, plot_replay, get_rot_mat, MAX_RELATIVE_ACTION
from IPython import embed


def run(env, replay_buffer, cfg, cam_dim, savebase):
    robot_name = cfg['robot']['robots'][0]

    env_type = cfg['experiment']['env_type']
    num_steps = 0
    total_steps = replay_buffer.max_size-1
    use_frames = cam_dim[0] > 0
    if use_frames:
        print('recording camera: %s'%args.camera)

    h, w, c = cam_dim
    torques = []
    rewards = []
    while num_steps < total_steps:
        done = False
        state, body =  env.reset()
        if use_frames:
            frame_compressed = compress_frame(env.render(camera_name=args.camera, height=h, width=w))
        ep_reward = 0
        e_step = 0

        while not done:# and e_step < args.max_eval_timesteps:
 
            action = np.array([0.00001, .0001, -.0001, .01, .0, .0, 0, 0])

            next_state, next_body, reward, done, info = env.step(action) # take a random action
            ep_reward += reward
            if use_frames:
                next_frame_compressed = compress_frame(env.render(camera_name=args.camera, height=h, width=w))
                replay_buffer.add(state, body, action, reward, next_state, next_body, done, 
                              frame_compressed=frame_compressed, 
                              next_frame_compressed=next_frame_compressed)
                frame_compressed = next_frame_compressed
            else:
                replay_buffer.add(state, body, action, reward, next_state, next_body, done)
            if e_step > 100:
                done = True
            torques.append(env.env.robots[0].torques)
            print(next_body[:7] - body[:7])
            print(torques[-1])
            state = next_state
            body = next_body
            num_steps+=1
            e_step+=1
        rewards.append(ep_reward)
    replay_buffer.torques = torques
    return rewards, replay_buffer

def rollout():
    print(cfg)
    if "kinematic_function" in cfg['experiment'].keys():
        kinematic_fn = cfg['experiment']['kinematic_function']
        print("setting kinematic function", kinematic_fn)
        robot_name = cfg['robot']['robots'][0]
        if 'robot_dh' in cfg['robot'].keys():
            robot_dh_name = cfg['robot']['robot_dh']
        else:
            robot_dh_name = cfg['robot']['robots'][0]


    env_type = cfg['experiment']['env_type']
    env = build_env(cfg['robot'], cfg['robot']['frame_stack'], skip_state_keys=skip_state_keys, env_type=env_type, default_camera=args.camera)
    if 'eval_seed' in cfg['experiment'].keys():
        eval_seed = cfg['experiment']['eval_seed'] + 1000
    else:
        eval_seed = cfg['experiment']['seed'] + 1000
    if args.frames: cam_dim = (240,240,3)
    else:
       cam_dim = (0,0,0)
 
    if 'eval_replay_buffer_size' in cfg['experiment'].keys():
        eval_replay_buffer_size = cfg['experiment']['eval_replay_buffer_size']
    else:
        eval_replay_buffer_size =  env.max_timesteps*args.num_eval_episodes
    print('running eval for %s steps'%eval_replay_buffer_size)
 
    savebase = '_show_%06d'%(eval_replay_buffer_size)
    replay_file = savebase+'.pkl' 
    movie_file = savebase+'_%s.mp4' %args.camera
    #if not os.path.exists(replay_file):
    if 1:
        replay_buffer = build_replay_buffer(cfg, env, eval_replay_buffer_size, cam_dim, eval_seed)
 
        rewards, replay_buffer = run(env, replay_buffer, cfg, cam_dim, savebase)
        pickle.dump(replay_buffer, open(replay_file, 'wb'))
        plt.figure()
        plt.plot(rewards)
        plt.title('eval episode rewards')
        plt.savefig(savebase+'.png')
 
    else:
        replay_buffer = pickle.load(open(replay_file, 'rb'))
    plot_replay(env, replay_buffer, savebase, frames=args.frames)

   
if __name__ == '__main__':
    import argparse
    from glob import glob
    parser = argparse.ArgumentParser()
    parser.add_argument('--cfg', default='experiments/base_robosuite.cfg')
    parser.add_argument('--eval', action='store_true', default=False)
    parser.add_argument('--frames', action='store_true', default=False)
    parser.add_argument('--camera', default='', choices=['default', 'frontview', 'sideview', 'birdview', 'agentview'])
    parser.add_argument('--num_eval_episodes', default=2, type=int)
#    parser.add_argument('--max_eval_timesteps', default=100, type=int)
    args = parser.parse_args()
    # keys that are robot specific
   
    cfg = json.load(open(args.cfg))
    rollout()

