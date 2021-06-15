from comet_ml import Experiment

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
import datetime

import json
from imageio import mimwrite

from replay_buffer import ReplayBuffer, compress_frame

from torch.utils.tensorboard import SummaryWriter
import torch
import robosuite.utils.macros as macros
torch.set_num_threads(3)
import TD3

from dh_utils import seed_everything, normalize_joints, skip_state_keys
from utils import build_replay_buffer, build_env, build_model, plot_replay
from logger import Logger
from IPython import embed


"""
eef_rot_offset? 
https://github.com/ARISE-Initiative/robosuite/blob/fc3738ca6361db73376e4c9d8a09b0571167bb2d/robosuite/models/robots/manipulators/manipulator_model.py
https://github.com/ARISE-Initiative/robosuite/blob/65d3b9ad28d6e7a006e9eef7c5a0330816483be4/robosuite/environments/manipulation/single_arm_env.py#L41
"""


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def eval_policy(env, policy, eval_episodes=10):

    avg_reward = 0.
    for _ in range(eval_episodes):
        done = False
        state, body = env.reset()
        while not done:
            action = policy.select_action(np.array(state))
            state, body, reward, done, _ = env.step(action)
            avg_reward += reward

    avg_reward /= eval_episodes

    print("---------------------------------------")
    print(f"Evaluation over {eval_episodes} episodes: {avg_reward:.3f}")
    print("---------------------------------------")
    return avg_reward


def run_train(env, eval_env, policy, replay_buffer, kwargs, savedir, exp_name, start_timesteps,
              eval_freq, num_steps=0, max_timesteps=2000, use_frames=False, expl_noise=0.1,
              batch_size=128, num_eval_episodes=10):
    L = Logger(savedir, use_tb=True, use_comet=args.use_comet, project_name="DH")
    evaluations = []
    steps = 0
    while num_steps < max_timesteps:
        #ts, reward, d, o = env.reset()
        done = False
        state, body = env.reset()
        if use_frames:
            frame_compressed = compress_frame(env.render(camera_name=args.camera, height=h, width=w))
        ep_reward = 0
        e_step = 0
        while not done:
            if num_steps < start_timesteps:
                action = random_state.uniform(low=kwargs['min_action'], high=kwargs['max_action'], size=kwargs['action_dim'])
            else:
                # Select action randomly or according to policy
                action = (
                        policy.select_action(state)
                        + random_state.normal(0, kwargs['max_action'] * expl_noise, size=kwargs['action_dim'])
                    ).clip(-kwargs['max_action'], kwargs['max_action'])

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
     
            state = next_state
            body = next_body
            if num_steps > start_timesteps:
                critic_loss, actor_loss = policy.train(num_steps, replay_buffer, batch_size)
                if actor_loss != 0:
                    L.log('actor_loss', actor_loss, num_steps)
                L.log('critic_loss', critic_loss, num_steps)
            if not num_steps % eval_freq:
                step_filepath = os.path.join(savedir, '{}_{:010d}'.format(exp_name, num_steps))
                policy.save(step_filepath+'.pt')

                # evaluate
                eval_reward = eval_policy(eval_env, policy, num_eval_episodes)
                evaluations.append(eval_reward)
                L.log('eval_reward', eval_reward, num_steps)
                np.save(f"{savedir}/evaluations", evaluations)

            num_steps+=1
            e_step+=1
        L.log('train_reward', ep_reward, num_steps)
        

    step_filepath = os.path.join(savedir, '{}_{:010d}'.format(exp_name, num_steps))
    pickle.dump(replay_buffer, open(step_filepath+'.pkl', 'wb'))
    policy.save(step_filepath+'.pt')
 
def make_savedir(cfg, new_log_dir=''):
    cnt = 0
    datetime_str = datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    log_dir = new_log_dir if new_log_dir else cfg['experiment']['log_dir']

    savedir = os.path.join(log_dir, "%s_%s_%05d_%s_%s_%s_%02d"%(cfg['experiment']['exp_name'],
                                                                cfg['robot']['env_name'],  cfg['experiment']['seed'],
                                                                cfg['robot']['robots'][0], cfg['robot']['controller'],
                                                                datetime_str, cnt))
    while len(glob(os.path.join(savedir, '*.pt'))):
        cnt +=1
        savedir = os.path.join(log_dir, "%s_%s_%05d_%s_%s_%s_%02d"%(cfg['experiment']['exp_name'],
                                                                    cfg['robot']['env_name'],  cfg['experiment']['seed'],
                                                                    cfg['robot']['robots'][0], cfg['robot']['controller'],
                                                                    datetime_str, cnt))
    if not os.path.exists(savedir):
        os.makedirs(savedir)
 
    os.system('cp -r %s %s'%(args.cfg, os.path.join(savedir, 'cfg.txt')))
    return savedir

def run_eval(env, policy, replay_buffer, kwargs, cfg, cam_dim, savebase):
    robot_name = cfg['robot']['robots'][0]
    num_steps = 0
    total_steps = replay_buffer.max_size-1
    use_frames = cam_dim[0] > 0
    if use_frames:
        print('recording camera: %s'%args.camera)

    h, w, c = cam_dim
    torques = []
    rewards = []
    while num_steps < total_steps:
        #ts, reward, d, o = env.reset()
        done = False
        state, body =  env.reset()
        if use_frames:
            frame_compressed = compress_frame(env.render(camera_name=args.camera, height=h, width=w))
        ep_reward = 0
        e_step = 0

        # IT SEEMS LIKE BASE_POS DOESNT CHANGE for DOOR/Jaco - will need to change things up if it does
        #print(env.env.robots[0].base_pos)
        #print(env.env.robots[0].base_ori)
        # while not done and e_step < args.max_eval_timesteps:
        while not done:
            # Select action randomly or according to policy
            action = (
                    policy.select_action(state)
                ).clip(-kwargs['max_action'], kwargs['max_action'])
 
            next_state, next_body, reward, done, info = env.step(action) # take a random action
            ep_reward += reward
            # if e_step+1 == args.max_eval_timesteps:
            #     done = True
            if use_frames:
                next_frame_compressed = compress_frame(env.render(camera_name=args.camera, height=h, width=w))
                replay_buffer.add(state, body, action, reward, next_state, next_body, done, 
                              frame_compressed=frame_compressed, 
                              next_frame_compressed=next_frame_compressed)
                frame_compressed = next_frame_compressed
            else:
                replay_buffer.add(state, body, action, reward, next_state, next_body, done)
            #torques.append(env.env.robots[0].torques)
            state = next_state
            body = next_body
            num_steps+=1
            e_step+=1
        rewards.append(ep_reward)
    #replay_buffer.torques = torques
    return rewards, replay_buffer


def rollout():
    if os.path.isdir(args.load_model):
        load_model = sorted(glob(os.path.join(args.load_model, '*.pt')))[-1]
        cfg_path = os.path.join(args.load_model, 'cfg.cfg')
    else:
        assert args.load_model.endswith('.pt')
        load_model = args.load_model
        load_dir, model_name = os.path.split(args.load_model)

    load_dir, model_name = os.path.split(load_model)
    print('loading model: %s'%load_model)
    cfg_path = os.path.join(load_dir, 'cfg.txt')
    if not os.path.exists(cfg_path):
        cfg_path = os.path.join(load_dir, 'cfg.cfg')
    print('loading cfg: %s'%cfg_path)
    cfg = json.load(open(cfg_path))
    print(cfg)
    env = build_env(cfg['robot'], cfg['robot']['frame_stack'], skip_state_keys=skip_state_keys, env_type=cfg['experiment']['env_type'], default_camera=args.camera)
    if 'eval_seed' in cfg['experiment'].keys():
        eval_seed = cfg['experiment']['eval_seed'] + 1000
    else:
        eval_seed = cfg['experiment']['seed'] + 1000
    cam_dim = (240,240,3) if args.frames else (0, 0, 0)

    if 'eval_replay_buffer_size' in cfg['experiment'].keys():
        eval_replay_buffer_size = cfg['experiment']['eval_replay_buffer_size']
    else:
        eval_replay_buffer_size = int(env.max_timesteps * args.num_eval_episodes)
    print('running eval for %s steps'%eval_replay_buffer_size)
 
    policy,  kwargs = build_model(cfg['experiment']['policy_name'], env)
    savebase = load_model.replace('.pt','_eval_%06d_S%06d'%(eval_replay_buffer_size, eval_seed))
    replay_file = savebase+'.pkl' 
    movie_file = savebase+'_%s.mp4' %args.camera
    if not os.path.exists(replay_file):
        policy.load(load_model)
        replay_buffer = build_replay_buffer(cfg, env, eval_replay_buffer_size, cam_dim, eval_seed)
 
        rewards, replay_buffer = run_eval(env, policy, replay_buffer, kwargs, cfg, cam_dim, savebase)
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
    parser.add_argument('--load_model', default='')
    parser.add_argument('--num_eval_episodes', default=10, type=int)
    parser.add_argument('--max_eval_timesteps', default=100, type=int)
    parser.add_argument('--log_dir', default='', type=str, help="Overwrites the log_dir in the config file (Needed for CC).")
    parser.add_argument('--use_comet', action='store_true', default=False)

    args = parser.parse_args()
    # keys that are robot specific
    
    if args.eval:
        rollout()
    else:
        cfg = json.load(open(args.cfg))
        print(cfg)
        seed_everything(cfg['experiment']['seed'])
        random_state = np.random.RandomState(cfg['experiment']['seed'])
        env = build_env(cfg['robot'], cfg['robot']['frame_stack'], skip_state_keys=skip_state_keys,
                        env_type=cfg['experiment']['env_type'], default_camera=args.camera)
        # TODO: Currently the seed in the evaluation is the same as the training
        eval_env = build_env(cfg['robot'], cfg['robot']['frame_stack'], skip_state_keys=skip_state_keys,
                             env_type=cfg['experiment']['env_type'], default_camera=args.camera)
        savedir = make_savedir(cfg, args.log_dir)
        policy, kwargs = build_model(cfg['experiment']['policy_name'], env)

        replay_buffer = build_replay_buffer(cfg, env, cfg['experiment']['replay_buffer_size'], cam_dim=(0,0,0),
                                            seed=cfg['experiment']['seed'])
 
        run_train(env, eval_env, policy, replay_buffer, kwargs, savedir,
                  cfg['experiment']['exp_name'], cfg['experiment']['start_training'],
                  cfg['experiment']['eval_freq'], num_steps=0, max_timesteps=cfg['experiment']['max_timesteps'],
                  expl_noise=cfg['experiment']['expl_noise'], batch_size=cfg['experiment']['batch_size'],
                  num_eval_episodes=args.num_eval_episodes)
