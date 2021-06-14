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
from utils import build_replay_buffer, build_env, build_model, plot_replay
from IPython import embed


"""
eef_rot_offset? 
https://github.com/ARISE-Initiative/robosuite/blob/fc3738ca6361db73376e4c9d8a09b0571167bb2d/robosuite/models/robots/manipulators/manipulator_model.py
https://github.com/ARISE-Initiative/robosuite/blob/65d3b9ad28d6e7a006e9eef7c5a0330816483be4/robosuite/environments/manipulation/single_arm_env.py#L41
"""
def reacher_kinematic_fn(joint_position, state):
    eef_rot = robot_dh.torch_angle2ee(robot_dh.base_matrix, joint_position)
    # obs is position, to_target, velocity
    eef_pos = eef_rot[:,:2,3]
    target_pos = eef_pos.detach() + state[:,2:4]
    return eef_pos, target_pos

def run_train(env, model, replay_buffer, kwargs, savedir, exp_name, start_timesteps, save_every, num_steps=0, max_timesteps=2000, use_frames=False, expl_noise=0.1, batch_size=128):
    tb_writer = SummaryWriter(savedir)
    steps = 0
    while num_steps < max_timesteps:
        #ts, reward, d, o = env.reset()
        done = False
        state, body =  env.reset()
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
     
 
            target_joint_position = body[:len(action)] + action
            env.sim.data.qpos[:len(action)] = target_joint_position
            next_state, next_body, reward, done, info = env.step(0*action) # take a random action
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
                critic_loss, actor_loss = policy.train(num_steps, replay_buffer, batch_size=batch_size)
                if actor_loss != 0:
                    tb_writer.add_scalar('actor_loss', actor_loss, num_steps)
                tb_writer.add_scalar('critic_loss', critic_loss, num_steps)
            if not num_steps % save_every:
                step_filepath = os.path.join(savedir, '{}_{:010d}'.format(exp_name, num_steps))
                policy.save(step_filepath+'.pt')
            num_steps+=1
            e_step+=1
        tb_writer.add_scalar('train_reward', ep_reward, num_steps)
        

    step_filepath = os.path.join(savedir, '{}_{:010d}'.format(exp_name, num_steps))
    pickle.dump(replay_buffer, open(step_filepath+'.pkl', 'wb'))
    policy.save(step_filepath+'.pt')
 
def make_savedir(cfg):
    cnt = 0

    savedir = os.path.join(cfg['experiment']['log_dir'], "%s_%s_%05d_%s_%s_%02d"%(cfg['experiment']['exp_name'], 
                                                                        cfg['robot']['env_name'],  cfg['experiment']['seed'], 
                                                                        cfg['robot']['robots'][0], cfg['robot']['controller'],  cnt))
    while len(glob(os.path.join(savedir, '*.pt'))):
        cnt +=1
        savedir = os.path.join(cfg['experiment']['log_dir'], "%s_%s_%05d_%s_%s_%02d"%(cfg['experiment']['exp_name'], 
                                                                        cfg['robot']['env_name'],  cfg['experiment']['seed'], 
                                                                        cfg['robot']['robots'][0], cfg['robot']['controller'],  cnt))
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
        while not done and e_step < args.max_eval_timesteps:
            # Select action randomly or according to policy
            action = (
                    policy.select_action(state)
                ).clip(-kwargs['max_action'], kwargs['max_action'])
 
            next_state, next_body, reward, done, info = env.step(action) # take a random action
            ep_reward += reward
            if e_step+1 == args.max_eval_timesteps:
                done = True
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
    if args.frames: cam_dim = (240,240,3)
    else:
       cam_dim = (0,0,0)
 
    if 'eval_replay_buffer_size' in cfg['experiment'].keys():
        eval_replay_buffer_size = cfg['experiment']['eval_replay_buffer_size']
    else:
        eval_replay_buffer_size =  int(min([env.max_timesteps, args.max_eval_timesteps])*args.num_eval_episodes)
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
    parser.add_argument('--num_eval_episodes', default=30, type=int)
    parser.add_argument('--max_eval_timesteps', default=100, type=int)
    args = parser.parse_args()
    # keys that are robot specific
    
    robot_dh = robotDH(robot_name='reacher', device='cpu')
    if args.eval:
        rollout()
    else:
        cfg = json.load(open(args.cfg))
        print(cfg)
        seed_everything(cfg['experiment']['seed'])
        random_state = np.random.RandomState(cfg['experiment']['seed'])
        env = build_env(cfg['robot'], cfg['robot']['frame_stack'], skip_state_keys=skip_state_keys, env_type=cfg['experiment']['env_type'], default_camera=args.camera)
        savedir = make_savedir(cfg)
        policy, kwargs = build_model(cfg['experiment']['policy_name'], env)
        policy.kinematic_fn = reacher_kinematic_fn

        replay_buffer = build_replay_buffer(cfg, env, cfg['experiment']['replay_buffer_size'], cam_dim=(0,0,0), seed=cfg['experiment']['seed'])
 
        robot_dh.base_matrix = torch.FloatTensor(replay_buffer.base_matrix)
        run_train(env, policy, replay_buffer, kwargs, savedir, cfg['experiment']['exp_name'], cfg['experiment']['start_training'], cfg['experiment']['eval_freq'], num_steps=0, max_timesteps=cfg['experiment']['max_timesteps'], expl_noise=cfg['experiment']['expl_noise'], batch_size=cfg['experiment']['batch_size'])

