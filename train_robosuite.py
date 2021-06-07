import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import robosuite
import imageio
import numpy as np
import os
from glob import glob
from gym.spaces import Box, Discrete
from gym import Env
import robosuite as suite
from robosuite.wrappers.gym_wrapper import GymWrapper
import json
from dh_utils import seed_everything
from imageio import mimwrite
from utils import EnvStackRobosuite
from replay_buffer_TD3 import ReplayBuffer, compress_frame
from torch.utils.tensorboard import SummaryWriter
import torch
from IPython import embed
import pickle
import TD3

import robosuite.utils.macros as macros
macros.IMAGE_CONVENTION = 'opencv'
torch.set_num_threads(4)

def build(cfg):
    controller_configs = suite.load_controller_config(default_controller=cfg['controller'])
    env = EnvStackRobosuite(suite.make(env_name=cfg['env_name'], 
                     robots=cfg['robots'], 
                     controller_configs=controller_configs,
                     use_camera_obs=cfg['use_camera_obs'], 
                     use_object_obs=cfg['use_object_obs'], 
                     reward_shaping=cfg['reward_shaping'], 
                     camera_names=cfg['camera_names'], 
                     horizon=cfg['horizon'], 
                     control_freq=cfg['control_freq'], 
                     ignore_done=False, 
                     hard_reset=False, 
                     reward_scale=1.0,
                       ), k=cfg['frame_stack'])
    
    return env

def make_model(cfg, env, replay_buffer_size, cam_dim=(0,0,0)):
    state_dim = env.observation_space.shape[0]
    action_dim = env.control_shape
    max_action = env.control_max
    min_action = env.control_min
    if cfg['policy_name'] == 'TD3':
        kwargs = {'tau':0.005, 
                'action_dim':action_dim, 'state_dim':state_dim, 
                'policy_noise':0.2, 'max_policy_action':1.0, 
                'noise_clip':0.5, 'policy_freq':2, 
                'discount':0.99, 'max_action':max_action, 'min_action':min_action}
        policy = TD3.TD3(**kwargs)

    replay_buffer = ReplayBuffer(state_dim, action_dim, 
                                 max_size=replay_buffer_size, 
                                 cam_dim=cam_dim, 
                                 seed=cfg['seed'])
 
    return policy, replay_buffer, kwargs


def run_train(env, model, replay_buffer, kwargs, savedir, exp_name, start_timesteps, save_every, num_steps=0, num_episodes=1000, use_frames=False, expl_noise=0.1, batch_size=128):
    tb_writer = SummaryWriter(savedir)
    for ep in range(num_episodes):
        #ts, reward, d, o = env.reset()
        done = False
        state, body =  env.reset()
        if use_frames:
            frame_compressed = compress_frame(env.physics.render(camera_name=args.camera, height=h, width=w)[::-1])
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
                next_frame_compressed = compress_frame(env.physics.render(camera_name=args.camera, height=h, width=w)[::-1])
     
                replay_buffer.add(state, action, reward, next_state, done, 
                              frame_compressed=frame_compressed, 
                              next_frame_compressed=next_frame_compressed)
                frame_compressed = next_frame_compressed
            else:
                replay_buffer.add(state, action, reward, next_state, done)
     
            state = next_state
            body = next_body
            if num_steps > start_timesteps:
                critic_loss, actor_loss = policy.train(num_steps, replay_buffer, batch_size)
                if actor_loss != 0:
                    tb_writer.add_scalar('actor_loss', actor_loss, num_steps)
                tb_writer.add_scalar('critic_loss', critic_loss, num_steps)
            if not num_steps % save_every:
                step_filepath = os.path.join(savedir, '{}_{:010d}'.format(exp_name, num_steps))
                policy.save(step_filepath+'.pt')
            num_steps+=1
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

def run_eval(env, policy, replay_buffer, kwargs, cam_dim):
    num_steps = 0
    total_steps = replay_buffer.max_size-1
    use_frames = cam_dim[0] > 0
    if use_frames:
        print('recording camera: %s'%args.camera)

    h, w, c = cam_dim
    rewards = []
    while num_steps < total_steps:
        #ts, reward, d, o = env.reset()
        done = False
        state, body =  env.reset()
        if use_frames:
            frame_compressed = compress_frame(env.env.sim.render(camera_name=args.camera,height=h, width=w)[::-1])
        ep_reward = 0
        e_step = 0
        while not done:
            # Select action randomly or according to policy
            action = (
                    policy.select_action(state)
                ).clip(-kwargs['max_action'], kwargs['max_action'])
 
            next_state, next_body, reward, done, info = env.step(action) # take a random action
            ep_reward += reward
            if use_frames:
               
                next_frame_compressed = compress_frame(env.env.sim.render(camera_name=args.camera, height=h, width=w)[::-1])
                replay_buffer.add(state, action, reward, next_state, done, 
                              frame_compressed=frame_compressed, 
                              next_frame_compressed=next_frame_compressed)
                frame_compressed = next_frame_compressed
            else:
                replay_buffer.add(state, action, reward, next_state, done)
     
            state = next_state
            body = next_body
            num_steps+=1
        rewards.append(ep_reward)
        

    return rewards, replay_buffer
 
def rollout():
    if os.path.isdir(args.load):
        load_model = sorted(glob(os.path.join(args.load, '*.pt')))[-1]
        cfg_path = os.path.join(args.load, 'cfg.cfg')
    else:
        assert args.load.endswith('.pt')
        load_model = args.load
        load_dir, model_name = os.path.split(args.load)

    load_dir, model_name = os.path.split(load_model)
    print('loading model: %s'%load_model)
    cfg_path = os.path.join(load_dir, 'cfg.txt')
    if not os.path.exists(cfg_path):
        cfg_path = os.path.join(load_dir, 'cfg.cfg')
    print('loading cfg: %s'%cfg_path)
    cfg = json.load(open(cfg_path))
    if 'eval_seed' in cfg['experiment'].keys():
        eval_seed = cfg['experiment']['eval_seed'] + 1000
    else:
        eval_seed = cfg['experiment']['seed'] + 1000
    if 'eval_replay_buffer_size' in cfg['experiment'].keys():
        eval_replay_buffer_size = cfg['experiment']['eval_replay_buffer_size']
    else:
        eval_replay_buffer_size =  cfg['robot']['horizon']*10
    if args.frames: cam_dim = (240,240,3)
    else:
       cam_dim = (0,0,0)
        
    print(cfg)
    env = build(cfg['robot'])
    policy, replay_buffer, kwargs = make_model(cfg['experiment'], env, eval_replay_buffer_size, cam_dim=cam_dim)
    savebase = load_model.replace('.pt','_eval_%06d_S%06d'%(eval_replay_buffer_size, eval_seed))
    replay_file = savebase+'.pkl' 
    movie_file = savebase+'_%s.mp4' %args.camera
    if not os.path.exists(replay_file):
        policy.load(load_model)
        rewards, replay_buffer = run_eval(env, policy, replay_buffer, kwargs, cam_dim)
        pickle.dump(replay_buffer, open(replay_file, 'wb'))
        plt.plot(rewards)
        plt.savefig(savebase+'.png')
        if args.frames:
            _, _, _, _, _, frames, next_frames = replay_buffer.get_indexes(np.arange(len(replay_buffer.frames)))
            mimwrite(movie_file, frames)

if __name__ == '__main__':
    import argparse
    from glob import glob
    parser = argparse.ArgumentParser()
    parser.add_argument('--cfg', default='robo_config/base.cfg')
    parser.add_argument('--eval', action='store_true', default=False)
    parser.add_argument('--frames', action='store_true', default=False)
    parser.add_argument('--camera', default='agentview', choices=['frontview', 'sideview', 'birdview', 'agentview'])
    parser.add_argument('--load', default='')
    args = parser.parse_args()
    
    if args.eval:
        rollout()
    else:
        cfg = json.load(open(args.cfg))
        print(cfg)
        seed_everything(cfg['experiment']['seed'])
        random_state = np.random.RandomState(cfg['experiment']['seed'])
        env = build(cfg['robot'])
        savedir = make_savedir(cfg)
        policy, replay_buffer, kwargs = make_model(cfg['experiment'], env, cfg['experiment']['replay_buffer_size'])
        run_train(env, policy, replay_buffer, kwargs, savedir, cfg['experiment']['exp_name'], cfg['experiment']['start_training'], cfg['experiment']['eval_freq'], num_steps=0, num_episodes=1000, expl_noise=cfg['experiment']['expl_noise'], batch_size=cfg['experiment']['batch_size'])

