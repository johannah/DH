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
from imageio import mimwrite
from utils import EnvStackRobosuite
from replay_buffer_TD3 import ReplayBuffer, compress_frame
from torch.utils.tensorboard import SummaryWriter
from copy import deepcopy
import torch
from IPython import embed
import pickle
import TD3
from dh_utils import seed_everything, normalize_joints
from dh_utils import robotDH, quaternion_matrix, quaternion_from_matrix, robot_attributes
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


def run_train(env, model, replay_buffer, kwargs, savedir, exp_name, start_timesteps, save_every, num_steps=0, num_episodes=2000, use_frames=False, expl_noise=0.1, batch_size=128):
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

def run_eval(env, policy, replay_buffer, kwargs, cfg, cam_dim, savebase):

    robot_name = cfg['robot']['robots'][0]
    num_steps = 0
    total_steps = replay_buffer.max_size-1
    use_frames = cam_dim[0] > 0
    if use_frames:
        print('recording camera: %s'%args.camera)

    h, w, c = cam_dim
    rewards = []
    joint_positions = []
    next_joint_positions = []
    while num_steps < total_steps:
        #ts, reward, d, o = env.reset()
        done = False
        state, body =  env.reset()
        if use_frames:
            frame_compressed = compress_frame(env.env.sim.render(camera_name=args.camera,height=h, width=w)[::-1])
        ep_reward = 0
        e_step = 0

        # IT SEEMS LIKE BASE_POS DOESNT CHANGE for DOOR/Jaco - will need to change things up if it does
        print(env.env.robots[0].base_pos)
        print(env.env.robots[0].base_ori)
        while not done:
            # Select action randomly or according to policy
            joint_positions.append(env.env.robots[0]._joint_positions)
            action = (
                    policy.select_action(state)
                ).clip(-kwargs['max_action'], kwargs['max_action'])
 
            next_state, next_body, reward, done, info = env.step(action) # take a random action
            next_joint_positions.append(env.env.robots[0]._joint_positions)
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
        
    n, ss = replay_buffer.states.shape
    bpos = env.env.robots[0].base_pos
    bori = env.env.robots[0].base_ori 
    replay_buffer.base_pos = bpos
    replay_buffer.base_ori = bori
    replay_buffer.k = env.k
    replay_buffer.obs_keys = env.obs_keys
    replay_buffer.obs_sizes = env.obs_sizes
    replay_buffer.obs_specs = env.obs_specs
    # hard code orientation 
    # TODO add conversion to rotation matrix
    base_matrix = quaternion_matrix(bori)
    base_matrix[:3, 3] = bpos
    # TODO this is hacky - but it seems the world needs to be flipped in y,z to be correct
    # Sanity checked in Jaco w/ Door  
    # ensure this holds for other robots
    base_matrix[1,1] = -1 
    base_matrix[2,2] = -1
    replay_buffer.base_matrix = base_matrix

    joint_positions = np.array(joint_positions)
    next_joint_positions = np.array(next_joint_positions)
    
    norm_joint_positions = normalize_joints(deepcopy(joint_positions))
    next_norm_joint_positions = normalize_joints(deepcopy(next_joint_positions))

    replay_buffer.norm_joint_positions = joint_positions
    replay_buffer.joint_positions = norm_joint_positions
    replay_buffer.next_norm_joint_positions = next_joint_positions
    replay_buffer.next_joint_positions = next_norm_joint_positions
    replay_buffer.robot_name = robot_name
    replay_buffer.cfg = cfg
    pickle.dump(replay_buffer, open(savebase + '.pkl', 'wb'))
    plt.figure()
    plt.plot(rewards)
    plt.title('eval episode rewards')
    plt.savefig(savebase+'.png')
 
    # find eef position according to DH
    if robot_name in robot_attributes.keys():
        n, ss = replay_buffer.states.shape
        k = replay_buffer.k
        idx = (k-1)*(ss//k) # start at most recent observation
        data = {}
        for key in replay_buffer.obs_keys:
            o_size = env.obs_sizes[key]
            data[key] = replay_buffer.states[:, idx:idx+o_size]
            idx += o_size

        rdh = robotDH(robot_name)
        f_eef = rdh.np_angle2ee(base_matrix, norm_joint_positions)
        # do the rotation in the beginning rather than end
        #f_eef = np.array([np.dot(base_matrix, r_eef[x]) for x in range(n)])
        dh_pos = f_eef[:,:3,3] 
        dh_ori = np.array([quaternion_from_matrix(f_eef[x]) for x in range(n)])

        f, ax = plt.subplots(3, figsize=(10,18))
        ax[0].plot(data['robot0_eef_pos'][:,0], label='state')
        ax[0].plot(dh_pos[:,0], label='dh calc')
        ax[0].set_title('pos x')
        ax[1].plot(data['robot0_eef_pos'][:,1])
        ax[1].plot(dh_pos[:,1])
        ax[1].set_title('pos y')
        ax[2].plot(data['robot0_eef_pos'][:,2])
        ax[2].plot(dh_pos[:,2])
        ax[2].set_title('pos z')
        plt.legend()
        plt.savefig(savebase+'eef.png')
        """
         From Robosuite paper
         rotations from the current end-effector orientation in the form of axis-angle coordinates (rx, ry, rz), where the direction represents the axis and the magnitude
         represents the angle (in radians). Note that for OSC, the rotation axes are taken
         relative to the global world coordinate frame, whereas for IK, the rotation axes
         are taken relative to the end-effector frame, NOT the global world coordinate
         frame!
        """
 
        # TODO quaternion is still not right! the errors occur when i hit 1 or 0 - this must be a common thing
        # CHECK DH parameters?
        f, ax = plt.subplots(4, figsize=(10,18))
        ax[0].plot(data['robot0_eef_quat'][:,0], label='sqx')
        ax[0].plot(dh_ori[:,0], label='dhqx')
        ax[0].set_title('qx')
        ax[1].plot(data['robot0_eef_quat'][:,1], label='sqy')
        ax[1].plot(dh_ori[:,1], label='dhqy')
        ax[1].set_title('qy')
        ax[2].plot(data['robot0_eef_quat'][:,2], label='sqz')
        ax[2].plot(dh_ori[:,2], label='dhqz')
        ax[2].set_title('qz')
        ax[3].plot(data['robot0_eef_quat'][:,3], label='sqw')
        ax[3].plot(dh_ori[:,3], label='dhqw')
        ax[3].set_title('qw')
        plt.legend()
        plt.savefig(savebase+'quat.png')
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
    print(cfg)
    env = build(cfg['robot'])
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
        eval_replay_buffer_size =  cfg['robot']['horizon']*args.num_eval_episodes
    print('running eval for %s steps'%eval_replay_buffer_size)
 
    policy, replay_buffer, kwargs = make_model(cfg['experiment'], env, eval_replay_buffer_size, cam_dim=cam_dim)
    savebase = load_model.replace('.pt','_eval_%06d_S%06d'%(eval_replay_buffer_size, eval_seed))
    replay_file = savebase+'.pkl' 
    movie_file = savebase+'_%s.mp4' %args.camera
    if not os.path.exists(replay_file):
        policy.load(load_model)
        rewards, replay_buffer = run_eval(env, policy, replay_buffer, kwargs, cfg, cam_dim, savebase)
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
    parser.add_argument('--num_eval_episodes', default=10, type=int)
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
        run_train(env, policy, replay_buffer, kwargs, savedir, cfg['experiment']['exp_name'], cfg['experiment']['start_training'], cfg['experiment']['eval_freq'], num_steps=0, num_episodes=2000, expl_noise=cfg['experiment']['expl_noise'], batch_size=cfg['experiment']['batch_size'])

