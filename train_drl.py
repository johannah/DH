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

from dh_utils import seed_everything, normalize_joints, skip_state_keys
from robosuite.wrappers import JacoSim2RealWrapper
from utils import build_replay_buffer, build_env, build_model, plot_replay, get_rot_mat
from IPython import embed


"""
eef_rot_offset?
https://github.com/ARISE-Initiative/robosuite/blob/fc3738ca6361db73376e4c9d8a09b0571167bb2d/robosuite/models/robots/manipulators/manipulator_model.py
https://github.com/ARISE-Initiative/robosuite/blob/65d3b9ad28d6e7a006e9eef7c5a0330816483be4/robosuite/environments/manipulation/single_arm_env.py#L41
"""
#def reacher_kinematic_fn(action, state, body, next_body):
#    bs,fs = action.shape
#    n_joints = len(robot_dh.npdh['DH_a'])
#    # turn relative action to abs action
#    # env.obs_keys = ['position', to_target', 'velocity']
#    joint_position = action+torch.FloatTensor(state[:,:2])
#    eef_rot = robot_dh.torch_angle2ee(robot_dh.base_matrix, joint_position)
#    eef_pos = eef_rot[:,:2,3]
#    st_target = n_joints+19+3
#    target_pos = next_body[:,st_target:st_target+16].reshape(bs, 4, 4)[:,:2,3]
#    target_pos = torch.FloatTensor(target_pos)
#    return eef_pos, target_pos
#
#def jaco_kinematic_fn(action, state, body, next_body):
#    # last dim is gripper
#    bs = action.shape[0]
#    n_joints = len(robot_dh.npdh['DH_a'])
#    # turn relative action to abs action
#    joint_position = action[:, :n_joints] + torch.FloatTensor(body[:, :n_joints])
#    eef_rot = robot_dh.torch_angle2ee(robot_dh.base_matrix, joint_position)
#    eef_pos = eef_rot[:,:3,3]
#
#    # second body n_joints + 3 + 16 + 3 = 29
#    handle_rot = next_body[:,29:].reshape(bs, 4, 4)
#    handle_pos = torch.FloatTensor(handle_rot[:,:3,3])
#    return eef_pos, handle_pos

def eval_policy(eval_env, policy, kwargs, eval_episodes=10):

    total_rewards = []
    for _ in range(eval_episodes):
        done = False
        state, body = eval_env.reset()
        ep_reward = 0
        while not done:
            action = policy.select_action(np.array(state)).clip(-kwargs['max_action'], kwargs['max_action'])
            state, body, reward, done, _ = eval_env.step(action)
            ep_reward += reward
        print(ep_reward)
        total_rewards.append(ep_reward)
    avg_reward = np.mean(total_rewards)

    print("---------------------------------------")
    print(f"Evaluation over {eval_episodes} episodes: {avg_reward:.3f}")
    print("---------------------------------------")
    return total_rewards

def run_train(env, eval_env, model, replay_buffer, kwargs, savedir, exp_name, start_timesteps, eval_freq, num_steps=0, max_timesteps=2000, use_frames=False, expl_noise=0.1, batch_size=128, num_eval_episodes=10):
    tb_writer = SummaryWriter(savedir)
    steps = 0
    evaluations = []
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


            if env_type == 'dm_control':
                # we are working on joint position and don't have a joint position controller
                target_joint_position = body[:len(action)] + action
                env.sim.data.qpos[:len(action)] = target_joint_position
                next_state, next_body, reward, done, info = env.step(0*action)
            else:
                next_state, next_body, reward, done, info = env.step(action)
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
            #if num_steps > 2000:
                loss_dict = policy.train(num_steps, replay_buffer, batch_size=batch_size)
                tb_writer.add_scalars('DRLloss', loss_dict, num_steps)
            if not num_steps % eval_freq:
                step_filepath = os.path.join(savedir, '{}_{:010d}'.format(exp_name, num_steps))
                policy.save(step_filepath+'.pt')
                # evaluate
                eval_rewards = eval_policy(eval_env, policy, kwargs, num_eval_episodes)
                evaluations.append(eval_rewards)
                tb_writer.add_scalar('eval', np.mean(eval_rewards), num_steps)
                np.save(f"{savedir}/evaluations", evaluations)

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
        while not done:
            # Select action randomly or according to policy
            action = (
                    policy.select_action(state)
                ).clip(-kwargs['max_action'], kwargs['max_action'])


            print(action)
            if robot_name == 'reacher':
                target_joint_position = body[:len(action)] + action
                env.sim.data.qpos[:len(action)] = target_joint_position
                next_state, next_body, reward, done, info = env.step(action)
            else:
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
            #print(action)
            #print(torques[-1])
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
    #if "kinematic_function" in cfg['experiment'].keys():
    #    kinematic_fn = cfg['experiment']['kinematic_function']
    #    print("setting kinematic function", kinematic_fn)
    #    robot_name = cfg['robot']['robots'][0]
    #    if 'robot_dh' in cfg['robot'].keys():
    #        robot_dh_name = cfg['robot']['robot_dh']
    #    else:
    #        robot_dh_name = cfg['robot']['robots'][0]


    env_type = cfg['experiment']['env_type']
    # TODO find skip_state_keys -

    env = build_env(cfg['robot'], cfg['robot']['frame_stack'], skip_state_keys=skip_state_keys, env_type=env_type, default_camera=args.camera)
    if 'eval_seed' in cfg['experiment'].keys():
        eval_seed = cfg['experiment']['eval_seed'] + 1000
    else:
        eval_seed = cfg['experiment']['seed'] + 1000
    if args.frames: cam_dim = (240,240,3)
    else:
       cam_dim = (0,0,0)

    #if 'eval_replay_buffer_size' in cfg['experiment'].keys():
    #    eval_replay_buffer_size = cfg['experiment']['eval_replay_buffer_size']
    #else:
    eval_replay_buffer_size =  args.max_eval_timesteps*args.num_eval_episodes
    print('running eval for %s steps'%eval_replay_buffer_size)

    policy, kwargs = build_model(cfg['experiment']['policy_name'], env, cfg)

    #if 'kinematic' in cfg['experiment']['policy_name']:
    #    policy.kinematic_fn = eval(kinematic_fn)
    #    policy.kine_loss_weight = cfg['experiment']['kine_loss_weight']
    #    policy.kine_loss_stop = cfg['experiment']['kine_loss_stop']
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

def rollout_real():
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
    #if "kinematic_function" in cfg['experiment'].keys():
    #    kinematic_fn = cfg['experiment']['kinematic_function']
    #    print("setting kinematic function", kinematic_fn)
    #    robot_name = cfg['robot']['robots'][0]
    #    if 'robot_dh' in cfg['robot'].keys():
    #        robot_dh_name = cfg['robot']['robot_dh']
    #    else:
    #        robot_dh_name = cfg['robot']['robots'][0]


    env_type = cfg['experiment']['env_type']
    # TODO find skip_state_keys -

    env = build_env(cfg['robot'], cfg['robot']['frame_stack'], skip_state_keys=skip_state_keys, env_type=env_type, default_camera=args.camera)
    env.env = JacoSim2RealWrapper(env.env)
    if 'eval_seed' in cfg['experiment'].keys():
        eval_seed = cfg['experiment']['eval_seed'] + 1000
    else:
        eval_seed = cfg['experiment']['seed'] + 1000
    if args.frames: cam_dim = (240,240,3)
    else:
       cam_dim = (0,0,0)

    #if 'eval_replay_buffer_size' in cfg['experiment'].keys():
    #    eval_replay_buffer_size = cfg['experiment']['eval_replay_buffer_size']
    #else:
    eval_replay_buffer_size =  args.max_eval_timesteps*args.num_eval_episodes
    print('running eval for %s steps'%eval_replay_buffer_size)

    policy,  kwargs = build_model(cfg['experiment']['policy_name'], env, cfg)

    #if 'kinematic' in cfg['experiment']['policy_name']:
    #    policy.kinematic_fn = eval(kinematic_fn)
    #    policy.kine_loss_weight = cfg['experiment']['kine_loss_weight']
    #    policy.kine_loss_stop = cfg['experiment']['kine_loss_stop']
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
    parser.add_argument('--real', action='store_true', default=False)
    parser.add_argument('--frames', action='store_true', default=False)
    parser.add_argument('--camera', default='', choices=['default', 'frontview', 'sideview', 'birdview', 'agentview'])
    parser.add_argument('--load_model', default='')
    parser.add_argument('--num_eval_episodes', default=10, type=int)
    parser.add_argument('--max_eval_timesteps', default=100, type=int)
    args = parser.parse_args()
    # keys that are robot specific

    if args.eval and args.real:
        rollout_real()
    elif args.eval:
        rollout()
    else:
        cfg = json.load(open(args.cfg))
        print(cfg)
        env_type = cfg['experiment']['env_type']
        seed_everything(cfg['experiment']['seed'])
        random_state = np.random.RandomState(cfg['experiment']['seed'])
        env = build_env(cfg['robot'], cfg['robot']['frame_stack'], skip_state_keys=skip_state_keys, env_type=env_type, default_camera=args.camera)
        eval_env = build_env(cfg['robot'], cfg['robot']['frame_stack'], skip_state_keys=skip_state_keys,
                             env_type=cfg['experiment']['env_type'], default_camera=args.camera)
        savedir = make_savedir(cfg)
        policy, kwargs = build_model(cfg['experiment']['policy_name'], env, cfg)
        replay_buffer = build_replay_buffer(cfg, env, cfg['experiment']['replay_buffer_size'], cam_dim=(0,0,0), seed=cfg['experiment']['seed'])
        #robot_name = cfg['robot']['robots'][0]
        #if 'robot_dh' in cfg['robot'].keys():
        #    robot_dh_name = cfg['robot']['robot_dh']
        #else:
        #    robot_dh_name = cfg['robot']['robots'][0]
        #robot_dh = robotDH(robot_name=robot_name, device='cpu')
        #robot_dh.base_matrix = torch.FloatTensor(replay_buffer.base_matrix)

        #if "kinematic_function" in cfg['experiment'].keys():
        #    kinematic_fn = cfg['experiment']['kinematic_function']
        #    policy.kine_loss_weight = cfg['experiment']['kine_loss_weight']
        #    policy.kine_loss_stop = cfg['experiment']['kine_loss_stop']
        #    print("setting kinematic function", kinematic_fn)
        #    policy.kinematic_fn = eval(kinematic_fn)

        run_train(env, eval_env, policy, replay_buffer, kwargs, savedir, cfg['experiment']['exp_name'], cfg['experiment']['start_training'], cfg['experiment']['eval_freq'], num_steps=0, max_timesteps=cfg['experiment']['max_timesteps'], expl_noise=cfg['experiment']['expl_noise'], batch_size=cfg['experiment']['batch_size'], num_eval_episodes=cfg['experiment']['n_eval_episodes'])

