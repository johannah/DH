from comet_ml import Experiment

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import json
from imageio import mimwrite
import pickle
import numpy as np
from copy import deepcopy
import time
import os, sys
import numpy as np
import shutil
from pathlib import Path


import torch
# torch.set_num_threads(2)
import torch.nn as nn
from torch.nn.utils import clip_grad_norm_
import torch.optim as optim
import torch.nn.init as init
import torch.nn.functional as F
from torch.utils.tensorboard import SummaryWriter

from utils import build_env, build_model, build_replay_buffer, plot_replay, get_replay_state_dict, get_hyperparameters, parse_slurm_task_bc
from replay_buffer import compress_frame
from dh_utils import find_latest_checkpoint, create_results_dir, skip_state_keys, mean_angle_btw_vectors, so3_relative_angle
from dh_utils import robotDH, robotDHLearnable, seed_everything, normalize_joints
from dh_utils import load_robosuite_data, get_data_norm_params, quaternion_from_matrix, quaternion_matrix, robot_attributes
from logger import Logger

from IPython import embed 

class LSTM(nn.Module):
    def __init__(self, input_size=14, output_size=7, hidden_size=1024):
        super(LSTM, self).__init__()
        self.input_size = input_size
        self.output_size = output_size
        self.hidden_size = hidden_size
        self.lstm1 = nn.LSTMCell(input_size, hidden_size)
        self.lstm2 = nn.LSTMCell(hidden_size, hidden_size)
        self.linear = nn.Linear(hidden_size, self.output_size)

    def forward(self, xt, h1_tm1, c1_tm1, h2_tm1, c2_tm1):
        h1_t, c1_t = self.lstm1(xt, (h1_tm1, c1_tm1))
        h2_t, c2_t = self.lstm2(h1_t, (h2_tm1, c2_tm1))
        output = self.linear(h2_t)
        return output, h1_t, c1_t, h2_t, c2_t

def forward_pass(x, phase='train'):
    #input_data = (input_data-train_mean)/train_std
    if phase == 'train':
        input_noise = torch.normal(torch.zeros_like(x), noise_std*torch.ones_like(x))
        x = x + input_noise
    bs = x.shape[1]
    ts = x.shape[0]
    h1_tm1 = torch.zeros((bs, hidden_size)).to(device)
    c1_tm1 = torch.zeros((bs, hidden_size)).to(device)
    h2_tm1 = torch.zeros((bs, hidden_size)).to(device)
    c2_tm1 = torch.zeros((bs, hidden_size)).to(device)
    y_pred = torch.zeros((ts, bs, output_size)).to(device)
    for step_num, i in enumerate(np.arange(ts)):
        output, h1_tm1, c1_tm1, h2_tm1, c2_tm1 = lstm(x[i], h1_tm1, c1_tm1, h2_tm1, c2_tm1)
        y_pred[i] = y_pred[i] + output
    return y_pred

def train(data, step=0, n_epochs=1e7):
    # todo add running avg for loss
    train_loss = 0; valid_loss = 0
    for epoch in range(n_epochs):
        for phase in ['valid', 'train']:
            # time, batch, features
            n_samples = data[phase]['states'].shape[1]
            indexes = np.arange(0, n_samples)
            random_state.shuffle(indexes)
            st = 0
            en = min([st+batch_size, n_samples])
            bs = en-st
            batch_cnt = 0
            while en <= n_samples and bs > 0:
                opt.zero_grad()
                if args.learn_dh:
                    dh_opt.zero_grad()
                joints = torch.FloatTensor(data[phase]['joints'][:,indexes[st:en]]).to(device)
                target_pos = torch.FloatTensor(data[phase]['target_pos'][:,indexes[st:en]]).to(device)
                target_rot = torch.FloatTensor(data[phase]['target_rot'][:,indexes[st:en]]).to(device)
                target_diff = torch.FloatTensor(data[phase]['actions'][:,indexes[st:en]]).to(device)

                x = torch.FloatTensor(data[phase]['states'][:,indexes[st:en]]).to(device)
                pred_diff = torch.tanh(forward_pass(x, phase))
                ts,bs,f = pred_diff.shape 
                pred_jt = pred_diff + joints
                pred_rot_mat = robot_dh.torch_angle2ee(base_matrix, pred_jt.contiguous().view(ts*bs,f)).contiguous().view(ts,bs,4,4)
                pred_pos = pred_rot_mat[:,:,:3,3]
                pred_rot = pred_rot_mat[:,:,:3,:3]
                if args.loss == 'DH':
                    dh_pos_loss = args.alpha*criterion(pred_pos, target_pos)
                    if args.drop_rot:
                        # DONT USE ROTATION
                        dh_rot_loss = mean_angle_btw_vectors(pred_rot.detach().contiguous().view(ts*bs,3,3), 
                                                             target_rot.detach().contiguous().view(ts*bs,3,3))
                        dh_loss = dh_pos_loss
                    else:
                        dh_rot_loss = mean_angle_btw_vectors(pred_rot.contiguous().view(ts*bs,3,3), 
                                                             target_rot.contiguous().view(ts*bs,3,3))
                        dh_loss = dh_pos_loss + dh_rot_loss
                    loss = dh_loss 
                    with torch.no_grad():
                        joint_loss = criterion(pred_diff.detach(), target_diff)
                elif args.loss == 'angle':
                    joint_loss = criterion(pred_diff, target_diff)
                    loss = joint_loss
                    with torch.no_grad():
                        dh_pos_loss =  args.alpha*criterion(pred_pos.detach(), target_pos.detach())
                        dh_rot_loss = mean_angle_btw_vectors(pred_rot.detach().contiguous().view(ts*bs,3,3), 
                                                             target_rot.detach().contiguous().view(ts*bs,3,3))
                        dh_loss = dh_pos_loss + dh_rot_loss
                loss_dict = {'dh_pos_%s'%(phase):dh_pos_loss,
                             'dh_rot_%s'%(phase):dh_rot_loss,
                             'dh_%s'%(phase):dh_loss,
                             'jt_%s'%(phase):joint_loss}
 
                if phase == 'train':
                    clip_grad_norm_(lstm.parameters(), grad_clip)
                    train_loss = loss
                    loss.backward()
                    opt.step()
                    if args.learn_dh:
                        dh_opt.step()
                        estimation_error_dict = robot_dh.get_dh_estimation_error()
                        estimated_dh_params = robot_dh.get_estimated_dh_params()
                        L.log('dh_estimation_error', estimation_error_dict, step)
                        L.log('dh_estimated_params', estimated_dh_params, step)

                    if not step % (bs*10):
                        L.log('BC_loss', loss_dict, step)
                    step+=bs
                else:
                    valid_loss = loss

                st = en
                en = min([st+batch_size, n_samples+1])
                bs = en-st
                batch_cnt +=1
            
            L.log('BC_loss', loss_dict, step)
 
        print('{} epoch:{} step:{} loss:{}'.format(phase, epoch, step, loss))
        if not epoch % save_every_epochs:
            model_dict = {'model':lstm.state_dict(), 'train_cnt':step}
            fbase = os.path.join(savebase, 'lstm_model_%010d'%(step))
            print('saving model', fbase)
            torch.save(model_dict, fbase+'.pt') 
    model_dict = {'model':lstm.state_dict(), 'train_cnt':step}
    fbase = os.path.join(savebase, 'lstm_model_%010d'%(step))
    print('saving model', fbase)
    torch.save(model_dict, fbase+'.pt') 


def load_data(use_states=['position', 'to_target']):
    # ASSUMES DATA DOES NOT WRAP (IT DOESN"T IN EVAL)
    print('loading data from', args.load_replay)
    replay = pickle.load(open(args.load_replay, 'rb'))
    cfg = replay.cfg
    starts = np.array(replay.episode_start_steps[:-1], dtype=np.int)
    random_state.shuffle(starts)
 
    max_ts = int(replay.max_timesteps)
    # bodies is joint_angles + world eef_pos + rotation in base
    j_size = replay.next_bodies.shape[1]-19
    # action is joint_diff
    if cfg['experiment']['env_type'] == 'robosuite':
        gripper = np.zeros((max_ts, len(starts), 1)) 

    actions = np.zeros((max_ts, len(starts), j_size)) 
    joints = np.zeros((max_ts, len(starts), j_size)) 
    target_joints = np.zeros((max_ts, len(starts), j_size)) 
    target_pos = np.zeros((max_ts, len(starts), 3)) 
    target_rot = np.zeros((max_ts, len(starts), 3, 3)) 
    state_data, next_state_data = get_replay_state_dict(replay, use_states)
    state_len = state_data['state'].shape[1]
    states = np.zeros((max_ts, len(starts), state_len)) 
    replay.frames_enabled = False
    #_n_eef = replay.next_states[:,data_idx['robot0_eef_pos'][0]:data_idx['robot0_eef_pos'][1]]
    # get pos, and quat for 
    sts = state_data['state']
    jts = replay.bodies[:,:-19]
    next_jts = replay.next_bodies[:,:-19]
    next_world_pos = replay.next_bodies[:,-19:-16]
    n_ts = replay.next_bodies.shape[0]
    next_mat = replay.next_bodies[:, -16:].reshape(n_ts, 4,4)
    next_pos = next_mat[:,:3,3]
    next_rot = next_mat[:,:3,:3]
    if cfg['experiment']['env_type'] == 'robosuite':
        grip = replay.actions[:,-1][:,None]
    #target_ee = robot_dh.angle2ee(torch.FloatTensor(target_joints).to(device)).cpu().numpy()
    # put data in time, batch, features
    for xx, s in enumerate(starts):
        # TODO hack to make same
        indexes = np.arange(s, s+max_ts, dtype=np.int)
        states[:,xx] = sts[s:s+max_ts]
        diff = next_jts[s:s+max_ts] - jts[s:s+max_ts]
        actions[:,xx] = diff
        joints[:,xx] = jts[s:s+max_ts]
        target_joints[:,xx] = next_jts[s:s+max_ts]
        target_pos[:,xx] = next_pos[s:s+max_ts]
        target_rot[:,xx] = next_rot[s:s+max_ts]
        if cfg['experiment']['env_type'] == 'robosuite':
            gripper[:,xx] = grip[s:s+max_ts]
    # position, to_target, velocity
    n_episodes = target_pos.shape[1]
    # take first episodes as test
    st_val = max([1,int(n_episodes*.15)])

    data = {'train':{}, 'valid':{}}
    data['valid']['states'] =  states[:max_ts,:st_val]
    data['train']['states'] =  states[:max_ts,st_val:]
    data['valid']['actions'] =  actions[:max_ts,:st_val]
    data['train']['actions'] =  actions[:max_ts,st_val:]
    data['valid']['joints'] =  joints[:max_ts,:st_val]
    data['train']['joints'] =  joints[:max_ts,st_val:]
    data['valid']['target_joints'] =  target_joints[:max_ts,:st_val]
    data['train']['target_joints'] =  target_joints[:max_ts,st_val:]
    data['valid']['target_pos'] =  target_pos[:max_ts,:st_val]
    data['train']['target_pos'] =  target_pos[:max_ts,st_val:]
    data['valid']['target_rot'] =  target_rot[:max_ts,:st_val]
    data['train']['target_rot'] =  target_rot[:max_ts,st_val:]
    if cfg['experiment']['env_type'] == 'robosuite':
        data['valid']['gripper'] =  gripper[:max_ts,:st_val]
        data['train']['gripper'] =  gripper[:max_ts,st_val:]

    data['base_matrix'] = replay.base_matrix 
    print('diffs')
    print((next_jts-jts).min(0))
    print((next_jts-jts).max(0))
    return data

def setup_eval():
   
    print('loading model: %s'%load_model)
    cfg['robot']['controller_config_file'] = 'configs/%s_joint_position.json'%args.target_robot_name.lower()
    env = build_env(cfg['robot'], k=1, skip_state_keys=skip_state_keys, env_type=cfg['experiment']['env_type'], default_camera=args.camera)    
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
    replay_buffer = build_replay_buffer(cfg, env, eval_replay_buffer_size, cam_dim, eval_seed)
 
    savebase = load_model.replace('.pt','_BC_eval_%06d_S%06d'%(eval_replay_buffer_size, eval_seed))
    replay_file = savebase+'.pkl' 
    movie_file = savebase+'_%s.mp4' %args.camera
  
    if not os.path.exists(replay_file) or args.force:
        rewards, replay_buffer = run_BC_eval(env, replay_buffer, cfg, cam_dim, savebase)
        pickle.dump(replay_buffer, open(replay_file, 'wb'))
        plt.figure()
        plt.plot(rewards)
        plt.title('eval episode rewards')
        plt.savefig(savebase+'.png')

    else:
        replay_buffer = pickle.load(open(replay_file, 'rb'))
    plot_replay(env, replay_buffer, savebase)
    if args.frames:
        frames = [replay_buffer.undo_frame_compression(replay_buffer.frames[f]) for f in np.arange(len(replay_buffer.frames))]
        mimwrite(movie_file, frames, fps=100)

def run_BC_eval(env, replay_buffer, cfg, cam_dim, savebase):
    robot_name = cfg['robot']['robots'][0]
    target_robot_name = cfg['experiment']['target_robot_name']
    num_steps = 0
    total_steps = replay_buffer.max_size-1
    use_frames = cam_dim[0] > 0
    if use_frames:
        print('recording camera: %s'%args.camera)

    h, w, c = cam_dim
    rewards = []
    
    data_action_trace = data['train']['actions'][:,0]
    if cfg['experiment']['env_type'] == 'robosuite':
        data_grip_trace = data['train']['gripper'][:,0]
    with torch.no_grad():
        while num_steps < total_steps:
            #ts, reward, d, o = env.reset()
            done = False
            state, body =  env.reset()
            if use_frames:
                frame_compressed = compress_frame(env.render(camera_name=args.camera,height=h, width=w))
            ep_reward = 0
            e_step = 0

            # IT SEEMS LIKE BASE_POS DOESNT CHANGE for DOOR/Jaco - will need to change things up if it does
            ts = replay_buffer.max_timesteps
            base_x = torch.zeros((ts, 1, input_size)).to(device)
            h1_tm1 = torch.zeros((1, hidden_size)).to(device)
            c1_tm1 = torch.zeros((1, hidden_size)).to(device)
            h2_tm1 = torch.zeros((1, hidden_size)).to(device)
            c2_tm1 = torch.zeros((1, hidden_size)).to(device)
            action_pred = torch.zeros((ts, 1, output_size)).to(device)

            #ex_trace = data['train']['states'][:,0:1]
            #ex_action = data['train']['actions'][:,0]
            switch = False
            while not done:
                # Select action randomly or according to policy
                # TODO select state names properly this only works for reacher
                base_x[e_step] = torch.FloatTensor(state[:4])
                output, h1_tm1, c1_tm1, h2_tm1, c2_tm1 = lstm(base_x[e_step], h1_tm1, c1_tm1, h2_tm1, c2_tm1)
                #output, h1_tm1, c1_tm1, h2_tm1, c2_tm1 = lstm(base_x[e_step], h1_tm1, c1_tm1, h2_tm1, c2_tm1)
                #if e_step < 250:
                #    pred_action = data_action_trace[e_step]
                #else:
                #    pred_action = output[0].cpu().numpy()
                pred_action = output[0].cpu().numpy()
                if cfg['experiment']['env_type'] == 'robosuite':
                    fake_grip = data_grip_trace[e_step]
                    action = np.hstack((pred_action, fake_grip))
                else:
                    action = pred_action
                env.sim.data.qpos[:len(pred_action)] = body[:-19]+pred_action
                action = np.zeros_like(action)
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
                e_step += 1
                num_steps+=1
            rewards.append(ep_reward)
    print('TOTAL REWARDS', np.sum(rewards))
    return rewards, replay_buffer
 

if __name__ == '__main__':
    import argparse
    from glob import glob
    parser = argparse.ArgumentParser()
    parser.add_argument('--load_replay')
    parser.add_argument('--target_robot_name', default='')
    parser.add_argument('--learn_dh', action='store_true', default=False)
    parser.add_argument('--eval', default=False, action='store_true')
    parser.add_argument('--force', default=False, action='store_true')
    parser.add_argument('--load_model', default='')
    parser.add_argument('--loss', default='DH', choices=['DH', 'angle'])
    parser.add_argument('--frames', action='store_true', default=False)
    parser.add_argument('--camera', default='')
    parser.add_argument('--num_eval_episodes', default=3, type=int)
    parser.add_argument('--learning_rate', default=0.0001, type=float)
    parser.add_argument('--alpha', default=2000, type=int)
    parser.add_argument('--drop_rot', default=False, action='store_true')
    parser.add_argument('--noise', default=1, type=float)
    parser.add_argument('--dh_noise', default=0.1, type=float)
    parser.add_argument('--use_comet', action='store_true', default=False)
    parser.add_argument('--slurm_task_id', default=-1, type=int)
    args = parser.parse_args()
    seed = 323
    seed_everything(seed)
    random_state = np.random.RandomState(seed)
    # some keys are robot-specifig!
    # TODO log where data was trained

    if args.slurm_task_id != -1:
        # overwrites args.load_replay to make the rest of script unchanged
        slurm_load_replay = parse_slurm_task_bc(args.load_replay, args.slurm_task_id)
        args.load_replay = slurm_load_replay

    if args.load_model != '':
        if os.path.isdir(args.load_model):
            load_model = sorted(glob(os.path.join(args.load_model, '*.pt')))[-1]
            load_dir = args.load_model
        else:
            assert args.load_model.endswith('.pt')
            load_model = args.load_model
            load_dir, model_name = os.path.split(args.load_model)

        agent_load_dir = os.path.split(os.path.split(os.path.split(load_dir)[0])[0])[0]
        args.load_replay = os.path.split(os.path.split(load_dir)[0])[0]+'.pkl'
    else:
        agent_load_dir, fname = os.path.split(args.load_replay)
        _, ddir = os.path.split(agent_load_dir)
        exp_name = 'BC_state_%s_lr%s_N%s_ROT%s'%(args.loss, args.learning_rate, args.noise, int(not args.drop_rot))

    agent_cfg_path = os.path.join(agent_load_dir, 'cfg.txt')
    print('cfg', agent_cfg_path)
    if not os.path.exists(agent_cfg_path):
        agent_cfg_path = os.path.join(agent_load_dir, 'cfg.cfg')
    cfg = json.load(open(agent_cfg_path))

    # Sahand: dirty hack to retrieve the correct seed the RL was trained on
    # Because I forgot to save the updated cfg after the SLURM task change in train_rl.py
    slurm_rl_args_path = os.path.join(agent_load_dir, 'args.json')
    if os.path.exists(slurm_rl_args_path):
        slurm_rl_args = json.load(open(slurm_rl_args_path))
        cfg['experiment']['seed'] = slurm_rl_args['seed']

    cfg['experiment']['target_robot_name'] = args.target_robot_name
    cfg['experiment']['bc_seed'] = cfg['experiment']['seed'] + random_state.randint(10000)
    cfg['robot']['controller'] = "JOINT_POSITION" 

    if args.target_robot_name == '':
        args.target_robot_name = cfg['robot']['robots'][0]
        if 'robot_dh' in cfg['robot'].keys():
            args.target_robot_name = cfg['robot']['robot_dh']
        else:
            args.target_robot_name = cfg['robot']['robots'][0]
    print('target robot', args.target_robot_name)
    cfg['robot']['robots'] = [args.target_robot_name]

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    results_dir = args.load_replay.replace('.pkl', '')
    # set random seed to 0
    noise_std = args.noise
    grad_clip = 3
    hidden_size = 512
    batch_size = 32
    save_every_epochs = 100

    if args.learn_dh:
        robot_dh = robotDHLearnable(robot_name=args.target_robot_name, dh_noise=args.dh_noise, device=device)
    else:
        robot_dh = robotDH(robot_name=args.target_robot_name, device=device)

    data = load_data()
    base_matrix = torch.FloatTensor((data['base_matrix'])).to(device)
    #_t,_b,_f = _d.shape
    #train_mean, train_std = get_data_norm_params(_d.reshape(_t*_b,_f), device=device)
     
    input_size = data['valid']['states'].shape[2]
    output_size =  data['valid']['actions'].shape[2]

    lstm = LSTM(input_size=input_size, output_size=output_size, hidden_size=hidden_size).to(device)
    criterion = nn.MSELoss()

    if not args.eval and args.load_model == '':
        step = 0 
        savebase = create_results_dir(exp_name, results_dir=results_dir)
    else:
        if os.path.isdir(args.load_model): 
            savebase = args.load_model 
            loadpath = find_latest_checkpoint(args.load_model)
        else:
            loadpath  = args.load_model
            savebase = os.path.split(args.load_model)[0]
            
    if args.load_model != '':
        print("LOADING MODEL FROM", loadpath) 
        modelbase = loadpath.replace('.pt', '_')
        load_dict = torch.load(loadpath, map_location=device)
        step = load_dict['train_cnt']
        lstm.load_state_dict(load_dict['model'])

    if args.eval:
        setup_eval()
    else:
        pickle.dump(data, open(savebase+'_data.pkl', 'wb'))
        L = Logger(savebase, use_tb=True, use_comet=args.use_comet, project_name="DH")
        hyperparameters = get_hyperparameters(args, cfg)
        L.log_hyper_params(hyperparameters)
        # use LBFGS as optimizer since we can load the whole data to train
        opt = optim.Adam(lstm.parameters(), lr=0.0001)
        if args.learn_dh:
            dh_opt = optim.Adam(robot_dh.parameters(), lr=0.001)
            print(f"Total number of DH parameters to learn: {sum(p.numel() for p in robot_dh.parameters())}")
        train(data, step, n_epochs=int(1e7))
