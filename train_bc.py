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


import torch
torch.set_num_threads(2)
import torch.nn as nn
from torch.nn.utils.clip_grad import clip_grad_norm
import torch.optim as optim
import torch.nn.init as init
import torch.nn.functional as F
from torch.utils.tensorboard import SummaryWriter

from utils import build_env, build_model, build_replay_buffer, plot_replay
from replay_buffer import compress_frame
from dh_utils import find_latest_checkpoint, create_results_dir
from dh_utils import robotDH, robotDHLearnable, seed_everything, normalize_joints
from dh_utils import load_robosuite_data, get_data_norm_params, quaternion_from_matrix, quaternion_matrix, robot_attributes

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

def forward_pass(input_data):
    #input_data = (input_data-train_mean)/train_std
    input_noise = torch.normal(torch.zeros_like(input_data), noise_std*torch.ones_like(input_data))
    x = input_data + input_noise
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

def train(data, step=0, n_epochs=2000):
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
            while en <= n_samples and bs > 0:
                opt.zero_grad()
                if args.learn_dh:
                    dh_opt.zero_grad()
                x = torch.FloatTensor(data[phase]['states'][:,indexes[st:en]]).to(device)
                pred_diff = torch.tanh(forward_pass(x))
                #pred_diff = forward_pass(x)
                ts,bs,f = pred_diff.shape 
                if args.loss == 'DH':
                    joints = torch.FloatTensor(data[phase]['joints'][:,indexes[st:en]]).to(device)
                    pred_jt = pred_diff + joints
                    target_pos = torch.FloatTensor(data[phase]['target_pos'][:,indexes[st:en]]).to(device)
                    pred_rot_mat = robot_dh.torch_angle2ee(base_matrix, pred_jt.contiguous().view(ts*bs,f)).contiguous().view(ts,bs,4,4)
                    pred_pos = pred_rot_mat[:,:,:3,3]
                    loss = criterion(pred_pos, target_pos)
                elif args.loss == 'angle':
                    target_diff = torch.FloatTensor(data[phase]['actions'][:,indexes[st:en]]).to(device)
                    loss = criterion(pred_diff, target_diff)

                if phase == 'train':
                    clip_grad_norm(lstm.parameters(), grad_clip)
                    loss.backward()
                    step+=bs
                    opt.step()
                    if args.learn_dh:
                        dh_opt.step()
                    train_loss = loss
                else:
                    valid_loss = loss
                if not step % (bs*10):
                    tb_writer.add_scalars('BC_loss',{'%s_train'%(args.loss):train_loss, '%s_valid'%(args.loss):valid_loss}, step)

                st = en
                en = min([st+batch_size, n_samples+1])
                bs = en-st
            tb_writer.add_scalars('BC_loss',{'%s_train'%(args.loss):train_loss, '%s_valid'%(args.loss):valid_loss}, step)
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
 
def load_data():
    # ASSUMES DATA DOES NOT WRAP (IT DOESN"T IN EVAL)
    print('loading data from', args.load_replay)
    replay = pickle.load(open(args.load_replay, 'rb'))
    cfg = replay.cfg
    starts = np.array(replay.episode_start_steps[:-1], dtype=np.int)
    random_state.shuffle(starts)
 
    max_ts = int(replay.max_timesteps)
    # bodies is joint_angles + eef_pose
    j_size = replay.next_bodies.shape[1]-7
    # action is joint_diff
    if cfg['experiment']['env_type'] == 'robosuite':
        gripper = np.zeros((max_ts, len(starts), 1)) 

    actions = np.zeros((max_ts, len(starts), j_size)) 
    joints = np.zeros((max_ts, len(starts), j_size)) 
    target_joints = np.zeros((max_ts, len(starts), j_size)) 
    target_pos = np.zeros((max_ts, len(starts), 3)) 
    target_quat = np.zeros((max_ts, len(starts), 4)) 
    n, ss = replay.states.shape
    k = replay.k
    idx = (k-1)*(ss//k) # start at most recent observation
    data_idx = {}

    states = np.zeros((max_ts, len(starts), ss)) 
    replay.frames_enabled = False
    #_n_eef = replay.next_states[:,data_idx['robot0_eef_pos'][0]:data_idx['robot0_eef_pos'][1]]
    # get pos, and quat for 
    sts = replay.states
    jts = replay.bodies[:,:-7]
    next_jts = replay.next_bodies[:,:-7]
    next_pos = replay.next_bodies[:,-7:-7+3]
    next_quat = replay.next_bodies[:,-7+3:]
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
        target_quat[:,xx] = next_quat[s:s+max_ts]
        if cfg['experiment']['env_type'] == 'robosuite':
            gripper[:,xx] = grip[s:s+max_ts]
    # position, to_target, velocity
    n_episodes = target_pos.shape[1]
    # take first episodes as test
    st_val = max([1,int(n_episodes*.15)])

    data = {'train':{}, 'valid':{}}
    data['valid']['states'] =  states[:,:st_val]
    data['train']['states'] =  states[:,st_val:]
    data['valid']['actions'] =  actions[:,:st_val]
    data['train']['actions'] =  actions[:,st_val:]
    data['valid']['joints'] =  joints[:,:st_val]
    data['train']['joints'] =  joints[:,st_val:]
    data['valid']['target_joints'] =  target_joints[:,:st_val]
    data['train']['target_joints'] =  target_joints[:,st_val:]
    data['valid']['target_pos'] =  target_pos[:,:st_val]
    data['train']['target_pos'] =  target_pos[:,st_val:]
    data['valid']['target_quat'] =  target_quat[:,:st_val]
    data['train']['target_quat'] =  target_quat[:,st_val:]
    if cfg['experiment']['env_type'] == 'robosuite':
        data['valid']['gripper'] =  gripper[:,:st_val]
        data['train']['gripper'] =  gripper[:,st_val:]

    data['base_matrix'] = replay.base_matrix 
    print('actions', actions.max(), actions.min())
    return data

def setup_eval():
   
    print('loading model: %s'%load_model)
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
    if not os.path.exists(replay_file):
        rewards, replay_buffer = run_BC_eval(env, replay_buffer, cfg, cam_dim, savebase)
        pickle.dump(replay_buffer, open(replay_file, 'wb'))
        plt.figure()
        plt.plot(rewards)
        plt.title('eval episode rewards')
        plt.savefig(savebase+'.png')

    else:
        replay_buffer = pickle.load(open(replay_file, 'rb'))
    plot_replay(replay_buffer, savebase)
    if args.frames:
        frames = [replay_buffer.undo_frame_compression(replay_buffer.frames[f]) for f in np.arange(len(replay_buffer.frames))]
        mimwrite(movie_file, frames)

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
            while not done:
                # Select action randomly or according to policy
                base_x[e_step] = torch.FloatTensor(state)
                output, h1_tm1, c1_tm1, h2_tm1, c2_tm1 = lstm(base_x[e_step], h1_tm1, c1_tm1, h2_tm1, c2_tm1)
                pred_action = output[0].cpu().numpy()
                if cfg['experiment']['env_type'] == 'robosuite':
                    fake_grip = np.ones(1)
                    action = np.hstack((pred_action, fake_grip))
                else:
                    action = pred_action
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
    return rewards, replay_buffer
 

if __name__ == '__main__':
    import argparse
    from glob import glob
    parser = argparse.ArgumentParser()
    parser.add_argument('--load_replay')
    parser.add_argument('--target_robot_name', default='')
    parser.add_argument('--learn_dh', action='store_true', default=False)
    parser.add_argument('--eval', default=False, action='store_true')
    parser.add_argument('--load_model', default='')
    parser.add_argument('--loss', default='DH', choices=['DH', 'angle'])
    parser.add_argument('--device', default='cuda')
    parser.add_argument('--frames', action='store_true', default=False)
    parser.add_argument('--camera', default='')
    parser.add_argument('--num_eval_episodes', default=10, type=int)
    args = parser.parse_args()
    seed = 323
    seed_everything(seed)
    random_state = np.random.RandomState(seed)
    # some keys are robot-specifig!
    skip_state_keys = ['robot0_joint_pos_cos', 'robot0_joint_pos_sin','robot0_joint_vel', 'robot0_proprio-state']
    # TODO log where data was trained 

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
        exp_name = 'roboBC_act_%s'%(args.loss)

    agent_cfg_path = os.path.join(agent_load_dir, 'cfg.txt')
    print('cfg', agent_cfg_path)
    if not os.path.exists(agent_cfg_path):
        agent_cfg_path = os.path.join(agent_load_dir, 'cfg.cfg')
    cfg = json.load(open(agent_cfg_path))
    cfg['experiment']['target_robot_name'] = args.target_robot_name
    cfg['experiment']['bc_seed'] = cfg['experiment']['seed'] + random_state.randint(10000)
    cfg['robot']['controller'] = "JOINT_POSITION" 
    if args.target_robot_name == '':
        args.target_robot_name = cfg['robot']['robots'][0]
    print('setting target robot', args.target_robot_name)
    cfg['robot']['robots'] = [args.target_robot_name] 

     
    device = args.device
    results_dir = args.load_replay.replace('.pkl', '')
    # set random seed to 0
    noise_std = 3
    grad_clip = 5
    hidden_size = 1024
    batch_size = 32
    save_every_epochs = 100

    # TODO
    if args.learn_dh:
        robot_dh = robotDHLearnable(robot_name=args.target_robot_name, device=device)
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
        tb_writer = SummaryWriter(savebase)
        # use LBFGS as optimizer since we can load the whole data to train
        opt = optim.Adam(lstm.parameters(), lr=0.0001)
        if args.learn_dh:
            dh_opt = optim.Adam(robot_dh.parameters(), lr=0.001)
            print(f"Total number of DH parameters to learn: {sum(p.numel() for p in robot_dh.parameters())}")
        train(data, step, n_epochs=2000)
