import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

import torch
import torch.nn as nn
from torch.nn.utils.clip_grad import clip_grad_norm
import torch.optim as optim
import torch.nn.init as init
import torch.nn.functional as F

import pickle
import numpy as np
from copy import deepcopy
import time
import os, sys
import numpy as np
import shutil

from utils import find_latest_checkpoint, create_results_dir, plot_losses
from utils import robotDH, angle2sincos, sincos2angle
from utils import load_robosuite_data, get_data_norm_params
from IPython import embed 
import replay_buffer
import imageio

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
    input_data = (input_data-train_mean)/train_std
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

def train(data, step=0, n_epochs=10000):
    for epoch in range(n_epochs):
        for phase in ['valid', 'train']:
            # time, batch, features
            n_samples = data[phase]['inputs'].shape[1]
            indexes = np.arange(0, n_samples)
            random_state.shuffle(indexes)
            st = 0
            en = min([st+batch_size, n_samples])
            running_loss = 0
            bs = en-st
            while en <= n_samples and bs > 0:
                opt.zero_grad()
                x = torch.FloatTensor(data[phase]['inputs'][:,indexes[st:en]]).to(device)

                diff_pred = forward_pass(x)
                if args.loss == 'DH':
                    joints = torch.FloatTensor(data[phase]['joints'][:,indexes[st:en]]).to(device)
                    jt_pred = diff_pred + joints
                    ee_target = torch.FloatTensor(data[phase]['target_ee'][:,indexes[st:en]]).to(device)
                    ee_pred = robot_dh.angle2ee(jt_pred)
                    jt_loss = criterion(ee_pred, ee_target)
                elif args.loss == 'angle':
                     diff_targets = torch.FloatTensor(data[phase]['target_states'][:,indexes[st:en]]).to(device)
                     jt_loss = criterion(diff_pred, diff_targets)

                loss = jt_loss 
                if phase == 'train':
                    clip_grad_norm(lstm.parameters(), grad_clip)
                    loss.backward()
                    step+=bs
                    opt.step()
                st = en
                en = min([st+batch_size, n_samples+1])
                bs = en-st
                running_loss += loss.item()

            avg_loss = running_loss/n_samples
            losses[phase].append([step, avg_loss])  
            print('{} epoch:{} step:{} loss:{}'.format(phase, epoch, step, avg_loss))
        if not epoch % save_every_epochs:
            model_dict = {'model':lstm.state_dict(), 'train_cnt':step}
            fbase = os.path.join(savebase, 'model_%010d'%(step))
            print('saving model', fbase)
            torch.save(model_dict, fbase+'.pt') 
            np.savez(fbase+'_losses.npz', train=losses['train'], valid=losses['valid'] )

 
def forward_pass_eval(input_data, teacher_force=True, lead_in=10):
    if teacher_force:
        x = input_data
    else:
        x = torch.zeros_like(input_data)
        # action input
        x[:,:,2:]= input_data[:,:,2:]
        # joint position input
        x[:lead_in,:,:2] = input_data[:lead_in,:,:2]
 
    bs = x.shape[1]
    ts = x.shape[0]
    h1_tm1 = torch.zeros((bs, hidden_size)).to(device)
    c1_tm1 = torch.zeros((bs, hidden_size)).to(device)
    h2_tm1 = torch.zeros((bs, hidden_size)).to(device)
    c2_tm1 = torch.zeros((bs, hidden_size)).to(device)
    y_pred = torch.zeros((ts, bs, output_size)).to(device)
    for step_num, i in enumerate(np.arange(ts)):
        xi = x[i]
        if not teacher_force and step_num > lead_in and step_num > 0:
            xi[:,:2] = output
        xi = (xi-train_mean)/train_std
        output, h1_tm1, c1_tm1, h2_tm1, c2_tm1 = lstm(xi, h1_tm1, c1_tm1, h2_tm1, c2_tm1)
        y_pred[i] = y_pred[i] + output
    return y_pred


def eval_model(data, phases=['train', 'valid'], n=20, shuffle=False, teacher_force=False, lead_in=10):
    """
    phase: list containing phases to evaluate ['train', 'valid']
    n int num examples to evaluate / plot. if None, all are plotted
    """
    #env = 'reacher'
    #task = 'easy'
    #env = suite.load(args.env, args.task)
    for phase in phases:
        # time, batch, features
        n_samples = data[phase]['inputs'].shape[1]
        indexes = np.arange(0, n_samples)
        if shuffle:
            random_state.shuffle(indexes)
        # determine how many samples to take
        if n == None:
             n = n_samples
        else:
             n = min([n, n_samples])
        st = 0
        en = min([st+batch_size, n])
        bs = en-st
        while en <= n and bs > 0:
            with torch.no_grad():

                x = torch.FloatTensor(data[phase]['inputs'][:,indexes[st:en]]).to(device)
                diff_pred = forward_pass_eval(x, teacher_force=teacher_force, lead_in=lead_in)
                joints = torch.FloatTensor(data[phase]['joints'][:,indexes[st:en]]).to(device)
                jt_pred = diff_pred + joints
                ee_pred = robot_dh.angle2ee(jt_pred)

            ee_target = data[phase]['target_ee'][:,indexes[st:en]]
            ee_pred = ee_pred.cpu().detach().numpy()

            for ii in range(ee_pred.shape[1]):
                plt.figure()
                plt.scatter(ee_target[0:1,ii,0], ee_target[0:1,ii,1], c='g', marker='o')
                plt.scatter(ee_target[-2:-1,ii,0], ee_target[-2:-1,ii,1], c='r', marker='x')
                plt.scatter(ee_target[:,ii,0], ee_target[:,ii,1], marker='o')
                plt.scatter(ee_pred[:,ii,0], ee_pred[:,ii,1], label='pred')
                plt.legend()
                if args.teacher_force:
                    fname = modelbase+'%s_tf_%04d.png'%(phase, ii)
                else:
                    fname = modelbase+'%s_li%02d_%04d.png'%(phase, args.lead_in, ii)
                print('plotting', fname)
                plt.savefig(fname)
                plt.close()
            st = en
            en = min([st+batch_size, n+1])
            bs = en-st


def plot_ee(ee, frames):
    rep_base = args.load_replay.replace('.pkl', '_plot_norm_angle_ee')
    if not os.path.exists(rep_base):
        os.makedirs(rep_base)
    n_eps = ee.shape[1]
    c = np.arange(ee.shape[0])
    for b in range(n_eps):
        plt.figure()
        plt.scatter(ee[:,b,0], ee[:,b,1], c=c)
        plt.savefig(os.path.join(rep_base, 'ee_%06d.png'%b))
        plt.close()
        imageio.mimsave(os.path.join(rep_base, 'ee_%06d.gif'%b), frames[:,b], fps=30)
     
    mov_search = os.path.join(rep_base, 'ee_%06d.png')
    mov_out = os.path.join(rep_base, '_out.mp4')
    cmd = 'ffmpeg -i %s -c:v libx264 -vf fps=25 -pix_fmt yuv420p %s'%(mov_search, mov_out)
    os.system(cmd)

def normalize_angles(angles):
    """
    :param np.array of angles in radians: (float)
    :return: (float) Angle in radian in [-pi, pi]
    """
    return (angles + np.pi) % (2 * np.pi) - np.pi

def load_data():
    # TODO - this is a hack - data is shorter than replay buffer, so we don't actually hit the boundary
    replay = pickle.load(open(args.load_replay, 'rb'))
    replay.frame_enabled=False
    starts = np.array(replay.episode_start_steps[:-1], dtype=np.int)
    random_state.shuffle(starts)
 
    # median first reward happens at ts 19, max occurs at 642. Choose a smallish number to start with
    max_ts = 100
    states = np.zeros((max_ts, len(starts), 4)) 
    #target_states = np.zeros((max_ts, len(starts), 4)) 
    target_states = np.zeros((max_ts, len(starts), 2)) 
    joints = np.zeros((max_ts, len(starts), 2)) 
    target_joints = np.zeros((max_ts, len(starts), 2)) 
    target_ee = np.zeros((max_ts, len(starts), 2)) 
    #frames = []
    for xx, s in enumerate(starts):
        # TODO hack to make same
        indexes = np.arange(s, s+max_ts, dtype=np.int)
        st, act, r, nst, nd, fr, nfr = replay.get_indexes(indexes)
        diff = nst[:,:2]-st[:,:2]
        states[1:,xx,:2] = diff[:-1] # good
        states[:,xx,2:] = act  # good
        target_states[:,xx] = diff
        joints[:,xx] = st[:,:2]
        target_joints[:,xx] = nst[:,:2]

        #target_states[:,xx]= ns[:,2:]
        #frames.append(fr)
    
    #frames = np.array(frames).transpose(1,0,2,3,4)
    target_ee = robot_dh.angle2ee(torch.FloatTensor(target_joints).to(device)).cpu().numpy()
    # position, to_target, velocity
    n_episodes = target_ee.shape[1]
    
    st_val = max([1,int(n_episodes*.15)])

    data = {'train':{}, 'valid':{}}
    data['valid']['inputs'] =  states[:,:st_val]
    data['train']['inputs'] =  states[:,st_val:]
    data['valid']['target_states'] =  target_states[:,:st_val]
    data['train']['target_states'] =  target_states[:,st_val:]
    data['valid']['joints'] =  joints[:,:st_val]
    data['train']['joints'] =  joints[:,st_val:]
    data['valid']['target_joints'] =  target_joints[:,:st_val]
    data['train']['target_joints'] =  target_joints[:,st_val:]
    data['valid']['target_ee'] =  target_ee[:,:st_val]
    data['train']['target_ee'] =  target_ee[:,st_val:]
    #data['valid']['frames'] =  frames[:,:st_val]
    #data['train']['frames'] =  frames[:,st_val:]
    #plot_ee(target_ee, frames)
    return data


if __name__ == '__main__':
    import argparse
    from glob import glob
    parser = argparse.ArgumentParser()
    parser.add_argument('--load_replay', default='results/21-02-02_reacher_easy_00000_05/reacher_easy_00000_S00000_0000980000eval_NE01200.pkl')
    parser.add_argument('--eval', default=False, action='store_true')
    parser.add_argument('-tf', '--teacher_force', default=False, action='store_true')
    parser.add_argument('--load_model', default='')
    parser.add_argument('--loss', default='DH')
    parser.add_argument('--device', default='cuda')
    parser.add_argument('-li', '--lead_in', default=5, type=int)
    parser.add_argument('--robot_name', default='dm_reacher', choices=['jaco27DOF', 'dm_reacher'])
    args = parser.parse_args()
    exp_name = 'v5_lstm_act_%s'%(args.loss)
    
    device = args.device
    results_dir = 'results'
    # set random seed to 0
    seed = 323
    noise_std = 4
    grad_clip = 5
    hidden_size = 1024
    batch_size = 32
    save_every_epochs = 100

    robot_dh = robotDH(robot_name=args.robot_name, device=device)
    np.random.seed(seed)
    torch.manual_seed(seed)
    random_state = np.random.RandomState(seed)
    losses = {'train':[], 'valid':[]}


    #data = load_robosuite_data('square.npy', random_state=random_state)


    data = load_data()
   
    _d = data['train']['inputs']
    _t,_b,_f = _d.shape
    train_mean, train_std = get_data_norm_params(_d.reshape(_t*_b,_f), device=device)
    input_size = 4
    output_size = 2

    lstm = LSTM(input_size=input_size, output_size=output_size, hidden_size=hidden_size).to(device)

    if not args.eval and args.load_model == '':
        savebase = create_results_dir(exp_name, results_dir=results_dir)
        step = 0
    else:
        if os.path.isdir(args.load_model):
            savebase = args.load_model
            loadpath = find_latest_checkpoint(args.load_model)
        else:
            loadpath  = args.load_model
            savebase = os.path.split(args.load_model)[0]
            
        modelbase = loadpath.replace('.pt', '_')
        load_dict = torch.load(loadpath, map_location=device)
        step = load_dict['train_cnt']
        lstm.load_state_dict(load_dict['model'])

    if args.eval:
        plot_losses(loadpath.replace('.pt', '_losses.npz'))
        eval_model(data, phases=['train', 'valid'], n=20, shuffle=False)
    else:
        criterion = nn.MSELoss()
        # use LBFGS as optimizer since we can load the whole data to train
        opt = optim.Adam(lstm.parameters(), lr=0.0001)
        train(data, step,  n_epochs=10000)
     
