import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

import torch
import torch.nn as nn
from torch.nn.utils.clip_grad import clip_grad_norm
import torch.optim as optim
import torch.nn.init as init
import torch.nn.functional as F

import numpy as np
from copy import deepcopy
import time
import os, sys
import numpy as np
import shutil
from utils import find_latest_checkpoint, create_results_dir
from IPython import embed 

# Params for Denavit-Hartenberg Reference Frame Layout (DH)
jaco27DOF_DH_lengths = {'D1':0.2755, 'D2':0.2050, 
               'D3':0.2050, 'D4':0.2073,
               'D5':0.1038, 'D6':0.1038, 
               'D7':0.1600, 'e2':0.0098}
 
DH_attributes_jaco27DOF = {
          'DH_a':[0, 0, 0, 0, 0, 0, 0],
           'DH_alpha':[np.pi/2.0, np.pi/2.0, np.pi/2.0, np.pi/2.0, np.pi/2.0, np.pi/2.0, np.pi],
           'DH_theta_sign':[1, 1, 1, 1, 1, 1, 1],
           'DH_theta_offset':[np.pi,0.0, 0.0, 0.0, 0.0,0.0,np.pi/2.0],
           'DH_d':(-jaco27DOF_DH_lengths['D1'], 
                    0, 
                    -(jaco27DOF_DH_lengths['D2']+jaco27DOF_DH_lengths['D3']), 
                    -jaco27DOF_DH_lengths['e2'], 
                    -(jaco27DOF_DH_lengths['D4']+jaco27DOF_DH_lengths['D5']), 
                    0, 
                    -(jaco27DOF_DH_lengths['D6']+jaco27DOF_DH_lengths['D7']))
           }


def get_torch_attributes(np_attribute_dict, device='cpu'):
    pt_attribute_dict = {}
    for key, item in np_attribute_dict.items():
        pt_attribute_dict[key] = torch.FloatTensor(item).to(device)
    return pt_attribute_dict


def load_data(data_file):
    # each episode is 500 steps long
    sq = np.load(data_file, allow_pickle=True)
    sq_fmt = []
    for episode in sq:
        ep_fmt = [np.hstack((x['robot0_joint_pos_cos'], x['robot0_joint_pos_sin'], x['robot0_eef_pos'])) for x in episode]
        sq_fmt.append(ep_fmt)
    # t, batch, features
    sq_fmt = np.array(sq_fmt).swapaxes(0,1)
    n_traces = sq_fmt.shape[1]

    # EE is at end of values
    indexes = np.arange(n_traces, dtype=np.int)
    random_state.shuffle(indexes)
    ttb = max([1,int(n_traces*.15)])
    t_inds = indexes[ttb:]
    v_inds = indexes[:ttb]
    train_input = sq_fmt[:-1,   t_inds, :input_size]
    train_target = sq_fmt[1:,   t_inds, :input_size]
    train_ee_target = sq_fmt[1:,t_inds, input_size:]

    valid_input = sq_fmt[:-1,    v_inds, :input_size]
    valid_target = sq_fmt[1:,    v_inds, :input_size]
    valid_ee_target = sq_fmt[1:, v_inds, input_size:]
    return train_input, train_target, train_ee_target, valid_input, valid_target, valid_ee_target

def torch_dh_transform(dh_index,angles):
    theta = tdh['DH_theta_sign'][dh_index]*angles+tdh['DH_theta_offset'][dh_index]
    d = tdh['DH_d'][dh_index]
    a = tdh['DH_a'][dh_index]
    alpha = tdh['DH_alpha'][dh_index]
    bs = angles.shape[0]
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

def forward_pass(x):
    bs = x.shape[1]
    h1_tm1 = torch.zeros((bs, hidden_size)).to(device)
    c1_tm1 = torch.zeros((bs, hidden_size)).to(device)
    h2_tm1 = torch.zeros((bs, hidden_size)).to(device)
    c2_tm1 = torch.zeros((bs, hidden_size)).to(device)
    ts = x.shape[0]
    y_pred = torch.zeros((ts, bs, output_size)).to(device)
    for i in np.arange(ts):
        output, h1_tm1, c1_tm1, h2_tm1, c2_tm1 = lstm(x[i], h1_tm1, c1_tm1, h2_tm1, c2_tm1)
        y_pred[i] = y_pred[i] + output
    return y_pred

def angle2ee(rec_angle):
    # ts, bs, feat
    ts, bs, fs = rec_angle.shape
    ee_pred = torch.zeros((ts,bs,3)).to(device)
    Tinit = torch.FloatTensor(([[1,0,0,0],[0,-1,0,0],[0,0,-1,0],[0,0,0,1]])).to(device)
    # TODO join the time/batch so i don't have to loop this
    for b in range(bs):
        _T0 = torch_dh_transform(0, rec_angle[:,b,0])
        # TODO is Tall really needed - it is just a constant
        
        T0_pred = torch.matmul(Tinit,_T0)
 
        _T1 = torch_dh_transform(1, rec_angle[:,b,1])
        T1_pred = torch.matmul(T0_pred,_T1)
 
        _T2 = torch_dh_transform(2, rec_angle[:,b,2])
        T2_pred = torch.matmul(T1_pred,_T2)

        _T3 = torch_dh_transform(3, rec_angle[:,b,3])
        T3_pred = torch.matmul(T2_pred,_T3)

        _T4 = torch_dh_transform(4, rec_angle[:,b,4])
        T4_pred = torch.matmul(T3_pred,_T4)

        _T5 = torch_dh_transform(5, rec_angle[:,b,5])
        T5_pred = torch.matmul(T4_pred,_T5)

        _T6 = torch_dh_transform(6, rec_angle[:,b,6])
        T6_pred = torch.matmul(T5_pred,_T6)
        # TODO - get full quat
        ee_pred[:,b] = ee_pred[:,b] + T6_pred[:,:3,3]
    return ee_pred
 
def train(data, step=0, n_epochs=10000):
    for epoch in range(n_epochs):
        for phase in ['valid', 'train']:
            # time, batch, features
            n_samples = data[phase]['input'].shape[1]
            indexes = np.arange(0, n_samples)
            random_state.shuffle(indexes)
            st = 0
            en = min([st+batch_size, n_samples])
            running_loss = 0
            bs = en-st
            while en <= n_samples and bs > 0:
                x = torch.FloatTensor(data[phase]['input'][:,indexes[st:en]]).to(device)
                ee_target = torch.FloatTensor(data[phase]['ee_target'][:,indexes[st:en]]).to(device)

                opt.zero_grad()
                rec_angle = np.pi*(torch.tanh(forward_pass(x))+1)
                ee_pred = angle2ee(rec_angle)
                loss = criterion(ee_pred, ee_target)
                if phase == 'train':
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


def eval_model(data, phases=['train', 'valid'], n=20, shuffle=False, teacher_force=False, lead_in=10):
    """
    phase: list containing phases to evaluate ['train', 'valid']
    n int num examples to evaluate / plot. if None, all are plotted
    """
    for phase in phases:
        # time, batch, features
        n_samples = data[phase]['input'].shape[1]
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
            x = torch.FloatTensor(data[phase]['input'][:,indexes[st:en]]).to(device)
            ee_target = torch.FloatTensor(data[phase]['ee_target'][:,indexes[st:en]]).to(device)

            rec_angle = np.pi*(torch.tanh(forward_pass(x))+1)
            ee_pred = angle2ee(rec_angle).detach().cpu().numpy()
            plt.figure()
            for ii in range(ee_pred.shape[1]):
                plt.scatter(ee_pred[:,ii,0], ee_pred[:,ii,1])
            fname = modelbase+'%s_%05d-%05d.png'%(phase,st,en)
            print('plotting', fname)
            plt.savefig(fname)
            plt.close()
            st = en
            en = min([st+batch_size, n+1])
            bs = en-st

# tan(theta) = sin(theta)/cos(theta) 
 
if __name__ == '__main__':
    import argparse
    from glob import glob
    parser = argparse.ArgumentParser()
    parser.add_argument('--exp_name', default='v2_lstm_ee')
    parser.add_argument('--eval', default=False, action='store_true')
    parser.add_argument('--load_model', default='')
    args = parser.parse_args()
    
    load_from = 'results/21-01-21_v1_lstm_ee_03/'
    device = 'cuda'
    results_dir = 'results'
    # set random seed to 0
    exp_name = args.exp_name
    seed = 323
    grad_clip = 5
    hidden_size = 1024
    input_size = 14
    output_size = 7
    batch_size = 32
    save_every_epochs = 10
    y_pred = torch.zeros((batch_size, output_size))

    tdh = get_torch_attributes(DH_attributes_jaco27DOF, device)

      
    losses = {'train':[], 'valid':[]}

    np.random.seed(seed)
    torch.manual_seed(seed)
    random_state = np.random.RandomState(seed)

    data = {'valid':{}, 'train':{}} 
    data['train']['input'], data['train']['target'], data['train']['ee_target'], data['valid']['input'], data['valid']['target'], data['valid']['ee_target'] = load_data('square.npy')
    

    lstm = LSTM(input_size=input_size, hidden_size=hidden_size).to(device)
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
        load_dict = torch.load(loadpath)
        step = load_dict['train_cnt']
        lstm.load_state_dict(load_dict['model'])

    if args.eval:
        eval_model(data, phases=['train', 'valid'], n=20, shuffle=False)
    else:
        criterion = nn.MSELoss()
        # use LBFGS as optimizer since we can load the whole data to train
        opt = optim.Adam(lstm.parameters(), lr=0.001)
        train(data, step,  n_epochs=10000)
     
embed()
