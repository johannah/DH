import matplotlib
matplotlib.use('Agg')
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
from util import plot_losses
from IPython import embed 

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


class LSTM(nn.Module):
    def __init__(self, input_size=14, hidden_size=1024):
        super(LSTM, self).__init__()
        self.input_size = input_size
        self.output_size = self.input_size
        self.hidden_size = hidden_size
        self.lstm1 = nn.LSTMCell(input_size, hidden_size)
        self.lstm2 = nn.LSTMCell(hidden_size, hidden_size)
        self.linear = nn.Linear(hidden_size, self.output_size)

    def forward(self, xt, h1_tm1, c1_tm1, h2_tm1, c2_tm1):
        h1_t, c1_t = self.lstm1(xt, (h1_tm1, c1_tm1))
        h2_t, c2_t = self.lstm2(h1_t, (h2_tm1, c2_tm1))
        output = self.linear(h2_t)
        return output, h1_t, c1_t, h2_t, c2_t

def forward_pass(x, target):
    bs = x.shape[1]
    h1_tm1 = torch.zeros((bs, hidden_size)).to(device)
    c1_tm1 = torch.zeros((bs, hidden_size)).to(device)
    h2_tm1 = torch.zeros((bs, hidden_size)).to(device)
    c2_tm1 = torch.zeros((bs, hidden_size)).to(device)
    y_pred = torch.zeros_like(target)
    timesteps = x.shape[0]
    for i in np.arange(0,timesteps):
        output, h1_tm1, c1_tm1, h2_tm1, c2_tm1 = lstm(x[i], h1_tm1, c1_tm1, h2_tm1, c2_tm1)
        y_pred[i] = y_pred[i] + output
    return y_pred

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
                target = torch.FloatTensor(data[phase]['target'][:,indexes[st:en]]).to(device)
                opt.zero_grad()
                pred = forward_pass(x, target)
                loss = criterion(pred, target)
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


 
if __name__ == '__main__':
    from datetime import date
    import argparse
    from glob import glob

    device = 'cuda'
    # set random seed to 0
    exp_name = 'v1_lstm_mse'
    seed = 323
    grad_clip = 5
    hidden_size = 1024
    input_size = output_size = 14
    batch_size = 32
    save_every_epochs = 100
    y_pred = torch.zeros((batch_size, output_size))

    today = date.today()
    today_str = today.strftime("%y-%m-%d")
 
    exp_cnt = 0
    savebase = os.path.join('results', '%s_%s_%02d'%(today_str, exp_name, exp_cnt))
    while len(glob(os.path.join(savebase, '*.pt'))):
         exp_cnt += 1
         savebase = os.path.join('results', '%s_%s_%02d'%(today_str, exp_name, exp_cnt))
    if not os.path.exists(savebase):
        os.makedirs(savebase)
    if not os.path.exists(os.path.join(savebase, 'python')):
        os.makedirs(os.path.join(savebase, 'python'))
        os.system('cp *.py %s/python'%savebase)


    step = 0
    losses = {'train':[], 'valid':[]}

    np.random.seed(seed)
    torch.manual_seed(seed)
    random_state = np.random.RandomState(seed)

    data = {'valid':{}, 'train':{}} 
    data['train']['input'], data['train']['target'], data['train']['target_ee'], data['valid']['input'], data['valid']['target'], data['valid']['target_ee'] = load_data('square.npy')

    lstm = LSTM(input_size=input_size, hidden_size=hidden_size).to(device)
    criterion = nn.MSELoss()
    # use LBFGS as optimizer since we can load the whole data to train
    opt = optim.Adam(lstm.parameters(), lr=0.001)
    train(data, step,  n_epochs=10000)
embed()
