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
from utils import find_latest_checkpoint, create_results_dir, plot_losses
from utils import robotDH, angle2sincos, sincos2angle
from utils import load_robosuite_data, get_data_norm_params
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

def forward_pass(input_data, use_output=False, lead_in=3):
    input_data = (input_data-train_mean)/train_std
    if use_output:
        x = torch.zeros_like(input_data)
        x[:lead_in] = input_data[:lead_in]
    else:
        # teacher force
        # add gaussian noise independent per feature and sampled 
        # from a 0 mean gaussian where std deviation is that of the data (normed to 1)
        x = input_data + torch.normal(torch.zeros_like(input_data), noise_std*torch.ones_like(input_data))
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
        if use_output and lead_in < step_num < ts-1:
            # use predicted output as input
            x[i+1] = torch.cat(angle2sincos(output), 1)
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

                opt.zero_grad()
                rec_angle = np.pi*(torch.tanh(forward_pass(x))+1)
                #rec_sincos = torch.tanh(forward_pass(x))
                 
                #rec_angle = sincos2angle(rec_sincos[:,:,:7], rec_sincos[:,:,7:])
                if args.loss == 'DH':
                    ee_target = torch.FloatTensor(data[phase]['ee_target'][:,indexes[st:en]]).to(device)
                    ee_pred = robot_dh.angle2ee(rec_angle)
                    loss = criterion(ee_pred, ee_target)
                elif args.loss == 'angle':
                    jt_target = torch.FloatTensor(data[phase]['jt_target'][:,indexes[st:en]]).to(device)
                    loss = criterion(rec_angle, jt_target)

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

            rec_angle = np.pi*(torch.tanh(forward_pass(x, use_output=not args.teacher_force, lead_in=args.lead_in))+1)
            ee_pred = robot_dh.angle2ee(rec_angle).detach().cpu().numpy()
            plt.figure()
            for ii in range(ee_pred.shape[1]):
                plt.scatter(ee_pred[:,ii,0], ee_pred[:,ii,1])
            if args.teacher_force:
                fname = modelbase+'%s_%05d-%05d_tf.png'%(phase,st,en)
            else:
                fname = modelbase+'%s_%05d-%05d_li%02d.png'%(phase,st,en, args.lead_in)
            print('plotting', fname)
            plt.savefig(fname)
            plt.close()
            st = en
            en = min([st+batch_size, n+1])
            bs = en-st
if __name__ == '__main__':
    import argparse
    from glob import glob
    parser = argparse.ArgumentParser()
    parser.add_argument('--eval', default=False, action='store_true')
    parser.add_argument('-tf', '--teacher_force', default=False, action='store_true')
    parser.add_argument('--load_model', default='')
    parser.add_argument('--loss', default='DH')
    parser.add_argument('--device', default='cuda')
    parser.add_argument('-li', '--lead_in', default=100, type=int)
    args = parser.parse_args()
    exp_name = 'v4_lstm_%s'%(args.loss)
    
    device = args.device
    results_dir = 'results'
    # set random seed to 0
    seed = 323
    noise_std = 4
    grad_clip = 5
    hidden_size = 1024
    input_size = 14
    output_size = 7
    batch_size = 32
    save_every_epochs = 100

    robot_dh = robotDH(robot_name='jaco27DOF', device=device)
    np.random.seed(seed)
    torch.manual_seed(seed)
    random_state = np.random.RandomState(seed)
    losses = {'train':[], 'valid':[]}


    data = load_robosuite_data('square.npy', random_state=random_state)
    # 0 mean and divide by std dev data 
    train_mean, train_std = get_data_norm_params(data, device=device)

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
        eval_model(data, phases=['train', 'valid'], n=20, shuffle=False)
        plot_losses(loadpath.replace('.pt', '_losses.npz'))
    else:
        criterion = nn.MSELoss()
        # use LBFGS as optimizer since we can load the whole data to train
        opt = optim.Adam(lstm.parameters(), lr=0.0001)
        train(data, step,  n_epochs=10000)
     
