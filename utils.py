import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import os
import copy
import numpy as np
import pickle
from datetime import datetime as date
from glob import glob
import torch

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
DH_attributes_dm_reacher = {
     'DH_a':[0.01,0.01],
     'DH_alpha':[0.0,0.0],
     'DH_theta_sign':[1.0,1.0], 
     'DH_theta_offset':[0,0],
     'DH_d':[0,0]}
       
robot_attributes = {'dm_reacher':DH_attributes_dm_reacher, 
                    'jaco27DOF':DH_attributes_jaco27DOF, 
                   }

def torch_dh_transform(theta, d, a, alpha, device):
    # TODO check this - it had a bug
    bs = theta.shape[0]
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


class robotDH():
    def __init__(self, robot_name, device='cpu'):
        self.device = device
        self.robot_name = robot_name
        self.robot_attribute_dict = robot_attributes[self.robot_name]
        self.tdh = {}
        for key, item in self.robot_attribute_dict.items():
            self.tdh[key] = torch.FloatTensor(item).to(self.device)

    def angle2ee(self, angles):
        """ 
            convert joint angle to end effector for reacher for ts,bs,f
        """
        # ts, bs, feat
        ts, bs, fs = angles.shape
        ee_pred = torch.zeros((ts,bs,2)).to(self.device)
        # TODO join the time/batch so i don't have to loop this
        # TODO this transform is pretty dependent on the 7 features in jaco
        for b in range(bs):
            T0 = self.dh_transform(0, angles[:,b,0])
            T1 = self.dh_transform(1, angles[:,b,1])
            T = torch.matmul(T0, T1)
            ee_pred[:,b] = ee_pred[:,b] + T[:,:2,3]
        return ee_pred

    def dh_transform(self, dh_index, angles):
        theta = self.tdh['DH_theta_sign'][dh_index]*angles+self.tdh['DH_theta_offset'][dh_index]
        d = self.tdh['DH_d'][dh_index]
        a = self.tdh['DH_a'][dh_index]
        alpha = self.tdh['DH_alpha'][dh_index]
        return torch_dh_transform(theta, d, a, alpha, self.device)     



#    def angle2ee(self, rec_angle):
#        """ 
#            convert joint angle to end effector for jaco 
#              TODO make this general
#        """
#        # ts, bs, feat
#        ts, bs, fs = rec_angle.shape
#        ee_pred = torch.zeros((ts,bs,3)).to(self.device)
#        Tinit = torch.FloatTensor(([[1,0,0,0],[0,-1,0,0],[0,0,-1,0],[0,0,0,1]])).to(self.device)
#        # TODO join the time/batch so i don't have to loop this
#        # TODO this transform is pretty dependent on the 7 features in jaco
#        for b in range(bs):
#          
#            _T0 = self.dh_transform(0, rec_angle[:,b,0])
#            # TODO is Tall really needed - it is just a constant
#            
#            T0_pred = torch.matmul(Tinit,_T0)
#     
#            _T1 = self.dh_transform(1, rec_angle[:,b,1])
#            T1_pred = torch.matmul(T0_pred,_T1)
#     
#            _T2 = self.dh_transform(2, rec_angle[:,b,2])
#            T2_pred = torch.matmul(T1_pred,_T2)
#    
#            _T3 = self.dh_transform(3, rec_angle[:,b,3])
#            T3_pred = torch.matmul(T2_pred,_T3)
#    
#            _T4 = self.dh_transform(4, rec_angle[:,b,4])
#            T4_pred = torch.matmul(T3_pred,_T4)
#    
#            _T5 = self.dh_transform(5, rec_angle[:,b,5])
#            T5_pred = torch.matmul(T4_pred,_T5)
#    
#            _T6 = self.dh_transform(6, rec_angle[:,b,6])
#            T6_pred = torch.matmul(T5_pred,_T6)
#            # TODO - get full quat
#            ee_pred[:,b] = ee_pred[:,b] + T6_pred[:,:3,3]
#        return ee_pred


def sincos2angle(sin_theta, cos_theta, use_numpy=False):
    """ robosuite outputs the joint angle in sin(theta) cos(angle) 
    This function converts it to angles in radians """
    if not use_numpy: 
        return torch.arctan(sin_theta/cos_theta)
    else:
        return np.arctan(sin_theta/cos_theta)

def angle2sincos(theta, use_numpy=False):
    """ convert an angle to sin(angle) cos(theta)  in radians """
    if not use_numpy: 
        return torch.sin(theta), torch.cos(theta) 
    else:
        return np.sin(theta), np.cos(theta) 
 


def get_data_norm_params(data, device='cpu'):
    input_size = data.shape[1]
    train_mean = torch.FloatTensor([data[:,x].mean() for x in range(input_size)]).to(device)
    train_std = torch.FloatTensor([data[:,x].std() for x in range(input_size)]).to(device)    
    return train_mean, train_std

def load_robosuite_data(data_file, random_state):
    # each episode is 500 steps long
    # data is 
    sq = np.load(data_file, allow_pickle=True)

    input_size = len(sq[0][0]['robot0_joint_pos_sin']) + len(sq[0][0]['robot0_joint_pos_cos'])
    sq_fmt = []
    for episode in sq:
        ep_fmt = [np.hstack((x['robot0_joint_pos_sin'], x['robot0_joint_pos_cos'], x['robot0_eef_pos'], x['robot0_joint_pos'])) for x in episode]
        #ep_fmt = [np.hstack((x['robot0_joint_pos_sin'], x['robot0_joint_pos_cos'], x['robot0_eef_pos'])) for x in episode]
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
    data = {'valid':{}, 'train':{}} 
    data['train']['input'] = sq_fmt[:-1,   t_inds, :input_size]
    data['train']['target'] = sq_fmt[1:,   t_inds, :input_size]
    data['train']['ee_target'] = sq_fmt[1:,t_inds, input_size:input_size+3]
    data['train']['jt_target'] = sq_fmt[1:,t_inds, input_size+3:]

    data['valid']['input'] = sq_fmt[:-1,    v_inds, :input_size]
    data['valid']['target'] = sq_fmt[1:,    v_inds, :input_size]
    data['valid']['ee_target']= sq_fmt[1:, v_inds, input_size:input_size+3]
    data['valid']['jt_target'] = sq_fmt[1:, v_inds, input_size+3:]
    return data


def plot_losses(loss_path):
    losses = np.load(loss_path)
    plt.figure()
    for phase, ll in [ ('valid',losses['valid']),('train', losses['train'])]:
        plt.plot(ll[1:,0], ll[1:,1], label=phase, marker='o')
    plt.title('losses')
    plt.legend()
    fname = loss_path.replace('.npz', '.png')
    print("saving loss image: {}".format(fname))
    plt.savefig(fname)
    plt.close()

def find_latest_checkpoint(basedir):
    assert os.path.isdir(basedir)
    search = os.path.join(basedir, '*.pt')
    print('searching {} for models'.format(search))
    found_models = sorted(glob(search))
    print('found {} models'.format(len(found_models)))
    # this is the latest model
    load_path = found_models[-1]
    print('using most recent - {}'.format(load_path))
    return load_path

def create_results_dir(exp_name, results_dir='results'):
    today = date.today()
    today_str = today.strftime("%y-%m-%d")
    exp_cnt = 0
    savebase = os.path.join('results', '%s_%s_%02d'%(today_str, exp_name, exp_cnt))
    while len(glob(os.path.join(savebase, '*.pt'))):
         exp_cnt += 1
         savebase = os.path.join(results_dir, '%s_%s_%02d'%(today_str, exp_name, exp_cnt))
    if not os.path.exists(savebase):
        os.makedirs(savebase)
    if not os.path.exists(os.path.join(savebase, '__python')):
        os.makedirs(os.path.join(savebase, '__python'))
        os.system('cp *.py %s/__python'%savebase)
    return savebase

def seed_everything(seed=1234):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)

if __name__ == '__main__':
    #fb = 'results/21-01-21_v1_lstm_ee_03'
    #pt_latest = find_latest_checkpoint(fb)
    #plot_losses(pt_latest.replace('.pt', '_losses.npz'))
    plot_losses('results/21-01-21_v1_lstm_ee_03/model_0000023136_losses.npz')
