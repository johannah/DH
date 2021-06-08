import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import os
import copy
import numpy as np
import pickle
import math
from datetime import datetime as date
from glob import glob
import torch
from robosuite.utils.transform_utils import mat2quat
from IPython import embed
# epsilon for testing whether a number is close to zero
_EPS = np.finfo(float).eps * 4.0

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
                    'Jaco':DH_attributes_jaco27DOF, 
                   }

def normalize_joints(angles):
    """
    This removes the wrapping from joint angles and ensures joint vals are bt -pi < vals < pi
    angles: np.array of joint angles in radians
    """
    while angles.max() > np.pi:
        angles[angles>np.pi] -= 2*np.pi
    while angles.min() < -np.pi:
        angles[angles<-np.pi] += 2*np.pi
    return angles


def quaternion_from_matrix(matrix):
    """Return quaternion from rotation matrix.
    from: https://github.com/BerkeleyAutomation/autolab_core/blob/master/autolab_core/transformations.py
    >>> R = rotation_matrix(0.123, (1, 2, 3))
    >>> q = quaternion_from_matrix(R)
    >>> numpy.allclose(q, [0.0164262, 0.0328524, 0.0492786, 0.9981095])
    True
    """
    q = np.empty((4,), dtype=np.float64)
    M = np.array(matrix, dtype=np.float64, copy=False)[:4, :4]
    t = np.trace(M)
    if t > M[3, 3]:
        q[3] = t
        q[2] = M[1, 0] - M[0, 1]
        q[1] = M[0, 2] - M[2, 0]
        q[0] = M[2, 1] - M[1, 2]
    else:
        i, j, k = 0, 1, 2
        if M[1, 1] > M[0, 0]:
            i, j, k = 1, 2, 0
        if M[2, 2] > M[i, i]:
            i, j, k = 2, 0, 1
        t = M[i, i] - (M[j, j] + M[k, k]) + M[3, 3]
        q[i] = t
        q[j] = M[i, j] + M[j, i]
        q[k] = M[k, i] + M[i, k]
        q[3] = M[k, j] - M[j, k]
    q *= 0.5 / math.sqrt(t * M[3, 3])
    return q

def np_dh_transform(theta, d, a, alpha):
    # TODO check this - it had a bug
    bs = theta.shape[0]
    T = np.zeros((bs,4,4))
    T[:,0,0] = T[:,0,0] +  np.cos(theta)
    T[:,0,1] = T[:,0,1] + -np.sin(theta)*np.cos(alpha)
    T[:,0,2] = T[:,0,2] +  np.sin(theta)*np.sin(alpha)
    T[:,0,3] = T[:,0,3] +  a*np.cos(theta)
    T[:,1,0] = T[:,1,0] +    np.sin(theta)
    T[:,1,1] = T[:,1,1] +    np.cos(theta)*np.cos(alpha)
    T[:,1,2] = T[:,1,2] +   -np.cos(theta)*np.sin(alpha)
    T[:,1,3] = T[:,1,3] +  a*np.sin(theta)
    T[:,2,1] = T[:,2,1] +  np.sin(alpha)
    T[:,2,2] = T[:,2,2] +   np.cos(alpha)
    T[:,2,3] = T[:,2,3] +  d
    T[:,3,3] = T[:,3,3] +  1.0
    return T 



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

def quaternion_matrix(quaternion):
    """Return homogeneous rotation matrix from quaternion.
    >>> R = quaternion_matrix([0.06146124, 0, 0, 0.99810947])
    >>> numpy.allclose(R, rotation_matrix(0.123, (1, 0, 0)))
    True
    """
    q = np.array(quaternion[:4], dtype=np.float64, copy=True)
    nq = np.dot(q, q)
    if nq < _EPS:
        return np.identity(4)
    q *= math.sqrt(2.0 / nq)
    q = np.outer(q, q)
    return np.array(
        (
            (
                1.0 - q[1, 1] - q[2, 2],
                q[0, 1] - q[2, 3],
                q[0, 2] + q[1, 3],
                0.0,
            ),
            (
                q[0, 1] + q[2, 3],
                1.0 - q[0, 0] - q[2, 2],
                q[1, 2] - q[0, 3],
                0.0,
            ),
            (
                q[0, 2] - q[1, 3],
                q[1, 2] + q[0, 3],
                1.0 - q[0, 0] - q[1, 1],
                0.0,
            ),
            (0.0, 0.0, 0.0, 1.0),
        ),
        dtype=np.float64,
    )


class robotDH():
    def __init__(self, robot_name, device='cpu'):
        self.device = device
        self.robot_name = robot_name
        self.npdh = robot_attributes[self.robot_name]
        self.tdh = {}
        for key, item in self.npdh.items():
            self.tdh[key] = torch.FloatTensor(item).to(self.device)

    def np_angle2ee(self, base_matrix, angles):
        """ 
            convert np joint angle to end effector for for ts,angles (in radians)
        """
        # ts, bs, feat
        ts, fs = angles.shape
        ee_pred = np.zeros((ts,7))
        # TODO join the time/batch so i don't have to loop this
        #_T = self.np_dh_transform(0, angles[:,0])
        #_T = np.array([[1,0,0,0],[0,-1,0,0],[0,0,-1,0],[0,0,0,1]], dtype=np.float)
        _T = base_matrix
        for _a in range(fs):        
            _T1 = self.np_dh_transform(_a, angles[:,_a])
            _T = np.matmul(_T, _T1)
        #ee_pred[:,:3] = ee_pred[:,:3] + _T[:, :3, 3] # position
        #ee_pred[:,3:] = [quaternion_from_matrix(_T[x]) for x in range(ts)] # quaternion
        #ee_pred[:,3:] = [mat2quat(_T[x]) for x in range(ts)] # quaternion
        return _T

    def torch_angle2ee(self, base_matrix, angles):
        """ 
            convert joint angle to end effector for reacher for ts,bs,f
        """
        # ts, bs, feat
        ts, fs = angles.shape
        ee_pred = torch.zeros((ts,4,4)).to(self.device)
        _T = base_matrix
        for _a in range(fs):        
            _T1 = self.torch_dh_transform(_a, angles[:,_a])
            _T = torch.matmul(_T, _T1)
        return _T


    def batch_angle2ee_reacher(self, angles):
        """ 
            convert joint angle to end effector for reacher for ts,bs,f
        """
        # ts, bs, feat
        ts, bs, fs = angles.shape
        ee_pred = torch.zeros((ts,bs,2)).to(self.device)
        # TODO join the time/batch so i don't have to loop this
        # TODO this transform is pretty dependent on the 7 features in jaco
        for b in range(bs):
            T0 = self.torch_dh_transform(0, angles[:,b,0])
            T1 = self.torch_dh_transform(1, angles[:,b,1])
            T = torch.matmul(T0, T1)
            ee_pred[:,b] = ee_pred[:,b] + T[:,:2,3]
        return ee_pred

    def np_dh_transform(self, dh_index, angles):
        theta = self.npdh['DH_theta_sign'][dh_index]*angles+self.npdh['DH_theta_offset'][dh_index]
        d = self.npdh['DH_d'][dh_index]
        a = self.npdh['DH_a'][dh_index]
        alpha = self.npdh['DH_alpha'][dh_index]
        return np_dh_transform(theta, d, a, alpha)     


    def torch_dh_transform(self, dh_index, angles):
        theta = self.tdh['DH_theta_sign'][dh_index]*angles+self.tdh['DH_theta_offset'][dh_index]
        d = self.tdh['DH_d'][dh_index]
        a = self.tdh['DH_a'][dh_index]
        alpha = self.tdh['DH_alpha'][dh_index]
        return torch_dh_transform(theta, d, a, alpha, self.device)     

# I"M NOT CONVINCED THESE WORK - 
#def sincos2angle(sin_theta, cos_theta, use_numpy=False):
#    """ robosuite outputs the joint angle in sin(theta) cos(angle) 
#    This function converts it to angles in radians """
#    if not use_numpy: 
#        return torch.arctan(sin_theta/cos_theta)
#    else:
#        return np.arctan(sin_theta/cos_theta)
#
#def angle2sincos(theta, use_numpy=False):
#    """ convert an angle to sin(angle) cos(theta)  in radians """
#    if not use_numpy: 
#        return torch.sin(theta), torch.cos(theta) 
#    else:
#        return np.sin(theta), np.cos(theta) 
# 


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
