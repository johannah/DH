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
import torch.nn as nn
import torch.nn.functional as F
import robosuite.utils.transform_utils as T
from dh_parameters import robot_attributes
from IPython import embed

# epsilon for testing whether a number is close to zero
_EPS = np.finfo(float).eps * 4.0

# some keys are robot-specifig!
skip_state_keys = ['robot0_joint_pos_cos', 'robot0_joint_pos_sin','robot0_joint_vel', 'robot0_proprio-state']



#c1 = tf.nn.l2_normalize(x[:, :3], axis=-1)
#c2 = tf.nn.l2_normalize(x[:, 3:] - self.dot(c1,x[:, 3:])*c1, axis=-1)
#x = tf.concat([c1,c2], axis=-1)
#self.add_metric(mean_angle_btw_vectors(inputs, self.get_rotated(x)), 
#                    name='mean_angular_distance', aggregation='mean')
# ideas for rotation losses from 
# https://towardsdatascience.com/better-rotation-representations-for-accurate-pose-estimation-e890a7e1317f
#def euler_loss(y_true, y_pred):
#    dist1 = tf.abs(y_true - y_pred)
#    dist2 = tf.abs(2*np.pi + y_true - y_pred)
#    dist3 = tf.abs(-2*np.pi + y_true - y_pred)
#    loss = tf.where(dist1<dist2, dist1, dist2)
#    loss = tf.where(loss<dist3, loss, dist3)
#    return tf.reduce_mean(loss)
#  
#def quaternion_loss(y_true, y_pred):
#    dist1 = tf.reduce_mean(tf.abs(y_true-y_pred), axis=-1)
#    dist2 = tf.reduce_mean(tf.abs(y_true+y_pred), axis=-1)
#    loss = tf.where(dist1<dist2, dist1, dist2)
#    return tf.reduce_mean(loss)
#  
#def mean_angle_btw_vectors(v1, v2, eps = 1e-8):
#    dot_product = tf.reduce_sum(v1*v2, axis=-1)
#    cos_a = dot_product / (tf.norm(v1, axis=-1) * tf.norm(v2, axis=-1))
#    cos_a = tf.clip_by_value(cos_a, -1 + eps, 1 - eps)
#    angle_dist = tf.math.acos(cos_a) / np.pi * 180.0
#    return tf.reduce_mean(angle_dist)

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


def so3_relative_angle(R1, R2, cos_angle: bool = False):
    """
    from: https://pytorch3d.readthedocs.io/en/latest/_modules/pytorch3d/transforms/so3.html#so3_relative_angle

    Calculates the relative angle (in radians) between pairs of
    rotation matrices `R1` and `R2` with `angle = acos(0.5 * (Trace(R1 R2^T)-1))`

    .. note::
        This corresponds to a geodesic distance on the 3D manifold of rotation
        matrices.

    Args:
        R1: Batch of rotation matrices of shape `(minibatch, 3, 3)`.
        R2: Batch of rotation matrices of shape `(minibatch, 3, 3)`.
        cos_angle: If==True return cosine of the relative angle rather than
                   the angle itself. This can avoid the unstable
                   calculation of `acos`.

    Returns:
        Corresponding rotation angles of shape `(minibatch,)`.
        If `cos_angle==True`, returns the cosine of the angles.

    Raises:
        ValueError if `R1` or `R2` is of incorrect shape.
        ValueError if `R1` or `R2` has an unexpected trace.
    """
    R12 = torch.bmm(R1, R2.permute(0, 2, 1))
    return so3_rotation_angle(R12, cos_angle=cos_angle)



def so3_rotation_angle(R, eps: float = 1e-4, cos_angle: bool = False):
    """
    Calculates angles (in radians) of a batch of rotation matrices `R` with
    `angle = acos(0.5 * (Trace(R)-1))`. The trace of the
    input matrices is checked to be in the valid range `[-1-eps,3+eps]`.
    The `eps` argument is a small constant that allows for small errors
    caused by limited machine precision.

    Args:
        R: Batch of rotation matrices of shape `(minibatch, 3, 3)`.
        eps: Tolerance for the valid trace check.
        cos_angle: If==True return cosine of the rotation angles rather than
                   the angle itself. This can avoid the unstable
                   calculation of `acos`.

    Returns:
        Corresponding rotation angles of shape `(minibatch,)`.
        If `cos_angle==True`, returns the cosine of the angles.

    Raises:
        ValueError if `R` is of incorrect shape.
        ValueError if `R` has an unexpected trace.
    """

    N, dim1, dim2 = R.shape
    if dim1 != 3 or dim2 != 3:
        raise ValueError("Input has to be a batch of 3x3 Tensors.")

    rot_trace = R[:, 0, 0] + R[:, 1, 1] + R[:, 2, 2]

    if ((rot_trace < -1.0 - eps) + (rot_trace > 3.0 + eps)).any():
        raise ValueError("A matrix has trace outside valid range [-1-eps,3+eps].")

    # clamp to valid range
    rot_trace = torch.clamp(rot_trace, -1.0, 3.0)

    # phi ... rotation angle
    phi = 0.5 * (rot_trace - 1.0)

    if cos_angle:
        return phi
    else:
        # pyre-fixme[16]: `float` has no attribute `acos`.
        return phi.acos()


def mean_angle_btw_vectors(v1, v2, eps = 1e-4):
    # https://towardsdatascience.com/better-rotation-representation    s-for-accurate-pose-estimation-e890a7e1317f
    dot_product = torch.sum(v1*v2, axis=-1)
    cos_a = dot_product / (torch.norm(v1, dim=-1) * torch.norm(v2,     dim=-1))
    cos_a = torch.clamp(cos_a, -1 + eps, 1 - eps)
    angle_dist = torch.acos(cos_a)
    return torch.mean(angle_dist)


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
            #print(_a, _T[0])
            #print(T.mat2euler(_T[0]))
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


class robotDHLearnable(nn.Module):
    def __init__(self, robot_name, dh_noise=0.1, device='cpu'):
        super().__init__()
        self.device = device
        self.learnable_params = ['DH_a']
        self.robot_name = robot_name
        self.npdh_true = robot_attributes[self.robot_name]
        self.tdh_true = {}
        self.tdh = nn.ParameterDict({})
        for key, item in self.npdh_true.items():
            self.tdh_true[key] = torch.FloatTensor(item).to(self.device)
            if key in self.learnable_params:
                param_noise = torch.normal(torch.zeros(len(item)), dh_noise * torch.ones(len(item))).to(device)
                self.tdh[key] = nn.Parameter(torch.abs(torch.FloatTensor(item).to(device) + param_noise), requires_grad=True)
            else:
                self.tdh[key] = nn.Parameter(torch.FloatTensor(item).to(device), requires_grad=False)

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

    def torch_dh_transform(self, dh_index, angles):
        theta = self.tdh['DH_theta_sign'][dh_index] * angles + self.tdh['DH_theta_offset'][dh_index]
        d = self.tdh['DH_d'][dh_index]
        a = self.tdh['DH_a'][dh_index]
        alpha = self.tdh['DH_alpha'][dh_index]
        return torch_dh_transform(theta, d, a, alpha, self.device)

    def get_dh_estimation_error(self):
        error_dict = {}
        mean_error = 0.
        with torch.no_grad():
            for k in self.learnable_params:
                error_dict[k] = F.mse_loss(self.tdh[k], self.tdh_true[k]).item()
                mean_error += error_dict[k]
            mean_error /= len(self.learnable_params)
            error_dict['mean'] = mean_error
            return error_dict

    def get_estimated_dh_params(self):
        estimated_dh_dict = {}
        for k in self.learnable_params:
            for i in range(self.tdh[k].shape[0]):
                estimated_dh_dict[f'{k}_j{i+1}'] = self.tdh[k][i].item()
        return estimated_dh_dict


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
    savebase = os.path.join(results_dir, '%s_%s_%02d'%(today_str, exp_name, exp_cnt))
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
