import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

import math
import os
import random
from collections import deque
import numpy as np
import scipy.linalg as sp_la
from imageio import mimwrite

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import distributions as pyd
from skimage.util.shape import view_as_windows

import gym
import gym.spaces as spaces

import robosuite
import robosuite.utils.macros as macros
macros.IMAGE_CONVENTION = 'opencv'
from robosuite.utils.transform_utils import mat2quat

from dm_control import suite

import TD3
from replay_buffer import ReplayBuffer, compress_frame
from dh_utils import robotDH, quaternion_matrix, quaternion_from_matrix, robot_attributes, normalize_joints

from IPython import embed; 


class eval_mode(object):
    def __init__(self, *models):
        self.models = models

    def __enter__(self):
        self.prev_states = []
        for model in self.models:
            self.prev_states.append(model.training)
            model.train(False)

    def __exit__(self, *args):
        for model, state in zip(self.models, self.prev_states):
            model.train(state)
        return False


def soft_update_params(net, target_net, tau):
    for param, target_param in zip(net.parameters(), target_net.parameters()):
        target_param.data.copy_(tau * param.data +
                                (1 - tau) * target_param.data)


def set_seed_everywhere(seed):
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)


def make_dir(*path_parts):
    dir_path = os.path.join(*path_parts)
    try:
        os.mkdir(dir_path)
    except OSError:
        pass
    return dir_path


def tie_weights(src, trg):
    assert type(src) == type(trg)
    trg.weight = src.weight
    trg.bias = src.bias


def weight_init(m):
    """Custom weight init for Conv2D and Linear layers."""
    if isinstance(m, nn.Linear):
        nn.init.orthogonal_(m.weight.data)
        if hasattr(m.bias, 'data'):
            m.bias.data.fill_(0.0)
    elif isinstance(m, nn.Conv2d) or isinstance(m, nn.ConvTranspose2d):
        gain = nn.init.calculate_gain('relu')
        nn.init.orthogonal_(m.weight.data, gain)
        if hasattr(m.bias, 'data'):
            m.bias.data.fill_(0.0)


def mlp(input_dim, hidden_dim, output_dim, hidden_depth, output_mod=None):
    if hidden_depth == 0:
        mods = [nn.Linear(input_dim, output_dim)]
    else:
        mods = [nn.Linear(input_dim, hidden_dim), nn.ReLU(inplace=True)]
        for i in range(hidden_depth - 1):
            mods += [nn.Linear(hidden_dim, hidden_dim), nn.ReLU(inplace=True)]
        mods.append(nn.Linear(hidden_dim, output_dim))
    if output_mod is not None:
        mods.append(output_mod)
    trunk = nn.Sequential(*mods)
    return trunk


def to_np(t):
    if t is None:
        return None
    elif t.nelement() == 0:
        return np.array([])
    else:
        return t.cpu().detach().numpy()

class EnvStack():
    def __init__(self, env, k, skip_state_keys=[], env_type='robosuite', default_camera='', xpos_target='', bpos="root"):
        assert env_type in ['robosuite', 'dm_control']
        # dm_control named.data to use for eef position 
        # see https://github.com/deepmind/dm_control/blob/5ca4094e963236d0b7b3b1829f9097ad865ebabe/dm_control/suite/reacher.py#L66 for example:
        env.reset()
        self.bpos = bpos
        self.xpos_target = xpos_target
        self.env_type = env_type
        self.env = env
        self.k = k
        self._body = deque([], maxlen=k)
        self._state = deque([], maxlen=k)
        self.body_shape = k*len(self.make_body())
        
        if self.env_type == 'robosuite':
            self.control_min = self.env.action_spec[0].min()
            self.control_max = self.env.action_spec[1].max()
            self.control_shape = self.env.action_spec[0].shape[0]
            self.max_timesteps = self.env.horizon
            self.sim = self.env.sim
            if default_camera == '':
                 self.default_camera = 'agentview'

            self.bpos = self.env.robots[0].base_pos
            self.bori = self.env.robots[0].base_ori
            # hard code orientation
            # TODO add conversion to rotation matrix
            self.base_matrix = quaternion_matrix(self.bori)
            self.base_matrix[:3, 3] = self.bpos
            # TODO this is hacky - but it seems the world needs to be flipped in y,z to be correct
            # Sanity checked in Jaco w/ Door
            # ensure this holds for other robots
            self.base_matrix[1,1] = -1
            self.base_matrix[2,2] = -1

        elif self.env_type == 'dm_control':
            self.control_min = self.env.action_spec().minimum[0]
            self.control_max = self.env.action_spec().maximum[0]
            self.control_shape = self.env.action_spec().shape[0]
            self.max_timesteps = int(self.env._step_limit)
            self.sim = self.env.physics
            if default_camera == '':
                 self.default_camera = -1
  
            self.base_matrix = np.eye(4)
            self.base_matrix[:3,:3] = self.env.physics.named.data.geom_xmat[self.bpos].reshape(3,3)
            self.bpos = self.env.physics.named.data.geom_xpos[self.bpos]
            self.base_matrix[:3, 3] = self.bpos
            self.base_matrix[1,1] = 1 # TODO FOUND EXPERIMENTALLY FOR REACHER
            self.bori = quaternion_from_matrix(self.base_matrix)
 
        total_size = 0
        self.skip_state_keys = skip_state_keys
        self.obs_keys = [o for o in list(self.env.observation_spec().keys()) if o not in self.skip_state_keys]

        self.obs_sizes = {}
        self.obs_specs = {}
        for i, j in  self.env.observation_spec().items():
            if i in self.obs_keys:
                if type(j) in [int, np.bool]: s = 1
                else:
                    l = len(j.shape)
                    if l == 0: s = 1
                    elif l == 1: s = j.shape[0]
                    elif l == 2: s = (j.shape[0]*j.shape[1])
                    else:
                         print("write code to handle this shape",j.shape); sys.exit()
                total_size +=s
                self.obs_sizes[i] = s
                self.obs_specs[i] = j
        self.observation_space = spaces.Box(-np.inf, np.inf, (total_size*k, ))

    def render(self, camera_name='', height=240, width=240, depth=False):
        if camera_name == '':
            camera_name = self.default_camera
        if self.env_type == 'dm_control':
            frame = self.sim.render(camera_id=camera_name, height=height, width=width, depth=depth)
        elif self.env_type == 'robosuite':
            frame = self.sim.render(camera_name=camera_name, height=height, width=width, depth=depth)[::-1]
        return frame

    def make_obs(self, obs):
        a = []
        for i in self.obs_keys:
            o = obs[i]
            if type(o) in [np.ndarray, np.array, list]:
               o = np.ravel(o)
            else:
              o = np.array([o])
            a.append(o)
        return np.concatenate(a)

    def make_body(self):
        if self.env_type == 'dm_control':
            bxq = np.hstack((self.env.physics.data.qpos, self.env.physics.named.data.xpos[self.xpos_target], self.env.physics.named.data.xquat[self.xpos_target]))
        if self.env_type == 'robosuite':
            r = self.env.robots[0]
            bxq = np.hstack((r._joint_positions, self.env.sim.data.site_xpos[r.eef_site_id], self.env.sim.data.body_xquat[r.eef_site_id]))
            #bx = np.hstack((r.eef_pos(), r.eef_quat()))
        return bxq

    def reset(self):
        o = self.env.reset()
        if self.env_type == 'dm_control':
            o = o.observation    
        o = self.make_obs(o)
        b = self.make_body()
        for _ in range(self.k):
            self._state.append(o)
            self._body.append(b)
        return self._get_obs(), self._get_body()

    def step(self, action):
        if self.env_type == 'robosuite':
            state, reward, done, info = self.env.step(action)
        elif self.env_type == 'dm_control':
            o  = self.env.step(action)
            done = o.step_type.last()
            state = o.observation
            reward = o.reward
            info = o.step_type
        self._state.append(self.make_obs(state))
        self._body.append(self.make_body())
        return self._get_obs(), self._get_body(), reward, done, info 

    def _get_obs(self):
        assert len(self._state) == self.k
        return np.concatenate(list(self._state), axis=0)

    def _get_body(self):
        assert len(self._body) == self.k
        return np.concatenate(list(self._body), axis=0)

def build_env(cfg, k, skip_state_keys, env_type='robosuite', default_camera=''):
    if env_type == 'robosuite':
        controller_configs = robosuite.load_controller_config(default_controller=cfg['controller'])
        if cfg['controller'] == 'JOINT_POSITION':
            controller_configs['kp'] = 150
        env = robosuite.make(env_name=cfg['env_name'], 
                         robots=cfg['robots'], 
                         controller_configs=controller_configs,
                         use_camera_obs=cfg['use_camera_obs'], 
                         use_object_obs=cfg['use_object_obs'], 
                         reward_shaping=cfg['reward_shaping'], 
                         camera_names=cfg['camera_names'], 
                         horizon=cfg['horizon'], 
                         control_freq=cfg['control_freq'], 
                         ignore_done=False, 
                         hard_reset=False, 
                         reward_scale=1.0,
                         has_offscreen_renderer=True,
                         has_renderer=False,
                           )
        xpos_target = ''
    elif env_type == 'dm_control':
        env = suite.load(cfg['robots'][0], cfg['env_name'])
        xpos_target = cfg['xpos_target']
    env = EnvStack(env, k=k, skip_state_keys=skip_state_keys, env_type=env_type, default_camera=default_camera, xpos_target=xpos_target)
    return env



def build_model(policy_name, env):
    state_dim = env.observation_space.shape[0]
    action_dim = env.control_shape
    body_dim = env.body_shape
    max_action = env.control_max
    min_action = env.control_min
    if policy_name == 'TD3':
        kwargs = {'tau':0.005, 
                'action_dim':action_dim, 'state_dim':state_dim, 'body_dim':body_dim,
                'policy_noise':0.2, 'max_policy_action':1.0, 
                'noise_clip':0.5, 'policy_freq':2, 
                'discount':0.99, 'max_action':max_action, 'min_action':min_action}
        policy = TD3.TD3(**kwargs)


    return policy, kwargs


def build_replay_buffer(cfg, env, max_size, cam_dim, seed):
    env_type = cfg['experiment']['env_type'] 
    state_dim = env.observation_space.shape[0]
    action_dim = env.control_shape
    body_dim = env.body_shape
    replay_buffer = ReplayBuffer(state_dim, body_dim, action_dim, 
                                 max_size=max_size, 
                                 cam_dim=cam_dim, 
                                 seed=seed)
    # this is a bit hacky! TODO
    replay_buffer.k = env.k
    replay_buffer.obs_keys = env.obs_keys
    replay_buffer.obs_sizes = env.obs_sizes
    replay_buffer.obs_specs = env.obs_specs
    replay_buffer.max_timesteps = env.max_timesteps
    replay_buffer.xpos_target = env.xpos_target
    replay_buffer.cfg = cfg

    replay_buffer.base_pos = env.bpos
    replay_buffer.base_ori = env.bori
    # hard code orientation 
    # TODO add conversion to rotation matrix
    replay_buffer.base_matrix = env.base_matrix
    return replay_buffer

def plot_replay(replay_buffer, savebase, frames=False):
    joint_positions = replay_buffer.bodies[:,:-7]
    next_joint_positions = replay_buffer.next_bodies[:,:-7]

    #norm_joint_positions = normalize_joints(deepcopy(joint_positions))
    #next_norm_joint_positions = normalize_joints(deepcopy(next_joint_positions))

    rpos = replay_buffer.bodies[:,-7:-7+3]
    rquat = replay_buffer.bodies[:,-7+3:]

    # find eef position according to DH
#    if robot_name in robot_attributes.keys():
#        n, ss = replay_buffer.states.shape
#        k = replay_buffer.k
#        idx = (k-1)*(ss//k) # start at most recent observation
#        data = {}
#        for key in replay_buffer.obs_keys:
#            o_size = env.obs_sizes[key]
#            data[key] = replay_buffer.states[:, idx:idx+o_size]
#            idx += o_size
#
    rdh = robotDH(replay_buffer.cfg['robot']['robots'][0])
    bm = replay_buffer.base_matrix
    f_eef = rdh.np_angle2ee(bm, joint_positions)
    # do the rotation in the beginning rather than end
    dh_pos = f_eef[:,:3,3] 
    #dh_ori = np.array([quaternion_from_matrix(f_eef[x]) for x in range(n)])
    dh_ori = np.array([mat2quat(f_eef[x]) for x in range(f_eef.shape[0])])
    f, ax = plt.subplots(3, figsize=(10,18))
    xdiff = rpos[:,0]-dh_pos[:,0]
    ydiff = rpos[:,1]-dh_pos[:,1]
    zdiff = rpos[:,2]-dh_pos[:,2]
    print('max xyzdiff', np.abs(xdiff).max(), np.abs(ydiff).max(), np.abs(zdiff).max())
    ax[0].plot(rpos[:,0], label='state')
    ax[0].plot(dh_pos[:,0], label='dh calc')
    ax[0].plot(xdiff, label='diff')
    ax[0].set_title('posx: max diff %.04f'%np.abs(xdiff).max())
    ax[0].legend()
    ax[1].plot(rpos[:,1])
    ax[1].plot(dh_pos[:,1])
    ax[1].plot(ydiff)
    ax[1].set_title('posy: max diff %.04f'%np.abs(ydiff).max())
    ax[2].plot(rpos[:,2])
    ax[2].plot(dh_pos[:,2])
    ax[2].plot(zdiff)
    ax[2].set_title('posz: max diff %.04f'%np.abs(zdiff).max())
    plt.savefig(savebase+'eef.png')
    print('saving', savebase+'eef.png')
 
    # TODO quaternion is still not right! the errors occur when i hit 1 or 0 - this must be a common thing
    # CHECK DH parameters?
    f, ax = plt.subplots(4, figsize=(10,18))
    qxdiff = rquat[:,0]-dh_ori[:,0]
    qzdiff = rquat[:,2]-dh_ori[:,2]
    qydiff = rquat[:,1]-dh_ori[:,1]
    qwdiff = rquat[:,3]-dh_ori[:,3]
    print('max qxyzwdiff',np.abs(qxdiff).max(), np.abs(qydiff).max(), np.abs(qzdiff).max(), np.abs(qwdiff).max())
    ax[0].plot(rquat[:,0], label='sqx')
    ax[0].plot(dh_ori[:,0], label='dhqx')
    ax[0].plot(qxdiff, label='diff')
    ax[0].set_title('qx: max diff %.04f'%np.abs(qxdiff).max())
    ax[0].legend()
    ax[1].plot(rquat[:,1], label='sqy')
    ax[1].plot(dh_ori[:,1], label='dhqy')
    ax[1].plot(qydiff)
    ax[1].set_title('qy: max diff %.04f'%np.abs(qydiff).max())
    ax[2].plot(rquat[:,2], label='sqz')
    ax[2].plot(dh_ori[:,2], label='dhqz')
    ax[2].plot(qzdiff)
    ax[2].set_title('qz: max diff %.04f'%np.abs(qzdiff).max())
    ax[3].plot(rquat[:,3])
    ax[3].plot(dh_ori[:,3])
    ax[3].plot(qwdiff)
    ax[3].set_title('qw: max diff %.04f'%np.abs(qwdiff).max())
    plt.savefig(savebase+'quat.png')
    print('saving', savebase+'quat.png')
    if frames:
        frames = [replay_buffer.undo_frame_compression(replay_buffer.frames[f]) for f in np.arange(len(replay_buffer.frames))]
        mimwrite(savebase+'.mp4', frames)
        print('writing', savebase+'.mp4')

class TanhTransform(pyd.transforms.Transform):
    domain = pyd.constraints.real
    codomain = pyd.constraints.interval(-1.0, 1.0)
    bijective = True
    sign = +1

    def __init__(self, cache_size=1):
        super().__init__(cache_size=cache_size)

    @staticmethod
    def atanh(x):
        return 0.5 * (x.log1p() - (-x).log1p())

    def __eq__(self, other):
        return isinstance(other, TanhTransform)

    def _call(self, x):
        return x.tanh()

    def _inverse(self, y):
        # We do not clamp to the boundary here as it may degrade the performance of certain algorithms.
        # one should use `cache_size=1` instead
        return self.atanh(y)

    def log_abs_det_jacobian(self, x, y):
        # We use a formula that is more numerically stable, see details in the following link
        # https://github.com/tensorflow/probability/commit/ef6bb176e0ebd1cf6e25c6b5cecdd2428c22963f#diff-e120f70e92e6741bca649f04fcd907b7
        return 2. * (math.log(2.) - x - F.softplus(-2. * x))


class SquashedNormal(pyd.transformed_distribution.TransformedDistribution):
    def __init__(self, loc, scale):
        self.loc = loc
        self.scale = scale

        self.base_dist = pyd.Normal(loc, scale)
        transforms = [TanhTransform()]
        super().__init__(self.base_dist, transforms)

    @property
    def mean(self):
        mu = self.loc
        for tr in self.transforms:
            mu = tr(mu)
        return mu
