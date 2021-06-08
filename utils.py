import math
import os
import random
from collections import deque

import numpy as np
import scipy.linalg as sp_la

import gym
import torch
import torch.nn as nn
import torch.nn.functional as F
from skimage.util.shape import view_as_windows
from torch import distributions as pyd
from IPython import embed; 
import gym.spaces as spaces

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

class EnvStackRobosuite():
    def __init__(self, env, k):
        self.env = env
        self.k = k
        self._body = deque([], maxlen=k)
        self._state = deque([], maxlen=k)
        self.body_space = k*len(self.env.robots[0]._joint_positions)
        self.control_min = self.env.action_spec[0].min()
        self.control_max = self.env.action_spec[1].max()
        self.control_shape = self.env.action_spec[0].shape[0]
        total_size = 0
        self.obs_keys = list(self.env.observation_spec().keys())
        self.obs_sizes = {}
        self.obs_specs = {}
        for i, j in  self.env.observation_spec().items():
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

    def reset(self):
        o = self.make_obs(self.env.reset())
        b = self.env.robots[0]._joint_positions
        for _ in range(self.k):
            self._state.append(o)
            self._body.append(b)
        return self._get_obs(), self._get_body()

    def step(self, action):
        state, reward, done, info = self.env.step(action)
        self._state.append(self.make_obs(state))
        b = self.env.robots[0]._joint_positions
        self._body.append(b)
        return self._get_obs(), self._get_body(), reward, done, info 

    def _get_obs(self):
        assert len(self._state) == self.k
        return np.concatenate(list(self._state), axis=0)

    def _get_body(self):
        assert len(self._body) == self.k
        return np.concatenate(list(self._body), axis=0)


class EnvStack():
    """ stack for dm_control """
    def __init__(self, env, k):
        self.env = env
        self.k = k
        self._body = deque([], maxlen=k)
        self._state = deque([], maxlen=k)
        self.body_space = k*len(self.env.physics.data.qpos)
        self.control_min = self.env.action_spec().minimum[0]
        self.control_max = self.env.action_spec().maximum[0]
        self.control_shape = self.env.action_spec().shape
        
        self.action_space = spaces.Box(self.control_min, self.control_max, self.control_shape)
        total_size = 0
        self.obs_keys = list(self.env.observation_spec().keys())
        for i, j in  self.env.observation_spec().items():
            l = len(j.shape)
            if l == 0: total_size +=1
            elif l == 1: total_size +=j.shape[0]
            elif l == 2: total_size +=(j.shape[0]*j.shape[1])
            else:
                 print("write code to handle this shape",j.shape); sys.exit()
          
        self.observation_space = spaces.Box(-np.inf, np.inf, (total_size*k, ))

    def make_obs(self, obs):
        a = []
        for i in self.obs_keys:
            a.append(obs[i].ravel())
        return np.concatenate(a)

    def reset(self):
        o = self.make_obs(self.env.reset().observation)
        b = self.env.physics.data.qpos
        for _ in range(self.k):
            self._state.append(o)
            self._body.append(b)
        return self._get_obs(), self._get_body()

    def step(self, action):
        o = self.env.step(action)
        done = o.step_type.last()
        self._state.append(self.make_obs(o.observation))
        b = self.env.physics.data.qpos
        self._body.append(b)
        return self._get_obs(), self._get_body(), o.reward, done, o.step_type

    def _get_obs(self):
        assert len(self._state) == self.k
        return np.concatenate(list(self._state), axis=0)

    def _get_body(self):
        assert len(self._body) == self.k
        return np.concatenate(list(self._body), axis=0)




class FrameStack(gym.Wrapper):
    def __init__(self, env, k):
        gym.Wrapper.__init__(self, env)
        self.k = k
        self._frames = deque([], maxlen=k)
        self._body = deque([], maxlen=k)
        shp = env.observation_space.shape
        self.body_space = k*len(self.env.env._env._physics.data.qpos)
        self.observation_space = gym.spaces.Box(
            low=0,
            high=1,
            shape=((shp[0] * k,) + shp[1:]),
            dtype=env.observation_space.dtype)
        self._max_episode_steps = env._max_episode_steps

    def reset(self):
        obs = self.env.reset()
        b = self.env.env._env._physics.data.qpos
        for _ in range(self.k):
            self._frames.append(obs)
            self._body.append(b)
        return self._get_obs(), self._get_body()

    def step(self, action):
        obs, reward, done, info = self.env.step(action)
        self._frames.append(obs)
        b = self.env.env._env._physics.data.qpos
        self._body.append(b)
        return self._get_obs(), self._get_body(), reward, done, info

    def _get_obs(self):
        assert len(self._frames) == self.k
        return np.concatenate(list(self._frames), axis=0)

    def _get_body(self):
        assert len(self._body) == self.k
        return np.concatenate(list(self._body), axis=0)



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
