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
import robosuite.utils.transform_utils as T
import robosuite.utils.macros as macros
macros.IMAGE_CONVENTION = 'opencv'
from robosuite.utils.transform_utils import mat2quat

from dm_control import suite

# import TD3, TD3_kinematic
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

def dm_site_pose_in_base_from_name(self, physics, root_body, name):
    """
    A helper function that takes in a named data field and returns the pose
    of that object in the base frame.
    Args:
        name (str): Name of site in sim to grab pose
    Returns:
        np.array: (4,4) array corresponding to the pose of @name in the base frame
    """
    pos_in_world = physics.named.data.xpos[name]
    rot_in_world = physics.named.data.xmat[name].reshape((3, 3))
    pose_in_world = T.make_pose(pos_in_world, rot_in_world)

    base_pos_in_world =  physics.named.data.xpos[root_body]
    base_rot_in_world =  physics.named.data.xmat[root_body].reshape((3, 3))
    base_pose_in_world = T.make_pose(base_pos_in_world, base_rot_in_world)
    world_pose_in_base = T.pose_inv(base_pose_in_world)

    pose_in_base = T.pose_in_A_to_pose_in_B(pose_in_world, world_pose_in_base)
    return pose_in_base



def site_pose_in_base_from_name(self, sim, root_body, name):
    """
    A helper function that takes in a named data field and returns the pose
    of that object in the base frame.
    Args:
        name (str): Name of site in sim to grab pose
    Returns:
        np.array: (4,4) array corresponding to the pose of @name in the base frame
    """

    pos_in_world = sim.data.get_site_xpos(name)
    rot_in_world = sim.data.get_site_xmat(name).reshape((3, 3))
    pose_in_world = T.make_pose(pos_in_world, rot_in_world)

    base_pos_in_world = sim.data.get_body_xpos(root_body)
    base_rot_in_world = sim.data.get_body_xmat(root_body).reshape((3, 3))
    base_pose_in_world = T.make_pose(base_pos_in_world, base_rot_in_world)
    world_pose_in_base = T.pose_inv(base_pose_in_world)

    pose_in_base = T.pose_in_A_to_pose_in_B(pose_in_world, world_pose_in_base)
    return pose_in_base

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
                 self.default_camera = 'frontview'

            self.bpos = self.env.robots[0].base_pos
            self.bori = self.env.robots[0].base_ori
            # hard code orientation
            # TODO add conversion to rotation matrix
            self.base_matrix = quaternion_matrix(self.bori)
            self.base_matrix[:3, 3] = self.bpos
            
            # TODO this is hacky - but it seems the world needs to be flipped in y,z to be correct
            # Sanity checked in Jaco w/ Door
            # ensure this holds for other robots
            #self.base_matrix[1,1] = -1
            #self.base_matrix[2,2] = -1

        elif self.env_type == 'dm_control':
            self.control_min = self.env.action_spec().minimum[0]
            self.control_max = self.env.action_spec().maximum[0]
            self.control_shape = self.env.action_spec().shape[0]
            self.max_timesteps = int(self.env._step_limit)
            self.sim = self.env.physics
            if default_camera == '':
                 self.default_camera = -1
  
            self.base_matrix = np.eye(4)
            # TODO hardcoded for reacher
            # eye is right rot for reachr
            #self.base_matrix[:3, :3] =  self.env.physics.named.data.geom_xmat['root'].reshape(3,3)
            self.bpos = self.env.physics.named.data.geom_xpos['root']
            self.base_matrix[:3, 3] = self.bpos
            #self.base_matrix[1,1] = 1 # TODO FOUND EXPERIMENTALLY FOR REACHER
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
            # TODO hardcoded arm debug reacher
            pos_in_world = self.env.physics.named.data.xpos[self.xpos_target]
            rot_in_world = self.env.physics.named.data.xmat[self.xpos_target].reshape((3, 3))
            targ_rmat = T.make_pose(pos_in_world, rot_in_world).reshape(16)
            #targ_rmat = dm_site_pose_in_base_from_name(self, self.env.physics, "arm", self.xpos_target).reshape(16)
            #targ_rmat = self.env.physics.named.data.geom_xmat[self.xpos_target].reshape(16)
            bxq = np.hstack((self.env.physics.data.qpos, self.env.physics.named.data.xpos[self.xpos_target], targ_rmat))
        if self.env_type == 'robosuite':
            r = self.env.robots[0]
            #bxq = np.hstack((r._joint_positions, self.env.sim.data.site_xpos[r.eef_site_id], self.env.sim.data.get_body_xquat[r.eef_site_id]))
            #bxq = np.hstack((r._joint_positions, self.env.sim.data.site_xpos[r.eef_site_id], r.pose_in_base_from_name('gripper0_eef').reshape(16)))
            grip_rmat = site_pose_in_base_from_name(self, self.env.sim, r.robot_model.root_body, r.gripper.important_sites['grip_site'])
            # joint pos, eef in world frame, grip site in base frame
            bxq = np.hstack((r._joint_positions, self.env.sim.data.site_xpos[r.eef_site_id], grip_rmat.reshape(16)))
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
        if 'controller_config_file' in cfg.keys():
            cfg_file = os.path.abspath(cfg['controller_config_file'])
            print('loading controller from', cfg_file)
            controller_configs = robosuite.load_controller_config(custom_fpath=cfg_file)
        else:
            controller_configs = robosuite.load_controller_config(default_controller=cfg['controller'])
        #from robosuite.models.grippers import JacoThreeFingerGripper
        #gripper = JacoThreeFingerGripper
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
                'discount':0.99, 'max_action':max_action, 'min_action':min_action,
                }
        policy = TD3.TD3(**kwargs)
    if policy_name == 'TD3_kinematic':
        kwargs = {'tau':0.005, 
                'action_dim':action_dim, 'state_dim':state_dim, 'body_dim':body_dim,
                'policy_noise':0.2, 'max_policy_action':1.0, 
                'noise_clip':0.5, 'policy_freq':2, 
                'discount':0.99, 'max_action':max_action, 'min_action':min_action}
        policy = TD3_kinematic.TD3(**kwargs)

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


def get_hyperparameters(args, cfg):
    hyperparameters = vars(args)
    for k, v in cfg.items():
        if isinstance(v, dict):
            hyperparameters.update(v)
    return hyperparameters

def get_replay_state_dict(replay_buffer, use_states=[]):
    # find eef position according to DH
    n, ss = replay_buffer.states.shape
    k = replay_buffer.k
    idx = (k-1)*(ss//k) # start at most recent observation
    state_data = {'state':np.empty((n,0))}
    next_state_data = {'next_state':np.empty((n,0))}
    for key in replay_buffer.obs_keys:
        o_size = replay_buffer.obs_sizes[key]
        state_data[key] = replay_buffer.states[:, idx:idx+o_size]
        next_state_data[key] = replay_buffer.next_states[:, idx:idx+o_size]
        if key in use_states:
            state_data['state'] = np.hstack((state_data['state'], state_data[key]))
            next_state_data['next_state'] = np.hstack((next_state_data['next_state'], next_state_data[key]))
        idx += o_size
    return state_data, next_state_data


def parse_slurm_task_id(cfg, slurm_task_id, n_seeds=5):
    cfg['experiment']['seed'] += (slurm_task_id % n_seeds) * 3000
    if cfg['robot']['robots'] == ['reacher']:
        return cfg
    if slurm_task_id // n_seeds == 0:
        cfg['robot']['env_name'] = 'Door'
    elif slurm_task_id // n_seeds == 1:
        cfg['robot']['env_name'] = 'Lift'
    else:
        cfg['robot']['env_name'] = 'NutAssembly'
    return cfg


def plot_replay(env, replay_buffer, savebase, frames=False):
    joint_positions = replay_buffer.bodies[:,:-19]
    next_joint_positions = replay_buffer.next_bodies[:,:-19]

    nt = replay_buffer.bodies.shape[0]
    true_rmat = replay_buffer.bodies[:,-16:].reshape(nt, 4,4)
    true_posquat = [T.mat2pose(true_rmat[x]) for x in range(true_rmat.shape[0])]
    true_pos = true_rmat[:,:3,3]
    true_euler = np.array([T.mat2euler(a) for a in true_rmat])
    true_quat = np.array([T.mat2quat(a) for a in true_rmat])
    
    if 'robot_dh' in replay_buffer.cfg['robot'].keys():
        robot_name = replay_buffer.cfg['robot']['robot_dh']
    else:
        robot_name = replay_buffer.cfg['robot']['robots'][0]



    rdh = robotDH(robot_name)
    bm = replay_buffer.base_matrix
    #bm = np.eye(4)
    #bm[1,1] =-1
    #bm[2,2] =-1

    #pm = np.eye(4)
    #pm[1,1] *=-1
    #pm[2,2] *=-1
    dh_rmat = rdh.np_angle2ee(bm, joint_positions)
    dh_pos = dh_rmat[:,:3,3]#np.array([a[0] for a in dh_posquat])
    dh_euler = np.array([T.mat2euler(a) for a in dh_rmat])
    dh_quat = np.array([T.mat2quat(a) for a in dh_rmat])
 
    #true_ = env.sim.data.get_body_xmat('gripper0_eef')
    #dh_ori = np.array([quaternion_from_matrix(f_eef[x]) for x in range(n)])
    #dh_ori = np.array([mat2quat(f_eef[x]) for x in range(f_eef.shape[0])])
    f, ax = plt.subplots(3, figsize=(10,18))
    xdiff = true_pos[:,0]-dh_pos[:,0]
    ydiff = true_pos[:,1]-dh_pos[:,1]
    zdiff = true_pos[:,2]-dh_pos[:,2]
    print('max xyzdiff', np.abs(xdiff).max(), np.abs(ydiff).max(), np.abs(zdiff).max())
    ax[0].plot(true_pos[:,0], label='state')
    ax[0].plot(dh_pos[:,0], label='dh calc')
    ax[0].plot(xdiff, label='diff')
    ax[0].set_title('posx: max diff %.04f'%np.abs(xdiff).max())
    ax[0].legend()
    ax[1].plot(true_pos[:,1])
    ax[1].plot(dh_pos[:,1])
    ax[1].plot(ydiff)
    ax[1].set_title('posy: max diff %.04f'%np.abs(ydiff).max())
    ax[2].plot(true_pos[:,2])
    ax[2].plot(dh_pos[:,2])
    ax[2].plot(zdiff)
    ax[2].set_title('posz: max diff %.04f'%np.abs(zdiff).max())
    plt.savefig(savebase+'pos.png')
    print('saving', savebase+'pos.png')
 
    f, ax = plt.subplots(4, figsize=(10,18))
    qxdiff = true_quat[:,0]-dh_quat[:,0]
    qydiff = true_quat[:,1]-dh_quat[:,1]
    qzdiff = true_quat[:,2]-dh_quat[:,2]
    qwdiff = true_quat[:,3]-dh_quat[:,3]
    print('max qxyzwdiff',np.abs(qxdiff).max(), np.abs(qydiff).max(), np.abs(qzdiff).max(), np.abs(qwdiff).max())
    ax[0].plot(true_quat[:,0], label='sqx')
    ax[0].plot(dh_quat[:,0], label='dhqx')
    ax[0].plot(qxdiff, label='diff')
    ax[0].set_title('qx: max diff %.04f'%np.abs(qxdiff).max())
    ax[0].legend()
    ax[1].plot(true_quat[:,1], label='sqy')
    ax[1].plot(dh_quat[:,1], label='dhqy')
    ax[1].plot(qydiff)
    ax[1].set_title('qy: max diff %.04f'%np.abs(qydiff).max())
    ax[2].plot(true_quat[:,2], label='sqz')
    ax[2].plot(dh_quat[:,2], label='dhqz')
    ax[2].plot(qzdiff)
    ax[2].set_title('qz: max diff %.04f'%np.abs(qzdiff).max())
    ax[3].plot(true_quat[:,3])
    ax[3].plot(dh_quat[:,3])
    ax[3].plot(qwdiff)
    ax[3].set_title('qw: max diff %.04f'%np.abs(qwdiff).max())
    plt.savefig(savebase+'quat.png')
    print('saving', savebase+'quat.png')

    exdiff = true_euler[:,0]-dh_euler[:,0]
    eydiff = true_euler[:,1]-dh_euler[:,1]
    ezdiff = true_euler[:,2]-dh_euler[:,2]
    print('max qxyzwdiff',np.abs(exdiff).max(), np.abs(eydiff).max(), np.abs(ezdiff).max())
 
    f, ax = plt.subplots(3, figsize=(10,18))
    ax[0].plot(true_euler[:,0], label='sqx')
    ax[0].plot(dh_euler[:,0], label='dhqx')
    ax[0].plot(exdiff, label='diff')
    ax[0].legend()
    ax[1].plot(true_euler[:,1])
    ax[1].plot(dh_euler[:,1])
    ax[1].plot(eydiff)
    ax[2].plot(true_euler[:,2])
    ax[2].plot(dh_euler[:,2])
    ax[2].plot(ezdiff)
    plt.savefig(savebase+'euler.png')

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
