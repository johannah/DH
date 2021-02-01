import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import plotting
from  dm_control import suite
import numpy as np
import os
import sys

import pickle
import torch
from IPython import embed
from utils import torch_dh_transform
from replay_buffer import ReplayBuffer, compress_frame
import TD3

   
"""
 assign Zi along the axis of joint i
 - for a revolute joint, the joint axis is along the axis of rotation
- for a prismatic joint, the joint axis is along the axis of translation

choose xi to point along the commong perpendicular of zi and zi+1, pointing otwards the next joint
- if zi and zi+1 intersect, then choose xi to be normal to the plane of intersection

choose yi to roud out the right hand coordinate system

ai-1 distance from zi-1 to zi along zi-1
ai-1 angle from zi-1 to zi about xi-1 
di distance from xi-1 to xi along zi
thetai angle from xi-1 to xi about zi

"""
# observation is 
# position, to_target
# veloicty


# action_spec is -1, 1
exp_name = 'td3_reacher'
device = 'cpu'
alpha = torch.FloatTensor([0.0, 0.0])
r = torch.FloatTensor([0.01, 0.01])
d = torch.FloatTensor([0.0, 0.0])
seed = 1020
h = 120; w=120
cam_dim = (h,w,3)
random_state = np.random.RandomState(seed)
frames = []
positions = []
pred_positions = []
to_targets = []
env = suite.load('reacher', 'easy')
state_dim = 6
action_dim = 2
save_freq = 5000
obs_placeholder = np.zeros((1,state_dim))
use_states = ['position', 'to_target', 'velocity']
replay_buffer = ReplayBuffer(state_dim, action_dim, 
                             max_size=20000, 
                             cam_dim=cam_dim, 
                             seed=seed)

exploration_noise = 0.1
batch_size = 256
start_timesteps = 1e3
kwargs = {'tau':0.005, 
                'action_dim':action_dim, 'state_dim':state_dim, 
                'policy_noise':0.2, 'max_policy_action':1.0, 
                'noise_clip':0.5, 'policy_freq':2}
policy = TD3.TD3(**kwargs)

def format_observation(o):
    obs = obs_placeholder*0.0
    cnt = 0
    for u in use_states:
        val = o[u].flatten()[None]
        s = val.shape[1]
        obs[0,cnt:cnt+s] = val
        cnt += s
    return obs[:1,:cnt]


num_steps = 0
for ep in range(10000):
    ts, reward, discount, o = env.reset()
    state = format_observation(o)
    frame_compressed = compress_frame(env.physics.render(height=h, width=w))
    e_step = 0
    while not ts.last():
        action = random_state.rand(2)
        ts, reward, _, next_o = env.step(action) # take a random action
        next_state = format_observation(next_o)
        #angles = torch.FloatTensor(o['position']).to(device)
        #T0 = torch_dh_transform(angles[0][None], d[0], r[0], alpha[0], device)
        #_T1 = torch_dh_transform(angles[1][None], d[1], r[1], alpha[1], device)
        #T1 = torch.matmul(T0, _T1) 
        #ee = T1[:,:3,3]
        next_frame_compressed = compress_frame(env.physics.render(height=h, width=w))
        replay_buffer.add(state, action, reward, next_state, int(ts.last()), 
                          frame_compressed=frame_compressed, 
                          next_frame_compressed=next_frame_compressed)
        frame_compressed = next_frame_compressed
        next_o = o
        num_steps+=1
        if num_steps > start_timesteps:
            policy.train(num_steps, replay_buffer, batch_size)
        step_filepath = '{}_S{:05d}_{:010d}'.format(exp_name, seed, num_steps)
        if not num_steps % save_freq:
            pickle.dump(replay_buffer, open(step_filepath+'.pkl', 'wb'))
            policy.save(step_filepath)
    
replay_buffer.shrink_to_last_step()

base = 'results/random_reacher'
if not os.path.exists('results'):
    os.makedirs('results')
pickle.dump(replay_buffer, open(base+'.pkl', 'wb'))    
plotting.plot_replay_reward(replay_buffer, base, start_step=0)
plotting.plot_frames(base+'movie.mp4', replay_buffer.get_last_steps(num_steps))
#cmd = 'ffmpeg -i images/%04d.png -c:v libx264 -vf fps=25 -pix_fmt yuv420p _out.mp4'
#os.system(cmd)
    
