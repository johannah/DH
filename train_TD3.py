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
from dh_utils import  create_results_dir, find_latest_checkpoint
from replay_buffer_TD3 import ReplayBuffer, compress_frame
import TD3
from collections import deque

   
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
import gym.spaces as spaces

class EnvStack():
    def __init__(self, env, k):
        self.env = env
        self._k = k
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
        for _ in range(self._k):
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
        assert len(self._state) == self._k
        return np.concatenate(list(self._state), axis=0)

    def _get_body(self):
        assert len(self._body) == self._k
        return np.concatenate(list(self._body), axis=0)



def run_eval(num_train_steps, num_eval_episodes=10):
    for ep in range(num_eval_episodes):
        state,body = env.reset()
        if use_frames:
            frame_compressed = compress_frame(env.physics.render(height=h, width=w))
        e_step = 0
        #while not ts.last():
        done = False
        while not done:
            # Select action randomly or according to policy
            action = (
                    policy.select_action(state)
                ).clip(-kwargs['max_action'], kwargs['max_action'])
     
            next_state, next_body, reward, done, info = env.step(action) # take a random action
            e_step += 1
            if e_step == 100:
                done = True
            if use_frames:
                next_frame_compressed = compress_frame(env.physics.render(height=h, width=w))
     
                eval_replay_buffer.add(state, body, action, reward, next_state, next_body, int(done), 
                              frame_compressed=frame_compressed, 
                              next_frame_compressed=next_frame_compressed)
                frame_compressed = next_frame_compressed
            else:
                eval_replay_buffer.add(state, action, reward, next_state, int(done))
     
            state = next_state

    pickle.dump(eval_replay_buffer, open(modelbase+'_eval.pkl', 'wb'))
    

def run_train(num_steps=0, num_episodes=1000):
    for ep in range(num_episodes):
        #ts, reward, d, o = env.reset()
        done = False
        state, body =  env.reset()
        if use_frames:
            frame_compressed = compress_frame(env.physics.render(height=h, width=w))
        e_step = 0
        while not done:
            if num_steps < start_timesteps:
                action = random_state.uniform(low=-kwargs['max_action'], high=kwargs['max_action'], size=action_dim)
            else:
                # Select action randomly or according to policy
                action = (
                        policy.select_action(state)
                        + random_state.normal(0, kwargs['max_action'] * expl_noise, size=kwargs['action_dim'])
                    ).clip(-kwargs['max_action'], kwargs['max_action'])
    
     
 
            next_state, next_body, reward, done, info = env.step(action) # take a random action
            #ts, reward, _, next_o = env.step(action) # take a random action
            #next_state = format_observation(next_o)
            #angles = torch.FloatTensor(o['position']).to(device)
            #T0 = torch_dh_transform(angles[0][None], d[0], r[0], alpha[0], device)
            #_T1 = torch_dh_transform(angles[1][None], d[1], r[1], alpha[1], device)
            #T1 = torch.matmul(T0, _T1) 
            #ee = T1[:,:3,2]
            if use_frames:
                next_frame_compressed = compress_frame(env.physics.render(height=h, width=w))
     
                replay_buffer.add(state, action, reward, next_state, done, 
                              frame_compressed=frame_compressed, 
                              next_frame_compressed=next_frame_compressed)
                frame_compressed = next_frame_compressed
            else:
                replay_buffer.add(state, action, reward, next_state, done)
     
            state = next_state
            body = next_body
            num_steps+=1
            if num_steps > start_timesteps:
                policy.train(num_steps, replay_buffer, batch_size)
            if not num_steps % save_freq:
                step_filepath = os.path.join(savebase, '{}_S{:05d}_{:010d}'.format(exp_name, seed, num_steps))
                pickle.dump(replay_buffer, open(step_filepath+'.pkl', 'wb'))
                policy.save(step_filepath+'.pt')

    step_filepath = os.path.join(savebase, '{}_S{:05d}_{:010d}'.format(exp_name, seed, num_steps))
    pickle.dump(replay_buffer, open(step_filepath+'.pkl', 'wb'))
    policy.save(step_filepath+'.pt')
 
    
def plot_all(rb, modifier='', use_frames=False):
    state_names_dict = {}
    st = 0 
    en = 2
    for k in use_states:
        state_names_dict[k] = np.arange(st, en)
        st = en 
        en += 2
    plotting.plot_loss_dict(policy, modelbase+modifier) 
    plotting.plot_replay_reward(rb, modelbase+modifier, start_step=0)
    plotting.plot_states(rb.get_last_steps(rb.size), 
                         modelbase+modifier, detail_dict=state_names_dict)

    plotting.plot_actions(rb.get_last_steps(rb.size), 
                         modelbase+modifier)
    if use_frames:
        plotting.plot_frames(modelbase+modifier+'_movie.mp4', rb.get_last_steps(min([1000, rb.size])), min_action=min_action, max_action=max_action, plot_action_frames=True)


if __name__ == '__main__':
    import argparse
    from glob import glob
    parser = argparse.ArgumentParser()
    parser.add_argument('--eval', default=False, action='store_true')
    parser.add_argument('--use_frames', default=False, action='store_true')
    parser.add_argument('--env', default='manipulator', type=str)
    parser.add_argument('--task', default='bring_ball', type=str)
    parser.add_argument('--load_model', default='')
    parser.add_argument('--device', default='cuda')
    parser.add_argument('--seed', default=0, type=int)
    parser.add_argument('--k', default=3, type=int, help='num timesteps to consider')
    parser.add_argument('--num_eval_episodes', default=10, type=int)
    parser.add_argument('--exp_name_modifier', default='')
    policy_name = 'TD3'
    args = parser.parse_args()
    exp_name = '%s_%s_%s_%05d_%s'%(args.env, args.task,  policy_name, args.seed, args.exp_name_modifier)
 
    device = args.device
    seed = args.seed
    results_dir = 'results'
    expl_noise = 0.1
    
    #alpha = torch.FloatTensor([0.0, 0.0])
    #r = torch.FloatTensor([0.01, 0.01])
    #d = torch.FloatTensor([0.0, 0.0])
    train_buffer_size = int(10e6)
    # TODO hack based on reacher 1000 length episodes
    eval_buffer_size = int(args.num_eval_episodes*100)
    h=120; w=120
    
    use_frames = args.use_frames
    if use_frames:
        cam_dim = (h,w,3)
    else:
        cam_dim = (0,0,0)

    random_state = np.random.RandomState(seed)
    env = EnvStack(suite.load(args.env, args.task), args.k)
    state_dim=env.observation_space.shape[0]
    action_dim = env.action_space.shape[0]
    ## TODO does not solve joint-dependent max/min
    max_action = 1 #env.action_spec().maximum.max()
    min_action = -1 #env.action_spec().minimum.min()

    save_freq = 10000
    start_timesteps = 25000
    batch_size = 256 
    max_timesteps = 1e6

    if policy_name == 'TD3':
        kwargs = {'tau':0.005, 
                'action_dim':action_dim, 'state_dim':state_dim, 
                'policy_noise':0.2, 'max_policy_action':1.0, 
                'noise_clip':0.5, 'policy_freq':2, 
                'discount':0.99, 'max_action':max_action}
        policy = TD3.TD3(**kwargs)

    if not os.path.exists(args.load_model) or args.load_model == '' and not args.eval:
        steps = 0
        savebase = create_results_dir(exp_name, results_dir=results_dir)
        replay_buffer = ReplayBuffer(state_dim, action_dim, 
                                 max_size=train_buffer_size, 
                                 cam_dim=cam_dim, 
                                 seed=seed)
 
    else:
        if os.path.isdir(args.load_model):
            savebase = args.load_model
            loadpath = find_latest_checkpoint(args.load_model)

        else:
            loadpath  = args.load_model
            savebase = os.path.split(args.load_model)[0]
        modelbase = loadpath.replace('.pt', '')
        policy.load(loadpath)
        steps = policy.step

        replay_buffer = pickle.load(open(modelbase+'.pkl', 'rb'))
 
    if args.eval:
        plot_all(replay_buffer)
        eval_replay_buffer = ReplayBuffer(state_dim, action_dim, 
                                 max_size=eval_buffer_size, 
                                 cam_dim=cam_dim, 
                                 seed=seed)
 
        run_eval(steps, args.num_eval_episodes)
        pickle.dump(eval_replay_buffer, open(modelbase+'eval_NE%05d.pkl'%args.num_eval_episodes, 'wb'))
        plot_all(eval_replay_buffer, modifier='_eval', use_frames=args.use_frames)
    else:
        run_train(steps)

 


#replay_buffer.shrink_to_last_step()
#pickle.dump(replay_buffer, open(step_filepath+'.pkl', 'wb'))    
#plotting.plot_replay_reward(replay_buffer, savebase, start_step=0)
#plotting.plot_frames(savebase+'movie.mp4', replay_buffer.get_last_steps(num_steps))
##cmd = 'ffmpeg -i images/%04d.png -c:v libx264 -vf fps=25 -pix_fmt yuv420p _out.mp4'
#os.system(cmd)
    
