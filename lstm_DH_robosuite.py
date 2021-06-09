import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import json
import torch
import torch.nn as nn
from torch.nn.utils.clip_grad import clip_grad_norm
import torch.optim as optim
import torch.nn.init as init
import torch.nn.functional as F
from replay_buffer_TD3 import ReplayBuffer, compress_frame
from train_robosuite import build, make_model
from imageio import mimwrite
import pickle
import numpy as np
from copy import deepcopy
import time
import os, sys
import numpy as np
import shutil

from dh_utils import find_latest_checkpoint, create_results_dir
from dh_utils import robotDH, seed_everything, normalize_joints
from dh_utils import load_robosuite_data, get_data_norm_params, quaternion_from_matrix, quaternion_matrix, robot_attributes
from robosuite.utils.transform_utils import mat2quat
from IPython import embed 
import imageio
torch.set_num_threads(2)

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

def forward_pass(input_data):
    #input_data = (input_data-train_mean)/train_std
    input_noise = torch.normal(torch.zeros_like(input_data), noise_std*torch.ones_like(input_data))
    x = input_data + input_noise
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
    return y_pred

def train(data, step=0, n_epochs=2000):
    for epoch in range(n_epochs):
        for phase in ['valid', 'train']:
            # time, batch, features
            n_samples = data[phase]['states'].shape[1]
            indexes = np.arange(0, n_samples)
            random_state.shuffle(indexes)
            st = 0
            en = min([st+batch_size, n_samples])
            running_loss = 0
            bs = en-st
            while en <= n_samples and bs > 0:
                opt.zero_grad()
                x = torch.FloatTensor(data[phase]['states'][:,indexes[st:en]]).to(device)

                #diff_pred = torch.tanh(forward_pass(x))
                diff_pred = forward_pass(x)
                ts,bs,f = diff_pred.shape 
                if args.loss == 'DH':
                    joints = torch.FloatTensor(data[phase]['joints'][:,indexes[st:en]]).to(device)
                    jt_pred = diff_pred + joints
                    ee_target = torch.FloatTensor(data[phase]['target_ee'][:,indexes[st:en]]).to(device)
                    rot_mat_pred = robot_dh.torch_angle2ee(base_matrix, jt_pred.contiguous().view(ts*bs,f)).contiguous().view(ts,bs,4,4)
                    ee_pred = rot_mat_pred[:,:,:3,3]
                    loss = criterion(ee_pred, ee_target)
                elif args.loss == 'angle':
                    diff_targets = torch.FloatTensor(data[phase]['actions'][:,indexes[st:en]]).to(device)
                    loss = criterion(diff_pred, diff_targets)
                    tb_writer.add_scalars('BC_DH_loss', { 'DH_%s_loss'%phase:loss,}, step)

                if phase == 'train':
                    clip_grad_norm(lstm.parameters(), grad_clip)
                    loss.backward()
                    step+=bs
                    opt.step()
                    if not step%(bs*10):
                        tb_writer.add_scalars('BC_DH_loss',{'%s_%s_loss'%(args.loss, phase):loss}, step)
                st = en
                en = min([st+batch_size, n_samples+1])
                bs = en-st
            tb_writer.add_scalars('BC_DH_loss',{'%s_%s_loss'%(args.loss, phase):loss}, step)

            print('{} epoch:{} step:{} loss:{}'.format(phase, epoch, step, loss))
        if not epoch % save_every_epochs:
            model_dict = {'model':lstm.state_dict(), 'train_cnt':step}
            fbase = os.path.join(savebase, 'lstm_model_%010d'%(step))
            print('saving model', fbase)
            torch.save(model_dict, fbase+'.pt') 
    model_dict = {'model':lstm.state_dict(), 'train_cnt':step}
    fbase = os.path.join(savebase, 'lstm_model_%010d'%(step))
    print('saving model', fbase)
    torch.save(model_dict, fbase+'.pt') 
 
def load_data():
     
    # ASSUMES DATA DOES NOT WRAP (IT DOESN"T IN EVAL)
    print('loading data from', args.load_replay)
    replay = pickle.load(open(args.load_replay, 'rb'))
    starts = np.array(replay.episode_start_steps[:-1], dtype=np.int)
    random_state.shuffle(starts)
 
    # median first reward happens at ts 19, max occurs at 642. Choose a smallish number to start with
    max_ts = replay.cfg['robot']['horizon']
    #target_states = np.zeros((max_ts, len(starts), 4)) 
    j_size = replay.next_joint_positions.shape[1]
    actions = np.zeros((max_ts, len(starts), j_size)) 
    joints = np.zeros((max_ts, len(starts), j_size)) 
    target_joints = np.zeros((max_ts, len(starts), j_size)) 
    target_ee = np.zeros((max_ts, len(starts), 3)) 
    n, ss = replay.states.shape
    k = replay.k
    idx = (k-1)*(ss//k) # start at most recent observation
    data_idx = {}
    for key in replay.obs_keys:
        o_size = replay.obs_sizes[key]
        data_idx[key] = (idx,idx+o_size)
        idx += o_size
    use_keys = [key for key in data_idx.keys() if key not in skip_state_keys] 
    for key_cnt, key in enumerate(use_keys):
        print('adding', key)
        if not key_cnt:
            _st = replay.states[:,data_idx[key][0]:data_idx[key][1]]
        else:
            _st = np.hstack((_st, replay.states[:,data_idx[key][0]:data_idx[key][1]]))
 
    s_size = _st.shape[1]
    states = np.zeros((max_ts, len(starts), s_size)) 
    _n_eef = replay.next_states[:,data_idx['robot0_eef_pos'][0]:data_idx['robot0_eef_pos'][1]]
    for xx, s in enumerate(starts):
        # TODO hack to make same
        indexes = np.arange(s, s+max_ts, dtype=np.int)
        jts = replay.joint_positions[s:s+max_ts]
        next_jts = replay.next_joint_positions[s:s+max_ts]
        states[:,xx] = _st[s:s+max_ts]
        diff = next_jts-jts
        #TODO specific to door! 
        # ignore 
        actions[:,xx] = diff
        joints[:,xx] = jts
        target_joints[:,xx] = next_jts
        target_ee[:,xx] = _n_eef[s:s+max_ts]
    #target_ee = robot_dh.angle2ee(torch.FloatTensor(target_joints).to(device)).cpu().numpy()
    # position, to_target, velocity
    n_episodes = target_ee.shape[1]
    
    st_val = max([1,int(n_episodes*.15)])

    data = {'train':{}, 'valid':{}}
    data['valid']['states'] =  states[:,:st_val]
    data['train']['states'] =  states[:,st_val:]
    data['valid']['actions'] =  actions[:,:st_val]
    data['train']['actions'] =  actions[:,st_val:]
    data['valid']['joints'] =  joints[:,:st_val]
    data['train']['joints'] =  joints[:,st_val:]
    data['valid']['target_joints'] =  target_joints[:,:st_val]
    data['train']['target_joints'] =  target_joints[:,st_val:]
    data['valid']['target_ee'] =  target_ee[:,:st_val]
    data['train']['target_ee'] =  target_ee[:,st_val:]
    data['base_matrix'] = replay.base_matrix 
    print('actions', actions.max(), actions.min())
    embed()
    return data

def setup_eval():
   
    print('loading model: %s'%load_model)
    env = build(cfg['robot'], 1,skip_state_keys)    
    if 'eval_seed' in cfg['experiment'].keys():
        eval_seed = cfg['experiment']['eval_seed'] + 1000
    else:
        eval_seed = cfg['experiment']['seed'] + 1000
    if args.frames: cam_dim = (240,240,3)
    else:
       cam_dim = (0,0,0)
 
    if 'eval_replay_buffer_size' in cfg['experiment'].keys():
        eval_replay_buffer_size = cfg['experiment']['eval_replay_buffer_size']
    else:
        eval_replay_buffer_size =  cfg['robot']['horizon']*args.num_eval_episodes
    print('running eval for %s steps'%eval_replay_buffer_size)
 
    state_dim = env.observation_space.shape[0]
    action_dim = env.control_shape
    max_action = env.control_max
    min_action = env.control_min
    kwargs = {'min_action':min_action, 'max_action':max_action, 'state_dim':state_dim, 'action_dim':action_dim}
    replay_buffer = ReplayBuffer(state_dim, action_dim, 
                                 max_size=eval_replay_buffer_size, 
                                 cam_dim=cam_dim, 
                                 seed=eval_seed)
 
    savebase = load_model.replace('.pt','_BC_eval_%06d_S%06d'%(eval_replay_buffer_size, eval_seed))
    replay_file = savebase+'.pkl' 
    movie_file = savebase+'_%s.mp4' %args.camera
    if not os.path.exists(replay_file):
        rewards, replay_buffer = run_BC_eval(env, replay_buffer, kwargs, cfg, cam_dim, savebase)
        if args.frames:
            _, _, _, _, _, frames, next_frames = replay_buffer.get_indexes(np.arange(len(replay_buffer.frames)))
            mimwrite(movie_file, frames)

def run_BC_eval(env, replay_buffer, kwargs, cfg, cam_dim, savebase):
    robot_name = cfg['robot']['robots'][0]
    target_robot_name = cfg['experiment']['target_robot_name']
    num_steps = 0
    total_steps = replay_buffer.max_size-1
    use_frames = cam_dim[0] > 0
    if use_frames:
        print('recording camera: %s'%args.camera)

    h, w, c = cam_dim
    rewards = []
    joint_positions = []
    next_joint_positions = []
    with torch.no_grad():
        while num_steps < total_steps:
            #ts, reward, d, o = env.reset()
            done = False
            state, body =  env.reset()
            if use_frames:
                frame_compressed = compress_frame(env.env.sim.render(camera_name=args.camera,height=h, width=w)[::-1])
            ep_reward = 0
            e_step = 0

            # IT SEEMS LIKE BASE_POS DOESNT CHANGE for DOOR/Jaco - will need to change things up if it does
            print(env.env.robots[0].base_pos)
            print(env.env.robots[0].base_ori)
            ts = cfg['robot']['horizon']
            base_x = torch.zeros((ts, 1, input_size)).to(device)
            h1_tm1 = torch.zeros((1, hidden_size)).to(device)
            c1_tm1 = torch.zeros((1, hidden_size)).to(device)
            h2_tm1 = torch.zeros((1, hidden_size)).to(device)
            c2_tm1 = torch.zeros((1, hidden_size)).to(device)
            action_pred = torch.zeros((ts, 1, output_size)).to(device)

            ex_trace = data['train']['states'][:,0:1]
            ex_action = data['train']['actions'][:,0]
            while not done:
                # Select action randomly or according to policy
                joint_positions.append(env.env.robots[0]._joint_positions)
                base_x[e_step] = torch.FloatTensor(state)
                #base_x[e_step] = torch.FloatTensor(ex_trace[e_step])
                output, h1_tm1, c1_tm1, h2_tm1, c2_tm1 = lstm(base_x[e_step], h1_tm1, c1_tm1, h2_tm1, c2_tm1)
                pred_action = output.clip(-kwargs['max_action'], kwargs['max_action'])[0].cpu().numpy()
                fake_grip = np.array([1.0])
                #pred_action = ex_action[e_step]
                pred_action = np.zeros_like(pred_action)
                pred_action[3] = .2
                pred_action[4] = .2
                action = np.hstack((pred_action, fake_grip))
                next_state, next_body, reward, done, info = env.step(action) # take a random action
                next_joint_positions.append(env.env.robots[0]._joint_positions)
                ep_reward += reward
                if use_frames:
                    next_frame_compressed = compress_frame(env.env.sim.render(camera_name=args.camera, height=h, width=w)[::-1])
                    replay_buffer.add(state, action, reward, next_state, done, 
                                  frame_compressed=frame_compressed, 
                                  next_frame_compressed=next_frame_compressed)
                    frame_compressed = next_frame_compressed
                else:
                    replay_buffer.add(state, action, reward, next_state, done)
                state = next_state
                body = next_body
                e_step += 1
                num_steps+=1
            rewards.append(ep_reward)
            
        n, ss = replay_buffer.states.shape
        bpos = env.env.robots[0].base_pos
        bori = env.env.robots[0].base_ori 
        replay_buffer.base_pos = bpos
        replay_buffer.base_ori = bori
        replay_buffer.k = env.k
        replay_buffer.obs_keys = env.obs_keys
        replay_buffer.obs_sizes = env.obs_sizes
        replay_buffer.obs_specs = env.obs_specs
        # hard code orientation 
        # TODO add conversion to rotation matrix
        base_matrix = quaternion_matrix(bori)
        base_matrix[:3, 3] = bpos
        # TODO this is hacky - but it seems the world needs to be flipped in y,z to be correct
        # Sanity checked in Jaco w/ Door  
        # ensure this holds for other robots
        base_matrix[1,1] = -1 
        base_matrix[2,2] = -1
        replay_buffer.base_matrix = base_matrix

        joint_positions = np.array(joint_positions)
        next_joint_positions = np.array(next_joint_positions)
        
        norm_joint_positions = normalize_joints(deepcopy(joint_positions))
        next_norm_joint_positions = normalize_joints(deepcopy(next_joint_positions))

        replay_buffer.norm_joint_positions = joint_positions
        replay_buffer.joint_positions = norm_joint_positions
        replay_buffer.next_norm_joint_positions = next_joint_positions
        replay_buffer.next_joint_positions = next_norm_joint_positions
        replay_buffer.robot_name = target_robot_name
        replay_buffer.cfg = cfg
        pickle.dump(replay_buffer, open(savebase + '.pkl', 'wb'))
        plt.figure()
        plt.plot(rewards)
        plt.title('eval episode rewards')
        plt.savefig(savebase+'.png')
 
        # find eef position according to DH
        if robot_name in robot_attributes.keys():
            n, ss = replay_buffer.states.shape
            k = replay_buffer.k
            idx = (k-1)*(ss//k) # start at most recent observation
            rdata = {}
            for key in replay_buffer.obs_keys:
                o_size = env.obs_sizes[key]
                rdata[key] = replay_buffer.states[:, idx:idx+o_size]
                idx += o_size

            rdh = robotDH(robot_name)
            f_eef = rdh.np_angle2ee(base_matrix, norm_joint_positions)
            # do the rotation in the beginning rather than end
            #f_eef = np.array([np.dot(base_matrix, r_eef[x]) for x in range(n)])
            dh_pos = f_eef[:,:3,3] 
            dh_ori = np.array([quaternion_from_matrix(f_eef[x]) for x in range(n)])
            dh_ori = np.array([mat2quat(f_eef[x]) for x in range(n)])

            f, ax = plt.subplots(3, figsize=(10,18))
            xdiff = rdata['robot0_eef_pos'][:,0]-dh_pos[:,0]
            ydiff = rdata['robot0_eef_pos'][:,1]-dh_pos[:,1]
            zdiff = rdata['robot0_eef_pos'][:,2]-dh_pos[:,2]
            print('max xyzdiff', np.abs(xdiff).max(), np.abs(ydiff).max(), np.abs(zdiff).max())
            ax[0].plot(rdata['robot0_eef_pos'][:,0], label='state')
            ax[0].plot(dh_pos[:,0], label='dh calc')
            ax[0].plot(xdiff, label='diff')
            ax[0].set_title('posx: max diff %.04f'%np.abs(xdiff).max())
            ax[0].legend()
            ax[1].plot(rdata['robot0_eef_pos'][:,1])
            ax[1].plot(dh_pos[:,1])
            ax[1].plot(ydiff)
            ax[1].set_title('posy: max diff %.04f'%np.abs(ydiff).max())
            ax[2].plot(rdata['robot0_eef_pos'][:,2])
            ax[2].plot(dh_pos[:,2])
            ax[2].plot(zdiff)
            ax[2].set_title('posz: max diff %.04f'%np.abs(zdiff).max())
            plt.savefig(savebase+'eef.png')
            """
             From Robosuite paper
             rotations from the current end-effector orientation in the form of axis-angle coordinates (rx, ry, rz), where the direction represents the axis and the magnitude
             represents the angle (in radians). Note that for OSC, the rotation axes are taken
             relative to the global world coordinate frame, whereas for IK, the rotation axes
             are taken relative to the end-effector frame, NOT the global world coordinate
             frame!
            """
 
            # TODO quaternion is still not right! the errors occur when i hit 1 or 0 - this must be a common thing
            # CHECK DH parameters?
            f, ax = plt.subplots(4, figsize=(10,18))
            qxdiff = rdata['robot0_eef_quat'][:,0]-dh_ori[:,0]
            qzdiff = rdata['robot0_eef_quat'][:,2]-dh_ori[:,2]
            qydiff = rdata['robot0_eef_quat'][:,1]-dh_ori[:,1]
            qwdiff = rdata['robot0_eef_quat'][:,3]-dh_ori[:,3]
            print('max qxyzwdiff',np.abs(qxdiff).max(), np.abs(qydiff).max(), np.abs(qzdiff).max(), np.abs(qwdiff).max())
            ax[0].plot(rdata['robot0_eef_quat'][:,0], label='sqx')
            ax[0].plot(dh_ori[:,0], label='dhqx')
            ax[0].plot(qxdiff, label='diff')
            ax[0].set_title('qx: max diff %.04f'%np.abs(qxdiff).max())
            ax[0].legend()
            ax[1].plot(rdata['robot0_eef_quat'][:,1], label='sqy')
            ax[1].plot(dh_ori[:,1], label='dhqy')
            ax[1].plot(qydiff)
            ax[1].set_title('qy: max diff %.04f'%np.abs(qydiff).max())
            ax[2].plot(rdata['robot0_eef_quat'][:,2], label='sqz')
            ax[2].plot(dh_ori[:,2], label='dhqz')
            ax[2].plot(qzdiff)
            ax[2].set_title('qz: max diff %.04f'%np.abs(qzdiff).max())
            ax[3].plot(rdata['robot0_eef_quat'][:,3])
            ax[3].plot(dh_ori[:,3])
            ax[3].plot(qwdiff)
            ax[3].set_title('qw: max diff %.04f'%np.abs(qwdiff).max())
            plt.savefig(savebase+'quat.png')
    return rewards, replay_buffer
 

if __name__ == '__main__':
    import argparse
    from glob import glob
    from torch.utils.tensorboard import SummaryWriter
    parser = argparse.ArgumentParser()
    parser.add_argument('--load_replay', default='')
    parser.add_argument('--eval', default=False, action='store_true')
    parser.add_argument('--load_model', default='')
    parser.add_argument('--loss', default='DH')
    parser.add_argument('--device', default='cuda')
    parser.add_argument('--target_robot_name', default='Jaco', choices=['Jaco'])
    parser.add_argument('--frames', action='store_true', default=False)
    parser.add_argument('--camera', default='agentview', choices=['frontview', 'sideview', 'birdview', 'agentview'])
    parser.add_argument('--num_eval_episodes', default=10, type=int)
 
    args = parser.parse_args()
    seed = 323
    seed_everything(seed)
    random_state = np.random.RandomState(seed)
    # some keys are robot-specifig!
    skip_state_keys = ['robot0_joint_pos_cos', 'robot0_joint_pos_sin','robot0_joint_vel', 'robot0_proprio-state']
    # TODO log where data was trained 

    if args.load_model != '':
        if os.path.isdir(args.load_model):
            load_model = sorted(glob(os.path.join(args.load_model, '*.pt')))[-1]
            load_dir = args.load_model
        else:
            assert args.load_model.endswith('.pt')
            load_model = args.load_model
            load_dir, model_name = os.path.split(args.load_model)

 
        args.load_replay = os.path.split(load_dir)[0] +'.pkl'
        agent_load_dir = os.path.split(os.path.split(load_dir)[0])[0]
        #agent_load_dir = os.path.split(os.path.split(os.path.split(load_dir)[0])[0])[0]

        agent_cfg_path = os.path.join(agent_load_dir, 'cfg.txt')
        print('cfg', agent_cfg_path)
        if not os.path.exists(agent_cfg_path):
            agent_cfg_path = os.path.join(agent_load_dir, 'cfg.cfg')
        cfg = json.load(open(agent_cfg_path))
        cfg['experiment']['target_robot_name'] = args.target_robot_name
        cfg['experiment']['bc_seed'] = cfg['experiment']['seed'] + random_state.randint(10000)
        cfg['robot']['controller'] = "JOINT_POSITION" 
        cfg['robot']['robots'] = [args.target_robot_name] 
    else: 
        fdir, fname = os.path.split(args.load_replay)
        _, ddir = os.path.split(fdir)
        exp_name = 'roboBC_act_%s'%(args.loss)
     
    device = args.device
    results_dir = args.load_replay.replace('.pkl', '')
    # set random seed to 0
    noise_std = 3
    grad_clip = 5
    hidden_size = 1024
    batch_size = 32
    save_every_epochs = 100

    # TODO 
    robot_dh = robotDH(robot_name=args.target_robot_name, device=device)

    data = load_data()
    base_matrix = torch.FloatTensor((data['base_matrix'])).to(device)
    #_t,_b,_f = _d.shape
    #train_mean, train_std = get_data_norm_params(_d.reshape(_t*_b,_f), device=device)
     
    input_size = data['valid']['states'].shape[2]
    output_size =  data['valid']['actions'].shape[2]

    lstm = LSTM(input_size=input_size, output_size=output_size, hidden_size=hidden_size).to(device)
    criterion = nn.MSELoss()

    if not args.eval and args.load_model == '':
        step = 0 
        savebase = create_results_dir(exp_name, results_dir=results_dir)
    else:
        if os.path.isdir(args.load_model): 
            savebase = args.load_model 
            loadpath = find_latest_checkpoint(args.load_model)
        else:
            loadpath  = args.load_model
            savebase = os.path.split(args.load_model)[0]
            
    if args.load_model != '':
        print("LOADING MODEL FROM", loadpath) 
        modelbase = loadpath.replace('.pt', '_')
        load_dict = torch.load(loadpath, map_location=device)
        step = load_dict['train_cnt']
        lstm.load_state_dict(load_dict['model'])

    if args.eval:
        setup_eval()
    else:
        pickle.dump(data, open(savebase+'_data.pkl', 'wb'))
        tb_writer = SummaryWriter(savebase)
        # use LBFGS as optimizer since we can load the whole data to train
        opt = optim.Adam(lstm.parameters(), lr=0.0001)
        train(data, step, n_epochs=2000)


   
