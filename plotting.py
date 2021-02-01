import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib import cm
import os
import shutil
from skvideo.io import vwrite
import shutil
from IPython import embed

import numpy as np
np.set_printoptions(precision=3, suppress=True)

def plot_rec_losses(load_path, losses):
    plt_base = load_path.replace('.pt', '')
    plt.figure()
    for lparam in losses['train'].keys():
        if lparam != 'steps':
            plt.plot(losses['train']['steps'][1:], losses['train'][lparam][1:], label='tr %s'%lparam, marker='x')
            plt.plot(losses['valid']['steps'][1:], losses['valid'][lparam][1:], label='va %s'%lparam, marker='o')
    plt.title('losses')
    plt.legend()
    plt.savefig(plt_base+'_losses.png')
    plt.close()

def plot_rec_results(results_fpath, phase, random_indexes=False, seed=10, num_plot_examples=256, plot_neighbors=0):
    
    data = np.load(results_fpath)
    
    ex = data['ex']
    rec_ex = data['rec_ex']
    ymin = min([data['st'].min(), data['rec_st'].min()])-.1
    ymax = max([data['st'].max(), data['rec_st'].max()])+.1
    exmin = min([ex[:,0].min(), rec_ex[:,0].min()])-.1
    exmax = max([ex[:,0].max(), rec_ex[:,0].max()])+.1
    eymin = min([ex[:,1].min(), rec_ex[:,1].min()])-.1
    eymax = max([ex[:,1].max(), rec_ex[:,1].max()])+.1
 
    if plot_neighbors: 
        blues = plt.cm.Blues(np.linspace(0.2,.8,plot_neighbors))[::-1]

    if random_indexes:
        random_state = np.random.RandomState(seed)
        inds = random_state.randint(0, data['st'].shape[0], num_plot_examples).astype(np.int)
    
        pltdir = results_fpath.replace('.npz', '_plots_random_%s'%phase)
    else:
        inds = np.arange(0, min([num_plot_examples,data['st'].shape[0]])).astype(np.int)
        pltdir = results_fpath.replace('.npz', '_plots_time_%s'%phase)
    print('plotting to', pltdir)

    if not os.path.exists(pltdir):
        os.makedirs(pltdir)

    sscale = 50
    diff = data['diffs']
    plt.figure()
    plt.plot(diff)
    plt.title('diff max:%.02f min:%.02f mean:%.02f std:%.02f'%(diff.max(), diff.min(), diff.mean(), diff.std()))
    plt.savefig(results_fpath.replace('.npz', '_'+phase+'_TP_dis.png'))
    plt.close()


    #plt.figure()
    #plt.plot(data['diffs'])
    #plt.title('Diff Max:%.02f Min:%.02f'%(diff.max(), diff.min()))
    #plt.savefig(results_fpath.replace('.npz', '_'+phase+'_TP_xyz_sqdis.png'))
    #plt.close()
    sscale = 100
    for num,i in enumerate(inds):
        pltname = os.path.join(pltdir, '%s_%04d.png'%(phase,num))
        if os.path.exists(pltname):
            print('skipping %s'%pltname)
        else:
            f,ax = plt.subplots(1,2,figsize=(8,5))
            ex_i = ex[i]
            rec_ex_i = rec_ex[i]
 
            ax[0].plot(data['st'][i], label='data', lw=2, marker='x', color='mediumorchid')
            ax[0].plot(data['rec_st'][i], label='rec', lw=1.5, marker='o', color='blue')

            ax[1].scatter(ex_i[0], ex_i[1], s=min([abs(ex_i[2]),.01])*sscale, marker='o', color='mediumorchid')
            ax[1].scatter(rec_ex_i[0], rec_ex_i[1], s=min([.01,abs(rec_ex_i[2])])*sscale, marker='o', color='blue')
            ax[0].set_ylim(ymin, ymax)                    
            ax[1].set_ylim(eymin, eymax)                    
            ax[1].set_xlim(exmin, exmax)                    
            if plot_neighbors:
                for nn, n in enumerate(data['neighbor_train_indexes'][i]):
                    if nn in [0, data['num_k']-1]: 
                        ax[0].plot(data['train_st'][n], label='k:%d i:%d'%(nn,n), lw=1., marker='.', color=blues[nn], alpha=.5)
                    else:
                        ax[0].plot(data['train_st'][n], lw=1., marker='.', color=blues[nn], alpha=.5)
                    ax[1].scatter(data['train_ex'][n][0], data['train_ex'][n][1], s=sscale*min([abs(data['train_ex'][n][2]),.01]), color=blues[nn])
 

            #f.suptitle('TP:[{:.2f},{:.2f},{:.2f}]\nETP:[{:.2f},{:.2f},{:.2f}]\ndiff:[{:.2f},{:.2f},{:.2f}],dis {:.2f}'.format(ex[0],ex[1],ex[2],rec_ex[0],rec_ex[1],rec_ex[2],d[0],d[1],d[2],diff[i]))
            f.suptitle('{} TP:[{:.2f},{:.2f},{:.2f}]\nETP:[{:.2f},{:.2f},{:.2f}]\ndis {:.2f}'.format(i,ex_i[0],ex_i[1],ex_i[2],rec_ex_i[0],rec_ex_i[1],rec_ex_i[2],diff[i]))
 
            #plt.legend()
            ax[0].legend()
            print('plotting %s'%pltname)
            plt.savefig(pltname)
            plt.close()
    
    pltsearch = os.path.join(pltdir, phase+'_%04d.png')
    outpath = os.path.join(pltdir, '_%s.mp4'%phase)
    cmd = 'ffmpeg -r 10 -f image2 -i %s -vcodec libx264 -crf 25 -pix_fmt yuv420p %s -y'%(pltsearch, outpath)
    print("calling: {}".format(cmd))
    os.system(cmd)

def plot_loss_dict(policy, load_model_base):
    loss_plot_path = load_model_base + '_loss.png'
    loss_dict = policy.get_loss_plot_data()
    plt.figure()
    for key, val in loss_dict.items():
        plt.plot(val[0], val[1], label=key)
    plt.title('Training Loss')
    plt.legend(loc=2)
    plt.savefig(loss_plot_path)
    plt.close()
        
def plot_frames(movie_fpath, last_steps, plot_frames=False, plot_action_frames=True, min_action=-.8, max_action=.8, min_reward=-1, max_reward=1):
    st, ac, re, nst, nd, fr, nfr = last_steps
    if fr.shape[1] == 0:
        raise ValueError; print("invalid frame shape, run with --state_pixels to ensure frames are rendered")
    n_steps = ac.shape[0]
    if plot_action_frames:
        n_actions = ac.shape[1]
        n_bars = n_actions + 1
        _, hsize, fr_wsize, nc = fr.shape
        wsize = int(fr_wsize*.20)
        n_bins = wsize
        cp = n_bins//2
        hw = hsize//n_bars
        action_bins = np.linspace(min_action, max_action, n_bins)
        reward_bins = np.linspace(min_reward, max_reward, n_bins)
        canvas = np.zeros((n_steps, hsize, wsize+1, nc), dtype=np.uint8)
        viridis = cm.get_cmap('viridis', n_actions)
        action_colors = np.array([np.array(viridis(x)[:nc]) for x in  np.linspace(0, 1, n_actions)])
        action_colors = (action_colors*255).astype(np.uint8)
        # red rewward
        reward_color = np.array([255,0,0])[:nc]
        try:
            for s in range(n_steps):
                for an,av in enumerate(ac[s]):
                    b = np.digitize(av, action_bins, right=True)
                    c = action_colors[an]
                    inds = np.linspace(cp-(cp-b), cp, np.abs(cp-b)+1, dtype=np.int)
                    canvas[s,hw*an:(hw*an)+hw,inds] = c
                rb = np.digitize(re[s][0], reward_bins, right=True)
                rinds = np.linspace(cp-(cp-rb), cp, np.abs(cp-rb)+1, dtype=np.int)
                canvas[s,hw*(n_bars-1):(hw*n_bars),rinds] = reward_color
            #output = np.concatenate((fr, canvas, nfr), 2)
            output = np.concatenate((canvas, nfr), 2)
            vwrite(movie_fpath, output)
        except Exception as e:
            print(e)
            embed()
    else:
        vwrite(movie_fpath, fr)
    if plot_frames:
        dir_path = movie_fpath.replace('.mp4', '')
        if os.path.exists(dir_path):
            shutil.rmtree(dir_path)
        os.makedirs(dir_path)
        TDIS = np.arange(0,3)
        init = 3 + 13+13+13
        # DEBUG DH PARAMS
        #DH1 = np.arange(init,              init+3)
        #DH2 = np.arange(init+3,            init+3+3)
        #DH3 = np.arange(init+3+3,          init+3+3+3)
        #DH4 = np.arange(init+3+3+3,        init+3+3+3+3) 
        #DH5 = np.arange(init+3+3+3+3,      init+3+3+3+3+3) 
        #DH6 = np.arange(init+3+3+3+3+3,    init+3+3+3+3+3+3) 
        #DH7 = np.arange(init+3+3+3+3+3+3,  init+3+3+3+3+3+3+3) 
        #TP = np.arange(init+(3*7), init+(3*7)+3)
        #minit = init+(3*7)+3
        #J4 = np.arange(minit, minit+3)
        #J6 = np.arange(minit+3, minit+3+3)
        #fing = np.arange(minit+3+3, minit+3+3+3)
        #target = np.arange(minit+3+3+3, minit+3+3+3+3)
        for n in range(fr.shape[0]):
            f,ax = plt.subplots(1,2, figsize=(14,10))
            ax[0].imshow(fr[n])
            ax[1].imshow(nfr[n])

            #TPn =  'TP:x{:.2f}y{:.2f}z{:.2f}'.format(*st[n,TP]) 
            #targetn = 'TAR:x{:.2f}y{:.2f}z{:.2f}'.format(*st[n,target]) 
            disn = 'S{}DISx{:.2f}y{:.2f}z{:.2f}'.format(n, *st[n,TDIS])
            #target_title = disn + '\n' + targetn + '\n' + TPn
            target_title = disn 
            act_title = ",".join(["{:.1f}".format(av) for av in ac[n][:7]])
            #dh1 =  'Dj1:x{:.2f}y{:.2f}z{:.2f}'.format(*st[n,DH1]) 
            #dh2 =  'Dj2:x{:.2f}y{:.2f}z{:.2f}'.format(*st[n,DH2]) 
            #dh3 =  'Dj3:x{:.2f}y{:.2f}z{:.2f}'.format(*st[n,DH3]) 
            #dh4 =  'Dj4:x{:.2f}y{:.2f}z{:.2f}'.format(*st[n,DH4])            
            #dh5 =  'Dj5:x{:.2f}y{:.2f}z{:.2f}'.format(*st[n,DH5])            
            #dh6 =  'Dj6:x{:.2f}y{:.2f}z{:.2f}'.format(*st[n,DH6])            
            #dh7 =  'Dj7:x{:.2f}y{:.2f}z{:.2f}'.format(*st[n,DH7])            
            #m4 =   'mj4:x{:.2f}y{:.2f}z{:.2f}'.format(*st[n,J4])
            #m6 =   'mj6:x{:.2f}y{:.2f}z{:.2f}'.format(*st[n,J6]) 
            #mfing= 'mfi:x{:.2f}y{:.2f}z{:.2f}'.format(*st[n,fing])
            #ax[0].set_title(target_title+'\n'+act_title)
            ax[1].set_title(act_title)
            #ax[1].set_title('\n'.join([dh1, dh2, dh3, dh4, dh5, dh6, dh7, m4, m6, mfing]))
            #ax[1].set_title(nst_title+"S:%s R:%s"%(n,re[n]))
            img_path = os.path.join(dir_path, 'frame_%05d.png'%n)
            if not n %20:
                print('writing', img_path)
            plt.savefig(img_path)
            plt.close()
        cmd = "ffmpeg -pattern_type glob -i '%s' -c:v libx264 '%s' -y"%(os.path.join(dir_path, '*.png'), os.path.join(dir_path, '_movie.mp4'))
        print('calling {}'.format(cmd))
        os.system(cmd)

def rolling_average(a, n=5) :
    if n == 0:
        return a
    ret = np.cumsum(a, dtype=float)
    ret[n:] = ret[n:] - ret[:-n]
    return ret[n - 1:] / n

def plot_states(last_steps, load_model_base, detail_dict):
    st, ac, re, nst, nd, fr, nfr = last_steps
    plt.figure()
    plt.title('reward')
    plt.plot(re)
    plt.savefig(load_model_base+'_reward.png')
    plt.xlabel('steps')
    plt.ylabel('step reward')
    plt.close()
   
    for key, indexer in detail_dict.items():
        plt.plot(st[:,indexer])
        plt.figure()
        plt.title(key)
        plt.plot(st[:,indexer])
        plt.savefig(load_model_base+'_state_%s.png'%(key))
        plt.xlabel('steps')
        plt.ylabel(key)
        plt.close()

def plot_position_actions(last_steps, load_model_base, relative=True):
    # only works for jaco
    st, actions, re, nst, nd, fr, nfr = last_steps
    joint_states = st[:,3:3+13]
    joint_next_states = nst[:,3:3+13]
    for an in range(actions.shape[1]):
        plt.figure()
        aname = 'action_%02d'%an
        plt.title(aname)
        if relative:
            plt.plot(actions[:,an], label='cmd rel', lw=3)
            cmd_action = actions[:,an]+joint_states[:,an]
        else:
            cmd_action = actions[:,an]
        plt.plot(cmd_action, label='cmd', lw=2.5)
        plt.plot(joint_next_states[:,an], label='next state', lw=2)
        error = joint_next_states[:,an] - cmd_action
        plt.plot(error, label='pos error', lw=1.5)
        plt.legend()
        plt.savefig(load_model_base+'_action_%02d.png'%(an))
        plt.xlabel('steps')
        plt.close()

def plot_replay_reward(replay_buffer, load_model_base, start_step=0, name_modifier=''):
    st = np.array(replay_buffer.episode_start_times)
    plt.figure()
    plt.title("Episode Time")
    plt.plot(st[1:]-st[:-1])
    plt.savefig(load_model_base+'_seconds_episode_%s.png'%name_modifier)
    plt.xlabel('episode')
    plt.ylabel('seconds')
    plt.close()

    plt.figure()
    plt.title("filtered reward")
    plt.plot(rolling_average(replay_buffer.episode_rewards, 10))
    plt.savefig(load_model_base+'_rewards_episode_filt_%s.png'%name_modifier)
    plt.xlabel('episode')
    plt.ylabel('reward')
    plt.close()

    plt.figure()
    plt.title("reward")
    plt.plot(replay_buffer.episode_rewards)
    plt.savefig(load_model_base+'_rewards_episode_%s.png'%name_modifier)
    plt.xlabel('episode')
    plt.ylabel('reward')
    plt.close()

    plt.figure()
    plt.title("cumulative reward")
    plt.plot(np.cumsum(replay_buffer.episode_rewards))
    plt.savefig(load_model_base+'_cumulative_episode_%s.png'%name_modifier)
    plt.xlabel('episode')
    plt.ylabel('total reward')
    plt.close()

    plt.figure()
    plt.title("reward")
    plt.plot(np.array(replay_buffer.episode_start_steps[1:])+start_step, replay_buffer.episode_rewards)
    plt.savefig(load_model_base+'_rewards_step_%s.png'%name_modifier)
    plt.xlabel('steps')
    plt.ylabel('reward')
    plt.close()

    plt.figure()
    plt.title("cumulative reward")
    plt.plot(np.array(replay_buffer.episode_start_steps[1:])+start_step, np.cumsum(replay_buffer.episode_rewards))
    plt.savefig(load_model_base+'_cumulative_step_%s.png'%name_modifier)
    plt.xlabel('steps')
    plt.ylabel('total reward')
    plt.close()
 

