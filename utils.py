import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import os
import copy
import numpy as np
import pickle
from datetime import datetime as date
from glob import glob

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
