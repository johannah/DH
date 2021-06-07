import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

def compare_plot_losses(mse_loss_path, dh_loss_path):
     plt.figure()
     for loss_name, loss_path in [('mse', mse_loss_path), ('DH', dh_loss_path)]:
         losses = np.load(loss_path)
         for phase, ll in [ ('valid',losses['valid']),('train', losses['train'])]:
             plt.plot(ll[1:,0], ll[1:,1], label=loss_name+phase, marker='o')
     plt.title('losses')
     plt.legend()
     fname = loss_path.replace('.npz', '_compare.png')
     print("saving loss image: {}".format(fname))
     plt.savefig('compare.png')
     plt.close()
     from IPython import embed; embed()


fp_mse= '../DH_old/results/21-02-09_v5_lstm_act_angle_00/model_0009821792.pt'
fp_dh = '../DH_old/results//21-02-08_v5_lstm_act_DH_00/model_0009821792.pt'
compare_plot_losses(fp_mse.replace('.pt', '_losses.npz'), fp_dh.replace('.pt', '_losses.npz'))
