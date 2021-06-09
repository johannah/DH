# train a new RL model:

`python train_rl.py --cfg experiments/base_dm.cfg`

## evaluate and plot the latest model from a training directory:  

`python train_rl.py --eval --frames --load_model results/base_easy_01111_reacher_1_00/ --frames`

This will result in plots for reward & target position, and a video of the episodes written as a derivitive of the most recent .pt file in the --load_model directory. 

## evaluate and plot a specific model:

`python train_rl.py --eval --frames --load_model results/base_easy_01111_reacher_1_00/base_0000355000.pt`

---

# train a behavior cloning model which predicts relative JOINT_POSITIONS for a given state 
 

### with DH on the end effector or target link:

`python train_bc.py --load_replay <path-to-rl-eval-replay-buffer>.pkl`

### on the joint angles:

`python train_bc.py --load_replay <path-to-rl-eval-replay-buffer>.pkl --loss "angle"`


## now plot and evaluate the behavior cloning 
### NOTE: This is not really relevant with dm_control - but used with robosuite and JOINT_POSITION controller

`python train_bc.py --load_model <path-to-saved-bc-model> --eval --frames`
