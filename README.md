


### train a new RL model:

`python train_rl.py --cfg experiments/base_dm.cfg`

### evaluate and plot the latest model from a training directory:  

`python train_rl.py --eval --frames --load_model results/base_easy_01111_reacher_1_00/ --frames`

This will result in plots for reward & target position, and a video of the episodes written as a derivitive of the most recent .pt file in the --load_model directory. 

### evaluate and plot a specific model:

`python train_rl.py --eval --frames --load_model results/base_easy_01111_reacher_1_00/base_0000355000.pt 

## train a behavior cloning model which predicts relative JOINT_POSITIONS for a given state 

### with DH on the end effector or target link:

`python train_bc.py --load_replay results/base_easy_01111_reacher_1_00/base_0000025000_eval_003000_S030000.pkl`

### with MSE on the joint angles:

`python train_bc.py --load_replay results/base_easy_01111_reacher_1_00/base_0000025000_eval_003000_S030000.pkl` --loss "angle"`


### now plot and evaluate the behavior cloning 

`python train_bc.py --load_model results/base_easy_01111_reacher_1_00/base_0000025000_eval_003000_S002111/21-06-09_roboBC_act_DH_00/ --eval --frames`
