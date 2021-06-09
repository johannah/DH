


### train a new RL model:

`python train_rl.py --cfg experiments/base_dm.cfg`

### evaluate and plot the latest model from a training directory:  

`python train_rl.py --eval --frames --load_model results/base_easy_01111_reacher_1_01/ --num_eval_episodes 1 --frames`
This will result in plots for reward & target position, and a video of the episodes written as a derivitive of the most recent .pt file in the --load_model directory. 

### evaluate and plot a specific model:

`python train_rl.py --eval --frames --load_model results/base_easy_01111_reacher_1_01/base_0000355000.pt 

## to train a behavior cloning model which predicts relative JOINT_POSITIONS for a given state with DH:

`python train_bc.py --load_replay results/base_easy_01111_reacher_1_04/base_0000025000_eval_003000_S030000.pkl`

### with MSE on the joint angles:

`python train_bc.py --load_replay results/base_easy_01111_reacher_1_04/base_0000025000_eval_003000_S030000.pkl` --loss "angle"`
