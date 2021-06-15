#!/bin/bash
#SBATCH --cpus-per-task=10               # Ask for 10 CPUs
#SBATCH --gres=gpu:1                     # Ask for 1 GPU
#SBATCH --mem=32G                        # Ask for 32 GB of RAM
#SBATCH --time=6:00:00                  # The job will run for 3 hours
#SBATCH -o /scratch/sahandr/DH-%j.out  # Write the log in $SCRATCH

unset display

module load httpproxy

source $HOME/.bashrc
source $HOME/env/robosuite/bin/activate
#export EGL_DEVICE_ID=$SLURM_JOB_GPUS
export MUJOCO_GL="egl"

# copy and unzip the dataset
# cp $SCRATCH/reacher_experiments.tar  $SLURM_TMPDIR/
# tar -xvf $SLURM_TMPDIR/reacher_experiments.tar -C $SLURM_TMPDIR/

# config
config="base_dm.cfg"
load_replay="base_easy_01111_reacher_1_00"

python $HOME/workspace/DH/train_rl.py --cfg $HOME/workspace/DH/experiments/${config} --log_dir $SLURM_TMPDIR/DH_logs/ --use_comet
# python $HOME/workspace/DH/train_bc.py --load_replay $SLURM_TMPDIR/collate_reacher/${load_replay} --use_comet


rsync -a  $SLURM_TMPDIR/DH_logs/  $SCRATCH/DH_logs
