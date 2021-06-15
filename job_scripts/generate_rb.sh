#!/bin/bash
#SBATCH --cpus-per-task=10
#SBATCH --gres=gpu:0
#SBATCH --mem=8G
#SBATCH --time=2:00:00
#SBATCH -o /scratch/sahandr/DH-rb-%j.out


unset display

module load httpproxy

source $HOME/.bashrc
source $HOME/env/robosuite/bin/activate
#export EGL_DEVICE_ID=$SLURM_JOB_GPUS
export MUJOCO_GL="egl"


# config
root_folder=$SCRATCH/DH_logs/reacher/
num_eval_episodes=100

# loop over all sub directories and generate replay buffers
for f in ${root_folder}*; do
    if [ -d "$f" ]; then
        python $HOME/workspace/DH/train_rl.py --eval --num_eval_episodes ${num_eval_episodes} --load_model "$f/"
    fi
done
