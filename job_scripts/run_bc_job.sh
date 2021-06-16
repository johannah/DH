#!/bin/bash
#SBATCH --array=0-14
#SBATCH --cpus-per-task=10
#SBATCH --gres=gpu:1
#SBATCH --mem=8G
#SBATCH --time=12:00:00
#SBATCH -o /scratch/sahandr/DH-bc-%j.out

unset display

module load httpproxy

source $HOME/.bashrc
source $HOME/env/robosuite/bin/activate
#export EGL_DEVICE_ID=$SLURM_JOB_GPUS
export MUJOCO_GL="egl"

# config
root_folder=$SCRATCH/DH_logs/reacher/

python $HOME/workspace/DH/train_bc.py --load_replay $root_folder --use_comet --slurm_task_id $SLURM_ARRAY_TASK_ID

rsync -a  $SLURM_TMPDIR/DH_logs/  $SCRATCH/DH_logs
