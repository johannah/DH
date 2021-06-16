#!/bin/bash
#SBATCH --array=0-4
#SBATCH --cpus-per-task=10
#SBATCH --gres=gpu:0
#SBATCH --mem=8G
#SBATCH --time=15:00:00
#SBATCH -o /scratch/sahandr/DH-drl-%j.out

unset display

module load httpproxy

source $HOME/.bashrc
source $HOME/env/robosuite/bin/activate
#export EGL_DEVICE_ID=$SLURM_JOB_GPUS
export MUJOCO_GL="egl"

# config
config="base_jaco_door.cfg"

python $HOME/workspace/DH/train_drl.py --cfg $HOME/workspace/DH/experiments/${config} --log_dir $SLURM_TMPDIR/DH_logs/ --use_comet --slurm_task_id $SLURM_ARRAY_TASK_ID

rsync -a  $SLURM_TMPDIR/DH_logs/  $SCRATCH/DH_logs
