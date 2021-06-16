#!/bin/bash
#SBATCH --array=0-4
#SBATCH --cpus-per-task=10
#SBATCH --gres=gpu:1
#SBATCH --mem=16G
#SBATCH --time=10:00:00
#SBATCH -o /scratch/sahandr/DH-bc-%j.out

unset display

module load httpproxy

source $HOME/.bashrc
source $HOME/env/robosuite/bin/activate
#export EGL_DEVICE_ID=$SLURM_JOB_GPUS
export MUJOCO_GL="egl"

# config
root_folder=$SCRATCH/DH_logs/reacher/
target_robot_name="double"
source_robot_name="reacher"

python $HOME/workspace/DH/train_bc.py --transfer --target_robot_name $target_robot_name --source_robot_name $source_robot_name --load_replay $root_folder --use_comet --slurm_task_id $SLURM_ARRAY_TASK_ID --learn_dh --dh_noise 0
