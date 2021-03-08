#!/bin/bash

# custom settings: jobname, partition_name; Note: can not pass the values via bash var to #SBATCH script
# one possible solution is https://stackoverflow.com/questions/27708656/pass-command-line-arguments-via-sbatch

# "#SBATCH" are valid SLURM commands or statements,
# "##SBATCH" are comments.  Uncomment

# job name
#SBATCH -J "LAOs"

# Set the maximum runtime
##SBATCH -t 8:00:00 #Maximum runtime of 48 hours

# email notification
#SBATCH --mail-user=cyinac@connect.ust.hk 
#SBATCH --mail-type=begin
#SBATCH --mail-type=end
#SBATCH --mail-type=fail
##SBATCH --mail-type=REQUEUE
##SBATCH --mail-type=ALL

# Choose partition (queue) "gpu" for hpc2 or "gpu-share" for hpc3
# check https://itsc.ust.hk/services/academic-teaching-support/high-performance-computing/hpc3-cluster/resource-limits/
#SBATCH -p "gpu-share"
# To use 24 cpu cores in a node, uncomment the statement below
##SBATCH -N 1 -n 24
# To use 24 cpu core and 4 gpu devices in a node, uncomment the statement below
##SBATCH -N 1 -n 24 --gres=gpu:4

# Setup runtime environment if necessary
# Or you can source ~/.bashrc or ~/.bash_profile
# conda activate pcp-tf
echo "start training..."

# Go to the job submission directory and run your application
# cd $HOME/code/pc-networks/pointnet

# we execute the job and time it
bash run-modelnet.sh
# bash run-s3dis.sh
echo "finish training"

