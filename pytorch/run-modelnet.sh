#!/bin/bash

# train 
time python -m torch.distributed.launch \
--master_port 12346 \
--nproc_per_node 1 \
function/train_modelnet_dist.py \
--cfg cfgs/modelnet/pospool_xyz_avg.yaml
# [--log_dir <log directory>]

# evaluate 
time python -m torch.distributed.launch \
--master_port 12346 \
--nproc_per_node 1 \
function/evaluate_modelnet_dist.py \
--cfg cfgs/modelnet/pospool_xyz_avg.yaml \
# --load_path <checkpoint>
# [--log_dir <log directory>]
