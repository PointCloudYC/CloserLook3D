#!/bin/bash

# train s3dis
time python
-m torch.distributed.launch \
--master_port 12346 \
--nproc_per_node 1 \
function/train_s3dis_dist.py \
--cfg cfgs/s3dis/pospool_xyz_avg.yaml
# [--log_dir <log directory>]

# evaluate s3dis
time python
-m torch.distributed.launch \
--master_port 12346 \
--nproc_per_node 1 \
function/evaluate_s3dis_dist.py \
--cfg cfgs/s3dis/pospool_xyz_avg.yaml \
--load_path <checkpoint>
# [--log_dir <log directory>]
