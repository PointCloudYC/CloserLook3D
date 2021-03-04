## TODOs

-  after set up the conda env, run `sh init.sh`, report `error: command '/usr/bin/nvcc' failed with exit status 1`
   -  this is due to gcc version is incompatible w. CUDA; Mine cuda is 9.1 and gcc is 7.5. check [build error:error: command '/usr/local/cuda/bin/nvcc' failed with exit status 1 · Issue #25 · facebookresearch/maskrcnn-benchmark](https://github.com/facebookresearch/maskrcnn-benchmark/issues/25)
  ```
  nvcc --version
  gcc --version
  ```



## run the code

- set up a conda env; create `closerlook` env with required packages(`pytorch 1.4 + cudnn 10.1`)(check `install.sh` file). based on the issue [RuntimeError: Error compiling objects for extension](https://github.com/zeliu98/CloserLook3D/issues/16)

```
conda create –n closerlook python=3.6.10 -y
source activate closerlook
conda install -c anaconda pillow=6.2 -y
conda install pytorch==1.4.0 torchvision==0.5.0 cudatoolkit=10.1 -c pytorch -y
~~conda install pytorch==1.5.0 torchvision==0.6.0 cudatoolkit=10.1 -c pytorch -y~~
conda install -c conda-forge opencv -y
pip3 install termcolor tensorboard h5py easydict
```

- prepare data; simlink  ModelNet40 dataset to data folder `modelnet40_normal_resampled` based on readme file.

```
cd pytorch
mkdir data
cd data && mkdir ModelNet40 S3DIS
# symlink for modelnet40 and s3dis
ln -s /media/yinchao/Mastery/dataset/ModelNet40/ ./ModelNet40/modelnet40_normal_resampled
ln -s /media/yinchao/Mastery/dataset/S3DIS/ ./S3DIS/Stanford3dDataset_v1.2
```

- config; check cfgs/datase_name/model_name_xx.yaml; take s3dis dataset as an example, five types of setting which just involve the five typs of setting mentioned in the paper.
```
pointwise mlp
pseudo grid
adaptive weight 
```

- datasets; dataset loaders.


- run code on s3dis(train, evaluate)

```
python 
-m torch.distributed.launch \
--master_port <port_num> \
--nproc_per_node <num_of_gpus_to_use> \
function/train_s3dis_dist.py \
--cfg <config file> \
[--log_dir <log directory>]

python 
-m torch.distributed.launch \ 
--master_port <port_num> \
--nproc_per_node 1 \
function/evaluate_s3dis_dist.py \
--cfg <config file> \
--load_path <checkpoint> \
[--log_dir <log directory>]
```

## code profiling

**main file for s3dis dataset: function/train_s3dis_dist.py**, then modules in the model(i.e. backbone, head, loss, etc.) and other utility modules are linked and used.

### code structure

```
cfgs; config for five types of networks on 3 different datasets(modelnet40,partnet,s3dis)
datasets; dataset loaders for the 3 datasets
function; train and evaluate files, main point for different datasets, take s3dis as an example
- s3dis; `train_s3dis_dist.py`, `evaluate_s3dis_dist.py` for training and evaluate respectively.
- modelnet40;
- partnet;
- shapenet;

model;
- backbones; resnet module.
- heads; classifier and segmentation module.
- losses; label_smoothing_CE, masked_CE, multi_shape_CE.
- `local_aggregation_operators.py`, local aggregation operators
- `build.py`, build the pipelines
- `utils.py`
ops; custom tensor ops, e.g. subsample ops.
utils; utility functions, including logger, config, lr_scheduler, etc.
init.sh; create pytorch ops, relevant to ops folder.
```

### analysis

## FAQ

### How to create a python module using c++? or how to implement an efficient the grid subsampling module?

**take subsampling module as an example**
- check ops/cpp_wrappers for this module;
- for define a python module using c++, u need do like `setup.py` (grid_sumsampling/setup.py) which need declare module requiring source files and compile arguments, and include directories(include_dirs)
- from the source files, u can know the core files of this module:
  - cloud for point and point cloud defining; cloud.h and cloud.cpp; 
    - define PointXYZ ctor, common methods, operators and inline operators; 
    - define PointCloud struct, including point collection, methods to get count, a pt's one dim's coordinate, and bbbox
  - sampling for point cloud;grid_subsampling.cpp
    - a PC include point, features and labels.
    - sub-sample method
  - wrapper to call and integrate the above methods;wrapper.cpp

### how to develop a cuda op?
**take pt_custom_ops as an example**
- the code structure is clear; include/src folder contain code for the 4 cuda ops;
- each op need a .h, .cpp and .cu file for declarement, source and CUDA code respectively
- setup.py will help create a py module