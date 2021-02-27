## TODOs

- run code;
    - set up the env; cuda 10.1 + pytorch 1.3+, etc.
    - run s3dis code
- 

## run the code

- conda env; use hpc3 `pytorch` env (pytorch 1.6) and install hydra-core

```
conda install -c conda-forge hydra-core
```

- prepare data; simlink hpc2 ModelNet40 dataset to local folder `modelnet40_normal_resampled`

```
ln -s /home/share/cejcheng/ModelNet40/ modelnet40_normal_resampled
```

- config; check config/config.yaml, model specific config check model/xx.config

- run succesfully (Feb 3, 12:00)

- consider multi-run for the 3 models; check https://hydra.cc/docs/next/tutorials/basic/running_your_app/multi-run/.

```
python train.py \

--multirun model=Menghao,Hengshuang,Nico
```

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