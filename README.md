
# FluidLab: A Differentiable Environment for Benchmarking Complex Fluid Manipulation

<p align="left">
    <a href='https://arxiv.org/abs/2301.06015'>
      <img src='https://img.shields.io/badge/Paper-arXiv-green?style=plastic&logo=arXiv&logoColor=green' alt='Paper arXiv'>
    </a>
    <a href='https://fluidlab2023.github.io/'>
      <img src='https://img.shields.io/badge/Project-Page-blue?style=plastic&logo=Google%20chrome&logoColor=blue' alt='Project Page'>
    </a>
</p>

This is the official repo of the paper:

> **[FluidLab: A Differentiable Environment for Benchmarking Complex Fluid Manipulation](https://fluidlab2023.github.io/)**  
> [Zhou Xian](https://zhou-xian.com/), [Bo Zhu](https://www.cs.dartmouth.edu/~bozhu/), [Zhenjia Xu](https://www.zhenjiaxu.com/), [Hsiao-Yu Tung](https://sfish0101.bitbucket.io/), [Antonio Torralba](https://groups.csail.mit.edu/vision/torralbalab/), [Katerina Fragkiadaki](https://www.cs.cmu.edu/~katef/), [Chuang Gan](https://people.csail.mit.edu/ganchuang/)   
> *ICLR 2023 (Spotlight)*

![](tasks.gif)

This codebase contains the following:
- **FluidEngine**, a multi-material fully-differentiable physics engine, supporting liquids, solids, and gasesous fluid simulation.
- **FluidLab**, a set of standardized complex (robotic) fluid manipulation tasks powered by FluidEngine.

## Setting up environment
Create conda env.
```
conda env create -f environment.yml
conda activate fluidlab
```
(I hope this env file is complete. If not, install required packages when prompted with error when running the code.)

Install FluidLab.
```
pip install -e .
```


## Compiling GLRenderer (our own renderer)
We have two types of renderers: GGUIRenderer and GLRenderer. GGUI is provided by Taichi. GL is our own, with better visual effects.

GLRenderer is developped in C++ and needs to be compiled first.
Compiling only works with cuda 9.2. So we need a docker.

Install docker-ce and nvidia-docker
- https://docs.docker.com/engine/install/ubuntu/
- https://docs.nvidia.com/datacenter/cloud-native/container-toolkit/install-guide.html#docker


Under FluidLab:
```
sudo docker build -t fluidlab-glrenderer fluidlab/engine/renderers/gl_renderer_src
sudo nvidia-docker run \
  -v ${PWD}/fluidlab/engine/renderers/gl_renderer_src:/workspace \
  -v /home/{username}/anaconda3:/home/{username}/anaconda3 \
  --gpus all \
  -e DISPLAY=$DISPLAY \
  -e QT_X11_NO_MITSHM=1 \
  -it fluidlab-glrenderer:latest bash

. /home/{username}/anaconda3/bin/activate fluidlab
. prepare.sh
. compile.sh
```

## Running exps (Use latte art as an example)
### Generate goal
```
python fluidlab/run.py --cfg_file configs/exp_latteart.yaml --record
```
A GUI window will pop up. Use your mouse to control the agent and generate a goal pattern you want.

### Solve trajectory with differentiable physics
```
python fluidlab/run.py --cfg_file configs/exp_latteart.yaml --exp_name exp_latteart
```
Optimized trajectories will be saved under ```FluidLab/logs/policies/{exp_name}```.

### Replay optimized trajectory
```
python fluidlab/run.py --cfg_file configs/exp_latteart.yaml --replay --path FluidLab/logs/policies/exp_latteart/100.pkl
```

Commands for running other exps are available in ```exp.sh```.
Commands for running demos are available in ```demo.sh```.

