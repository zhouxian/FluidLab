
# FluidLab

This repo contains FluidLab, a simulation platform containing both FluidEngine and a set of standardized fluid manipulation tasks proposed in the FluidLab ICLR 2023 submission. 

Paper submission: https://openreview.net/forum?id=Cp-io_BoFaE&noteId=EE5yG3TBxS

Project site: https://sites.google.com/view/fluidlab

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

