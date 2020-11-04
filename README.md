----

# Recurrent Reinforcement Learning

## Purpose

This repo contains recurrent implementations of state-of-the-art RL algorithms. Its purpose is to be clean, legible, and easy to understand. Many RL algorithms treat recurrence as an afterthought - it is my opinion that recurrence has important theoretical and practical benefits to many RL problems, especially partially observable ones in the real world.

In addition to recurrent reinforcement learning, it also provides algorithms for extracting interesting information out of recurrent policy networks. Implemented are system-ID decoding networks for use with policy networks trained with dynamics randomization (described [here](https://arxiv.org/abs/2006.02402)) and also for Quantized-Bottleneck Network insertion (described [here](https://arxiv.org/abs/1811.12530)).

## First-time setup
This repo assumes that you have [OpenAI Gym](https://gym.openai.com/) and [MuJoCo 2.0](http://www.mujoco.org/) installed, and that you are using Ubuntu 18.04, as this is my development environment (similar distros may also work). If you would like to do experiments with the simulated Cassie environment, you will also need my [Cassie](https://github.com/siekmanj/cassie) repository.

You will need to install several packages:
```bash
pip3 install --user torch numpy ray gym tensorboard
sudo apt-get install -y curl git libgl1-mesa-dev libgl1-mesa-glx libglew-dev libosmesa6-dev net-tools unzip vim wget xpra xserver-xorg-dev patchelf
```


If you haven't already, clone my repo:

```bash
git clone https://github.com/siekmanj/r2l
cd r2l
```

Optionally, you can clone the Cassie directory to do experiments with the Cassie simulator.
```bash
git clone https://github.com/siekmanj/cassie
```

Now, you will need to install MuJoCo. You will also need to obtain a license key `mjkey.txt` from the [official website](https://www.roboti.us/license.html). You can get a free 30-day trial if necessary.
```bash
wget https://www.roboti.us/download/mujoco200_linux.zip
unzip mujoco200_linux.zip
mkdir ~/.mujoco
mv mujoco200_linux ~/.mujoco/mujoco200
cp [YOUR KEY FILE] ~/.mujoco/mjkey.txt
```

You will need to create an environment variable `LD_LIBRARY_PATH` to allow mujoco-py to find your mujoco directory. You can add it to your `~/.bashrc` or just enter it into the terminal every time you wish to use mujoco.
```bash
export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:~/.mujoco/mujoco200/bin
```

Now that the prerequisites are installed, you can install `mujoco-py`.
```bash
pip3 install --user mujoco-py
```

## Running experiments

### Basics
Any algorithm can be run from the r2l.py entry point.

To train a GRU agent on the HalfCheetah-v2 environment and save the resulting policy to some directory, simply run:

```bash
python3 r2l.py ppo --env 'HalfCheetah-v2' --arch gru --save_actor cheetah.pt --layers 64 --batch_size 6 --num_steps 10000 --prenormalize_steps 100
```

Then, to do QBN insertion on this policy, simply run

```bash
python3 r2l.py qbn --policy cheetah.pt
```

### Logging details / Monitoring live training progress
Tensorboard logging is enabled by default for all algorithms. By default, logs are stored in ```logs/[algo]/[env]/[experiment hash]```.

After initiating an experiment, your directory structure would look like this:

```
logs/
├── [algo]
│     └── <env name> 
│             └── [New Experiment Logdir]
└── ddpg
```

To see live training progress, run ```$ tensorboard --logdir=logs``` then navigate to ```http://localhost:6006/``` in your browser

## Features:
* Parallelism with [Ray](https://github.com/ray-project/ray)
* [DDPG](https://arxiv.org/abs/1509.02971)
* [RDPG](https://arxiv.org/abs/1512.04455)
* [ARS](https://arxiv.org/abs/1803.07055)
* [PPO](https://arxiv.org/abs/1707.06347)
* [TD3](https://arxiv.org/abs/1802.09477)
* [SAC](https://arxiv.org/abs/1801.01290)
* [QBN Insertion](https://arxiv.org/abs/1811.12530)

## Acknowledgements

This repo was originally based on the Oregon State University DRL's Apex library: https://github.com/osudrl/apex (authored by my fellow researchers Yesh Godse and Pedro Morais), which was in turn inspired by @ikostrikov's implementations of RL algorithms. Thanks to @sfujim for the clean implementations of TD3 and DDPG in PyTorch. Thanks @modestyachts for the easy to understand ARS implementation.
