----

# Recurrent Reinforcement Learning

## Purpose

This repo contains recurrent implementations of state-of-the-art RL algorithms. Its purpose is to be clean, legible, and easy to understand. Many RL algorithms treat recurrence as an afterthought - it is my opinion that recurrence has important theoretical and practical benefits to many RL problems, especially partially observable ones in the real world.


## Running experiments

### Basics
Any algorithm can be run from the rll.py entry point.

To run recurrent PPO on Walker2d-v2,

```bash
python rll.py ppo --env_name Walker2d-v2 --recurrent
```

### Logging details / Monitoring live training progress
Tensorboard logging is enabled by default for all algorithms. By default, logs are stored in ```logs/[algo]/[env]/[experiment hash]```.

After initiating an experiment, your directory structure would look like this:

```
logs/
├── ars
│   └── <env_name> 
│           └── [New Experiment Logdir]
├── ddpg
└── rdpg
```

To see live training progress, run ```$ tensorboard --logdir=logs``` then navigate to ```http://localhost:6006/``` in your browser

### To Do
- [ ] Soft Actor-Critic

## Features:
* Parallelism with [Ray](https://github.com/ray-project/ray)
* [DDPG](https://arxiv.org/abs/1509.02971)
* [RDPG](https://arxiv.org/abs/1512.04455)
* [ARS](https://arxiv.org/abs/1803.07055)
* [PPO](https://arxiv.org/abs/1707.06347)
* [TD3](https://arxiv.org/abs/1802.09477)

#### To be implemented soon:
* [SAC](https://arxiv.org/abs/1801.01290)
* [SVG](https://arxiv.org/abs/1510.09142)

## Acknowledgements

This repo was cloned from the Oregon State University DRL's Apex library: https://github.com/osudrl/apex (authored by my fellow researchers Yesh Godse and Pedro Morais), which was in turn inspired by @ikostrikov's implementations of RL algorithms. Thanks to @sfujim for the clean implementations of TD3 and DDPG in PyTorch. Thanks @modestyachts for the easy to understand ARS implementation.
