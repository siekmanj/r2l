import ray
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import locale, os, time

from torch.utils.data.sampler import BatchSampler, SubsetRandomSampler
from copy import deepcopy
from policies.autoencoder import QBN
from util.env import env_factory

def get_hiddens(policy):
  hiddens = []
  if hasattr(policy, 'hidden'):
    hiddens += [h.data for h in policy.hidden]
  
  #if hasattr(policy, 'cells'):
    #hiddens += [c.data for c in policy.cells]
  
  for layer in hiddens:
    if layer.size(0) != 1:
      print("Invalid batch dim.")
      raise RuntimeError

  return torch.cat([layer.view(-1) for layer in hiddens]).numpy()
  
@ray.remote
def collect_data(policy, max_traj_len=200, points=500):
  torch.set_num_threads(1)
  with torch.no_grad():
    env = env_factory(policy.env_name)()
    env.dynamics_randomization = True

    done = True

    latent = []
    label  = []

    while len(label) < points:
      if done or timesteps >= max_traj_len:
        timesteps = 0
        state = env.reset()
        done = False
        if hasattr(policy, 'init_hidden_state'):
          policy.init_hidden_state()

      state = policy.normalize_state(state, update=True)
      action = policy(state).numpy()
      state, _, done, _ = env.step(action)
      timesteps += 1

      if timesteps > 50 and np.random.randint(25) == 0:
        latent += [get_hiddens(policy)]
        label  += [env.get_dynamics()]
    return latent, label
  
def concat(datalist):
  latents = []
  labels  = []
  for l in datalist:
    latent, label = l
    latents += latent
    labels  += label
  return latents, labels

def run_experiment(args):
  locale.setlocale(locale.LC_ALL, '')

  policy = torch.load(args.policy)

  env_fn     = env_factory(policy.env_name)
  obs_dim    = env_fn().observation_space.shape[0]
  action_dim = env_fn().action_space.shape[0]
  hidden_dim = policy.actor_layers[0].hidden_size

  env = env_fn()

  ray.init()
  points_per_worker = max(args.points / args.workers, 1)
  for i in range(args.iterations):
    x, y = concat(ray.get([collect_data.remote(policy, points=points_per_worker) for _ in range(args.workers)]))

    for i, x_i in enumerate(y[0]):
      print(i, x_i)

    input()


