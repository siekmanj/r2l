import ray
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import locale, os

from torch.utils.data.sampler import BatchSampler, SubsetRandomSampler
from copy import deepcopy
from policies.autoencoder import QBN
from util.env import env_factory


def get_hiddens(policy):
  hiddens = []
  if hasattr(policy, 'hidden'):
    hiddens += policy.hidden
  
  if hasattr(policy, 'cells'):
    hiddens += policy.cells
  
  for layer in hiddens:
    if layer.size(0) != 1:
      print("Invalid batch dim.")
      raise RuntimeError

  return torch.cat([layer.view(-1) for layer in hiddens])
  
#@ray.remote
def collect_data(policy, max_traj_len=400, points=2):
  torch.set_num_threads(1)
  with torch.no_grad():
    env = env_factory(policy.env_name)()
    env.dynamics_randomization = True

    total_t = 0
    while total_t < 800:
      state = env.reset()
      done = False
      timesteps = 0

      if hasattr(policy, 'init_hidden_state'):
        policy.init_hidden_state()

      while not done and timesteps < max_traj_len:
        state = policy.normalize_state(state, update=True)
        action = policy(state).numpy()
        state, _, done, _ = env.step(action)
        timesteps += 1
        total_t += 1

        latent = get_hiddens(policy)
        label  = env.get_dynamics()
  

def run_experiment(args):
  locale.setlocale(locale.LC_ALL, '')

  policy = torch.load(args.policy)

  env_fn     = env_factory(policy.env_name)
  obs_dim    = env_fn().observation_space.shape[0]
  action_dim = env_fn().action_space.shape[0]
  hidden_dim = policy.actor_layers[0].hidden_size

  env = env_fn()

  collect_data(policy)


