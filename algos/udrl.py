"""Proximal Policy Optimization (clip objective)."""
import os
import ray
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import numpy as np

from torch.utils.data.sampler import BatchSampler, SubsetRandomSampler
from torch.distributions import kl_divergence
from torch.nn.utils.rnn import pad_sequence
from copy import deepcopy
from time import time

class Buffer:
  def __init__(self):
    self.states            = []
    self.actions           = []
    self.rewards           = []
    self.commanded_rewards = []
    self.traj_idx          = [0]

    self.buffer_ready = False

  def __len__(self):
    return len(self.states)

  #def push(self, state, action, reward, commanded_reward, horizon, done):
  def push(self, state, action, reward, horizon, done):
    self.states            += [state]
    self.actions           += [action]
    self.reward            += [reward]
    #self.commanded_rewards += [commanded_rewards]
    self.horizon           += [horizon]

    if done:
      self.traj_idx += [len(states)]

  def sample(self, batch_size=64):
    if not self.buffer_ready:
      self.states            = torch.Tensor(self.states)
      self.actions           = torch.Tensor(self.actions)
      self.rewards           = torch.Tensor(self.rewards)
      #self.commanded_rewards = torch.Tensor(self.commanded_rewards)
      self.horizon           = torch.Tensor(self.horizon)

      random_indices = SubsetRandomSampler(range(len(self.traj_idx)-1))
      sampler = BatchSampler(random_indices, batch_size, drop_last=True)

      for traj_indices in sampler:
        states      = [self.states[self.traj_idx[i]:self.traj_idx[i+1]]      for i in traj_indices]
        actions     = [self.actions[self.traj_idx[i]:self.traj_idx[i+1]]     for i in traj_indices]
        rewards     = [self.rewards[self.traj_idx[i]:self.traj_idx[i+1]]     for i in traj_indices]
        #cmd_rewards = [self.cmd_rewards[self.traj_idx[i]:self.traj_idx[i+1]] for i in traj_indices]
        horizon     = [self.horizon[self.traj_idx[i]:self.traj_idx[i+1]]     for i in traj_indices]
        traj_mask   = [torch.ones_like(r) for r in rewards]
        
        states      = pad_sequence(states,      batch_first=False)
        actions     = pad_sequence(actions,     batch_first=False)
        rewards     = pad_sequence(rewards,     batch_first=False)
        #cmd_rewards = pad_sequence(cmd_rewards, batch_first=False)
        horizon     = pad_sequence(horizon,     batch_first=False)
        traj_mask   = pad_sequence(traj_mask,   batch_first=False)

        yield states, actions, rewards, horizon, traj_mask

@ray.remote
class UDRL_Worker:
  def __init__(self, behavioral_fn, env_fn):
    torch.set_num_threads(1)
    self.behavioral_fn = deepcopy(behavioral_fn)
    self.env = env_fn()

    if hasattr(self.env, 'dynamics_randomization'):
      self.dynamics_randomization = self.env.dynamics_randomization
    else:
      self.dynamics_randomization = False

  def update_policy(self, new_behavioral_fn_params, input_norm=None):
    for p, new_p in zip(self.behavioral_fn.parameters(), new_behavioral_fn_params):
      p.data.copy_(new_p)

    if input_norm is not None:
      self.behavioral_fn.welford_state_mean, self.behavioral_fn.welford_state_mean_diff, self.behavioral_fn.welford_state_n = input_norm

  def collect_experience(self, max_traj_len, min_steps):
    with torch.no_grad():
      start = time()

      num_steps = 0
      memory = Buffer()
      behavioral_fn  = self.behavioral_fn

      while num_steps < min_steps:
        self.env.dynamics_randomization = self.dynamics_randomization
        state = torch.Tensor(self.env.reset())

        done = False
        value = 0
        traj_len = 0

        if hasattr(behavioral_fn, 'init_hidden_state'):
          behavioral_fn.init_hidden_state()

        while not done and traj_len < max_traj_len:
            state      = torch.Tensor(state)
            norm_state = behavioral_fn.normalize_state(state, update=False)
            action     = behavioral_fn(norm_state, False)

            next_state, reward, done, _ = self.env.step(action.numpy())

            reward = np.array([reward])

            memory.push(state.numpy(), action.numpy(), reward, horizon)

            state = next_state

            traj_len += 1
            num_steps += 1

      return memory

class UDRL:
  def __init__(self, behavior_fn, 

def run_experiment(args):
  pass






