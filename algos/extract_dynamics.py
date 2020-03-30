import ray
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import locale, os, time

from torch.utils.data.sampler import BatchSampler, SubsetRandomSampler
from copy import deepcopy
from policies.fit import Model
from util.env import env_factory

def get_hiddens(policy):
  hiddens = []
  if hasattr(policy, 'hidden'):
    hiddens += [h.data for h in policy.hidden]
  
  if hasattr(policy, 'cells'):
    hiddens += [c.data for c in policy.cells]
  
  for layer in hiddens:
    if layer.size(0) != 1:
      print("Invalid batch dim.")
      raise RuntimeError

  return torch.cat([layer.view(-1) for layer in hiddens]).numpy()
  
def collect_point(policy, max_traj_len):
  env = env_factory(policy.env_name)()
  env.dynamics_randomization = True

  chosen_timestep = np.random.randint(40, max_traj_len)
  timesteps = 0
  done = False

  if hasattr(policy, 'init_hidden_state'):
    policy.init_hidden_state()

  state = env.reset()
  while not done and timesteps < chosen_timestep:

    state = policy.normalize_state(state, update=True)
    action = policy(state).numpy()
    state, _, done, _ = env.step(action)
    timesteps += 1

  return get_hiddens(policy), env.get_dynamics()

@ray.remote
def collect_data(policy, max_traj_len=200, points=500):
  torch.set_num_threads(1)
  with torch.no_grad():
    done = True

    latent = []
    label  = []

    last = time.time()
    while len(label) < points:
      x, y = collect_point(policy, max_traj_len)
      """
      if done or timesteps >= max_traj_len:
        env = env_factory(policy.env_name)()
        env.dynamics_randomization = True

        timesteps = 0
        state = env.reset()
        done = False
        chosen_timestep = np.random.randint(40, max_traj_len)

        if hasattr(policy, 'init_hidden_state'):
          policy.init_hidden_state()

        time.sleep(0.1)

      state = policy.normalize_state(state, update=True)
      action = policy(state).numpy()
      state, _, done, _ = env.step(action)
      timesteps += 1

      if timesteps == chosen_timestep or (done and timesteps < chosen_timestep):
        latent += [get_hiddens(policy)]
        label  += [env.get_dynamics()]
        done = True
        
        #print("Collected {:3d} of {:3d} points ({:5.2f} seconds since last, trajlen {:3d})".format(len(latent), points, time.time() - last, timesteps))
        last = time.time()
      """
    #print("DONE!")
    return latent, label
  
def concat(datalist):
  latents = []
  labels  = []
  for l in datalist:
    latent, label = l
    latents += latent
    labels  += label
  latents = torch.tensor(latents)
  labels  = torch.tensor(labels)
  return latents, labels

def run_experiment(args):
  locale.setlocale(locale.LC_ALL, '')

  policy = torch.load(args.policy)

  env_fn     = env_factory(policy.env_name)

  layers = [int(x) for x in args.layers.split(',')]

  env = env_fn()
  policy.init_hidden_state()
  policy(torch.tensor(env.reset()).float())

  latent_dim = get_hiddens(policy).shape[0]
  output_dim = env.get_dynamics().shape[0]
  model      = Model(latent_dim, output_dim)#, layers=layers)

  opt = optim.Adam(model.parameters(), lr=args.lr, eps=1e-5)

  ray.init()
  points_per_worker = max(args.points // args.workers, 1)
  for i in range(args.iterations):
    start = time.time()
    print("Collecting {:4d} timesteps of data.".format(args.points))

    x, y = concat(ray.get([collect_data.remote(policy, points=points_per_worker) for _ in range(args.workers)]))

    print("{:3.2f} to collect {} timesteps".format(time.time() - start, len(x)))

    for epoch in range(args.epochs):

      print("Starting epoch {:3d}".format(epoch))

      random_indices = SubsetRandomSampler(range(len(x)-1))
      sampler = BatchSampler(random_indices, args.batch_size, drop_last=False)

      for j, batch_idx in enumerate(sampler):
        batch_x = x[batch_idx]
        batch_y = y[batch_idx]
        loss = 0.5 * (batch_y - model(batch_x)).pow(2).mean()

        opt.zero_grad()
        loss.backward()
        opt.step()

        print("Batch {:4d}/{:4d}: {:6.5f}".format(j, len(sampler)-1, loss.item()), end='\r')
      print()

    input()
