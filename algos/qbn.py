import ray
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np

from copy import deepcopy
from policies.autoencoder import QBN
from util.env import env_factory

@ray.remote
def collect_data(actor, timesteps, max_traj_len, seed):
  np.random.seed(seed)
  policy = deepcopy(actor)

  with torch.no_grad():
    env = env_factory(policy.env_name)()
    num_steps = 0
    states  = []
    actions = []
    hiddens = []
    cells   = []

    while num_steps < timesteps:
      state = torch.as_tensor(env.reset())

      done = False
      traj_len = 0
      if hasattr(actor, 'init_hidden_state'):
        actor.init_hidden_state()

      while not done and traj_len < max_traj_len:
        state        = torch.as_tensor(state).float()
        norm_state   = actor.normalize_state(state, update=False)
        action       = actor(norm_state)
        hidden, cell = actor.hidden[0].view(-1), actor.cells[0].view(-1) # get hidden only for first layer and remove batch dim

        states.append(state.numpy())
        actions.append(action.numpy())
        hiddens.append(hidden.numpy())
        cells.append(cell.numpy())

        state, _, done, _ = env.step(action.numpy())
        traj_len += 1
        num_steps += 1

    states  = states
    actions = actions
    hiddens = hiddens
    cells   = cells

    data = [states, actions, hiddens, cells]

    return data

def run_experiment(args):
  import locale, os
  locale.setlocale(locale.LC_ALL, '')

  from util.env import env_factory
  from util.log import create_logger

  if args.policy is None:
    print("You must provide a policy with --policy.")
    exit(1)

  policy = torch.load(args.policy)

  if policy.actor_layers[0].__class__.__name__ != 'LSTMCell':
    print("Cannot do QBN insertion on a non-recurrent policy.")
    raise NotImplementedError 

  if len(policy.actor_layers) > 1:
    print("Cannot do QBN insertion on a policy with more than one hidden layer.")
    raise NotImplementedError

  env_fn     = env_factory(policy.env_name)
  obs_dim    = env_fn().observation_space.shape[0]
  action_dim = env_fn().action_space.shape[0]
  hidden_dim = policy.actor_layers[0].hidden_size

  if not args.nolog:
    logger = create_logger(args)
  else:
    logger = None

  # Generate data for training QBN
  if args.data is None:
    ray.init()
    data = np.array(ray.get([collect_data.remote(policy, 1000, 1000, np.random.randint(65536)) for _ in range(args.workers)]))
    states  = np.vstack(data[:,0])
    actions = np.vstack(data[:,1])
    hiddens = np.vstack(data[:,2])
    cells   = np.vstack(data[:,3])

    data = [states, actions, hiddens, cells]
    if logger is not None:
      np.savez(os.path.join(logger.dir, 'data.npz'), states=states, actions=actions, hiddens=hiddens, cells=cells)

