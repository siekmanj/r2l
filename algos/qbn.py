import ray
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np

from torch.utils.data.sampler import BatchSampler, SubsetRandomSampler
from copy import deepcopy
from policies.autoencoder import QBN
from util.env import env_factory

@ray.remote
def collect_data(actor, timesteps, max_traj_len, seed):
  torch.set_num_threads(1)
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

        #states.append(state.numpy())
        states.append(norm_state.numpy())
        actions.append(action.numpy())
        hiddens.append(hidden.numpy())
        cells.append(cell.numpy())

        state, _, done, _ = env.step(action.numpy())
        traj_len += 1
        num_steps += 1

    states  = np.array(states)
    actions = np.array(actions)
    hiddens = np.array(hiddens)
    cells   = np.array(cells)

    return [states, actions, hiddens, cells]

def eval_qbn(actor, obs_qbn, hid_qbn, cel_qbn, episodes=10, max_traj_len=1000):
  with torch.no_grad():
    env = env_factory(actor.env_name)()
    reward = 0

    for _ in range(episodes):
      done = False
      traj_len = 0

      if hasattr(actor, 'init_hidden_state'):
        actor.init_hidden_state()

      state = torch.as_tensor(env.reset())
      while not done and traj_len < max_traj_len:
        state        = torch.as_tensor(state).float()
        norm_state   = actor.normalize_state(state, update=False)
        hidden_state = actor.hidden[0]
        cell_state   = actor.cells[0]

        norm_state   = obs_qbn(norm_state) # discretize state
        actor.hidden = [hid_qbn(hidden_state)]
        actor.cells  = [cel_qbn(cell_state)]

        action       = actor(norm_state)
        state, r, done, _ = env.step(action.numpy())
        traj_len += 1
        reward += r/episodes
    return reward
  

def train_lstm_qbn(policy, states, actions, hiddens, cells, epochs=500, batch_size=64, logger=None):
  obs_qbn    = QBN(states.shape[-1])
  hidden_qbn = QBN(hiddens.shape[-1])
  cell_qbn   = QBN(cells.shape[-1])

  obs_optim = optim.Adam(obs_qbn.parameters(), lr=1e-4, eps=1e-6)
  hidden_optim = optim.Adam(hidden_qbn.parameters(), lr=1e-4, eps=1e-6)
  cell_optim = optim.Adam(cell_qbn.parameters(), lr=1e-4, eps=1e-6)

  for epoch in range(epochs):
    random_indices = SubsetRandomSampler(range(states.shape[0]))
    sampler = BatchSampler(random_indices, batch_size, drop_last=True)

    epoch_obs_losses = []
    epoch_hid_losses = []
    epoch_cel_losses = []
    for i, batch in enumerate(sampler):
      batch_states  = states[batch]
      batch_actions = actions[batch]
      batch_hiddens = hiddens[batch]
      batch_cells   = cells[batch]

      obs_loss = 0.5 * (batch_states  - obs_qbn(batch_states)).pow(2).mean()
      hid_loss = 0.5 * (batch_hiddens - hidden_qbn(batch_hiddens)).pow(2).mean()
      cel_loss = 0.5 * (batch_cells   - cell_qbn(batch_cells)).pow(2).mean()

      obs_optim.zero_grad()
      obs_loss.backward()
      obs_optim.step()

      hidden_optim.zero_grad()
      hid_loss.backward()
      hidden_optim.step()

      cell_optim.zero_grad()
      cel_loss.backward()
      cell_optim.step()

      epoch_obs_losses.append(obs_loss.item())
      epoch_hid_losses.append(hid_loss.item())
      epoch_cel_losses.append(cel_loss.item())
      print("batch {:3d} / {:3d}".format(i, len(sampler)), end='\r')

    epoch_obs_losses = np.mean(epoch_obs_losses)
    epoch_hid_losses = np.mean(epoch_hid_losses)
    epoch_cel_losses = np.mean(epoch_cel_losses)
    reward = eval_qbn(policy, obs_qbn, hidden_qbn, cell_qbn)
    print("{:5.3f} | {:5.3f} | {:5.3f} | reward: {:5.3f}".format(epoch_obs_losses, epoch_hid_losses, epoch_cel_losses, reward))
    if logger is not None:
      logger.add_scalar('qbn/obsloss', epoch_obs_losses, epoch)
      logger.add_scalar('qbn/hidloss', epoch_hid_losses, epoch)
      logger.add_scalar('qbn/celloss', epoch_cel_losses, epoch)
      logger.add_scalar('qbn/reward', reward, epoch)

  return obs_qbn, hidden_qbn, cell_qbn
  

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
    print("Collecting dataset of size {:n} with {} workers...".format(args.dataset_size, args.workers))
    ray.init()
    data = ray.get([collect_data.remote(policy, args.dataset_size/args.workers, args.traj_len, np.random.randint(65536)) for _ in range(args.workers)])
    states  = np.vstack([r[0] for r in data])
    actions = np.vstack([r[1] for r in data])
    hiddens = np.vstack([r[2] for r in data])
    cells   = np.vstack([r[3] for r in data])

    data = [states, actions, hiddens, cells]
    if logger is not None:
      args.data = os.path.join(logger.dir, 'data.npz')
      np.savez(args.data, states=states, actions=actions, hiddens=hiddens, cells=cells)
  data = np.load(args.data)

  states  = torch.as_tensor(data['states'])
  actions = torch.as_tensor(data['actions'])
  hiddens = torch.as_tensor(data['hiddens'])
  cells   = torch.as_tensor(data['cells'])

  obs_qbn, hid_qbn, cel_qbn = train_lstm_qbn(policy, states, actions, hiddens, cells)

