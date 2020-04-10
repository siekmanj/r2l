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

def evaluate(actor, obs_qbn=None, hid_qbn=None, cel_qbn=None, episodes=5, max_traj_len=1000):
  with torch.no_grad():
    env = env_factory(actor.env_name)()
    reward = 0

    obs_states = {}
    hid_states = {}
    cel_states = {}
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

        if obs_qbn is not None:
          norm_state   = obs_qbn(norm_state) # discretize state
          disc_state = np.round(obs_qbn.encode(norm_state).numpy()).astype(int)
          key = ''.join(map(str, disc_state))
          if key not in obs_states:
            obs_states[key] = True

        if hid_qbn is not None:
          actor.hidden = [hid_qbn(hidden_state)]
          disc_state = np.round(hid_qbn.encode(hidden_state).numpy()).astype(int)
          key = ''.join(map(str, disc_state[0]))
          if key not in hid_states:
            hid_states[key] = True

        if cel_qbn is not None:
          actor.cells  = [cel_qbn(cell_state)]
          disc_state = np.round(cel_qbn.encode(cell_state).numpy()).astype(int)
          key = ''.join(map(str, disc_state[0]))
          if key not in cel_states:
            cel_states[key] = True

        action       = actor(norm_state)
        state, r, done, _ = env.step(action.numpy())
        traj_len += 1
        reward += r/episodes
    return reward, len(obs_states), len(hid_states), len(cel_states)
  

def run_experiment(args):
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

  layers = [int(x) for x in args.layers.split(',')]

  states     = env_fn().reset()
  obs_qbn    = QBN(obs_dim,    layers=layers)
  hidden_qbn = QBN(hidden_dim, layers=layers)
  cell_qbn   = QBN(hidden_dim, layers=layers)

  obs_optim    = optim.Adam(obs_qbn.parameters(), lr=args.lr, eps=1e-6)
  hidden_optim = optim.Adam(hidden_qbn.parameters(), lr=args.lr, eps=1e-6)
  cell_optim   = optim.Adam(cell_qbn.parameters(), lr=args.lr, eps=1e-6)

  best_reward = None

  if not args.nolog:
    logger = create_logger(args)
  else:
    logger = None

  ray.init()

  n_reward, _, _, _                      = evaluate(policy, episodes=20)
  logger.add_scalar(policy.env_name + '_qbn/nominal_reward', n_reward, 0)
  for iteration in range(args.iterations):
    data = ray.get([collect_data.remote(policy, args.points/args.workers, args.traj_len, np.random.randint(65535)) for _ in range(args.workers)])
    states  = np.vstack([r[0] for r in data])
    actions = np.vstack([r[1] for r in data])
    hiddens = np.vstack([r[2] for r in data])
    cells   = np.vstack([r[3] for r in data])

    for epoch in range(args.epochs):
      random_indices = SubsetRandomSampler(range(states.shape[0]))
      sampler        = BatchSampler(random_indices, args.batch_size, drop_last=False)

      epoch_obs_losses = []
      epoch_hid_losses = []
      epoch_cel_losses = []
      for i, batch in enumerate(sampler):
        batch_states  = torch.from_numpy(states[batch])
        batch_actions = torch.from_numpy(actions[batch])
        batch_hiddens = torch.from_numpy(hiddens[batch])
        batch_cells   = torch.from_numpy(cells[batch])

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
        print("epoch {:3d} / {:3d}, batch {:3d} / {:3d}".format(epoch+1, args.epochs, i+1, len(sampler)), end='\r')

    epoch_obs_losses = np.mean(epoch_obs_losses)
    epoch_hid_losses = np.mean(epoch_hid_losses)
    epoch_cel_losses = np.mean(epoch_cel_losses)

    print("\nEvaluating...")
    d_reward, s_states, h_states, c_states = evaluate(policy, obs_qbn=obs_qbn, hid_qbn=hidden_qbn, cel_qbn=cell_qbn)
    a_reward, _, _, _                      = evaluate(policy, obs_qbn=None,    hid_qbn=None,       cel_qbn=cell_qbn)
    b_reward, _, _, _                      = evaluate(policy, obs_qbn=None,    hid_qbn=hidden_qbn, cel_qbn=None)
    c_reward, _, _, _                      = evaluate(policy, obs_qbn=obs_qbn, hid_qbn=None,       cel_qbn=None)

    if best_reward is None or d_reward > best_reward:
      torch.save(obs_qbn, os.path.join(logger.dir, 'obsqbn.pt'))
      torch.save(hidden_qbn, os.path.join(logger.dir, 'hidqbn.pt'))
      torch.save(cell_qbn, os.path.join(logger.dir, 'celqbn.pt'))

    print("Losses: {:7.5f} {:7.5f} {:7.5f} | States: {:5d} {:5d} {:5d} | QBN reward: {:5.1f} ({:5.1f}, {:5.1f}, {:5.1f}) | Nominal reward {:5.0f} ".format(epoch_obs_losses, epoch_hid_losses, epoch_cel_losses, s_states, h_states, c_states, d_reward, a_reward, b_reward, c_reward, n_reward))
    if logger is not None:
      logger.add_scalar(policy.env_name + '_qbn/obs_loss',           epoch_obs_losses,iteration)
      logger.add_scalar(policy.env_name + '_qbn/hidden_loss',        epoch_hid_losses,iteration)
      logger.add_scalar(policy.env_name + '_qbn/cell_loss',          epoch_cel_losses,iteration)
      logger.add_scalar(policy.env_name + '_qbn/qbn_reward',         d_reward, iteration)
      logger.add_scalar(policy.env_name + '_qbn/cellonly_reward',    a_reward, iteration)
      logger.add_scalar(policy.env_name + '_qbn/hiddenonly_reward',  b_reward, iteration)
      logger.add_scalar(policy.env_name + '_qbn/stateonly_reward',   c_reward, iteration)
      logger.add_scalar(policy.env_name + '_qbn/observation_states', s_states, iteration)
      logger.add_scalar(policy.env_name + '_qbn/hidden_states',      h_states, iteration)
      logger.add_scalar(policy.env_name + '_qbn/cell_states',        c_states, iteration)

  print("Training phase over. Beginning finetuning.")

  for fine_iter in range(args.finetuning):
    losses = []
    for ep in range(args.episodes):
      env = env_fn()
      state = torch.as_tensor(env.reset())

      done = False
      traj_len = 0
      if hasattr(policy, 'init_hidden_state'):
        policy.init_hidden_state()

      reward = 0
      while not done and traj_len < args.traj_len:
        with torch.no_grad():
          state        = torch.as_tensor(state).float()
          norm_state   = policy.normalize_state(state, update=False)

        hidden, cell = policy.hidden[0], policy.cells[0] # save hidden and cell states

        # Compute qbn values
        policy.hidden, policy.cells = [hidden_qbn(hidden)], [cell_qbn(cell)]
        qbn_action                  = policy(obs_qbn(norm_state))

        with torch.no_grad():
          policy.hidden, policy.cells = [hidden], [cell]
          action       = policy(norm_state)

        state, r, done, _ = env.step(action.numpy())
        reward += r
        traj_len += 1

        step_loss = 0.5 * (action - qbn_action).pow(2)
        losses += [step_loss]
    
    losses = torch.stack(losses).mean()
    losses.backward()
    for opt in [obs_optim, hidden_optim, cell_optim]:
      opt.zero_grad()
      opt.step()

    print("\nEvaluating...")
    d_reward, s_states, h_states, c_states = evaluate(policy, obs_qbn=obs_qbn, hid_qbn=hidden_qbn, cel_qbn=cell_qbn)
    a_reward, _, _, _                      = evaluate(policy, obs_qbn=None,    hid_qbn=None,       cel_qbn=cell_qbn)
    b_reward, _, _, _                      = evaluate(policy, obs_qbn=None,    hid_qbn=hidden_qbn, cel_qbn=None)
    c_reward, _, _, _                      = evaluate(policy, obs_qbn=obs_qbn, hid_qbn=None,       cel_qbn=None)

    logger.add_scalar(policy.env_name + '_qbn/finetune_loss',      losses.item(), iteration + fine_iter)
    logger.add_scalar(policy.env_name + '_qbn/qbn_reward',         d_reward,      iteration + fine_iter)
    logger.add_scalar(policy.env_name + '_qbn/cellonly_reward',    a_reward,      iteration + fine_iter)
    logger.add_scalar(policy.env_name + '_qbn/hiddenonly_reward',  b_reward,      iteration + fine_iter)
    logger.add_scalar(policy.env_name + '_qbn/stateonly_reward',   c_reward,      iteration + fine_iter)
    logger.add_scalar(policy.env_name + '_qbn/observation_states', s_states,      iteration + fine_iter)
    logger.add_scalar(policy.env_name + '_qbn/hidden_states',      h_states,      iteration + fine_iter)
    logger.add_scalar(policy.env_name + '_qbn/cell_states',        c_states,      iteration + fine_iter)

    print("Finetuning loss: {:7.5f} | States: {:5d} {:5d} {:5d} | QBN reward: {:5.1f} ({:5.1f}, {:5.1f}, {:5.1f}) | Nominal reward {:5.0f} ".format(losses, s_states, h_states, c_states, d_reward, a_reward, b_reward, c_reward, n_reward))

    if best_reward is None or d_reward > best_reward:
      torch.save(obs_qbn, os.path.join(logger.dir, 'obsqbn.pt'))
      torch.save(hidden_qbn, os.path.join(logger.dir, 'hidqbn.pt'))
      torch.save(cell_qbn, os.path.join(logger.dir, 'celqbn.pt'))
