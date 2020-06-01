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

@ray.remote
def collect_data(actor, timesteps, max_traj_len, seed):
  """
  A Ray remote function which collects information to be used when training QBNs.
  Specifically, it returns a list of states, actions, hidden states, and if the
  actor is an LSTM policy, cell states.
  """
  torch.set_num_threads(1)
  np.random.seed(seed)
  policy = deepcopy(actor)
  layertype = policy.layers[0].__class__.__name__

  with torch.no_grad(): # no gradients necessary
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
      if hasattr(actor, 'init_hidden_state'): # initialize the memory of the network if recurrent
        actor.init_hidden_state()

      while not done and traj_len < max_traj_len:
        state        = torch.as_tensor(state).float()
        action       = actor(state)
        hidden = actor.hidden[0].view(-1)

        if layertype == 'LSTMCell':
          cell = actor.cells[0].view(-1)

        # append the information we care about to lists to be returned.
        states.append(state.numpy())
        actions.append(action.numpy())
        hiddens.append(hidden.numpy())
        if layertype == 'LSTMCell':
          cells.append(cell.numpy())

        state, _, done, _ = env.step(action.numpy())
        traj_len += 1
        num_steps += 1

    states  = np.array(states)
    actions = np.array(actions)
    hiddens = np.array(hiddens)
    cells   = np.array(cells)
    return [states, actions, hiddens, cells]


# Function used for evaluating the performance of the policy networks when any, some, or none of the QBNs are inserted.
def evaluate(actor, obs_qbn=None, hid_qbn=None, cel_qbn=None, act_qbn=None, episodes=5, max_traj_len=1000):
  """ 
  A function used for evaluating the performance of a policy network when any, some,
  or none of the QBNs are inserted into key points of its execution.
  """
  with torch.no_grad(): # No gradients necessary
    env = env_factory(actor.env_name)()
    reward = 0

    layertype = actor.layers[0].__class__.__name__
    obs_states = {}
    hid_states = {}
    cel_states = {}
    act_states = {}
    for _ in range(episodes):
      done = False
      traj_len = 0

      if hasattr(actor, 'init_hidden_state'):
        actor.init_hidden_state()

      state = torch.as_tensor(env.reset())
      while not done and traj_len < max_traj_len:
        state        = torch.as_tensor(state).float()
        hidden_state = actor.hidden[0]
        if layertype == 'LSTMCell':
          cell_state   = actor.cells[0]

        if obs_qbn is not None:
          state   = obs_qbn(state) # discretize state
          disc_state = np.round(obs_qbn.encode(state).numpy()).astype(int)
          key = ''.join(map(str, disc_state))
          if key not in obs_states:
            obs_states[key] = True

        if hid_qbn is not None: # handle the hidden state QBN
          actor.hidden = [hid_qbn(hidden_state)] # discretize hidden state
          disc_state = np.round(hid_qbn.encode(hidden_state).numpy()).astype(int)
          key = ''.join(map(str, disc_state[0]))
          if key not in hid_states:
            hid_states[key] = True

        if cel_qbn is not None: # handle the cell state QBN
          actor.cells  = [cel_qbn(cell_state)] # discretize cell state if using LSTM
          disc_state = np.round(cel_qbn.encode(cell_state).numpy()).astype(int)
          key = ''.join(map(str, disc_state[0]))
          if key not in cel_states:
            cel_states[key] = True

        action       = actor(state) # run the policy to get an action

        if act_qbn is not None: # handle the action QBN
          action = act_qbn(action) # discretize action 
          disc_state = np.round(act_qbn.encode(action).numpy()).astype(int)
          key = ''.join(map(str, disc_state))
          if key not in act_states:
            act_states[key] = True

        state, r, done, _ = env.step(action.numpy())
        traj_len += 1
        reward += r/episodes
    if layertype == 'LSTMCell': # return cell states in addition to others if using LSTM
      return reward, len(obs_states), len(hid_states), len(cel_states), len(act_states)
    else:
      return reward, len(obs_states), len(hid_states), None, len(act_states)
  
def run_experiment(args):
  """
  The entry point for the QBN insertion algorithm. This function is called by r2l.py,
  and passed an args dictionary which contains hyperparameters for running the experiment.
  """
  locale.setlocale(locale.LC_ALL, '')

  from util.env import env_factory
  from util.log import create_logger

  if args.policy is None:
    print("You must provide a policy with --policy.")
    exit(1)

  policy = torch.load(args.policy) # load policy to be discretized

  layertype = policy.layers[0].__class__.__name__ 
  if layertype != 'LSTMCell' and layertype != 'GRUCell': # ensure that the policy loaded is actually recurrent
    print("Cannot do QBN insertion on a non-recurrent policy.")
    raise NotImplementedError 

  if len(policy.layers) > 1: # ensure that the policy only has one hidden layer
    print("Cannot do QBN insertion on a policy with more than one hidden layer.")
    raise NotImplementedError

  # retrieve dimensions of relevant quantities
  env_fn     = env_factory(policy.env_name)
  obs_dim    = env_fn().observation_space.shape[0]
  action_dim = env_fn().action_space.shape[0]
  hidden_dim = policy.layers[0].hidden_size

  # parse QBN layer sizes from command line arg
  layers = [int(x) for x in args.layers.split(',')]

  # create QBNs
  obs_qbn    = QBN(obs_dim,    layers=layers)
  hidden_qbn = QBN(hidden_dim, layers=layers)
  action_qbn = QBN(action_dim, layers=layers)
  if layertype == 'LSTMCell':
    cell_qbn   = QBN(hidden_dim, layers=layers)
  else:
    cell_qbn = None

  # create optimizers for all QBNs
  obs_optim    = optim.Adam(obs_qbn.parameters(), lr=args.lr, eps=1e-6)
  hidden_optim = optim.Adam(hidden_qbn.parameters(), lr=args.lr, eps=1e-6)
  action_optim = optim.Adam(action_qbn.parameters(), lr=args.lr, eps=1e-6)
  if layertype == 'LSTMCell':
    cell_optim   = optim.Adam(cell_qbn.parameters(), lr=args.lr, eps=1e-6)

  best_reward = None

  if not args.nolog:
    logger = create_logger(args)
  else:
    logger = None

  actor_dir = os.path.split(args.policy)[0]

  ray.init()

  # evaluate policy without QBNs inserted to get baseline reward
  n_reward, _, _, _, _                      = evaluate(policy, episodes=20)
  logger.add_scalar(policy.env_name + '_qbn/nominal_reward', n_reward, 0)

  # if generated data already exists at this directory, then just load that
  if os.path.exists(os.path.join(actor_dir, 'train_states.pt')):
    train_states  = torch.load(os.path.join(actor_dir, 'train_states.pt'))
    train_actions = torch.load(os.path.join(actor_dir, 'train_actions.pt'))
    train_hiddens = torch.load(os.path.join(actor_dir, 'train_hiddens.pt'))

    test_states  = torch.load(os.path.join(actor_dir, 'test_states.pt'))
    test_actions = torch.load(os.path.join(actor_dir, 'test_actions.pt'))
    test_hiddens = torch.load(os.path.join(actor_dir, 'test_hiddens.pt'))

    if layertype == 'LSTMCell':
      train_cells      = torch.load(os.path.join(actor_dir, 'train_cells.pt'))
      test_cells = torch.load(os.path.join(actor_dir, 'test_cells.pt'))
  else: # if no data exists and we need to generate some
    start = time.time()
    data = ray.get([collect_data.remote(policy, args.dataset/args.workers, 400, np.random.randint(65535)) for _ in range(args.workers)])
    states  = torch.from_numpy(np.vstack([r[0] for r in data]))
    actions = torch.from_numpy(np.vstack([r[1] for r in data]))
    hiddens = torch.from_numpy(np.vstack([r[2] for r in data]))
    if layertype == 'LSTMCell':
      cells   = torch.from_numpy(np.vstack([r[3] for r in data]))

    split = int(0.8 * len(states)) # 80/20 train test split 

    train_states, test_states   = states[:split], states[split:]
    train_actions, test_actions = actions[:split], actions[split:]
    train_hiddens, test_hiddens = hiddens[:split], hiddens[split:]
    if layertype == 'LSTMCell':
      train_cells, test_cells = cells[:split], cells[split:]

    print("{:3.2f} to collect {} timesteps.  Training set is {}, test set is {}".format(time.time() - start, len(states), len(train_states), len(test_states)))

    torch.save(train_states, os.path.join(actor_dir, 'train_states.pt'))
    torch.save(train_actions, os.path.join(actor_dir, 'train_actions.pt'))
    torch.save(train_hiddens, os.path.join(actor_dir, 'train_hiddens.pt'))
    if layertype == 'LSTMCell':
      torch.save(train_cells, os.path.join(actor_dir, 'train_cells.pt'))

    torch.save(test_states, os.path.join(actor_dir, 'test_states.pt'))
    torch.save(test_actions, os.path.join(actor_dir, 'test_actions.pt'))
    torch.save(test_hiddens, os.path.join(actor_dir, 'test_hiddens.pt'))
    if layertype == 'LSTMCell':
      torch.save(test_cells, os.path.join(actor_dir, 'test_cells.pt'))

  # run the nominal QBN training algorithm via unsupervised learning on the dataset
  for epoch in range(args.epochs):
    random_indices = SubsetRandomSampler(range(train_states.shape[0]))
    sampler        = BatchSampler(random_indices, args.batch_size, drop_last=False)

    epoch_obs_losses = []
    epoch_hid_losses = []
    epoch_act_losses = []
    epoch_cel_losses = []
    for i, batch in enumerate(sampler):

      # get batch inputs from dataset
      batch_states  = train_states[batch]
      batch_actions = train_actions[batch]
      batch_hiddens = train_hiddens[batch]
      if layertype == 'LSTMCell':
        batch_cells   = train_cells[batch]

      # do forward pass to create derivative graph
      obs_loss = 0.5 * (batch_states  - obs_qbn(batch_states)).pow(2).mean()
      hid_loss = 0.5 * (batch_hiddens - hidden_qbn(batch_hiddens)).pow(2).mean()
      act_loss = 0.5 * (batch_actions - action_qbn(batch_actions)).pow(2).mean()
      if layertype == 'LSTMCell':
        cel_loss = 0.5 * (batch_cells   - cell_qbn(batch_cells)).pow(2).mean()

      # gradient calculation and parameter updates
      obs_optim.zero_grad()
      obs_loss.backward()
      obs_optim.step()

      hidden_optim.zero_grad()
      hid_loss.backward()
      hidden_optim.step()

      action_optim.zero_grad()
      act_loss.backward()
      action_optim.step()

      if layertype == 'LSTMCell':
        cell_optim.zero_grad()
        cel_loss.backward()
        cell_optim.step()

      epoch_obs_losses.append(obs_loss.item())
      epoch_hid_losses.append(hid_loss.item())
      epoch_act_losses.append(act_loss.item())
      if layertype == 'LSTMCell':
        epoch_cel_losses.append(cel_loss.item())
      print("epoch {:3d} / {:3d}, batch {:3d} / {:3d}".format(epoch+1, args.epochs, i+1, len(sampler)), end='\r')

    
    epoch_obs_losses = np.mean(epoch_obs_losses)
    epoch_hid_losses = np.mean(epoch_hid_losses)
    epoch_act_losses = np.mean(epoch_act_losses)
    if layertype == 'LSTMCell':
      epoch_cel_losses = np.mean(epoch_cel_losses)

    # collect some statistics about performance on the test set
    with torch.no_grad():
      state_loss  = 0.5 * (test_states  - obs_qbn(test_states)).pow(2).mean()
      hidden_loss = 0.5 * (test_hiddens - hidden_qbn(test_hiddens)).pow(2).mean()
      act_loss    = 0.5 * (test_actions - action_qbn(test_actions)).pow(2).mean()
      if layertype == 'LSTMCell':
        cell_loss = 0.5 * (test_cells - cell_qbn(test_cells)).pow(2).mean()

    # evaluate QBN performance one-by-one
    print("\nEvaluating...")
    d_reward, s_states, h_states, c_states, a_states = evaluate(policy, obs_qbn=obs_qbn, hid_qbn=hidden_qbn, cel_qbn=cell_qbn, act_qbn=action_qbn)
    c_reward = 0.0
    if layertype == 'LSTMCell':
      c_reward, _, _, _, _                    = evaluate(policy, obs_qbn=None,    hid_qbn=None,       cel_qbn=cell_qbn, act_qbn=None)
    h_reward, _, _, _, _                      = evaluate(policy, obs_qbn=None,    hid_qbn=hidden_qbn, cel_qbn=None, act_qbn=None)
    s_reward, _, _, _, _                      = evaluate(policy, obs_qbn=obs_qbn, hid_qbn=None,       cel_qbn=None, act_qbn=None)
    a_reward, _, _, _, _                      = evaluate(policy, obs_qbn=None,    hid_qbn=None,       cel_qbn=None, act_qbn=action_qbn)

    if best_reward is None or d_reward > best_reward:
      torch.save(obs_qbn, os.path.join(logger.dir, 'obsqbn.pt'))
      torch.save(hidden_qbn, os.path.join(logger.dir, 'hidqbn.pt'))
      if layertype == 'LSTMCell':
        torch.save(cell_qbn, os.path.join(logger.dir, 'celqbn.pt'))

    if layertype == 'LSTMCell':
      print("Losses: {:7.5f} {:7.5f} {:7.5f}".format(state_loss, hidden_loss, cell_loss))
      print("States: {:5d} {:5d} {:5d}".format(s_states, h_states, c_states)) 
      print("QBN reward: {:5.1f} ({:5.1f}, {:5.1f}, {:5.1f}, {:5.1f}) | Nominal reward {:5.0f} ".format(d_reward, h_reward, s_reward, c_reward, a_reward, n_reward))
    else:
      print("Losses: {:7.5f} {:7.5f}".format(state_loss, hidden_loss))
      print("States: {:5d} {:5d} ".format(s_states, h_states))
      print("QBN reward: {:5.1f} ({:5.1f}, {:5.1f}, {:5.1f}) | Nominal reward {:5.0f} ".format(d_reward, h_reward, s_reward, a_reward, n_reward))

    if logger is not None:
      logger.add_scalar(policy.env_name + '_qbn/obs_loss',           state_loss, epoch)
      logger.add_scalar(policy.env_name + '_qbn/hidden_loss',        hidden_loss, epoch)
      logger.add_scalar(policy.env_name + '_qbn/qbn_reward',         d_reward, epoch)
      if layertype == 'LSTMCell':
        logger.add_scalar(policy.env_name + '_qbn/cell_loss',        cell_loss, epoch)
        logger.add_scalar(policy.env_name + '_qbn/cellonly_reward',  c_reward, epoch)
        logger.add_scalar(policy.env_name + '_qbn/cell_states',      c_states, epoch)

      logger.add_scalar(policy.env_name + '_qbn/obsonly_reward',     s_reward, epoch)
      logger.add_scalar(policy.env_name + '_qbn/hiddenonly_reward',  h_reward, epoch)
      logger.add_scalar(policy.env_name + '_qbn/actiononly_reward',  a_reward, epoch)

      logger.add_scalar(policy.env_name + '_qbn/observation_states', s_states, epoch)
      logger.add_scalar(policy.env_name + '_qbn/hidden_states',      h_states, epoch)
      logger.add_scalar(policy.env_name + '_qbn/action_states',      a_states, epoch)

  print("Training phase over. Beginning finetuning.")

  # initialize new optimizers, since the gradient magnitudes will likely change as we are calculating a different quantity.
  obs_optim    = optim.Adam(obs_qbn.parameters(), lr=args.lr, eps=1e-6)
  hidden_optim = optim.Adam(hidden_qbn.parameters(), lr=args.lr, eps=1e-6)
  if layertype == 'LSTMCell':
    cell_optim   = optim.Adam(cell_qbn.parameters(), lr=args.lr, eps=1e-6)
    optims = [obs_optim, hidden_optim, cell_optim, action_optim]
  else:
    optims = [obs_optim, hidden_optim, action_optim]

  optims = [action_optim]

  # run the finetuning portion of the QBN algorithm.
  for fine_iter in range(args.iterations):
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

        hidden        = policy.hidden[0]
        #policy.hidden = [hidden_qbn(hidden)]

        if layertype == 'LSTMCell':
          cell        = policy.cells[0]
          #policy.cells = [cell_qbn(cell)]

        # Compute qbn values
        qbn_action = action_qbn(policy(obs_qbn(state)))

        with torch.no_grad():
          policy.hidden = [hidden]
          if layertype == 'LSTMCell':
            policy.cells  = [cell]
          action       = policy(state)

        state, r, done, _ = env.step(action.numpy())
        reward += r
        traj_len += 1

        step_loss = 0.5 * (action - qbn_action).pow(2) # this creates the derivative graph for our backwards pass
        losses += [step_loss]
    
    # clear our parameter gradients
    for opt in optims:
      opt.zero_grad()

    # run the backwards pass
    losses = torch.stack(losses).mean()
    losses.backward()

    # update parameters
    for opt in optims:
      opt.step()

    # evaluate our QBN performance one-by-one
    print("\nEvaluating...")
    d_reward, s_states, h_states, c_states, a_states = evaluate(policy, obs_qbn=obs_qbn, hid_qbn=hidden_qbn, cel_qbn=cell_qbn, act_qbn=action_qbn)
    c_reward = 0.0
    if layertype == 'LSTMCell':
      c_reward, _, _, _, _                    = evaluate(policy, obs_qbn=None,    hid_qbn=None,       cel_qbn=cell_qbn, act_qbn=None)
    h_reward, _, _, _, _                      = evaluate(policy, obs_qbn=None,    hid_qbn=hidden_qbn, cel_qbn=None, act_qbn=None)
    s_reward, _, _, _, _                      = evaluate(policy, obs_qbn=obs_qbn, hid_qbn=None,       cel_qbn=None, act_qbn=None)
    a_reward, _, _, _, _                      = evaluate(policy, obs_qbn=None,    hid_qbn=None,       cel_qbn=None, act_qbn=action_qbn)

    if layertype == 'LSTMCell':
      print("Finetuning loss: {:7.5f}".format(losses))
      print("States: {:5d} {:5d} {:5d}".format(s_states, h_states, c_states)) 
      print("QBN reward: {:5.1f} ({:5.1f}, {:5.1f}, {:5.1f}) | Nominal reward {:5.0f} ".format(d_reward, h_reward, s_reward, c_reward, a_reward, n_reward))
    else:
      print("Losses: {:7.5f} {:7.5f}".format(epoch_obs_losses, epoch_hid_losses))
      print("States: {:5d} {:5d} ".format(s_states, h_states))
      print("QBN reward: {:5.1f} ({:5.1f}, {:5.1f}) | Nominal reward {:5.0f} ".format(d_reward, h_reward, s_reward, a_reward, n_reward))

    if logger is not None:
      logger.add_scalar(policy.env_name + '_qbn/finetune_loss',      losses.item(), epoch + fine_iter)
      logger.add_scalar(policy.env_name + '_qbn/qbn_reward',         d_reward,      epoch + fine_iter)
      if layertype == 'LSTMCell':
        logger.add_scalar(policy.env_name + '_qbn/cellonly_reward',    c_reward, epoch + fine_iter)
        logger.add_scalar(policy.env_name + '_qbn/cell_states',        c_states, epoch + fine_iter)
      
      logger.add_scalar(policy.env_name + '_qbn/obsonly_reward',     s_reward, epoch + fine_iter)
      logger.add_scalar(policy.env_name + '_qbn/hiddenonly_reward',  h_reward, epoch + fine_iter)
      logger.add_scalar(policy.env_name + '_qbn/actiononly_reward',  a_reward, epoch + fine_iter)

      logger.add_scalar(policy.env_name + '_qbn/observation_states', s_states, epoch + fine_iter)
      logger.add_scalar(policy.env_name + '_qbn/hidden_states',      h_states, epoch + fine_iter)
      logger.add_scalar(policy.env_name + '_qbn/action_states',      a_states, epoch + fine_iter)

    if best_reward is None or d_reward > best_reward:
      torch.save(obs_qbn, os.path.join(logger.dir, 'obsqbn.pt'))
      torch.save(hidden_qbn, os.path.join(logger.dir, 'hidqbn.pt'))
      torch.save(cell_qbn, os.path.join(logger.dir, 'celqbn.pt'))
