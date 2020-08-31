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
  def __init__(self, discount=0.99):
    self.discount = discount

    self.clear()

  def __len__(self):
    return len(self.states)

  def clear(self):
    self.states     = []
    self.actions    = []
    self.rewards    = []
    self.values     = []
    self.returns    = []
    self.advantages = []

    self.ep_returns = []
    self.ep_lens = []

    self.size = 0

    self.traj_idx = [0]
    self.buffer_ready = False

  def push(self, state, action, reward, value, done=False):
    self.states  += [state]
    self.actions += [action]
    self.rewards += [reward]
    self.values  += [value]

    self.size += 1

  def end_trajectory(self, terminal_value=0):
    self.traj_idx += [self.size]
    rewards = self.rewards[self.traj_idx[-2]:self.traj_idx[-1]]

    returns = []

    R = terminal_value
    for reward in reversed(rewards):
        R = self.discount * R + reward
        returns.insert(0, R)

    self.returns += returns

    self.ep_returns += [np.sum(rewards)]
    self.ep_lens    += [len(rewards)]

  def _finish_buffer(self, mirror):
    with torch.no_grad():
      self.states  = torch.Tensor(self.states)
      self.actions = torch.Tensor(self.actions)
      self.rewards = torch.Tensor(self.rewards)
      self.returns = torch.Tensor(self.returns)
      self.values  = torch.Tensor(self.values)

      if mirror is not None:
        start = time()
        squished           = np.prod(list(self.states.size())[:-1]) 
        state_dim          = self.states.size()[-1]
        state_squished     = self.states.view(squished, state_dim).numpy()
        self.mirror_states = torch.from_numpy(mirror(state_squished)).view(self.states.size())

      a = self.returns - self.values
      a = (a - a.mean()) / (a.std() + 1e-4)
      self.advantages = a
      self.buffer_ready = True

  def sample(self, batch_size=64, recurrent=False, mirror=None):
    if not self.buffer_ready:
      self._finish_buffer(mirror)

    if recurrent:
      random_indices = SubsetRandomSampler(range(len(self.traj_idx)-1))
      sampler = BatchSampler(random_indices, batch_size, drop_last=True)

      for traj_indices in sampler:
        states     = [self.states[self.traj_idx[i]:self.traj_idx[i+1]]     for i in traj_indices]
        actions    = [self.actions[self.traj_idx[i]:self.traj_idx[i+1]]    for i in traj_indices]
        returns    = [self.returns[self.traj_idx[i]:self.traj_idx[i+1]]    for i in traj_indices]
        advantages = [self.advantages[self.traj_idx[i]:self.traj_idx[i+1]] for i in traj_indices]
        traj_mask  = [torch.ones_like(r) for r in returns]

        states     = pad_sequence(states,     batch_first=False)
        actions    = pad_sequence(actions,    batch_first=False)
        returns    = pad_sequence(returns,    batch_first=False)
        advantages = pad_sequence(advantages, batch_first=False)
        traj_mask  = pad_sequence(traj_mask,  batch_first=False)

        if mirror is None:
          yield states, actions, returns, advantages, traj_mask
        else:
          mirror_states = [self.mirror_states[self.traj_idx[i]:self.traj_idx[i+1]] for i in traj_indices]
          mirror_states = pad_sequence(mirror_states, batch_first=False)
          yield states, mirror_states, actions, returns, advantages, traj_mask

    else:
      random_indices = SubsetRandomSampler(range(self.size))
      sampler = BatchSampler(random_indices, batch_size, drop_last=True)

      for i, idxs in enumerate(sampler):
        states     = self.states[idxs]
        actions    = self.actions[idxs]
        returns    = self.returns[idxs]
        advantages = self.advantages[idxs]

        if mirror is None:
          yield states, actions, returns, advantages, 1
        else:
          mirror_states = self.mirror_states[idxs]
          yield states, mirror_states, actions, returns, advantages, 1


def merge_buffers(buffers):
  memory = Buffer()

  for b in buffers:
    offset = len(memory)

    memory.states  += b.states
    memory.actions += b.actions
    memory.rewards += b.rewards
    memory.values  += b.values
    memory.returns += b.returns

    memory.ep_returns += b.ep_returns
    memory.ep_lens    += b.ep_lens

    memory.traj_idx += [offset + i for i in b.traj_idx[1:]]
    memory.size     += b.size
  return memory

@ray.remote
class PPO_Worker:
  def __init__(self, actor, critic, env_fn, gamma):
    self.gamma  = gamma
    self.actor  = deepcopy(actor)
    self.critic = deepcopy(critic)
    self.env    = env_fn()

    if hasattr(self.env, 'dynamics_randomization'):
      self.dynamics_randomization = self.env.dynamics_randomization
    else:
      self.dynamics_randomization = False

  def sync_policy(self, new_actor_params, new_critic_params, input_norm=None):
    for p, new_p in zip(self.actor.parameters(), new_actor_params):
      p.data.copy_(new_p)

    for p, new_p in zip(self.critic.parameters(), new_critic_params):
      p.data.copy_(new_p)

    if input_norm is not None:
      self.actor.welford_state_mean, self.actor.welford_state_mean_diff, self.actor.welford_state_n = input_norm
      self.critic.copy_normalizer_stats(self.actor)

  def collect_experience(self, max_traj_len, min_steps):
    torch.set_num_threads(1)
    with torch.no_grad():
      start = time()

      num_steps = 0
      memory = Buffer(self.gamma)
      actor  = self.actor
      critic = self.critic

      while num_steps < min_steps:
        self.env.dynamics_randomization = self.dynamics_randomization
        state = torch.Tensor(self.env.reset())

        done = False
        value = 0
        traj_len = 0

        if hasattr(actor, 'init_hidden_state'):
          actor.init_hidden_state()

        if hasattr(critic, 'init_hidden_state'):
          critic.init_hidden_state()

        while not done and traj_len < max_traj_len:
            state = torch.Tensor(state)
            action = actor(state, deterministic=False)
            value = critic(state)

            next_state, reward, done, _ = self.env.step(action.numpy())

            reward = np.array([reward])

            memory.push(state.numpy(), action.numpy(), reward, value.numpy())

            state = next_state

            traj_len += 1
            num_steps += 1

        value = (not done) * critic(torch.Tensor(state)).numpy()
        memory.end_trajectory(terminal_value=value)

      return memory

  def evaluate(self, trajs=1, max_traj_len=400):
    torch.set_num_threads(1)
    with torch.no_grad():
      ep_returns = []
      for traj in range(trajs):
        self.env.dynamics_randomization = False
        state = torch.Tensor(self.env.reset())

        done = False
        traj_len = 0
        ep_return = 0

        if hasattr(self.actor, 'init_hidden_state'):
          self.actor.init_hidden_state()

        while not done and traj_len < max_traj_len:
          action = self.actor(state, deterministic=True)

          next_state, reward, done, _ = self.env.step(action.numpy())

          state = torch.Tensor(next_state)
          ep_return += reward
          traj_len += 1
        ep_returns += [ep_return]

    return np.mean(ep_returns)

class PPO:
    def __init__(self, actor, critic, env_fn, args):

      self.actor = actor
      self.old_actor = deepcopy(actor)
      self.critic = critic

      if actor.is_recurrent or critic.is_recurrent:
        self.recurrent = True
      else:
        self.recurrent = False

      self.actor_optim   = optim.Adam(self.actor.parameters(), lr=args.a_lr, eps=args.eps)
      self.critic_optim  = optim.Adam(self.critic.parameters(), lr=args.c_lr, eps=args.eps)
      self.env_fn        = env_fn
      self.discount      = args.discount
      self.entropy_coeff = args.entropy_coeff
      self.grad_clip     = args.grad_clip
      self.sparsity      = args.sparsity
      self.mirror        = args.mirror
      self.env           = env_fn()

      if not ray.is_initialized():
        if args.redis is not None:
          ray.init(redis_address=args.redis)
        else:
          ray.init(num_cpus=args.workers)

      self.workers = [PPO_Worker.remote(actor, critic, env_fn, args.discount) for _ in range(args.workers)]

    def update_policy(self, states, actions, returns, advantages, mask, mirror_states=None):

      # get old action distribution and log probabilities
      with torch.no_grad():
        old_pdf       = self.old_actor.pdf(states)
        old_log_probs = old_pdf.log_prob(actions).sum(-1, keepdim=True)

      # if we are using sparsity constraint, set the internal boolean for saving memory to true
      if self.sparsity > 0:
        self.actor.calculate_norm = True

      # get new action distribution and log probabilities
      pdf       = self.actor.pdf(states)
      log_probs = pdf.log_prob(actions).sum(-1, keepdim=True)

      if self.sparsity > 0:
        self.actor.calculate_norm = False
        latent_norm = self.actor.get_latent_norm()
      else:
        latent_norm = torch.zeros(1)

      sparsity_loss = self.sparsity * latent_norm

      ratio      = ((log_probs - old_log_probs)).exp()
      cpi_loss   = ratio * advantages * mask
      clip_loss  = ratio.clamp(0.8, 1.2) * advantages * mask
      actor_loss = -torch.min(cpi_loss, clip_loss).mean()

      critic_loss = 0.5 * ((returns - self.critic(states)) * mask).pow(2).mean()

      entropy_penalty = -(self.entropy_coeff * pdf.entropy() * mask).mean()

      if mirror_states is not None:
        mirror_time = time()
        with torch.no_grad():
          action_fn        = self.env.mirror_action
          squished         = np.prod(list(states.size())[:-1]) 
          action_dim       = actions.size()[-1]
          action_squished  = self.actor(mirror_states).view(squished, action_dim).numpy()
          mirrored_actions = torch.from_numpy(action_fn(action_squished)).view(actions.size())

        unmirrored_actions = pdf.mean
        mirror_loss = self.mirror * 4 * (unmirrored_actions - mirrored_actions).pow(2).mean()
        #print("{:3.2f}s to calculate mirror loss".format(time() - mirror_time))
      else:
        mirror_loss = torch.zeros(1)

      self.actor_optim.zero_grad()
      self.critic_optim.zero_grad()

      (actor_loss + entropy_penalty + mirror_loss + sparsity_loss).backward()
      critic_loss.backward()

      torch.nn.utils.clip_grad_norm_(self.critic.parameters(), max_norm=self.grad_clip)
      torch.nn.utils.clip_grad_norm_(self.actor.parameters(), max_norm=self.grad_clip)
      self.actor_optim.step()
      self.critic_optim.step()

      with torch.no_grad():
        return kl_divergence(pdf, old_pdf).mean().numpy(), ((actor_loss + entropy_penalty).item(), critic_loss.item(), mirror_loss.item(), latent_norm.item())

    def do_iteration(self, num_steps, max_traj_len, epochs, kl_thresh=0.02, verbose=True, batch_size=64, mirror=False):
      self.old_actor.load_state_dict(self.actor.state_dict())

      start = time()
      actor_param_id  = ray.put(list(self.actor.parameters()))
      critic_param_id = ray.put(list(self.critic.parameters()))
      norm_id = ray.put([self.actor.welford_state_mean, self.actor.welford_state_mean_diff, self.actor.welford_state_n])

      steps = max(num_steps // len(self.workers), max_traj_len)

      for w in self.workers:
        w.sync_policy.remote(actor_param_id, critic_param_id, input_norm=norm_id)
      if verbose:
        print("\t{:5.4f}s to copy policy params to workers.".format(time() - start))

      eval_reward = np.mean(ray.get([w.evaluate.remote(trajs=1, max_traj_len=max_traj_len) for w in self.workers]))

      start = time()
      buffers = ray.get([w.collect_experience.remote(max_traj_len, steps) for w in self.workers])
      memory = merge_buffers(buffers)

      total_steps = len(memory)
      elapsed = time() - start
      if verbose:
        print("\t{:3.2f}s to collect {:6n} timesteps | {:3.2}k/s.".format(elapsed, total_steps, (total_steps/1000)/elapsed))
      sample_rate = (total_steps/1000)/elapsed

      if self.mirror > 0:
        state_fn = self.env.mirror_state
      else:
        state_fn = None

      done = False
      a_loss = []
      c_loss = []
      m_loss = []
      s_loss = []
      kls    = []
      update_time = time()
      torch.set_num_threads(4)
      for epoch in range(epochs):
        epoch_start = time()
        for batch in memory.sample(batch_size=batch_size, recurrent=self.recurrent, mirror=state_fn):

          if state_fn is not None:
            states, mirror_states, actions, returns, advantages, mask = batch
          else:
            mirror_states = None
            states, actions, returns, advantages, mask = batch

          kl, losses = self.update_policy(states, actions, returns, advantages, mask, mirror_states=mirror_states)
          kls    += [kl]
          a_loss += [losses[0]]
          c_loss += [losses[1]]
          m_loss += [losses[2]]
          s_loss += [losses[3]]

          if max(kls) > kl_thresh:
              done = True
              print("\t\tbatch had kl of {} (threshold {}), stopping optimization early.".format(max(kls), kl_thresh))
              break

        if verbose:
          print("\t\tepoch {:2d} in {:3.2f}s, kl {:6.5f}, actor loss {:6.3f}, critic loss {:6.3f}".format(epoch+1, time() - epoch_start, np.mean(kls), np.mean(a_loss), np.mean(c_loss)))

        if done:
          break

      update_time = time() - update_time
      if verbose:
        print("\t{:3.2f}s to update policy.".format(update_time))

      return eval_reward, np.mean(kls), np.mean(a_loss), np.mean(c_loss), np.mean(m_loss), np.mean(s_loss), len(memory), (sample_rate, update_time)

def run_experiment(args):

  from util.env import env_factory, train_normalizer
  from util.log import create_logger

  from policies.critic import FF_V, LSTM_V, GRU_V
  from policies.actor import FF_Stochastic_Actor, LSTM_Stochastic_Actor, GRU_Stochastic_Actor, QBN_GRU_Stochastic_Actor

  import locale, os
  locale.setlocale(locale.LC_ALL, '')

  # wrapper function for creating parallelized envs
  env_fn = env_factory(args.env)
  obs_dim = env_fn().observation_space.shape[0]
  action_dim = env_fn().action_space.shape[0]

  # Set seeds
  torch.manual_seed(args.seed)
  np.random.seed(args.seed)

  std = torch.ones(action_dim)*args.std

  layers = [int(x) for x in args.layers.split(',')]

  if args.arch.lower() == 'lstm':
    policy = LSTM_Stochastic_Actor(obs_dim, action_dim, env_name=args.env, fixed_std=std, bounded=False, layers=layers)
    critic = LSTM_V(obs_dim, layers=layers)
  elif args.arch.lower() == 'gru':
    policy = GRU_Stochastic_Actor(obs_dim, action_dim, env_name=args.env, fixed_std=std, bounded=False, layers=layers)
    critic = GRU_V(obs_dim, layers=layers)
  elif args.arch.lower() == 'qbngru':
    policy = QBN_GRU_Stochastic_Actor(obs_dim, action_dim, env_name=args.env, fixed_std=std, bounded=False, layers=layers)
    critic = GRU_V(obs_dim, layers=layers)
  elif args.arch.lower() == 'ff':
    policy = FF_Stochastic_Actor(obs_dim, action_dim, env_name=args.env, fixed_std=std, bounded=False, layers=layers)
    critic = FF_V(obs_dim, layers=layers)
  else:
    raise RuntimeError
  policy.legacy = False
  env = env_fn()

  print("Collecting normalization statistics with {} states...".format(args.prenormalize_steps))
  train_normalizer(policy, args.prenormalize_steps, max_traj_len=args.traj_len, noise=1)

  critic.copy_normalizer_stats(policy)

  policy.train(0)
  critic.train(0)

  algo = PPO(policy, critic, env_fn, args)

  # create a tensorboard logging object
  if not args.nolog:
    logger = create_logger(args)
  else:
    logger = None

  if args.save_actor is None and logger is not None:
    args.save_actor = os.path.join(logger.dir, 'actor.pt')

  if args.save_critic is None and logger is not None:
    args.save_critic = os.path.join(logger.dir, 'critic.pt')

  print()
  print("Proximal Policy Optimization:")
  print("\tseed:               {}".format(args.seed))
  print("\tenv:                {}".format(args.env))
  print("\ttimesteps:          {:n}".format(int(args.timesteps)))
  print("\titeration steps:    {:n}".format(int(args.num_steps)))
  print("\tprenormalize steps: {}".format(int(args.prenormalize_steps)))
  print("\ttraj_len:           {}".format(args.traj_len))
  print("\tdiscount:           {}".format(args.discount))
  print("\tactor_lr:           {}".format(args.a_lr))
  print("\tcritic_lr:          {}".format(args.c_lr))
  print("\tadam eps:           {}".format(args.eps))
  print("\tentropy coeff:      {}".format(args.entropy_coeff))
  print("\tgrad clip:          {}".format(args.grad_clip))
  print("\tbatch size:         {}".format(args.batch_size))
  print("\tepochs:             {}".format(args.epochs))
  print("\tworkers:            {}".format(args.workers))
  print()

  itr = 0
  timesteps = 0
  best_reward = None
  while timesteps < args.timesteps:
    eval_reward, kl, a_loss, c_loss, m_loss, s_loss, steps, (times) = algo.do_iteration(args.num_steps, args.traj_len, args.epochs, batch_size=args.batch_size, kl_thresh=args.kl, mirror=args.mirror)

    timesteps += steps
    print("iter {:4d} | return: {:5.2f} | KL {:5.4f} | ".format(itr, eval_reward, kl, timesteps), end='')
    if m_loss != 0:
      print("mirror {:6.5f} | ".format(m_loss), end='')

    if s_loss != 0:
      print("sparsity {:6.5f} | ".format(s_loss), end='')

    print("timesteps {:n}".format(timesteps))

    if best_reward is None or eval_reward > best_reward:
      print("\t(best policy so far! saving to {})".format(args.save_actor))
      best_reward = eval_reward
      if args.save_actor is not None:
        torch.save(algo.actor, args.save_actor)
      
      if args.save_critic is not None:
        torch.save(algo.critic, args.save_critic)

    if logger is not None:
      logger.add_scalar(args.env + '/kl', kl, timesteps)
      logger.add_scalar(args.env + '/return', eval_reward, timesteps)
      logger.add_scalar(args.env + '/actor loss', a_loss, timesteps)
      logger.add_scalar(args.env + '/critic loss', c_loss, timesteps)
      logger.add_scalar(args.env + '/mirror loss', m_loss, timesteps)
      logger.add_scalar(args.env + '/sparsity loss', s_loss, timesteps)
      logger.add_scalar(args.env + '/sample rate', times[0], timesteps)
      logger.add_scalar(args.env + '/update time', times[1], timesteps)
    itr += 1
  print("Finished ({} of {}).".format(timesteps, args.timesteps))
