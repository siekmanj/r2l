from algos.ddpg import DDPG
from algos.td3  import TD3
from algos.sac  import SAC
import locale, os, random, torch, time, ray
import numpy as np

from util.env import env_factory, train_normalizer
from util.log import create_logger

from torch.nn.utils.rnn import pad_sequence

from copy import deepcopy

class ReplayBuffer():
  def __init__(self, max_size):
    self.max_size    = int(max_size)
    self.states      = [] 
    self.actions     = [] 
    self.next_states = []
    self.rewards     = [] 
    self.not_dones   = []
    self.traj_idx    = [0]
    self.size        = 0

  def _cull_buffer(self):
    while len(self.states) > self.max_size:
      length           = self.traj_idx[1]
      self.states      = self.states[self.traj_idx[1]:]
      self.actions     = self.actions[self.traj_idx[1]:]
      self.next_states = self.next_states[self.traj_idx[1]:]
      self.rewards     = self.rewards[self.traj_idx[1]:]
      self.not_dones   = self.not_dones[self.traj_idx[1]:]
      self.traj_idx    = [idx - length for idx in self.traj_idx[1:]]

    self.size = len(self.states)

  def push(self, state, action, next_state, reward, done):
    self.states      += [state]
    self.actions     += [action]
    self.next_states += [next_state]
    self.rewards     += [[reward]]
    self.not_dones   += [[1] if done is False else [0]]

    if done:
      self.traj_idx.append(len(self.states))

    self._cull_buffer()

  def merge_with(self, buffers):
    for b in buffers:
      offset            = len(self.states)
      self.states      += b.states
      self.actions     += b.actions
      self.rewards     += b.rewards
      self.next_states += b.next_states
      self.not_dones   += b.not_dones
      self.traj_idx    += [offset + i for i in b.traj_idx[1:]]
      self.size        += b.size

    self._cull_buffer()

  def sample(self, batch_size, recurrent=False):
    if recurrent:
      # Collect raw trajectories from replay buffer
      traj_indices = np.random.randint(0, len(self.traj_idx)-1, size=batch_size)

      # Extract trajectory info into separate lists to be padded and batched
      states       = [torch.Tensor(self.states     [self.traj_idx[i]:self.traj_idx[i+1]]) for i in traj_indices]
      actions      = [torch.Tensor(self.actions    [self.traj_idx[i]:self.traj_idx[i+1]]) for i in traj_indices]
      rewards      = [torch.Tensor(self.rewards    [self.traj_idx[i]:self.traj_idx[i+1]]) for i in traj_indices]
      not_dones    = [torch.Tensor(self.not_dones  [self.traj_idx[i]:self.traj_idx[i+1]]) for i in traj_indices]
      next_states  = [torch.Tensor(self.next_states[self.traj_idx[i]:self.traj_idx[i+1]]) for i in traj_indices]

      # Get the trajectory mask
      traj_mask = [torch.ones_like(reward) for reward in rewards]

      # Pad all trajectories to be the same length, shape is (traj_len x batch_size x dim)
      states      = pad_sequence(states,      batch_first=False)
      actions     = pad_sequence(actions,     batch_first=False)
      next_states = pad_sequence(next_states, batch_first=False)
      rewards     = pad_sequence(rewards,     batch_first=False)
      not_dones   = pad_sequence(not_dones,   batch_first=False)
      traj_mask   = pad_sequence(traj_mask,   batch_first=False)

      return states, actions, next_states, rewards, not_dones, traj_mask
    else:
      idx = np.random.randint(0, self.size-1, size=batch_size)
      return self.states[idx], self.actions[idx], self.next_states[next_idx], self.rewards[idx], self.not_dones[idx], 1

@ray.remote
class Off_Policy_Worker:
  def __init__(self, actor, env_fn):
    torch.set_num_threads(1)
    self.actor = deepcopy(actor)
    self.env   = env_fn()

    if hasattr(self.env, 'dynamics_randomization'):
      self.dynamics_randomization = self.env.dynamics_randomization
    else:
      self.dynamics_randomization = False
  
  def sync_policy(self, new_actor_params, input_norm=None):
    for p, new_p in zip(self.actor.parameters(), new_actor_params):
      p.data.copy_(new_p)

    if input_norm is not None:
      self.actor.welford_state_mean, \
      self.actor.welford_state_mean_diff, \
      self.actor.welford_state_n = input_norm

  def collect_episode(self, noise, max_len):
    with torch.no_grad():
      buff  = ReplayBuffer(1e6)
      done  = False
      state = self.env.reset()
      steps = 1

      if hasattr(self.actor, 'init_hidden_state'):
        self.actor.init_hidden_state()

      while not done:
        steps += 1
        action = self.actor.forward(state).numpy()
        if noise is not None:
          action += np.random.normal(0, noise, size=self.actor.action_dim)

        next_state, reward, done, _ = self.env.step(action)

        if steps > max_len:
          done = True
          #print("sending done, max len,", max_len, "steps: ", steps)

        buff.push(state, action, next_state, reward, done)

        state = next_state

      return buff

def eval_policy(policy, env, episodes, traj_len):
  reward_sum = 0
  episode = 0

  if hasattr(env, 'dynamics_randomization'):
    rand = env.dynamics_randomization
  else:
    rand = False
  env.dynamics_randomization = False

  with torch.no_grad():
    while episode < episodes:
      state = env.reset()
      done = False
      timesteps = 0
      episode += 1

      if hasattr(policy, 'init_hidden_state'):
        policy.init_hidden_state()

      while not done and timesteps < traj_len:
        action = policy.forward(torch.Tensor(state)).numpy()
        state, reward, done, _ = env.step(action)
        reward_sum += reward
        timesteps += 1

    env.dynamics_randomization = rand
    return reward_sum / episodes

def run_experiment(args):
  from policies.critic import FF_Q, LSTM_Q, GRU_Q
  from policies.actor import FF_Stochastic_Actor, LSTM_Stochastic_Actor, GRU_Stochastic_Actor, FF_Actor, LSTM_Actor, GRU_Actor

  locale.setlocale(locale.LC_ALL, '')

  # wrapper function for creating parallelized envs
  env = env_factory(args.env)()

  random.seed(args.seed)
  np.random.seed(args.seed)
  torch.manual_seed(args.seed)
  if hasattr(env, 'seed'):
    env.seed(args.seed)

  obs_space = env.observation_space.shape[0]
  act_space = env.action_space.shape[0]

  replay_buff = ReplayBuffer(args.buffer)

  layers = [int(x) for x in args.layers.split(',')]

  if args.arch == 'lstm':
    q1 = LSTM_Q(obs_space, act_space, env_name=args.env, layers=layers)
    q2 = LSTM_Q(obs_space, act_space, env_name=args.env, layers=layers)

    if args.algo == 'sac':
      actor = LSTM_Stochastic_Actor(obs_space, act_space, env_name=args.env, bounded=True, layers=layers)
    else:
      actor = LSTM_Actor(obs_space, act_space, env_name=args.env, layers=layers)
  elif args.arch == 'gru':
    q1 = GRU_Q(obs_space, act_space, env_name=args.env, layers=layers)
    q2 = GRU_Q(obs_space, act_space, env_name=args.env, layers=layers)

    if args.algo == 'sac':
      actor = GRU_Stochastic_Actor(obs_space, act_space, env_name=args.env, bounded=True, layers=layers)
    else:
      actor = GRU_Actor(obs_space, act_space, env_name=args.env, layers=layers)
  elif args.arch == 'ff':
    q1 = FF_Q(obs_space, act_space, env_name=args.env, layers=layers)
    q2 = FF_Q(obs_space, act_space, env_name=args.env, layers=layers)

    if args.algo == 'sac':
      actor = FF_Stochastic_Actor(obs_space, act_space, env_name=args.env, bounded=True, layers=layers)
    else:
      actor = FF_Actor(obs_space, act_space, env_name=args.env, layers=layers)

  if args.algo == 'sac':
    print('Soft Actor-Critic')
    algo = SAC(actor, q1, q2, torch.prod(torch.Tensor(env.reset().shape)), args)
  elif args.algo == 'td3':
    print('Twin-Delayed Deep Deterministic Policy Gradient')
    algo = TD3(actor, q1, q2, args)
  elif args.algo == 'ddpg':
    print('Deep Deterministic Policy Gradient')
    algo = DDPG(actor, q1, args)

  print("\tenv:            {}".format(args.env))
  print("\tseed:           {}".format(args.seed))
  print("\ttimesteps:      {:n}".format(args.timesteps))
  print("\tactor_lr:       {}".format(args.a_lr))
  print("\tcritic_lr:      {}".format(args.c_lr))
  print("\tdiscount:       {}".format(args.discount))
  print("\ttau:            {}".format(args.tau))
  print("\tbatch_size:     {}".format(args.batch_size))
  print("\twarmup period:  {:n}".format(args.start_timesteps))
  print("\tworkers:        {}".format(args.workers))
  print("\tlayers:         {}".format(args.layers))
  print()

  # create a tensorboard logging object
  logger = create_logger(args)

  if args.save_actor is None:
    args.save_actor = os.path.join(logger.dir, 'actor.pt')

  # Keep track of some statistics for each episode
  training_start = time.time()
  episode_start = time.time()
  best_reward = None

  if not ray.is_initialized():
    #if args.redis is not None:
    #  ray.init(redis_address=args.redis)
    #else:
    #  ray.init(num_cpus=args.workers)
    ray.init(num_cpus=args.workers)

  workers = [Off_Policy_Worker.remote(actor, env_factory(args.env)) for _ in range(args.workers)]

  train_normalizer(algo.actor, args.prenormalize_steps, noise=algo.expl_noise)

  # Fill replay buffer, update policy until n timesteps have passed
  timesteps, i = 0, 0
  state = env.reset().astype(np.float32)
  while i < args.iterations:
    if timesteps < args.timesteps:
      actor_param_id  = ray.put(list(algo.actor.parameters()))
      norm_id = ray.put([algo.actor.welford_state_mean, algo.actor.welford_state_mean_diff, algo.actor.welford_state_n])

      for w in workers:
        w.sync_policy.remote(actor_param_id, input_norm=norm_id)

      buffers = ray.get([w.collect_episode.remote(args.expl_noise, args.traj_len) for w in workers])

      replay_buff.merge_with(buffers)

      timesteps += sum(len(b.states) for b in buffers)

      #for i in range(len(replay_buff.traj_idx)-1):
      #  for j in range(replay_buff.traj_idx[i], replay_buff.traj_idx[i+1]):
      #    print("traj {:2d} timestep {:3d}, not done {}, reward {},".format(i, j, replay_buff.not_dones[j], replay_buff.rewards[j]))

    if (algo.recurrent and len(replay_buff.traj_idx) > args.batch_size) or (not algo.recurrent and replay_buff.size > args.batch_size):
      i += 1
      loss = []
      for _ in range(args.updates):
        loss.append(algo.update_policy(replay_buff, batch_size=args.batch_size))
      loss = np.mean(loss, axis=0)

      print('algo {:4s} | explored: {:5n} of {:5n}'.format(args.algo, timesteps, args.timesteps), end=' | ')
      if args.algo == 'ddpg':
        print('iteration {:6n} | actor loss {:6.4f} | critic loss {:6.4f} | buffer size {:6n} / {:6n} ({:4n} trajectories) | {:60s}'.format(
          i+1, 
          loss[0], 
          loss[1], 
          replay_buff.size, 
          replay_buff.max_size,
          len(replay_buff.traj_idx), 
          ''
        ))
        logger.add_scalar(args.env + '/actor loss', loss[0], i)
        logger.add_scalar(args.env + '/critic loss', loss[1], i)
      if args.algo == 'td3':
        print('iteration {:6n} | actor loss {:6.4f} | critic loss {:6.4f} | buffer size {:6n} / {:6n} ({:4n} trajectories) | {:60s}'.format(
          i+1, 
          loss[0], 
          loss[1], 
          replay_buff.size, 
          replay_buff.max_size,
          len(replay_buff.traj_idx), 
          ''
        ))
        logger.add_scalar(args.env + '/actor loss', loss[0], i)
        logger.add_scalar(args.env + '/critic loss', loss[1], i)

      if i % args.eval_every == 0 and iter != 0:
        eval_reward = eval_policy(algo.actor, env, 5, args.traj_len)
        logger.add_scalar(args.env + '/return', eval_reward, i)

        print("evaluation after {:4d} iterations | return: {:7.3f}".format(i, eval_reward, ''))
        if best_reward is None or eval_reward > best_reward:
          torch.save(algo.actor, args.save_actor)
          best_reward = eval_reward
          print("\t(best policy so far! saving to {})".format(args.save_actor))

    else:
        if algo.recurrent:
            print("Collected {:5d} of {:5d} warmup trajectories \t\t".format(len(replay_buff.traj_idx), args.batch_size), end='\r')
        else:
            print("Collected {:5d} of {:5d} warmup trajectories \t\t".format(replay_buff.size, args.batch_size), end='\r')
    """
      if args.algo == 'sac':
        alpha_loss = np.mean([loss[2] for loss in losses])
        logger.add_scalar(args.env + '/alpha loss', alpha_loss, timesteps - args.start_timesteps)

      completion     = 1 - float(timesteps) / args.timesteps
      avg_sample_r   = (time.time() - training_start)/timesteps
      secs_remaining = avg_sample_r * args.timesteps * completion
      hrs_remaining  = int(secs_remaining//(60*60))
      min_remaining  = int(secs_remaining - hrs_remaining*60*60)//60

    """
