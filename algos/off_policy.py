from algos.ddpg import DDPG
from algos.td3  import TD3
from algos.sac  import SAC
import locale, os, random, torch, time
import numpy as np

from util.env import env_factory, eval_policy, train_normalizer
from util.log import create_logger

from torch.nn.utils.rnn import pad_sequence

class ReplayBuffer():
  def __init__(self, max_size):
    self.max_size  = int(max_size)
    self.states    = [] 
    self.actions   = [] 
    self.next_states = []
    self.rewards   = [] 
    self.not_dones = []
    self.traj_idx  = [0]
    self.size      = 0
    self.buffer_ready = False

  def push(self, state, action, next_state, reward, done):
    while len(self.states) >= self.max_size:
      print("Buffer full, removing {} / {}".format(len(self.states), self.max_size))
      self.states      = self.states[self.traj_idx[1]:]
      self.actions     = self.actions[self.traj_idx[1]:]
      self.next_states = self.next_states[self.traj_idx[1]:]
      self.rewards     = self.rewards[self.traj_idx[1]:]
      self.not_dones   = self.not_dones[self.traj_idx[1]:]

    self.states      += [state]
    self.actions     += [action]
    self.next_states += [next_state]
    self.rewards     += [reward]
    self.not_dones   += [1 if done is False else 0]

    if done:
      self.traj_idx.append(len(self.states))

    self.size = len(self.states)

  def merge_with(self, buffers):

    for b in buffers:
      offset = len(self.states)
      self.states      += b.states
      self.actions     += b.actions
      self.rewards     += b.rewards
      self.next_states += b.next_states
      self.not_dones   += b.not_dones
      self.traj_idx    += [offset + i for i in b.traj_idx[1:]]
      self.size        += b.size

  def _finish_buffer(self):
    with torch.no_grad():
      self.states = torch.Tensor(self.states)
      self.actions = torch.Tensor(self.actions)
      self.next_states = torch.Tensor(self.next_states)
      self.rewards = torch.Tensor(self.rewards)
      self.not_dones = torch.Tensor(self.not_dones)
      
      self.buffer_ready = True

  def sample(self, batch_size, recurrent=False):
    if not self.buffer_ready:
      self._finish_buffer()

    if recurrent:
      # Collect raw trajectories from replay buffer
      traj_indices = np.random.randint(0, len(self.traj_idx)-1, size=batch_size)

      # Extract trajectory info into separate lists to be padded and batched
      states       = [self.states     [self.traj_idx[i]:self.traj_idx[i+1]] for i in traj_indices]
      actions      = [self.actions    [self.traj_idx[i]:self.traj_idx[i+1]] for i in traj_indices]
      rewards      = [self.rewards    [self.traj_idx[i]:self.traj_idx[i+1]] for i in traj_indices]
      not_dones    = [self.not_dones  [self.traj_idx[i]:self.traj_idx[i+1]] for i in traj_indices]
      next_states  = [self.next_states[self.traj_idx[i]:self.traj_idx[i+1]] for i in traj_indices]

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
      idx      = np.random.randint(0, self.size-1, size=batch_size)
      next_idx = idx + 1
      return self.state[idx], self.action[idx], self.next_state[next_idx], self.reward[idx], self.not_done[idx], 1

def collect_episode(policy, env, noise=0.2, max_len=1000):
  with torch.no_grad():
    buff  = ReplayBuffer(1e6)
    done  = False
    state = env.reset()
    steps = 1

    while not done:
      action = policy.forward(state).numpy()
      if noise is not None:
        action += np.random.normal(0, noise, size=policy.action_dim)

      next_state, reward, done, _ = env.step(action)

      if steps > max_len:
        done = True

      buff.push(state, action, next_state, reward, done)

      state = next_state

    return buff

def run_experiment(args):
  from policies.critic import FF_Q, LSTM_Q
  from policies.actor import FF_Stochastic_Actor, LSTM_Stochastic_Actor, FF_Actor, LSTM_Actor

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

  replay_buff = ReplayBuffer(args.timesteps)

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
  print()

  iter = 0
  episode_reward = 0
  episode_timesteps = 0

  # create a tensorboard logging object
  logger = create_logger(args)

  if args.save_actor is None:
    args.save_actor = os.path.join(logger.dir, 'actor.pt')

  # Keep track of some statistics for each episode
  training_start = time.time()
  episode_start = time.time()
  episode_loss = 0
  update_steps = 0
  best_reward = None

  train_normalizer(algo.actor, args.prenormalize_steps, noise=algo.expl_noise)

  # Fill replay buffer, update policy until n timesteps have passed
  timesteps, episodes = 0, 0
  state = env.reset().astype(np.float32)
  while timesteps < args.timesteps:
    buf = [collect_episode(algo.actor, env, max_len=args.traj_len, noise=algo.expl_noise)]
    replay_buff.merge_with(buf)

    timesteps += sum(len(b.states) for b in buf)
    episodes += 1

    loss = algo.update_policy(replay_buff, batch_size=min(episodes, args.batch_size))
    print("done.")

    """
    buffer_ready = (algo.recurrent and len(replay_buff.traj_idx) > args.batch_size) or (not algo.recurrent and replay_buff.size > args.batch_size)
    episode_timesteps += 1
    timesteps += 1

    if not buffer_ready or warmup:
      iter = 0

    # Update the policy once our replay buffer is big enough
    if buffer_ready and done and not warmup:
      update_steps = 0
      if not algo.recurrent:
        num_updates = episode_timesteps
      else:
        num_updates = 1

      losses = []
      for _ in range(num_updates):
        losses.append(algo.update_policy(replay_buff, args.batch_size, traj_len=args.traj_len))

      episode_elapsed = (time.time() - episode_start)
      episode_secs_per_sample = episode_elapsed / episode_timesteps

      actor_loss   = np.mean([loss[0] for loss in losses])
      critic_loss  = np.mean([loss[1] for loss in losses])
      update_steps = sum([loss[-1] for loss in losses])

      logger.add_scalar(args.env + '/actor loss', actor_loss, timesteps - args.start_timesteps)
      logger.add_scalar(args.env + '/critic loss', critic_loss, timesteps - args.start_timesteps)
      logger.add_scalar(args.env + '/update steps', update_steps, timesteps - args.start_timesteps)

      if args.algo == 'sac':
        alpha_loss = np.mean([loss[2] for loss in losses])
        logger.add_scalar(args.env + '/alpha loss', alpha_loss, timesteps - args.start_timesteps)

      completion     = 1 - float(timesteps) / args.timesteps
      avg_sample_r   = (time.time() - training_start)/timesteps
      secs_remaining = avg_sample_r * args.timesteps * completion
      hrs_remaining  = int(secs_remaining//(60*60))
      min_remaining  = int(secs_remaining - hrs_remaining*60*60)//60

      if iter % args.eval_every == 0 and iter != 0:
        eval_reward = eval_policy(algo.actor, min_timesteps=1000, verbose=False, visualize=False, max_traj_len=args.traj_len)
        logger.add_scalar(args.env + '/return', eval_reward, timesteps - args.start_timesteps)

        print("evaluation after {:4d} episodes | return: {:7.3f} | timesteps {:9n}{:100s}".format(iter, eval_reward, timesteps - args.start_timesteps, ''))
        if best_reward is None or eval_reward > best_reward:
          torch.save(algo.actor, args.save_actor)
          best_reward = eval_reward
          print("\t(best policy so far! saving to {})".format(args.save_actor))

    try:
      print("episode {:5d} | episode timestep {:5d}/{:5d} | return {:5.1f} | update timesteps: {:7n} | {:3.1f}s/1k samples | approx. {:3d}h {:02d}m remain\t\t\t\t".format(
        iter, 
        episode_timesteps, 
        args.traj_len, 
        episode_reward, 
        update_steps, 
        1000*episode_secs_per_sample, 
        hrs_remaining, 
        min_remaining), end='\r')

    except NameError:
      pass

    if done:
      if hasattr(algo.actor, 'init_hidden_state'):
        algo.actor.init_hidden_state()

      episode_start, episode_reward, episode_timesteps, episode_loss = time.time(), 0, 0, 0
      iter += 1
    """
