from algos.ddpg import DDPG
from algos.td3  import TD3
from algos.sac  import SAC
import locale, os, random, torch, time
import numpy as np

from util.env import env_factory, eval_policy, train_normalizer
from util.log import create_logger

from torch.nn.utils.rnn import pad_sequence

class ReplayBuffer():
  def __init__(self, state_dim, action_dim, max_size):
    self.max_size   = int(max_size)
    self.state      = torch.zeros((self.max_size, state_dim))
    self.next_state = torch.zeros((self.max_size, state_dim))
    self.action     = torch.zeros((self.max_size, action_dim))
    self.reward     = torch.zeros((self.max_size, 1))
    self.not_done   = torch.zeros((self.max_size, 1))

    self.trajectory_idx = [0]
    self.trajectories = 0

    self.size = 1

  def push(self, state, action, next_state, reward, done):
    if self.size == self.max_size:
      print("\nBuffer full.")
      exit(1)

    idx = self.size-1

    self.state[idx]      = torch.Tensor(state)
    self.next_state[idx] = torch.Tensor(next_state)
    self.action[idx]     = torch.Tensor(action)
    self.reward[idx]     = reward
    self.not_done[idx]   = 1 - done

    if done:
      self.trajectory_idx.append(self.size)
      self.trajectories += 1

    self.size = min(self.size+1, self.max_size)

  def sample_trajectory(self, max_len):

    traj_idx = np.random.randint(0, self.trajectories-1)
    start_idx = self.trajectory_idx[traj_idx]
    end_idx   = self.trajectory_idx[traj_idx+1]
    #end_idx = start_idx + 1

    #while self.not_done[end_idx] == 1 and end_idx - start_idx < max_len:
    #  end_idx += 1
    #end_idx += 1

    traj_states = self.state[start_idx:end_idx]
    next_states = self.next_state[start_idx:end_idx]
    actions     = self.action[start_idx:end_idx]
    rewards     = self.reward[start_idx:end_idx]
    not_dones   = self.not_done[start_idx:end_idx]

    # Return an entire episode
    return traj_states, actions, next_states, rewards, not_dones, 1

  def sample(self, batch_size, sample_trajectories=False, max_len=1000):
    if sample_trajectories:
      # Collect raw trajectories from replay buffer
      raw_traj = [self.sample_trajectory(max_len) for _ in range(batch_size)]
      steps = sum([len(traj[0]) for traj in raw_traj])

      # Extract trajectory info into separate lists to be padded and batched
      states      = [traj[0] for traj in raw_traj]
      actions     = [traj[1] for traj in raw_traj]
      next_states = [traj[2] for traj in raw_traj]
      rewards     = [traj[3] for traj in raw_traj]
      not_dones   = [traj[4] for traj in raw_traj]

      # Get the trajectory mask for the critic
      traj_mask = [torch.ones_like(reward) for reward in rewards]

      # Pad all trajectories to be the same length, shape is (traj_len x batch_size x dim)
      states      = pad_sequence(states, batch_first=False)
      actions     = pad_sequence(actions, batch_first=False)
      next_states = pad_sequence(next_states, batch_first=False)
      rewards     = pad_sequence(rewards, batch_first=False)
      not_dones   = pad_sequence(not_dones, batch_first=False)
      traj_mask   = pad_sequence(traj_mask, batch_first=False)

      return states, actions, next_states, rewards, not_dones, steps, traj_mask

    else:
      idx = np.random.randint(0, self.size, size=batch_size)
      return self.state[idx], self.action[idx], self.next_state[idx], self.reward[idx], self.not_done[idx], batch_size, 1

def collect_experience(policy, env, replay_buffer, initial_state, steps, noise=0.2, max_len=1000):
  with torch.no_grad():
    state = policy.normalize_state(torch.Tensor(initial_state))

    if noise is None:
      a = policy.forward(state, deterministic=False).numpy()
    else:
      a = policy.forward(state).numpy() + np.random.normal(0, noise, size=policy.action_dim)

    state_t1, r, done, _ = env.step(a)

    if done or steps > max_len:
      state_t1 = env.reset()
      done = True
      if hasattr(policy, 'init_hidden_state'):
        policy.init_hidden_state()

    replay_buffer.push(initial_state, a, state_t1.astype(np.float32), r, done)

    return state_t1, r, done

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

  replay_buff = ReplayBuffer(obs_space, act_space, args.timesteps)

  layers = [int(x) for x in args.layers.split(',')]

  raise NotImplementedError

  if args.recurrent:
    print('Recurrent ', end='')
    q1 = LSTM_Q(obs_space, act_space, env_name=args.env, layers=layers)
    q2 = LSTM_Q(obs_space, act_space, env_name=args.env, layers=layers)

    if args.algo == 'sac':
      actor = LSTM_Stochastic_Actor(obs_space, act_space, env_name=args.env, bounded=True, layers=layers)
    else:
      actor = LSTM_Actor(obs_space, act_space, env_name=args.env, layers=layers)
  else:
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

  #eval_policy(algo.actor, min_timesteps=args.prenormalize_steps, max_traj_len=args.max_traj_len, visualize=False
  train_normalizer(algo.actor, args.prenormalize_steps, noise=algo.expl_noise)

  # Fill replay buffer, update policy until n timesteps have passed
  timesteps = 0
  state = env.reset().astype(np.float32)
  while timesteps < args.timesteps:
    buffer_ready = (algo.recurrent and replay_buff.trajectories > args.batch_size) or (not algo.recurrent and replay_buff.size > args.batch_size)
    warmup = timesteps < args.start_timesteps

    state, r, done = collect_experience(algo.actor, env, replay_buff, state, episode_timesteps,
                                        max_len=args.traj_len,
                                        noise=algo.expl_noise)
    episode_reward += r
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

      actor_loss = np.mean([loss[0] for loss in losses])
      critic_loss = np.mean([loss[1] for loss in losses])
      update_steps = sum([loss[-1] for loss in losses])

      logger.add_scalar(args.env + '/actor loss', actor_loss, timesteps - args.start_timesteps)
      logger.add_scalar(args.env + '/critic loss', critic_loss, timesteps - args.start_timesteps)
      logger.add_scalar(args.env + '/update steps', update_steps, timesteps - args.start_timesteps)

      if args.algo == 'sac':
        alpha_loss = np.mean([loss[2] for loss in losses])
        logger.add_scalar(args.env + '/alpha loss', alpha_loss, timesteps - args.start_timesteps)

      completion = 1 - float(timesteps) / args.timesteps
      avg_sample_r = (time.time() - training_start)/timesteps
      secs_remaining = avg_sample_r * args.timesteps * completion
      hrs_remaining = int(secs_remaining//(60*60))
      min_remaining = int(secs_remaining - hrs_remaining*60*60)//60

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
