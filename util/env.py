import os
import time 
import torch
import numpy as np

def env_factory(path, verbose=False, **kwargs):
    from functools import partial

    """
    Returns an *uninstantiated* environment constructor.

    Since environments containing cpointers (e.g. Mujoco envs) can't be serialized, 
    this allows us to pass their constructors to Ray remote functions instead 
    (since the gym registry isn't shared across ray subprocesses we can't simply 
    pass gym.make() either)

    Note: env.unwrapped.spec is never set, if that matters for some reason.
    """
    if 'cassie' in path.lower():
      if not os.path.isdir('cassie'):
        print("You appear to be missing a './cassie' directory.")
        print("You can clone the cassie environment repository with:")
        print("git clone https://github.com/siekmanj/cassie")
        exit(1)

      from cassie.cassie import CassieEnv_v2
      path = path.lower()

      if 'random_dynamics' in path or 'dynamics_random' in path or 'randomdynamics' in path or 'dynamicsrandom' in path:
        dynamics_randomization = True
      else:
        dynamics_randomization = False
      
      if 'nodelta' in path or 'no_delta' in path:
        no_delta = True
      else:
        no_delta = False
      
      if 'stateest' in path or 'state_est' in path:
        state_est = True
      else:
        state_est = False

      if 'clock_based' in path or 'clockbased' in path:
        clock = True
      else:
        clock = False

      if 'statehistory' in path or 'state_history' in path:
        history=1
      else:
        history=0

      if 'legacy' in path:
        legacy = True
      else:
        legacy = False

      if verbose:
        print("Created cassie env with arguments:")
        print("\tdynamics randomization: {}".format(dynamics_randomization))
        print("\tstate estimation:       {}".format(state_est))
        print("\tno delta:               {}".format(no_delta))
        print("\tclock based:            {}".format(clock))
      return partial(CassieEnv_v2, 'walking', clock=clock, state_est=state_est, no_delta=no_delta, dynamics_randomization=dynamics_randomization, history=history, legacy=legacy)

    import gym
    spec = gym.envs.registry.spec(path)
    _kwargs = spec._kwargs.copy()
    _kwargs.update(kwargs)

    try:
      if callable(spec._entry_point):
        cls = spec._entry_point(**_kwargs)
      else:
        cls = gym.envs.registration.load(spec._entry_point)
    except AttributeError:
      if callable(spec.entry_point):
        cls = spec.entry_point(**_kwargs)
      else:
        cls = gym.envs.registration.load(spec.entry_point)

    return partial(cls, **_kwargs)

def eval_policy(policy, min_timesteps=1000, max_traj_len=1000, visualize=True, env=None, verbose=True):
  env_name = env
  legacy = 'legacy' if not (hasattr(policy, 'legacy') and policy.legacy == False) else ''
  with torch.no_grad():
    if env_name is None:
      env = env_factory(policy.env_name + legacy)()
    else:
      env = env_factory(env_name)()

    print("Policy is a: {}".format(policy.__class__.__name__))
    reward_sum = 0
    env.dynamics_randomization = False
    #env.dynamics_randomization = True
    total_t = 0
    episodes = 0
    while total_t < min_timesteps:
      state = env.reset()
      done = False
      timesteps = 0
      eval_reward = 0
      episodes += 1

      if hasattr(policy, 'init_hidden_state'):
        policy.init_hidden_state()

      while not done and timesteps < max_traj_len:
        if (hasattr(env, 'simrate') or hasattr(env, 'dt')) and visualize:
          start = time.time()

        env.speed = 0.5
        env.side_speed = 0.0
        #state = policy.normalize_state(state, update=False)
        action = policy.forward(torch.Tensor(state)).detach().numpy()
        state, reward, done, _ = env.step(action)
        if visualize:
          env.render()
        eval_reward += reward
        timesteps += 1
        total_t += 1

        if visualize:
          if hasattr(env, 'simrate'):
            # assume 30hz (hack)
            end = time.time()
            delaytime = max(0, 1000 / 30000 - (end-start))
            time.sleep(delaytime)

          if hasattr(env, 'dt'):
            while time.time() - start < env.dt:
              time.sleep(0.0005)
              #print("dt: {}, rt: {}".format(env.dt, end - start))
              #input()

      reward_sum += eval_reward
      if verbose:
        print("Eval reward: ", eval_reward)
    return reward_sum / episodes

def train_normalizer(policy, min_timesteps, max_traj_len=1000, noise=0.5):
  with torch.no_grad():
    env = env_factory(policy.env_name)()
    env.dynamics_randomization = False

    total_t = 0
    while total_t < min_timesteps:
      state = env.reset()
      done = False
      timesteps = 0

      if hasattr(policy, 'init_hidden_state'):
        policy.init_hidden_state()

      while not done and timesteps < max_traj_len:
        if noise is None:
          action = policy.forward(state, update_norm=True, deterministic=False).numpy()
        else:
          action = policy.forward(state, update_norm=True).numpy() + np.random.normal(0, noise, size=policy.action_dim)
        state, _, done, _ = env.step(action)
        timesteps += 1
        total_t += 1
