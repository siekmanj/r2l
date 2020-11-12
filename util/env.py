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
    if 'digit' in path.lower():
      if not os.path.isdir('digit'):
        print("You appear to be missing a './digit' directory.")
        print("You can clone the cassie environment repository with:")
        print("git clone https://github.com/siekmanj/digit")
        exit(1)

      from digit.digit import DigitEnv
      path = path.lower()

      if 'random_dynamics' in path or 'dynamics_random' in path or 'randomdynamics' in path or 'dynamicsrandom' in path:
        dynamics_randomization = True
      else:
        dynamics_randomization = False

      if 'impedance' in path:
        impedance = True
      else:
        impedance = False

      if 'standing' in path:
        standing = True
      else:
        standing = False

      if 'footpos' in path:
        footpos = True
      else:
        footpos = False

      if 'perception' in path:
        perception = True
      else:
        perception = False
      
      if 'stairs' in path:
        stairs = True
      else:
        stairs = False

      return partial(DigitEnv, dynamics_randomization=dynamics_randomization, impedance=impedance, standing=standing, footpos=footpos, perception=perception, stairs=stairs)

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
      no_delta = True
      
      if 'stateest' in path or 'state_est' in path:
        state_est = True
      else:
        state_est = False
      state_est = True

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
      legacy = False

      if 'impedance' in path:
        impedance = True
      else:
        impedance = False

      if 'height' in path:
        height = True
      else:
        height = False

      if verbose:
        print("Created cassie env with arguments:")
        print("\tdynamics randomization: {}".format(dynamics_randomization))
        print("\tstate estimation:       {}".format(state_est))
        print("\tno delta:               {}".format(no_delta))
        print("\tclock based:            {}".format(clock))
        print("\timpedance control:      {}".format(impedance))
        print("\theight control:         {}".format(height))
      return partial(CassieEnv_v2, 'walking', clock=clock, state_est=state_est, no_delta=no_delta, dynamics_randomization=dynamics_randomization, history=history, legacy=legacy, impedance=impedance, height=height)

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
  with torch.no_grad():
    if env_name is None:
      env = env_factory(policy.env_name)()
    else:
      env = env_factory(env_name)()

    if verbose:
      print("Policy is a: {}".format(policy.__class__.__name__))
    reward_sum = 0
    env.dynamics_randomization = False
    total_t = 0
    episodes = 0

    obs_states = {}
    mem_states = {}

    while total_t < min_timesteps:
      state = env.reset()
      done = False
      timesteps = 0
      eval_reward = 0
      episodes += 1

      if hasattr(policy, 'init_hidden_state'):
        policy.init_hidden_state()

      #speeds = [(0, 0), (0.5, 0), (2.0, 0)]
      speeds = list(zip(np.array(range(0, 350)) / 100, np.zeros(350)))
      pelvis_vel = 0
      while not done and timesteps < max_traj_len:
        if (hasattr(env, 'simrate') or hasattr(env, 'dt')) and visualize:
          start = time.time()

        action = policy.forward(torch.Tensor(state)).detach().numpy()
        state, reward, done, _ = env.step(action)
        if visualize:
          env.render()
        eval_reward += reward
        timesteps += 1
        total_t += 1

        if hasattr(policy, 'get_quantized_states'):
          obs, mem = policy.get_quantized_states()
          obs_states[obs] = True
          mem_states[mem] = True
          print(policy.get_quantized_states(), len(obs_states), len(mem_states))

        if visualize:
          if hasattr(env, 'simrate'):
            # assume 30hz (hack)
            end = time.time()
            delaytime = max(0, 1000 / 30000 - (end-start))
            time.sleep(delaytime)

          if hasattr(env, 'dt'):
            while time.time() - start < env.dt:
              time.sleep(0.0005)

      reward_sum += eval_reward
      if verbose:
        print("Eval reward: ", eval_reward)
    return reward_sum / episodes

def interactive_eval(policy_name):
    from copy import deepcopy
    import termios, sys
    import tty
    import select
    with torch.no_grad():
        policy = torch.load(policy_name)
        m_policy = torch.load(policy_name)
        #args, run_args = self.args, self.run_args
        #run_args = run_args

        print("env name: ", policy.env_name)
        #if run_args.env is None:
        #    env_name = run_args.env
        #else:
        #    env_name = run_args.env

        env = env_factory(policy.env_name)()
        print(env)
        env.dynamics_randomization = False
        env.evaluation_mode = True

        #if self.run_args.pca:
        #    from util.pca import PCA_Plot
        #    pca_plot = PCA_Plot(policy, env)

        #if self.run_args.pds:
        #    from util.pds import PD_Plot
        #    pd_plot = PD_Plot(policy, env)
        #    print("DOING PDS??")

        if hasattr(policy, 'init_hidden_state'):
            policy.init_hidden_state()
            m_policy.init_hidden_state()

        old_settings = termios.tcgetattr(sys.stdin)

        env.render()
        render_state = True
        slowmo = True

        try:
            tty.setcbreak(sys.stdin.fileno())

            state = env.reset()
            env.speed = 0
            env.side_speed = 0
            env.phase_add = 50
            env.period_shift = [0, 0.5]
            #env.ratio = [0.4, 0.6]
            env.eval_mode = True
            done = False
            timesteps = 0
            eval_reward = 0
            mirror = False

            def isData():
                return select.select([sys.stdin], [], [], 0) == ([sys.stdin], [], [])

            while render_state:
            
                if isData():
                    c = sys.stdin.read(1)
                    if c == 'w':
                        env.speed = np.clip(env.speed + 0.1, env.min_speed, env.max_speed)
                    if c == 's':
                        env.speed = np.clip(env.speed - 0.1, env.min_speed, env.max_speed)
                    if c == 'q':
                        env.orient_add -= 0.005 * np.pi
                    if c == 'e':
                        env.orient_add += 0.005 * np.pi
                    if c == 'a':
                        env.side_speed = np.clip(env.side_speed + 0.05, env.min_side_speed, env.max_side_speed)
                    if c == 'd':
                        env.side_speed = np.clip(env.side_speed - 0.05, env.min_side_speed, env.max_side_speed)
                    if c == 'r':
                        state = env.reset()
                        if hasattr(policy, 'init_hidden_state'):
                            policy.init_hidden_state()
                            m_policy.init_hidden_state()
                        print("Resetting environment via env.reset()")
                        env.speed = 0
                        env.side_speed = 0
                        env.phase_add = env.simrate
                        env.period_shift = [0, 0.5]
                    if c == 't':
                        env.phase_add = np.clip(env.phase_add + 1, int(env.simrate * env.min_step_freq), int(env.simrate * env.max_step_freq))
                    if c == 'g':
                        env.phase_add = np.clip(env.phase_add - 1, int(env.simrate * env.min_step_freq), int(env.simrate * env.max_step_freq))
                    if c == 'm':
                        mirror = not mirror
                    if c == 'o':
                        # increase ratio of phase 1
                        env.ratio[0] = np.clip(env.ratio[0] + 0.01, 0, env.max_swing_ratio)
                        env.ratio[1] = 1 - env.ratio[0]
                    if c == 'l':
                        env.ratio[0] = np.clip(env.ratio[0] - 0.01, 0, env.max_swing_ratio)
                        env.ratio[1] = 1 - env.ratio[0]
                    if c == 'p':
                        # switch ratio shift from mirrored to matching and back
                        env.period_shift[0] = np.clip(env.period_shift[0] + 0.05, 0, 0.5)
                    if c == ';':
                        env.period_shift[0] = np.clip(env.period_shift[0] - 0.05, 0, 0.5)
                    if c == 'x':
                        if hasattr(policy, 'init_hidden_state'):
                            policy.init_hidden_state()
                            m_policy.init_hidden_state()

                start = time.time()

                if (not env.vis.ispaused()):

                    action   = policy(torch.Tensor(state)).numpy()
                    if hasattr(env, 'mirror_state') and hasattr(env, 'mirror_action'):
                        m_state = env.mirror_state(state)
                        m_m_state = env.mirror_state(m_state)
                        for i, (m_x, x) in enumerate(zip(m_m_state, state)):
                            if m_x - x != 0:
                                print("STATE DISCREPANCY:\n\n")
                                print(i, m_x - x)

                        action_ = m_policy(torch.Tensor(env.mirror_state(state))).numpy()
                        m_action = env.mirror_action(action_)
                        m_m_action = env.mirror_action(m_action)
                        for i, (m_x, x) in enumerate(zip(m_m_action, action_)):
                            if m_x - x != 0:
                                print("ACTION DISCREPANCY:\n\n")
                                print(i, m_x - x)


                    if mirror:
                        pass
                        action = m_action

                    state, reward, done, _ = env.step(action)

                    eval_reward += reward
                    timesteps += 1
                    qvel = env.sim.qvel()
                    actual_speed = np.linalg.norm(qvel[0:2])

                    #if run_args.pca:
                    #    pca_plot.update(policy)
                    #if run_args.pds:
                    #    pd_plot.update(action, env.l_foot_frc, env.r_foot_frc)
                    print("Mirror: {} | Des. Spd. {:5.2f} | Speed {:5.1f} | Sidespeed {:4.2f} | Heading {:5.2f} | Freq. {:3d} | Coeff {},{} | Ratio {:3.2f},{:3.2f} | RShift {:3.2f},{:3.2f} | {:20s}".format(mirror, env.speed, actual_speed, env.side_speed, env.orient_add, int(env.phase_add), *env.coeff, *env.ratio, *env.period_shift, ''), end='\r')

                render_state = env.render()
                if hasattr(env, 'simrate'):
                    # assume 40hz
                    end = time.time()
                    delaytime = max(0, 1000 / 40000 - (end-start))
                    if slowmo:
                        while(time.time() - end < delaytime*10):
                            env.render()
                            time.sleep(delaytime)
                    else:
                        time.sleep(delaytime)


            print("Eval reward: ", eval_reward)

        finally:
            termios.tcsetattr(sys.stdin, termios.TCSADRAIN, old_settings)

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
