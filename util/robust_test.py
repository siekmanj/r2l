import torch, time
import numpy as np
from pathlib import Path
from util.env import env_factory

torch.manual_seed(0)
np.random.seed(0)
policies = []
#for filename in Path('rss2').rglob('actor.pt'):
for filename in Path('rss2').rglob('actor.pt'):
  policies.append(filename)

max_traj_len=1200
visualize=True
env = env_factory('cassie-dynamicsrandom-nodelta-stateest-clockbased')()
trial_damp = []
trial_mass = []
trial_ipos = []
trial_fric = []

trials = 10

for trial in range(trials):
  env.reset()
  trial_damp.append(env.sim.get_dof_damping())
  trial_mass.append(env.sim.get_body_mass())
  trial_ipos.append(env.sim.get_body_ipos())
  trial_fric.append(env.sim.get_ground_friction())

with torch.no_grad():
  for policy_name in policies:
    print("EVALUATING POLICY '{}'".format(policy_name))
    policy = torch.load(policy_name)
    avg_reward = 0
    avg_life = 0
    for trial in range(trials):
      state = env.reset()
      env.sim.set_dof_damping(trial_damp[trial])
      env.sim.set_body_mass(trial_mass[trial])
      env.sim.set_body_ipos(trial_ipos[trial])
      env.sim.set_ground_friction(trial_fric[trial])
      env.sim.set_const()
      env.cassie_state = env.sim.step_pd(env.u)
      state = env.get_full_state()


      done = False
      timesteps = 0
      eval_reward = 0

      if hasattr(policy, 'init_hidden_state'):
        policy.init_hidden_state()

      while not done and timesteps < max_traj_len:

        state = policy.normalize_state(state, update=False)
        action = policy.forward(torch.Tensor(state)).detach().numpy()
        state, reward, done, _ = env.step(action)
        eval_reward += reward
        timesteps += 1

      #print("\t\ttrial {:2d}: {:4.3f}".format(trial, eval_reward))
      avg_reward += eval_reward / trials
      avg_life += (timesteps/30) / trials
    print("\tavg reward: {:4.3f} | avg time alive {:3.2f}s".format(avg_reward, avg_life))
