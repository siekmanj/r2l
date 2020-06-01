import ray
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import locale, os, time

from torch.utils.data.sampler import BatchSampler, SubsetRandomSampler
from copy import deepcopy
from policies.fit import Model
from util.env import env_factory

def get_hiddens(policy):
  """
  A helper function for flattening the memory of a recurrent
  policy into a vector.
  """
  hiddens = []
  if hasattr(policy, 'hidden'):
    hiddens += [h.data for h in policy.hidden]
  
  if hasattr(policy, 'cells'):
    hiddens += [c.data for c in policy.cells]

  if hasattr(policy, 'latent'):
    hiddens += [l for l in policy.latent]
  
  return torch.cat([layer.view(-1) for layer in hiddens]).numpy()
  
def collect_point(policy, max_traj_len):
  """
  A helper function which collects a single memory-dynamics parameter pair
  from a trajectory.
  """
  legacy = 'legacy' if not (hasattr(policy, 'legacy') and policy.legacy == False) else ''
  env = env_factory(policy.env_name + legacy)()
  env.dynamics_randomization = True

  chosen_timestep = np.random.randint(15, max_traj_len)
  timesteps = 0
  done = False

  if hasattr(policy, 'init_hidden_state'):
    policy.init_hidden_state()

  state = env.reset()
  while not done and timesteps < chosen_timestep:

    state = policy.normalize_state(state, update=True)
    action = policy(state).numpy()
    state, _, done, _ = env.step(action)
    timesteps += 1

  return get_hiddens(policy), env.get_friction(), env.get_damping(), env.get_mass(), env.get_quat()

@ray.remote
def collect_data(policy, max_traj_len=45, points=500):
  """
  A ray remote function which collects a series of memory-dynamics pairs
  and returns a dataset.
  """
  policy = deepcopy(policy)
  torch.set_num_threads(1)
  with torch.no_grad():
    done = True

    frics  = []
    damps  = []
    masses = []
    quats  = []

    latent = []
    ts     = []

    last = time.time()
    while len(latent) < points:
      x, f, d, m, q = collect_point(policy, max_traj_len)
      frics   += [f]
      damps   += [d]
      masses  += [m]
      quats   += [q]
      latent  += [x]
    return frics, damps, masses, quats, latent
  
def concat(datalist):
  """
  Concatenates several datasets into one larger
  dataset.
  """
  frics    = []
  damps    = []
  masses   = []
  quats    = []
  latents  = []
  for l in datalist:
    fric, damp, mass, quat, latent = l
    frics  += fric
    damps  += damp
    masses += mass
    quats  += quat

    latents += latent
  frics   = torch.tensor(frics).float()
  damps   = torch.tensor(damps).float()
  masses  = torch.tensor(masses).float()
  quats   = torch.tensor(quats).float()
  latents = torch.tensor(latents).float()
  return frics, damps, masses, quats, latents

def run_experiment(args):
  """
  The entry point for the dynamics extraction algorithm.
  """
  from util.log import create_logger

  locale.setlocale(locale.LC_ALL, '')

  policy = torch.load(args.policy)

  legacy = 'legacy' if not (hasattr(policy, 'legacy') and policy.legacy == False) else ''
  env_fn = env_factory(policy.env_name + legacy)

  layers = [int(x) for x in args.layers.split(',')]

  env = env_fn()
  policy.init_hidden_state()
  policy(torch.tensor(env.reset()).float())
  latent_dim = get_hiddens(policy).shape[0]

  models = []
  opts   = []
  for fn in [env.get_friction, env.get_damping, env.get_mass, env.get_quat]:
    output_dim = fn().shape[0]
    model = Model(latent_dim, output_dim, layers=layers)
    models += [model]
    opts   += [optim.Adam(model.parameters(), lr=args.lr, eps=1e-5)]

  logger = create_logger(args)

  best_loss = None
  actor_dir = os.path.split(args.policy)[0]
  create_new = True
  #if os.path.exists(os.path.join(actor_dir, 'test_latents.pt')):
  if False:
    x      = torch.load(os.path.join(logger.dir, 'train_latents.pt'))
    test_x = torch.load(os.path.join(logger.dir, 'test_latents.pt'))

    train_frics  = torch.load(os.path.join(logger.dir, 'train_frics.pt'))
    test_frics   = torch.load(os.path.join(logger.dir, 'test_frics.pt'))

    train_damps  = torch.load(os.path.join(logger.dir, 'train_damps.pt'))
    test_damps   = torch.load(os.path.join(logger.dir, 'test_damps.pt'))

    train_masses = torch.load(os.path.join(logger.dir, 'train_masses.pt'))
    test_masses  = torch.load(os.path.join(logger.dir, 'test_masses.pt'))

    train_quats  = torch.load(os.path.join(logger.dir, 'train_quats.pt'))
    test_quats   = torch.load(os.path.join(logger.dir, 'test_quats.pt'))

    if args.points > len(x) + len(y):
      create_new = True
    else:
      create_new = False
  
  if create_new:
    if not ray.is_initialized():
      if args.redis is not None:
        ray.init(redis_address=args.redis)
      else:
        ray.init(num_cpus=args.workers)

    print("Collecting {:4d} timesteps of data.".format(args.points))
    points_per_worker = max(args.points // args.workers, 1)
    start = time.time()

    frics, damps, masses, quats, x = concat(ray.get([collect_data.remote(policy, points=points_per_worker) for _ in range(args.workers)]))

    split = int(0.8 * len(x))

    test_x = x[split:]
    x = x[:split]

    test_frics = frics[split:]
    frics = frics[:split]

    test_damps = damps[split:]
    damps = damps[:split]

    test_masses = masses[split:]
    masses = masses[:split]

    test_quats = quats[split:]
    quats = quats[:split]

    print("{:3.2f} to collect {} timesteps.  Training set is {}, test set is {}".format(time.time() - start, len(x)+len(test_x), len(x), len(test_x)))
    torch.save(x, os.path.join(logger.dir, 'train_latents.pt'))
    torch.save(test_x, os.path.join(logger.dir, 'test_latents.pt'))

    torch.save(frics, os.path.join(logger.dir, 'train_frics.pt'))
    torch.save(test_frics, os.path.join(logger.dir, 'test_frics.pt'))

    torch.save(damps, os.path.join(logger.dir, 'train_damps.pt'))
    torch.save(test_damps, os.path.join(logger.dir, 'test_damps.pt'))

    torch.save(masses, os.path.join(logger.dir, 'train_masses.pt'))
    torch.save(test_masses, os.path.join(logger.dir, 'test_masses.pt'))

    torch.save(quats, os.path.join(logger.dir, 'train_quats.pt'))
    torch.save(test_quats, os.path.join(logger.dir, 'test_quats.pt'))

  for epoch in range(args.epochs):

    random_indices = SubsetRandomSampler(range(len(x)-1))
    sampler = BatchSampler(random_indices, args.batch_size, drop_last=False)

    for j, batch_idx in enumerate(sampler):
      batch_x = x[batch_idx]#.float()
      #batch_fric = frics[batch_idx]
      #batch_damp = damps[batch_idx]
      #batch_mass = masses[batch_idx]
      #batch_quat = quats[batch_idx]
      batch = [frics[batch_idx], damps[batch_idx], masses[batch_idx], quats[batch_idx]]

      losses = []
      for model, batch_y, opt in zip(models, batch, opts):
        loss = 0.5 * (batch_y - model(batch_x)).pow(2).mean()

        opt.zero_grad()
        loss.backward()
        opt.step()
        
        losses.append(loss.item())

      print("Epoch {:3d} batch {:4d}/{:4d}      ".format(epoch, j, len(sampler)-1), end='\r')

    train_y = [frics, damps, masses, quats]
    test_y  = [test_frics, test_damps, test_masses, test_quats]
    order   = ['friction', 'damping', 'mass', 'slope']

    with torch.no_grad():
      print("\nEpoch {:3d} losses:".format(epoch))
      for model, y_tr, y_te, name in zip(models, train_y, test_y, order):
        loss_total = 0.5 * (y_tr - model(x)).pow(2).mean().item()

        preds = model(test_x)
        test_loss  = 0.5 * (y_te - preds).pow(2).mean().item()
        pce = torch.mean(torch.abs((y_te - preds)/y_te))
        err = torch.mean(torch.abs((y_te - preds)))
        
        logger.add_scalar(logger.arg_hash + '/' + name + '_loss', test_loss, epoch)
        logger.add_scalar(logger.arg_hash + '/' + name + '_percenterr', pce, epoch)
        logger.add_scalar(logger.arg_hash + '/' + name + '_abserr', err, epoch)
        torch.save(model, os.path.join(logger.dir, name + '_extractor.pt'))
        print("\t{:16s}: train {:7.6f} test {:7.6f}".format(name, loss_total, test_loss))

