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
  hiddens = []
  if hasattr(policy, 'hidden'):
    hiddens += [h.data for h in policy.hidden]
  
  if hasattr(policy, 'cells'):
    hiddens += [c.data for c in policy.cells]
  
  for layer in hiddens:
    if layer.size(0) != 1:
      print("Invalid batch dim.")
      raise RuntimeError

  return torch.cat([layer.view(-1) for layer in hiddens]).numpy()
  
def collect_point(policy, max_traj_len):
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

  return get_hiddens(policy), env.get_dynamics()

@ray.remote
def collect_data(policy, max_traj_len=45, points=500):
  policy = deepcopy(policy)
  torch.set_num_threads(1)
  with torch.no_grad():
    done = True

    latent = []
    label  = []
    ts     = []

    last = time.time()
    while len(label) < points:
      x, y = collect_point(policy, max_traj_len)
      latent += [x]
      label  += [y]
    return latent, label
  
def concat(datalist):
  latents = []
  labels  = []
  for l in datalist:
    latent, label = l
    latents += latent
    labels  += label
  latents = torch.tensor(latents).float()
  labels  = torch.tensor(labels).float()
  return latents, labels

def run_experiment(args):
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
  output_dim = env.get_dynamics().shape[0]
  model      = Model(latent_dim, output_dim, layers=layers)

  opt = optim.Adam(model.parameters(), lr=args.lr, eps=1e-5)

  logger = create_logger(args)

  best_loss = None
  actor_dir = os.path.split(args.policy)[0]
  create_new = True
  if os.path.exists(os.path.join(actor_dir, 'test_latents.pt')):
    x      = torch.load(os.path.join(actor_dir, 'train_latents.pt'))
    y      = torch.load(os.path.join(actor_dir, 'train_labels.pt'))
    test_x = torch.load(os.path.join(actor_dir, 'test_latents.pt'))
    test_y = torch.load(os.path.join(actor_dir, 'test_labels.pt'))

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

    x, y = concat(ray.get([collect_data.remote(policy, points=points_per_worker) for _ in range(args.workers)]))

    split = int(0.8 * len(x))

    test_x = x[split:]
    test_y = y[split:]
    x = x[:split]
    y = y[:split]

    print("{:3.2f} to collect {} timesteps.  Training set is {}, test set is {}".format(time.time() - start, len(x)+len(test_x), len(x), len(test_x)))
    torch.save(x, os.path.join(actor_dir, 'train_latents.pt'))
    torch.save(y, os.path.join(actor_dir, 'train_labels.pt'))
    torch.save(test_x, os.path.join(actor_dir, 'test_latents.pt'))
    torch.save(test_y, os.path.join(actor_dir, 'test_labels.pt'))

  for epoch in range(args.epochs):

    random_indices = SubsetRandomSampler(range(len(x)-1))
    sampler = BatchSampler(random_indices, args.batch_size, drop_last=False)

    for j, batch_idx in enumerate(sampler):
      batch_x = x[batch_idx]#.float()
      batch_y = y[batch_idx]#.float()
      loss = 0.5 * (batch_y - model(batch_x)).pow(2).mean()

      opt.zero_grad()
      loss.backward()
      opt.step()

      print("Epoch {:3d} batch {:4d}/{:4d}: {:6.5f}".format(epoch, j, len(sampler)-1, loss.item()), end='\r')
    loss_total = 0.5 * (y - model(x)).pow(2).mean().item()
    test_loss  = 0.5 * (test_y - model(test_x)).pow(2).mean().item()
    print("Epoch {:3d} loss: train {:7.6f} test {:7.6f} {:64s}".format(epoch, loss_total, test_loss, ''))
    logger.add_scalar(logger.arg_hash + '/loss', test_loss, epoch)

    if best_loss is None or test_loss < best_loss:
      print("New best loss!")
      torch.save(model, os.path.join(actor_dir, 'extractor.pt'))
      best_loss = test_loss

