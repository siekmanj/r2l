import torch
import torch.nn as nn
import torch.nn.functional as F

from torch import sqrt

def normc_fn(m):
  """
  This function multiplies the weights of a pytorch linear layer by a small
  number so that outputs early in training are close to zero, which means 
  that gradients are larger in magnitude. This means a richer gradient signal
  is propagated back and speeds up learning (probably).
  """
  if m.__class__.__name__.find('Linear') != -1:
    m.weight.data.normal_(0, 1)
    m.weight.data *= 1 / torch.sqrt(m.weight.data.pow(2).sum(1, keepdim=True))
    if m.bias is not None:
      m.bias.data.fill_(0)

# The base class for an actor. Includes functions for normalizing state (optional)
class Net(nn.Module):
  """
  The base class which all policy networks inherit from. It includes methods
  for normalizing states, initializing network parameters, and a prototype forward function.
  """
  def __init__(self):
    super(Net, self).__init__()
    self.is_recurrent = False

    self.welford_state_mean = torch.zeros(1)
    self.welford_state_mean_diff = torch.ones(1)
    self.welford_state_n = 1

    self.env_name = None

  def forward(self):
    raise NotImplementedError

  def normalize_state(self, state, update=True):
    """
    Use Welford's algorithm to normalize a state, and optionally update the statistics
    for normalizing states using the new state, online.
    """
    state = torch.Tensor(state)

    if self.welford_state_n == 1:
      self.welford_state_mean = torch.zeros(state.size(-1))
      self.welford_state_mean_diff = torch.ones(state.size(-1))

    if update:
      if len(state.size()) == 1: # If we get a single state vector 
        state_old = self.welford_state_mean
        self.welford_state_mean += (state - state_old) / self.welford_state_n
        self.welford_state_mean_diff += (state - state_old) * (state - state_old)
        self.welford_state_n += 1
      else:
        raise RuntimeError
    return (state - self.welford_state_mean) / sqrt(self.welford_state_mean_diff / self.welford_state_n)

  def copy_normalizer_stats(self, net):
    self.welford_state_mean = net.self_state_mean
    self.welford_state_mean_diff = net.welford_state_mean_diff
    self.welford_state_n = net.welford_state_n
  
  def initialize_parameters(self):
    self.apply(normc_fn)
    if hasattr(self, 'network_out'):
      self.network_out.weight.data.mul_(0.01)

