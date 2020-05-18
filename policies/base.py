import torch
import torch.nn as nn
import torch.nn.functional as F

from torch import sqrt

def normc_fn(m):
  if m.__class__.__name__.find('Linear') != -1:
    m.weight.data.normal_(0, 1)
    m.weight.data *= 1 / torch.sqrt(m.weight.data.pow(2).sum(1, keepdim=True))
    if m.bias is not None:
      m.bias.data.fill_(0)

def create_layers(layer_fn, input_dim, layer_sizes):
  ret = nn.ModuleList()
  ret += [layer_fn(input_dim, layer_sizes[0])]
  for i in range(len(layer_sizes)-1):
    ret += [layer_fn(layer_sizes[i], layer_sizes[i+1])]
  return ret

#def ff_forward(layers, nonlinearity, x):
#  for idx, layer in enumerate(layers):
#    x = nonlinearity(layer(x))
#  return x
#
#def lstm_forward(layers, x):

class Net(nn.Module):
  """
  The base class for neural networks. Includes optional
  functions for state normalization.
  """
  def __init__(self):
    super(Net, self).__init__()
    #nn.Module.__init__(self)
    self.is_recurrent = False

    self.welford_state_mean = torch.zeros(1)
    self.welford_state_mean_diff = torch.ones(1)
    self.welford_state_n = 1

    self.env_name = None

  def normalize_state(self, state, update=True):
    state = torch.Tensor(state)

    if self.welford_state_n == 1:
      self.welford_state_mean = torch.zeros(state.size(-1))
      self.welford_state_mean_diff = torch.ones(state.size(-1))

    if update:
      if len(state.size()) == 1: # if we get a single state vector 
        state_old = self.welford_state_mean
        self.welford_state_mean += (state - state_old) / self.welford_state_n
        self.welford_state_mean_diff += (state - state_old) * (state - state_old)
        self.welford_state_n += 1
      else:
        raise RuntimeError # this really should not happen

    return (state - self.welford_state_mean) / sqrt(self.welford_state_mean_diff / self.welford_state_n)

  def copy_normalizer_stats(self, net):
    self.welford_state_mean      = net.self_state_mean
    self.welford_state_mean_diff = net.welford_state_mean_diff
    self.welford_state_n         = net.welford_state_n
  
  def initialize_parameters(self):
    self.apply(normc_fn)
    if hasattr(self, 'network_out'):
      self.network_out.weight.data.mul_(0.01)

class FF_Base(Net):
  """
  The base class for feedforward networks.
  """
  def __init__(self, in_dim, layers, nonlinearity):
    super(FF_Base, self).__init__()
    self.layers       = create_layers(nn.Linear, in_dim, layers)
    self.nonlinearity = nonlinearity

  def _base_forward(self, x):
    for idx, layer in enumerate(self.layers):
      x = self.nonlinearity(layer(x))
    return x

class LSTM_Base(Net):
  """
  The base class for LSTM networks.
  """
  def __init__(self, in_dim, layers):
    super(LSTM_Base, self).__init__()
    self.layers = create_layers(nn.LSTMCell, in_dim, layers)

  def get_hidden_state(self):
    return self.hidden, self.cells

  def init_hidden_state(self, batch_size=1):
    self.hidden = [torch.zeros(batch_size, l.hidden_size) for l in self.layers]
    self.cells  = [torch.zeros(batch_size, l.hidden_size) for l in self.layers]

  def _base_forward(self, x):
    dims = len(x.size())

    if dims == 3: # if we get a batch of trajectories
      self.init_hidden_state(batch_size=x.size(1))
      y = []
      for t, x_t in enumerate(x):
        for idx, layer in enumerate(self.layers):
          c, h = self.cells[idx], self.hidden[idx]
          self.hidden[idx], self.cells[idx] = layer(x_t, (h, c))
          x_t = self.hidden[idx]
        y.append(x_t)
      x = torch.stack([x_t for x_t in y])

    else:
      if dims == 1: # if we get a single timestep (if not, assume we got a batch of single timesteps)
        x = x.view(1, -1)

      for idx, layer in enumerate(self.layers):
        h, c = self.hidden[idx], self.cells[idx]
        self.hidden[idx], self.cells[idx] = layer(x, (h, c))
        x = self.hidden[idx]

      if dims == 1:
        x = x.view(-1)
    return x

class GRU_Base(Net):
  """
  The base class for GRU networks.
  """
  def __init__(self, in_dim, layers):
    super(GRU_Base, self).__init__()
    #Net.__init__(self)
    self.layers = create_layers(nn.GRUCell, in_dim, layers)

  def get_hidden_state(self):
    return self.hidden

  def init_hidden_state(self, batch_size=1):
    self.hidden = [torch.zeros(batch_size, l.hidden_size) for l in self.layers]

  def _base_forward(self, x):
    dims = len(x.size())

    if dims == 3: # if we get a batch of trajectories
      self.init_hidden_state(batch_size=x.size(1))
      y = []
      for t, x_t in enumerate(x):
        for idx, layer in enumerate(self.layers):
          h = self.hidden[idx]
          self.hidden[idx] = layer(x_t, h)
          x_t = self.hidden[idx]
        y.append(x_t)
      x = torch.stack([x_t for x_t in y])

    else:
      if dims == 1: # if we get a single timestep (if not, assume we got a batch of single timesteps)
        x = x.view(1, -1)

      for idx, layer in enumerate(self.layers):
        h = self.hidden[idx]
        self.hidden[idx] = layer(x, h)
        x = self.hidden[idx]

      if dims == 1:
        x = x.view(-1)
    return x
