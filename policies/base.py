import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

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

def create_layers(layer_fn, input_dim, layer_sizes):
  """
  This function creates a pytorch modulelist and appends
  pytorch modules like nn.Linear or nn.LSTMCell passed
  in through the layer_fn argument, using the sizes
  specified in the layer_sizes list.
  """
  ret = nn.ModuleList()
  ret += [layer_fn(input_dim, layer_sizes[0])]
  for i in range(len(layer_sizes)-1):
    ret += [layer_fn(layer_sizes[i], layer_sizes[i+1])]
  return ret

class Net(nn.Module):
  """
  The base class which all policy networks inherit from. It includes methods
  for normalizing states.
  """
  def __init__(self):
    super(Net, self).__init__()
    #nn.Module.__init__(self)
    self.is_recurrent = False

    self.welford_state_mean = torch.zeros(1)
    self.welford_state_mean_diff = torch.ones(1)
    self.welford_state_n = 1

    self.env_name = None

    self.calculate_norm = False

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
      if len(state.size()) == 1: # if we get a single state vector 
        state_old = self.welford_state_mean
        self.welford_state_mean += (state - state_old) / self.welford_state_n
        self.welford_state_mean_diff += (state - state_old) * (state - state_old)
        self.welford_state_n += 1
      else:
        raise RuntimeError # this really should not happen
    return (state - self.welford_state_mean) / sqrt(self.welford_state_mean_diff / self.welford_state_n)

  def copy_normalizer_stats(self, net):
    self.welford_state_mean      = net.welford_state_mean
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

  def get_latent_norm(self):
    raise NotImplementedError

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

  def get_latent_norm(self):
    return self.latent_norm

  def init_hidden_state(self, batch_size=1):
    self.hidden = [torch.zeros(batch_size, l.hidden_size) for l in self.layers]
    self.cells  = [torch.zeros(batch_size, l.hidden_size) for l in self.layers]

  def _base_forward(self, x):
    dims = len(x.size())

    if dims == 3: # if we get a batch of trajectories
      self.init_hidden_state(batch_size=x.size(1))

      if self.calculate_norm:
        self.latent_norm = 0

      y = []
      for t, x_t in enumerate(x):
        for idx, layer in enumerate(self.layers):
          c, h = self.cells[idx], self.hidden[idx]
          self.hidden[idx], self.cells[idx] = layer(x_t, (h, c))
          x_t = self.hidden[idx]

          if self.calculate_norm:
            self.latent_norm += (torch.mean(torch.abs(x_t)) + torch.mean(torch.abs(self.cells[idx])))

        y.append(x_t)
      x = torch.stack([x_t for x_t in y])

      if self.calculate_norm:
        self.latent_norm /= len(x) * len(self.layers)

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
    self.layers = create_layers(nn.GRUCell, in_dim, layers)

  def get_latent_norm(self):
    return self.latent_norm

  def init_hidden_state(self, batch_size=1):
    self.hidden = [torch.zeros(batch_size, l.hidden_size) for l in self.layers]

  def _base_forward(self, x):
    dims = len(x.size())

    if dims == 3: # if we get a batch of trajectories
      self.init_hidden_state(batch_size=x.size(1))

      if self.calculate_norm:
        self.latent_norm = 0

      y = []
      for t, x_t in enumerate(x):
        for idx, layer in enumerate(self.layers):
          h = self.hidden[idx]
          self.hidden[idx] = layer(x_t, h)
          x_t = self.hidden[idx]

          if self.calculate_norm:
            self.latent_norm += torch.mean(torch.abs(x_t))

        y.append(x_t)
      x = torch.stack([x_t for x_t in y])

      if self.calculate_norm:
        self.latent_norm /= len(x) * len(self.layers)

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

from policies.autoencoder import ternary_tanh
class QBN_GRU_Base(Net):
  """
  The base class for QBN-GRU networks.
  """
  def __init__(self, in_dim):
    super(QBN_GRU_Base, self).__init__()

    obs_bottleneck = 8
    m_bottleneck   = 8
    m_size         = 16
    latent         = 16
    obs_intermediate = int(abs(in_dim - obs_bottleneck)/2)

    self.obs_qbn       = create_layers(nn.Linear, in_dim, (obs_intermediate, obs_intermediate, obs_bottleneck))
    self.memory        = nn.GRUCell(obs_bottleneck, m_size)
    self.memory2qbn    = create_layers(nn.Linear, m_size, [latent, latent, m_bottleneck])
    self.qbn2memory    = create_layers(nn.Linear, m_bottleneck, [latent, latent, m_size])
    self.action_latent = create_layers(nn.Linear, m_size, [latent])
    
    self.memory_size        = m_size
    self.latent_output_size = latent

  def get_latent_norm(self):
    raise NotImplementedError

  def init_hidden_state(self, batch_size=1):
    self.hidden = torch.zeros(batch_size, self.memory_size)

  def get_quantized_states(self):
    obs_encoding = ''.join(map(str, np.round(self.obs_states.numpy() + 1).astype(int)))
    mem_encoding = ''.join(map(str, np.round(self.memory_states[0].numpy() + 1).astype(int)))
    #act_encoding = ''.join(map(str, np.round(self.action_states.numpy() + 1).astype(int)))
    return obs_encoding, mem_encoding#, act_encoding

  def _base_forward(self, x):
    for layer in self.obs_qbn[:-1]:
      x = torch.tanh(layer(x))
    x = ternary_tanh(self.obs_qbn[-1](x))
    self.obs_states = x

    dims = len(x.size())
    if dims == 3: # if we get a batch of trajectories
      self.init_hidden_state(batch_size=x.size(1))
      y = []
      self.memory_states = []
      for t, x_t in enumerate(x):
        h_t = self.hidden
        h_t = self.memory(x_t, h_t)
        for layer in self.memory2qbn[:-1]:
          h_t = torch.tanh(layer(h_t))
        h_t = ternary_tanh(self.memory2qbn[-1](h_t))

        self.memory_states.append(h_t)

        for layer in self.qbn2memory:
          h_t = torch.tanh(layer(h_t))

        self.hidden = h_t
        y.append(h_t)
      x = torch.stack([h_t for h_t in y])
      self.memory_states = torch.stack([h_q for h_q in self.memory_states])

    else:
      if dims == 1: # if we get a single timestep (if not, assume we got a batch of single timesteps)
        x = x.view(1, -1)

      h_t = self.hidden
      h_t = self.memory(x, h_t)

      for layer in self.memory2qbn[:-1]:
        h_t = torch.tanh(layer(h_t))
      h_t = ternary_tanh(self.memory2qbn[-1](h_t))

      self.memory_states = h_t

      for layer in self.qbn2memory:
        h_t = torch.tanh(layer(h_t))

      self.hidden = h_t

      if dims == 1:
        x = h_t.view(-1)

    for layer in self.action_latent:
      x = torch.tanh(layer(x))
    #x = ternary_tanh(self.action_latent[-1](x))

    return x
