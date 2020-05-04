import torch
import torch.nn as nn
import torch.nn.functional as F

from policies.base import Net

class Actor(Net):
  def __init__(self):
    super(Actor, self).__init__()

  def forward(self):
    raise NotImplementedError

class Linear_Actor(Actor):
  def __init__(self, state_dim, action_dim, layers=(64)):
    super(Linear_Actor, self).__init__()

    self.actor_layers = nn.ModuleList()
    self.actor_layers += [nn.Linear(state_dim, layers[0])]
    for i in range(len(layers)-1):
        self.actor_layers += [nn.Linear(layers[i], layers[i+1])]
    self.network_out = nn.Linear(layers[-1], action_dim)

    self.action_dim = action_dim

    for p in self.parameters():
      p.data = torch.zeros(p.shape)

  def forward(self, state):
    x = state
    for layer in self.actor_layers:
      x = layer(x)
    action = self.network_out(x)
    return action

class FF_Actor(Actor):
  def __init__(self, state_dim, action_dim, layers=(256, 256), env_name=None, nonlinearity=torch.tanh, normc_init=False, max_action=1):
    super(FF_Actor, self).__init__()

    self.actor_layers = nn.ModuleList()
    self.actor_layers += [nn.Linear(state_dim, layers[0])]
    for i in range(len(layers)-1):
        self.actor_layers += [nn.Linear(layers[i], layers[i+1])]
    self.network_out = nn.Linear(layers[-1], action_dim)

    self.action_dim = action_dim
    self.env_name = env_name
    self.nonlinearity = nonlinearity
    self.max_action = max_action

    if normc_init:
      self.initialize_parameters()

  def forward(self, state):
    x = state
    for idx, layer in enumerate(self.actor_layers):
      x = self.nonlinearity(layer(x))

    action = torch.tanh(self.network_out(x))
    return action

class FF_Stochastic_Actor(Actor):
  def __init__(self, state_dim, action_dim, layers=(256, 256), env_name=None, nonlinearity=torch.tanh, normc_init=True, bounded=False, fixed_std=None):
    super(FF_Stochastic_Actor, self).__init__()

    self.actor_layers = nn.ModuleList()
    self.actor_layers += [nn.Linear(state_dim, layers[0])]
    for i in range(len(layers)-1):
        self.actor_layers += [nn.Linear(layers[i], layers[i+1])]
    self.means = nn.Linear(layers[-1], action_dim)

    if fixed_std is None:
      self.log_stds = nn.Linear(layers[-1], action_dim)
      self.learn_std = True
    else:
      self.fixed_std = fixed_std
      self.learn_std = False

    self.action_dim = action_dim
    self.env_name = env_name
    self.nonlinearity = nonlinearity
    self.bounded = bounded

    if normc_init:
      self.initialize_parameters()

  def _get_dist_params(self, state):
    x = state
    self.latent = []
    for idx, layer in enumerate(self.actor_layers):
      x = self.nonlinearity(layer(x))
      self.latent.append(x)

    mu = self.means(x)

    if self.learn_std:
      sd = torch.clamp(self.log_stds(x), -2, 1).exp()
    else:
      sd = self.fixed_std

    return mu, sd

  def forward(self, state, deterministic=True, return_log_probs=False):
    mu, sd = self._get_dist_params(state)

    if not deterministic or return_log_probs:
      dist = torch.distributions.Normal(mu, sd)
      sample = dist.rsample()

    if self.bounded:
      action = torch.tanh(mu) if deterministic else torch.tanh(sample)
    else:
      action = mu if deterministic else sample

    if return_log_probs:
      log_prob = dist.log_prob(sample)
      if self.bounded:
        log_prob -= torch.log((1 - torch.tanh(sample).pow(2)) + 1e-6)

      return action, log_prob.sum(1, keepdim=True)
    else:
      return action

  def pdf(self, state):
    mu, sd = self._get_dist_params(state)
    return torch.distributions.Normal(mu, sd)

class LSTM_Actor(Actor):
  def __init__(self, input_dim, action_dim, layers=(128, 128), env_name=None, nonlinearity=torch.tanh, normc_init=False, bounded=False):
    super(LSTM_Actor, self).__init__()

    self.actor_layers = nn.ModuleList()
    self.actor_layers += [nn.LSTMCell(input_dim, layers[0])]
    for i in range(len(layers)-1):
        self.actor_layers += [nn.LSTMCell(layers[i], layers[i+1])]
    self.network_out = nn.Linear(layers[-1], action_dim)

    self.action_dim = action_dim
    self.init_hidden_state()
    self.env_name = env_name
    self.nonlinearity = nonlinearity
    self.bounded = bounded
    
    self.is_recurrent = True

    if normc_init:
      self.initialize_parameters()

  def get_hidden_state(self):
    return self.hidden, self.cells

  def set_hidden_state(self, data):
    if len(data) != 2:
      print("Got invalid hidden state data.")
      exit(1)

    self.hidden, self.cells = data
    
  def init_hidden_state(self, batch_size=1):
    self.hidden = [torch.zeros(batch_size, l.hidden_size) for l in self.actor_layers]
    self.cells  = [torch.zeros(batch_size, l.hidden_size) for l in self.actor_layers]

  def forward(self, x):
    dims = len(x.size())

    if dims == 3: # if we get a batch of trajectories
      self.init_hidden_state(batch_size=x.size(1))
      y = []
      for t, x_t in enumerate(x):
        for idx, layer in enumerate(self.actor_layers):
          c, h = self.cells[idx], self.hidden[idx]
          self.hidden[idx], self.cells[idx] = layer(x_t, (h, c))
          x_t = self.hidden[idx]
        y.append(x_t)
      x = torch.stack([x_t for x_t in y])

    else:
      if dims == 1: # if we get a single timestep (if not, assume we got a batch of single timesteps)
        x = x.view(1, -1)

      for idx, layer in enumerate(self.actor_layers):
        h, c = self.hidden[idx], self.cells[idx]
        self.hidden[idx], self.cells[idx] = layer(x, (h, c))
        x = self.hidden[idx]

      if dims == 1:
        x = x.view(-1)

    action = self.network_out(x)
    return action

class LSTM_Stochastic_Actor(Actor):
  def __init__(self, state_dim, action_dim, layers=(128, 128), env_name=None, normc_init=False, bounded=False, fixed_std=None):
    super(LSTM_Stochastic_Actor, self).__init__()

    self.actor_layers = nn.ModuleList()
    self.actor_layers += [nn.LSTMCell(state_dim, layers[0])]
    for i in range(len(layers)-1):
        self.actor_layers += [nn.LSTMCell(layers[i], layers[i+1])]
    self.network_out = nn.Linear(layers[-1], action_dim)

    self.action_dim = action_dim
    self.init_hidden_state()
    self.env_name = env_name
    self.bounded = bounded
    self.is_recurrent = True

    if fixed_std is None:
      self.log_stds = nn.Linear(layers[-1], action_dim)
      self.learn_std = True
    else:
      self.fixed_std = fixed_std
      self.learn_std = False

    if normc_init:
      self.initialize_parameters()

  def get_hidden_state(self):
    return self.hidden, self.cells


  def _get_dist_params(self, state):
    dims = len(state.size())

    x = state
    if dims == 3: # if we get a batch of trajectories
      self.init_hidden_state(batch_size=x.size(1))
      action = []
      y = []
      for t, x_t in enumerate(x):
        for idx, layer in enumerate(self.actor_layers):
          c, h = self.cells[idx], self.hidden[idx]
          self.hidden[idx], self.cells[idx] = layer(x_t, (h, c))
          x_t = self.hidden[idx]
        y.append(x_t)
      x = torch.stack([x_t for x_t in y])

    else:
      if dims == 1: # if we get a single timestep (if not, assume we got a batch of single timesteps)
        x = x.view(1, -1)

      for idx, layer in enumerate(self.actor_layers):
        h, c = self.hidden[idx], self.cells[idx]
        self.hidden[idx], self.cells[idx] = layer(x, (h, c))
        x = self.hidden[idx]

      if dims == 1:
        x = x.view(-1)

    mu = self.network_out(x)
    if self.learn_std:
      sd = torch.clamp(self.log_stds(x), -2, 2).exp()
    else:
      sd = self.fixed_std

    return mu, sd

  def init_hidden_state(self, batch_size=1):
    self.hidden = [torch.zeros(batch_size, l.hidden_size) for l in self.actor_layers]
    self.cells  = [torch.zeros(batch_size, l.hidden_size) for l in self.actor_layers]

  def forward(self, state, deterministic=True, return_log_probs=False):
    mu, sd = self._get_dist_params(state)

    if not deterministic or return_log_probs:
      dist = torch.distributions.Normal(mu, sd)
      sample = dist.rsample()

    if hasattr(self, 'bounded') and self.bounded:
      action = torch.tanh(mu) if deterministic else torch.tanh(sample)
    else:
      action = mu if deterministic else sample

    if return_log_probs:
      return action, dist.log_prob(sample)
    else:
      return action

  def pdf(self, state):
    mu, sd = self._get_dist_params(state)
    return torch.distributions.Normal(mu, sd)

class GRU_Actor(Actor):
  def __init__(self, input_dim, action_dim, layers=(128, 128), env_name=None, nonlinearity=torch.tanh, normc_init=False, bounded=False):
    super(GRU_Actor, self).__init__()

    self.actor_layers = nn.ModuleList()
    self.actor_layers += [nn.GRUCell(input_dim, layers[0])]
    for i in range(len(layers)-1):
        self.actor_layers += [nn.GRUCell(layers[i], layers[i+1])]
    self.network_out = nn.Linear(layers[-1], action_dim)

    self.action_dim = action_dim
    self.init_hidden_state()
    self.env_name = env_name
    self.nonlinearity = nonlinearity
    self.bounded = bounded
    
    self.is_recurrent = True

    if normc_init:
      self.initialize_parameters()

  def get_hidden_state(self):
    return self.hidden

  def set_hidden_state(self, data):
    if len(data) != 2:
      print("Got invalid hidden state data.")
      exit(1)

    self.hidden = data
    
  def init_hidden_state(self, batch_size=1):
    self.hidden = [torch.zeros(batch_size, l.hidden_size) for l in self.actor_layers]

  def forward(self, x):
    dims = len(x.size())

    if dims == 3: # if we get a batch of trajectories
      self.init_hidden_state(batch_size=x.size(1))
      y = []
      for t, x_t in enumerate(x):
        for idx, layer in enumerate(self.actor_layers):
          h = self.hidden[idx]
          self.hidden[idx] = layer(x_t, h)
          x_t = self.hidden[idx]
        y.append(x_t)
      x = torch.stack([x_t for x_t in y])

    else:
      if dims == 1: # if we get a single timestep (if not, assume we got a batch of single timesteps)
        x = x.view(1, -1)

      for idx, layer in enumerate(self.actor_layers):
        h = self.hidden[idx]
        self.hidden[idx] = layer(x, h)
        x = self.hidden[idx]

      if dims == 1:
        x = x.view(-1)

    action = self.network_out(x)
    return action

class GRU_Stochastic_Actor(Actor):
  def __init__(self, state_dim, action_dim, layers=(128, 128), env_name=None, normc_init=False, bounded=False, fixed_std=None):
    super(GRU_Stochastic_Actor, self).__init__()

    self.actor_layers = nn.ModuleList()
    self.actor_layers += [nn.GRUCell(state_dim, layers[0])]
    for i in range(len(layers)-1):
        self.actor_layers += [nn.GRUCell(layers[i], layers[i+1])]
    self.network_out = nn.Linear(layers[-1], action_dim)

    self.action_dim = action_dim
    self.init_hidden_state()
    self.env_name = env_name
    self.bounded = bounded
    self.is_recurrent = True

    if fixed_std is None:
      self.log_stds = nn.Linear(layers[-1], action_dim)
      self.learn_std = True
    else:
      self.fixed_std = fixed_std
      self.learn_std = False

    if normc_init:
      self.initialize_parameters()

  def get_hidden_state(self):
    return self.hidden

  def _get_dist_params(self, state):
    dims = len(state.size())

    x = state
    if dims == 3: # if we get a batch of trajectories
      self.init_hidden_state(batch_size=x.size(1))
      action = []
      y = []
      for t, x_t in enumerate(x):
        for idx, layer in enumerate(self.actor_layers):
          h = self.hidden[idx]
          self.hidden[idx] = layer(x_t, h)
          x_t = self.hidden[idx]
        y.append(x_t)
      x = torch.stack([x_t for x_t in y])

    else:
      if dims == 1: # if we get a single timestep (if not, assume we got a batch of single timesteps)
        x = x.view(1, -1)

      for idx, layer in enumerate(self.actor_layers):
        h = self.hidden[idx]
        self.hidden[idx] = layer(x, h)
        x = self.hidden[idx]

      if dims == 1:
        x = x.view(-1)

    mu = self.network_out(x)
    if self.learn_std:
      sd = torch.clamp(self.log_stds(x), -2, 2).exp()
    else:
      sd = self.fixed_std

    return mu, sd

  def init_hidden_state(self, batch_size=1):
    self.hidden = [torch.zeros(batch_size, l.hidden_size) for l in self.actor_layers]

  def forward(self, state, deterministic=True, return_log_probs=False):
    mu, sd = self._get_dist_params(state)

    if not deterministic or return_log_probs:
      dist = torch.distributions.Normal(mu, sd)
      sample = dist.rsample()

    if hasattr(self, 'bounded') and self.bounded:
      action = torch.tanh(mu) if deterministic else torch.tanh(sample)
    else:
      action = mu if deterministic else sample

    if return_log_probs:
      return action, dist.log_prob(sample)
    else:
      return action

  def pdf(self, state):
    mu, sd = self._get_dist_params(state)
    return torch.distributions.Normal(mu, sd)
