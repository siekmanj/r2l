import torch
import torch.nn as nn
import torch.nn.functional as F

from policies.base import FF_Base, LSTM_Base, GRU_Base, QBN_GRU_Base

class Actor:
  """
  The base class for deterministic actors.
  """
  def __init__(self, latent, action_dim, env_name):
    self.action_dim        = action_dim
    self.env_name          = env_name
    self.network_out       = nn.Linear(latent, action_dim)

  def deterministic_forward(self, state, update=False):
    state = self.normalize_state(state, update=update)
    x = self._base_forward(state)
    return self.network_out(x)


class FF_Actor(FF_Base, Actor):
  """
  A class inheriting from FF_Base and Actor
  which implements a feedforward deterministic policy.
  """
  def __init__(self, input_dim, action_dim, layers=(256, 256), env_name=None, nonlinearity=torch.tanh):

    FF_Base.__init__(self, input_dim, layers, nonlinearity)
    Actor.__init__(self, layers[-1], action_dim, env_name)

  def forward(self, x, update_norm=False):
    return self.deterministic_forward(x, update=update_norm)


class LSTM_Actor(LSTM_Base, Actor):
  """
  A class inheriting from LSTM_Base and Actor
  which implements a recurrent deterministic policy.
  """
  def __init__(self, input_dim, action_dim, layers=(128, 128), env_name=None):
    
    LSTM_Base.__init__(self, input_dim, layers)
    Actor.__init__(self, layers[-1], action_dim, env_name)

    self.is_recurrent = True
    self.init_hidden_state()

  def forward(self, x, update_norm=False):
    return self.deterministic_forward(x, update=update_norm)


class GRU_Actor(GRU_Base, Actor):
  """
  A class inheriting from GRU_Base and Actor
  which implements a recurrent deterministic policy.
  """
  def __init__(self, input_dim, action_dim, layers=(128, 128), env_name=None, nonlinearity=torch.tanh):

    GRU_Base.__init__(self, input_dim, layers)
    Actor.__init__(self, layers[-1], action_dim, env_name)

    self.is_recurrent = True
    self.init_hidden_state()

  def forward(self, x, update_norm=False):
    return self.deterministic_forward(x, update=update_norm)


class Stochastic_Actor:
  """
  The base class for stochastic actors.
  """
  def __init__(self, latent, action_dim, env_name, bounded, fixed_std=None):

    self.action_dim        = action_dim
    self.env_name          = env_name
    self.means             = nn.Linear(latent, action_dim)
    self.bounded           = bounded

    if fixed_std is None:
      self.log_stds = nn.Linear(latent, action_dim)
    self.fixed_std = fixed_std

  def _get_dist_params(self, state, update=False):
    state = self.normalize_state(state, update=update)
    x = self._base_forward(state)

    mu = self.means(x)

    if self.fixed_std is None:
      std = torch.clamp(self.log_stds(x), -2, 1).exp()
    else:
      std = self.fixed_std

    return mu, std

  def stochastic_forward(self, state, deterministic=True, update=False, log_probs=False):
    mu, sd = self._get_dist_params(state, update=update)

    if not deterministic or log_probs:
      dist = torch.distributions.Normal(mu, sd)
      sample = dist.rsample()

    if self.bounded:
      action = torch.tanh(mu) if deterministic else torch.tanh(sample)
    else:
      action = mu if deterministic else sample

    if log_probs:
      log_prob = dist.log_prob(sample)
      if self.bounded:
        log_prob -= torch.log((1 - torch.tanh(sample).pow(2)) + 1e-6)

      return action, log_prob.sum(1, keepdim=True)
    else:
      return action

  def pdf(self, state):
    mu, sd = self._get_dist_params(state)
    return torch.distributions.Normal(mu, sd)


class FF_Stochastic_Actor(FF_Base, Stochastic_Actor):
  """
  A class inheriting from FF_Base and Stochastic_Actor
  which implements a feedforward stochastic policy.
  """
  def __init__(self, input_dim, action_dim, layers=(256, 256), env_name=None, nonlinearity=torch.tanh, bounded=False, fixed_std=None):

    FF_Base.__init__(self, input_dim, layers, nonlinearity)
    Stochastic_Actor.__init__(self, layers[-1], action_dim, env_name, bounded, fixed_std=fixed_std)

  def forward(self, x, deterministic=True, update_norm=False, return_log_probs=False):
    return self.stochastic_forward(x, deterministic=deterministic, update=update_norm, log_probs=return_log_probs)


class LSTM_Stochastic_Actor(LSTM_Base, Stochastic_Actor):
  """
  A class inheriting from LSTM_Base and Stochastic_Actor
  which implements a recurrent stochastic policy.
  """
  def __init__(self, input_dim, action_dim, layers=(128, 128), env_name=None, bounded=False, fixed_std=None):

    LSTM_Base.__init__(self, input_dim, layers)
    Stochastic_Actor.__init__(self, layers[-1], action_dim, env_name, bounded, fixed_std=fixed_std)

    self.is_recurrent = True
    self.init_hidden_state()

  def forward(self, x, deterministic=True, update_norm=False, return_log_probs=False):
    return self.stochastic_forward(x, deterministic=deterministic, update=update_norm, log_probs=return_log_probs)


class GRU_Stochastic_Actor(GRU_Base, Stochastic_Actor):
  """
  A class inheriting from GRU_Base and Stochastic_Actor
  which implements a recurrent stochastic policy.
  """
  def __init__(self, input_dim, action_dim, layers=(128, 128), env_name=None, bounded=False, fixed_std=None):

    GRU_Base.__init__(self, input_dim, layers)
    Stochastic_Actor.__init__(self, layers[-1], action_dim, env_name, bounded, fixed_std=fixed_std)

    self.is_recurrent = True
    self.init_hidden_state()

  def forward(self, x, deterministic=True, update_norm=False, return_log_probs=False):
    return self.stochastic_forward(x, deterministic=deterministic, update=update_norm, log_probs=return_log_probs)

class QBN_GRU_Stochastic_Actor(QBN_GRU_Base, Stochastic_Actor):
  """
  A class inheriting from QBN_GRU_Base and Stochastic_Actor
  which implements a discrete, recurrent stochastic policy.
  """
  def __init__(self, input_dim, action_dim, layers=(128, 128), env_name=None, bounded=False, fixed_std=None):

    QBN_GRU_Base.__init__(self, input_dim)
    Stochastic_Actor.__init__(self, self.latent_output_size, action_dim, env_name, bounded, fixed_std=fixed_std)

    self.is_recurrent = True
    self.init_hidden_state()

  def forward(self, x, deterministic=True, update_norm=False, return_log_probs=False):
    return self.stochastic_forward(x, deterministic=deterministic, update=update_norm, log_probs=return_log_probs)


