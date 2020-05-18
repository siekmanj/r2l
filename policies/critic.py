import torch
import torch.nn as nn
import torch.nn.functional as F

from policies.base import FF_Base, LSTM_Base, GRU_Base

class Q:
  """
  The base class for Q functions.
  """
  def __init__(self, latent, env_name):
    self.env_name          = env_name
    self.network_out       = nn.Linear(latent, 1)

  def q_forward(self, state, action, update=False):
    state = self.normalize_state(state, update=update)
    x = torch.cat([state, action], len(state.size())-1)
    x = self._base_forward(x)
    return self.network_out(x)


class FF_Q(FF_Base, Q):
  """
  A class inheriting from FF_Base and Q
  which implements a feedforward Q function.
  """
  def __init__(self, state_dim, action_dim, layers=(256, 256), env_name=None):
    FF_Base.__init__(self, state_dim + action_dim, layers, F.relu)
    Q.__init__(self, layers[-1], env_name)

  def forward(self, state, action):
    return self.q_forward(state, action)


class LSTM_Q(LSTM_Base, Q):
  """
  A class inheriting from LSTM_Base and Q
  which implements a recurrent Q function.
  """
  def __init__(self, state_dim, action_dim, layers=(128, 128), env_name=None):
    LSTM_Base.__init__(self, state_dim + action_dim, layers)
    Q.__init__(self, layers[-1], env_name)

    self.is_recurrent = True
    self.init_hidden_state()

  def forward(self, state, action):
    return self.q_forward(state, action)

class GRU_Q(GRU_Base, Q):
  """
  A class inheriting from GRU_Base and Q
  which implements a recurrent Q function.
  """
  def __init__(self, state_dim, action_dim, layers=(128, 128), env_name=None):
    GRU_Base.__init__(self, state_dim + action_dim, layers)
    Q.__init__(self, layers[-1], env_name)

    self.is_recurrent = True
    self.init_hidden_state()

  def forward(self, state, action):
    return self.q_forward(state, action)


class V:
  """
  The base class for Value functions.
  """
  def __init__(self, latent, env_name):
    self.env_name          = env_name
    self.network_out       = nn.Linear(latent, 1)

  def v_forward(self, state, update=False):
    state = self.normalize_state(state, update=update)
    x = self._base_forward(state)
    return self.network_out(x)


class FF_V(FF_Base, V):
  """
  A class inheriting from FF_Base and V
  which implements a feedforward value function.
  """
  def __init__(self, input_dim, layers=(256, 256), env_name=None):
    FF_Base.__init__(self, input_dim, layers, F.relu)
    V.__init__(self, layers[-1], env_name)

  def forward(self, state):
    return self.v_forward(state)


class LSTM_V(LSTM_Base, V):
  """
  A class inheriting from LSTM_Base and V
  which implements a recurrent value function.
  """
  def __init__(self, input_dim, layers=(128, 128), env_name=None):
    LSTM_Base.__init__(self, input_dim, layers)
    V.__init__(self, layers[-1], env_name)

    self.is_recurrent = True
    self.init_hidden_state()

  def forward(self, state):
    return self.v_forward(state)

class GRU_V(GRU_Base, V):
  """
  A class inheriting from GRU_Base and V
  which implements a recurrent value function.
  """
  def __init__(self, input_dim, layers=(128, 128), env_name=None):
    GRU_Base.__init__(self, input_dim, layers)
    V.__init__(self, layers[-1], env_name)

    self.is_recurrent = True
    self.init_hidden_state()

  def forward(self, state):
    return self.v_forward(state)
