import torch
import torch.nn as nn
import torch.nn.functional as F

from policies.base import Net

class Model(FF_Base):
  """
  A very simple feedforward network to be used for
  vanilla supervised learning problems.
  """
  def __init__(self, state_dim, output_dim, layers=(512,256), nonlinearity=torch.tanh):
    super(Model, self).__init__(state_dim, layers, nonlinearity)

    self.network_out = nn.Linear(layers[-1], output_dim)

    self.output_dim = output_dim
    self.nonlinearity = nonlinearity

  def forward(self, x):
    x = self._base_forward(x)
    return self.network_out(x)
