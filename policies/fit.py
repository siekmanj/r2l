import torch
import torch.nn as nn
import torch.nn.functional as F

from policies.base import Net

class Model(Net):
  def __init__(self, state_dim, output_dim, layers=(512,256), nonlinearity=torch.tanh):
    super(Model, self).__init__()

    self.layers = nn.ModuleList()
    self.layers += [nn.Linear(state_dim, layers[0])]
    for i in range(len(layers)-1):
        self.layers += [nn.Linear(layers[i], layers[i+1])]
    self.network_out = nn.Linear(layers[-1], output_dim)

    self.output_dim = output_dim
    self.nonlinearity = nonlinearity

  def forward(self, x):
    for l in self.layers:
      x = self.nonlinearity(l(x))
    return self.network_out(x)
