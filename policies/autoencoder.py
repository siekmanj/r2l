import torch
import torch.nn as nn

class TernaryTanh(torch.autograd.Function):
  """
  A class with functions thatternarize tanh by rounding to either -1, 0, or 1.
  To be used to accomplish autoencoding values to a discretized latent
  representation.
  """
  @staticmethod
  def forward(cxt, x):
    y = x.new(x.size())
    y.data = x.data
    y.round_()
    return y

  @staticmethod
  def backward(cxt, dy):
    dx = dy.clone()
    return dx
ternaryTanh = TernaryTanh.apply

def ternary_tanh(x):
  return ternaryTanh(1.5 * torch.tanh(x) + 0.5 * torch.tanh(-3 * x))

class QBN(nn.Module):
  """
  A neural network autoencoder which compresses information using
  a structure defined by the 'layers' argument in the constructor.
  The discretized, latent representation will be the size of the last
  element in the layers tuple.
  """
  def __init__(self, dim, layers=(128, 64, 16)):
    super(QBN, self).__init__()

    self.encoder_layers = nn.ModuleList()
    self.encoder_layers += [nn.Linear(dim, layers[0])]
    for i in range(len(layers)-1):
      self.encoder_layers += [nn.Linear(layers[i], layers[i+1])]

    self.decoder_layers = nn.ModuleList()
    self.decoder_layers += [nn.Linear(layers[0], dim)]
    for i in range(len(layers)-1):
      self.decoder_layers += [nn.Linear(layers[i+1], layers[i])]
    self.decoder_layers = self.decoder_layers[::-1]

  def encode(self, x):
    for layer in self.encoder_layers[:-1]:
      x = torch.tanh(layer(x))
    x = self.encoder_layers[-1](x)
    #return ternaryTanh(1.5 * torch.tanh(x) + 0.5 * torch.tanh(-3 * x))
    return ternary_tanh(x)

  def decode(self, x):
    for layer in self.decoder_layers[:-1]:
      x = torch.tanh(layer(x))
    x = self.decoder_layers[-1](x)
    return x

  def forward(self, x):
    return self.decode(self.encode(x))
