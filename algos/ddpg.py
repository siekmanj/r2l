import copy
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import random
import time

class DDPG():
  """
  Deep Deterministic Policy Gradients is an off-policy policy gradient
  algorithm. This implementation can also act as RDPG, the recurrent
  variant of DDPG. It is called from off_policy.py, 
  """
  def __init__(self, actor, critic, args):

    self.recurrent = actor.is_recurrent or critic.is_recurrent

    self.actor = actor
    self.critic = critic

    self.target_actor = copy.deepcopy(actor)
    self.target_critic = copy.deepcopy(critic)

    self.soft_update(1.0)

    self.actor_optimizer  = torch.optim.Adam(self.actor.parameters(), lr=args.a_lr)
    self.critic_optimizer = torch.optim.Adam(self.critic.parameters(), lr=args.c_lr)

    self.discount   = args.discount
    self.tau        = args.tau
    self.expl_noise = args.expl_noise

  def soft_update(self, tau):
    for param, target_param in zip(self.critic.parameters(), self.target_critic.parameters()):
      target_param.data.copy_(tau * param.data + (1 - tau) * target_param.data)

    for param, target_param in zip(self.actor.parameters(), self.target_actor.parameters()):
      target_param.data.copy_(tau * param.data + (1 - tau) * target_param.data)

  def update_policy(self, replay_buffer, batch_size=32, traj_len=1000, grad_clip=None):
    states, actions, next_states, rewards, not_dones, mask = replay_buffer.sample(batch_size, recurrent=self.recurrent)

    with torch.no_grad():
      #print(rewards.shape, '+', not_dones.shape, '*', self.discount, '* critic(', next_states.shape, ',', self.target_actor(next_states).shape, ') which has shape', self.target_critic(next_states, self.target_actor(next_states)).shape)
      target_q = rewards + (not_dones * self.discount * self.target_critic(next_states, self.target_actor(next_states))) * mask

    current_q = self.critic(states, actions) * mask
    #print(target_q.shape, current_q.shape, '\n')

    critic_loss = F.mse_loss(current_q, target_q)

    self.critic_optimizer.zero_grad()
    critic_loss.backward()
    self.critic_optimizer.step()

    actor_loss = -(self.critic(states, self.actor(states)) * mask).mean()

    self.actor_optimizer.zero_grad()
    actor_loss.backward()
    self.actor_optimizer.step()

    self.soft_update(1-self.tau)

    return actor_loss.item(), critic_loss.item()
