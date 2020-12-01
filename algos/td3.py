import copy
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import random

class TD3():
  def __init__(self, actor, q1, q2, args):
    self.recurrent = args.arch.lower() == 'lstm' or args.arch.lower() == 'gru'

    self.actor  = actor
    self.q1 = q1
    self.q2 = q2

    self.target_actor = copy.deepcopy(actor)
    self.target_q1 = copy.deepcopy(q1)
    self.target_q2 = copy.deepcopy(q2)

    self.soft_update(1.0)

    self.actor_optimizer  = torch.optim.Adam(self.actor.parameters(), lr=args.a_lr)
    self.q1_optimizer = torch.optim.Adam(self.q1.parameters(), lr=args.c_lr, weight_decay=1e-2)
    self.q2_optimizer = torch.optim.Adam(self.q2.parameters(), lr=args.c_lr, weight_decay=1e-2)

    self.discount   = args.discount
    self.tau        = args.tau
    self.update_every = args.update_freq
    self.expl_noise = args.expl_noise

    self.policy_noise = args.policy_noise

    self.n = 0

  def soft_update(self, tau):
    for param, target_param in zip(self.q1.parameters(), self.target_q1.parameters()):
      target_param.data.copy_(tau * param.data + (1 - tau) * target_param.data)

    for param, target_param in zip(self.q2.parameters(), self.target_q2.parameters()):
      target_param.data.copy_(tau * param.data + (1 - tau) * target_param.data)

    for param, target_param in zip(self.actor.parameters(), self.target_actor.parameters()):
      target_param.data.copy_(tau * param.data + (1 - tau) * target_param.data)

  def update_policy(self, replay_buffer, batch_size=256, grad_clip=None, noise_clip=0.2):
    self.n += 1

    states, actions, next_states, rewards, not_dones, mask = replay_buffer.sample(batch_size, recurrent=self.recurrent)

    with torch.no_grad():
      noise        = (torch.randn_like(actions) * self.policy_noise).clamp(-noise_clip, noise_clip)
      next_actions = self.target_actor(next_states)

      next_actions += noise

      target_q1 = self.target_q1(next_states, next_actions)
      target_q2 = self.target_q2(next_states, next_actions)

      target_q = rewards + not_dones * self.discount * torch.min(target_q1, target_q2)

    current_q1 = self.q1(states, actions) * mask
    current_q2 = self.q2(states, actions) * mask

    critic_loss = F.mse_loss(current_q1, target_q) + F.mse_loss(current_q2, target_q)

    self.q1_optimizer.zero_grad()
    self.q2_optimizer.zero_grad()

    critic_loss.backward()

    self.q1_optimizer.step()
    self.q2_optimizer.step()

    if self.n % self.update_every == 0:
      actor_loss = -(self.q1(states, self.actor(states)) * mask).mean()

      self.actor_optimizer.zero_grad()
      actor_loss.backward()

      self.actor_optimizer.step()
      
      self.soft_update(self.tau)

    return actor_loss.item(), critic_loss.item()
