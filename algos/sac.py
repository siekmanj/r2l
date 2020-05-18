import copy
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import random

class SAC():
  def __init__(self, actor, q1, q2, target_entropy, args):
    self.actor  = actor
    self.q1 = q1
    self.q2 = q2

    self.recurrent = False
    self.normalize=False

    self.target_q1 = copy.deepcopy(q1)
    self.target_q2 = copy.deepcopy(q2)

    self.actor_optim  = torch.optim.Adam(self.actor.parameters(), lr=args.a_lr)
    self.q1_optim = torch.optim.Adam(self.q1.parameters(), lr=args.c_lr)
    self.q2_optim = torch.optim.Adam(self.q2.parameters(), lr=args.c_lr)

    self.target_entropy = target_entropy
    if args.alpha is None:
      self.log_alpha = torch.zeros(1, requires_grad=True)
      self.alpha = self.log_alpha.exp()
      self.alpha_optim = torch.optim.Adam([self.log_alpha], lr=args.a_lr)
      self.tune_alpha = True
    else:
      self.alpha = args.alpha
      self.tune_alpha = False

    self.gamma = args.discount
    self.tau = args.tau
    self.expl_noise = None

  def soft_update(self, tau):
    for param, target_param in zip(self.q1.parameters(), self.target_q1.parameters()):
      target_param.data.copy_(tau * param.data + (1 - tau) * target_param.data)

    for param, target_param in zip(self.q2.parameters(), self.target_q2.parameters()):
      target_param.data.copy_(tau * param.data + (1 - tau) * target_param.data)

  def update_policy(self, buff, batch_size=64, traj_len=400):
    state, action, next_state, reward, not_done, steps, mask = buff.sample(batch_size)

    with torch.no_grad():
      #state      = self.actor.normalize_state(state, update=False)
      #next_state = self.actor.normalize_state(next_state, update=False)

      next_action, next_log_prob = self.actor(next_state, deterministic=False, return_log_probs=True)
      next_target_q1 = self.target_q1(next_state, next_action)
      next_target_q2 = self.target_q2(next_state, next_action)
      next_target_q  = torch.min(next_target_q1, next_target_q2) - self.alpha * next_log_prob
      next_q = reward + not_done * self.gamma * next_target_q

    q1 = self.q1(state, action)
    q2 = self.q2(state, action)
    q1_loss = F.mse_loss(q1, next_q)
    q2_loss = F.mse_loss(q2, next_q)

    pi, log_prob = self.actor(state, deterministic=False, return_log_probs=True)
    q1_pi = self.q1(state, pi)
    q2_pi = self.q2(state, pi)
    q_pi  = torch.min(q1_pi, q2_pi)

    actor_loss = (self.alpha * log_prob - q_pi).mean()

    self.q1_optim.zero_grad()
    q1_loss.backward()
    self.q1_optim.step()

    self.q2_optim.zero_grad()
    q2_loss.backward()
    self.q2_optim.step()

    self.actor_optim.zero_grad()
    actor_loss.backward()
    self.actor_optim.step()

    self.soft_update(self.tau)

    if self.tune_alpha:
      alpha_loss = -(self.log_alpha * (log_prob + self.target_entropy).detach()).mean()

      self.alpha_optim.zero_grad()
      alpha_loss.backward()
      self.alpha_optim.step()

      self.alpha = self.log_alpha.exp()
    else:
      alpha_loss = torch.zeros(1)
    
    with torch.no_grad():
      return actor_loss.item(), torch.mean(q1_loss + q2_loss).item(), alpha_loss.item(), steps
