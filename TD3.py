import copy
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from IPython import embed
# Implementation of Twin Delayed Deep Deterministic Policy Gradients (TD3)
# Paper: https://arxiv.org/abs/1802.09477


class Actor(nn.Module):
    def __init__(self, state_dim, action_dim, max_policy_action):
        super(Actor, self).__init__()

        self.l1 = nn.Linear(state_dim, 256)
        self.l2 = nn.Linear(256, 256)
        self.l3 = nn.Linear(256, action_dim)

        self.max_policy_action = max_policy_action


    def forward(self, state):
        a = F.relu(self.l1(state))
        a = F.relu(self.l2(a))
        return self.max_policy_action * torch.tanh(self.l3(a))


class Critic(nn.Module):
    def __init__(self, state_dim, action_dim):
        super(Critic, self).__init__()

        # Q1 architecture
        self.l1 = nn.Linear(state_dim + action_dim, 256)
        self.l2 = nn.Linear(256, 256)
        self.l3 = nn.Linear(256, 1)

        # Q2 architecture
        self.l4 = nn.Linear(state_dim + action_dim, 256)
        self.l5 = nn.Linear(256, 256)
        self.l6 = nn.Linear(256, 1)


    def forward(self, state, action):
        sa = torch.cat([state, action], 1)

        q1 = F.relu(self.l1(sa))
        q1 = F.relu(self.l2(q1))
        q1 = self.l3(q1)

        q2 = F.relu(self.l4(sa))
        q2 = F.relu(self.l5(q2))
        q2 = self.l6(q2)
        return q1, q2


    def Q1(self, state, action):
        sa = torch.cat([state, action], 1)

        q1 = F.relu(self.l1(sa))
        q1 = F.relu(self.l2(q1))
        q1 = self.l3(q1)
        return q1


class TD3(object):
    def __init__(
        self,
        state_dim,
        action_dim,
        max_policy_action=1.0,
        discount=0.99,
        tau=0.005,
        policy_noise=0.2,
        noise_clip=0.5,
        policy_freq=2,
        device='cpu',
        **kwargs
    ):

        self.step = 0
        self.device = device
        self.max_policy_action = max_policy_action
        self.discount = discount
        self.tau = tau
        self.policy_noise = policy_noise
        self.noise_clip = noise_clip
        self.policy_freq = policy_freq


        self.actor = Actor(state_dim, action_dim, self.max_policy_action).to(self.device)
        self.actor_target = copy.deepcopy(self.actor)
        self.actor_optimizer = torch.optim.Adam(self.actor.parameters(), lr=3e-4)

        self.critic = Critic(state_dim, action_dim).to(self.device)
        self.critic_target = copy.deepcopy(self.critic)
        self.critic_optimizer = torch.optim.Adam(self.critic.parameters(), lr=3e-4)

        self.total_it = 0
        self.loss_dict = {'actor':[], 'critic':[], 'critic_step':[], 'actor_step':[]}


    def select_action(self, state):
        state = torch.FloatTensor(state.reshape(1, -1)).to(self.device)
        return self.actor(state).cpu().data.numpy().flatten()


    def train(self, step, replay_buffer, batch_size=100):
        self.step = step
        self.total_it += 1
        # Sample replay buffer
        state, action, reward, next_state, not_done, _, _ = replay_buffer.sample(batch_size)
        state = torch.FloatTensor(state).to(self.device)
        action = torch.FloatTensor(action).to(self.device)
        reward = torch.FloatTensor(reward).to(self.device)
        next_state = torch.FloatTensor(next_state).to(self.device)
        not_done = torch.FloatTensor(not_done).to(self.device)

        with torch.no_grad():
            # Select action according to policy and add clipped noise
            noise = (
                torch.randn_like(action) * self.policy_noise
            ).clamp(-self.noise_clip, self.noise_clip)

            # TODO - maybe clamp bt known min/max actions
            next_action = (
                self.actor_target(next_state) + noise
            ).clamp(-self.max_policy_action, self.max_policy_action)

            # Compute the target Q value
            target_Q1, target_Q2 = self.critic_target(next_state, next_action)
            target_Q = torch.min(target_Q1, target_Q2)
            target_Q = reward + not_done * self.discount * target_Q

        # Get current Q estimates
        current_Q1, current_Q2 = self.critic(state, action)

        # Compute critic loss
        critic_loss = F.mse_loss(current_Q1, target_Q) + F.mse_loss(current_Q2, target_Q)
        self.loss_dict['critic'].append(critic_loss.item())
        self.loss_dict['critic_step'].append(step)

        # Optimize the critic
        self.critic_optimizer.zero_grad()
        critic_loss.backward()
        self.critic_optimizer.step()

        # Delayed policy updates
        actor_loss = 0
        if self.total_it % self.policy_freq == 0:

            # Compute actor losse
            actor_loss = -self.critic.Q1(state, self.actor(state)).mean()
            self.loss_dict['actor'].append(actor_loss.item())
            self.loss_dict['actor_step'].append(step)
            # Optimize the actor
            self.actor_optimizer.zero_grad()
            actor_loss.backward()
            self.actor_optimizer.step()

            # Update the frozen target models
            for param, target_param in zip(self.critic.parameters(), self.critic_target.parameters()):
                target_param.data.copy_(self.tau * param.data + (1 - self.tau) * target_param.data)

            for param, target_param in zip(self.actor.parameters(), self.actor_target.parameters()):
                target_param.data.copy_(self.tau * param.data + (1 - self.tau) * target_param.data)


    def save(self, filepath):
        model_dict =  {'critic':self.critic.state_dict(), 
                      'actor':self.actor.state_dict(), 
                      'critic_optimizer':self.critic_optimizer.state_dict(), 
                      'actor_optimizer':self.actor_optimizer.state_dict(),
                      'loss_dict':self.loss_dict, 
                      'total_it':self.total_it}
        torch.save(model_dict, filepath)

    def get_loss_plot_data(self):
        plot_dict =  {'critic':(self.loss_dict['critic_step'], self.loss_dict['critic']), 
                'actor':(self.loss_dict['actor_step'], self.loss_dict['actor'])}
        return plot_dict

    def load(self, filepath):
        print("TD3 loading {}".format(filepath))
        model_dict = torch.load(filepath)
        self.critic.load_state_dict(model_dict['critic'])
        self.actor.load_state_dict(model_dict['actor'])
        self.critic_optimizer.load_state_dict(model_dict['critic_optimizer'])
        self.actor_optimizer.load_state_dict(model_dict['actor_optimizer'])
        self.total_it = model_dict['total_it']
        try:
            self.loss_dict = model_dict['loss_dict']
        except:
            self.loss_dict = {'actor':[0], 'critic':[0], 'critic_step':[0], 'actor_step':[0]}

        # TODO this is wrong
        if len(self.loss_dict['critic_step']):
            self.step = self.loss_dict['critic_step'][-1]
        else:
            self.step = 0
