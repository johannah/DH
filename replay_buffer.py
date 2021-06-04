import numpy as np

import kornia
import torch
import torch.nn as nn
import torch.nn.functional as F
import utils


class ReplayBuffer(object):
    """Buffer to store environment transitions."""
    def __init__(self, obs_shape, body_shape, action_shape, capacity, image_pad, device):
        self.capacity = capacity
        self.device = device

        self.aug_trans = nn.Sequential(
            nn.ReplicationPad2d(image_pad),
            kornia.augmentation.RandomCrop((obs_shape[-1], obs_shape[-1])))

        self.obses = np.empty((capacity, *obs_shape), dtype=np.uint8)
        self.next_obses = np.empty((capacity, *obs_shape), dtype=np.uint8)
        self.bodies = np.empty((capacity, body_shape), dtype=np.float32)
        self.next_bodies = np.empty((capacity, body_shape), dtype=np.float32)
        self.actions = np.empty((capacity, *action_shape), dtype=np.float32)
        self.rewards = np.empty((capacity, 1), dtype=np.float32)
        self.not_dones = np.empty((capacity, 1), dtype=np.float32)
        self.not_dones_no_max = np.empty((capacity, 1), dtype=np.float32)

        self.idx = 0
        self.full = False

    def __len__(self):
        return self.capacity if self.full else self.idx

    def save(self, filepath):
        thru = self.idx
        if self.full:
            thru = self.capacity
        np.savez(filepath, idx=self.idx, full=self.full, device=self.device, 
                          capacity=self.capacity, 
                          obses=self.obses[:thru], next_obses=self.next_obses[:thru], 
                          bodies=self.bodies[:thru], next_bodies=self.next_bodies[:thru], 
                          actions=self.actions[:thru], rewards=self.rewards[:thru], 
                          not_dones=self.not_dones[:thru], 
                          not_dones_no_max=self.not_dones_no_max[:thru])

    def load(self, filepath):
        thru = self.idx
        ff = np.load(filepath)
        capacity = ff['capacity']
        self.idx=ff['idx']
        self.full=ff['full']
        if self.full:
            thru = self.capacity
        else:
            thru = idx 
        assert thru <= self.capacity
        print("loading %s examples from replay buffer"%thru)
        self.obses=ff['obses'][:thru]
        self.next_obses=ff['next_obses'][:thru] 
        self.bodies=ff['bodies'][:thru]
        self.next_bodies=ff['next_bodies'][:thru]
        self.actions=ff['actions'][:thru]
        self.rewards=ff['rewards'][:thru]
        self.not_dones=ff['not_dones'][:thru]
        self.not_dones_no_max=ff['not_dones_no_max'][:thru]


    def add(self, obs, body, action, reward, next_obs, next_body, done, done_no_max):
        np.copyto(self.obses[self.idx], obs)
        np.copyto(self.bodies[self.idx], body)
        np.copyto(self.actions[self.idx], action)
        np.copyto(self.rewards[self.idx], reward)
        np.copyto(self.next_obses[self.idx], next_obs)
        np.copyto(self.next_bodies[self.idx], next_body)
        np.copyto(self.not_dones[self.idx], not done)
        np.copyto(self.not_dones_no_max[self.idx], not done_no_max)

        self.idx = (self.idx + 1) % self.capacity
        self.full = self.full or self.idx == 0

    def sample(self, batch_size):
        idxs = np.random.randint(0,
                                 self.capacity if self.full else self.idx,
                                 size=batch_size)

        obses = self.obses[idxs]
        bodies = self.bodies[idxs]
        next_obses = self.next_obses[idxs]
        next_bodies = self.next_bodies[idxs]
        obses_aug = obses.copy()
        next_obses_aug = next_obses.copy()

        obses = torch.as_tensor(obses, device=self.device).float()
        bodies = torch.as_tensor(bodies, device=self.device).float()
        next_obses = torch.as_tensor(next_obses, device=self.device).float()
        obses_aug = torch.as_tensor(obses_aug, device=self.device).float()
        next_obses_aug = torch.as_tensor(next_obses_aug,
                                         device=self.device).float()
        next_bodies = torch.as_tensor(next_bodies, device=self.device).float()
        actions = torch.as_tensor(self.actions[idxs], device=self.device)
        rewards = torch.as_tensor(self.rewards[idxs], device=self.device)
        not_dones_no_max = torch.as_tensor(self.not_dones_no_max[idxs],
                                           device=self.device)

        obses = self.aug_trans(obses)
        next_obses = self.aug_trans(next_obses)

        obses_aug = self.aug_trans(obses_aug)
        next_obses_aug = self.aug_trans(next_obses_aug)

        return obses, bodies, actions, rewards, next_obses, next_bodies, not_dones_no_max, obses_aug, next_obses_aug
