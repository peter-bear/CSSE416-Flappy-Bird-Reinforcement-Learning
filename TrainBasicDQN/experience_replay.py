import random
import numpy as np
from collections import deque

# Define replay buffer for experience replay
class Replay_Buffer():
    def __init__(self, arg):
        self.capacity = arg.capacity  # Maximum size of buffer
        self.action_dim = arg.action_dim  # Action dimensions
        self.env_obs_space = arg.obs_dim  # Observation space dimensions
        self.data = {
            'action': np.zeros((self.capacity, 1)),
            'obs': np.zeros((self.capacity, self.env_obs_space[0], self.env_obs_space[1], self.env_obs_space[2])),
            'next_obs': np.zeros((self.capacity, self.env_obs_space[0], self.env_obs_space[1], self.env_obs_space[2])),
            'done': np.zeros((self.capacity, 1)),
            'reward': np.zeros((self.capacity, 1)),
        }
        self.ptr = 0  # Pointer to keep track of current position
        self.isfull = 0  # Flag to check if buffer is full

    # Store transition data in the buffer
    def store_data(self, transition, length):
        if self.ptr + length > self.capacity:  # If buffer exceeds capacity
            rest = self.capacity - self.ptr
            for key in self.data:
                store_tmp = np.array(transition[key][:], dtype=object)
                store_tmp = np.expand_dims(store_tmp, -1) if len(store_tmp.shape) == 1 else store_tmp
                self.data[key][self.ptr:] = store_tmp[:rest]  # Store data until capacity
                transition[key] = transition[key][rest:]
            self.ptr = 0  # Reset pointer
            length -= rest
            self.isfull = 1
        for key in self.data:
            store_tmp = np.array(transition[key][:], dtype=object)
            self.data[key][self.ptr:self.ptr + length] = np.expand_dims(store_tmp, -1) if len(store_tmp.shape) == 1 else store_tmp
        self.ptr += length

    # Sample a batch of transitions from the buffer
    def sample(self, batch):
        if self.isfull:
            batch_index = np.random.choice(self.capacity, size=batch)  # Randomly sample from full buffer
        else:
            batch_index = np.random.choice(self.ptr, size=batch)  # Randomly sample from available data
        return {key: self.data[key][batch_index, :] for key in self.data}