import torch
import torch.nn as nn
from dqn import DQN
from experience_replay import Replay_Buffer
# from experience_replay import ReplayMemory
import random
import numpy as np

# Define a Deep Q-Network (DQN) agent
class Agent():
    def __init__(self, env, arg):
        self.arg = arg
        self.Buffer = Replay_Buffer(arg)  # Initialize replay buffer
        self.Net = DQN(arg.Frames, arg.action_dim).to(self.arg.cuda)  # Main Q-network
        self.targetNet = DQN(arg.Frames, arg.action_dim).to(self.arg.cuda)  # Target Q-network
        self.learnstep = 0  # Track learning steps
        self.optimizer = torch.optim.Adam(self.Net.parameters(), lr=arg.learning_rate)  # Optimizer for Q-network
        self.loss_func = nn.MSELoss()  # Loss function

    # Get action based on epsilon-greedy strategy
    def get_action(self, obs):
        if random.random() > self.arg.epsilon:  # Greedy action
            return self.greedy_action(obs)
        else:  # Random action
            return random.randint(0, 1)

    # Get action based on Q-values (greedy)
    def greedy_action(self, obs):
        obs = torch.tensor(obs, device=self.arg.cuda, dtype=torch.float32)
        if len(obs.shape) == 3:  # If observation does not have batch dimension
            obs = obs.unsqueeze(0)
        obs = self.Net(obs)  # Pass observation through Q-network
        action = np.argmax(obs.detach().cpu().numpy(), axis=-1)  # Get action with highest Q-value
        return action

    # Update the Q-network using experience replay
    def update(self, data):
        action, obs, next_obs, done, reward = data['action'], data['obs'], data['next_obs'], data['done'], data['reward']
        if self.learnstep % self.arg.Q_NETWORK_ITERATION == 0:  # Update target network periodically
            self.targetNet.load_state_dict(self.Net.state_dict())
        self.learnstep += 1
        
        # Convert data to PyTorch tensors
        obs = torch.tensor(obs, device=self.arg.cuda, dtype=torch.float32)
        next_obs = torch.tensor(next_obs, device=self.arg.cuda, dtype=torch.float32)
        action = torch.tensor(action, device=self.arg.cuda, dtype=torch.long)
        reward = torch.tensor(reward, device=self.arg.cuda, dtype=torch.float32)
        
        q_eval = self.Net(obs).gather(1, action)  # Q-values for the actions taken
        q_next = self.targetNet(next_obs).detach()  # Q-values from target network
        q_target = reward + self.arg.gamma * q_next.max(1)[0].view(action.shape[0], 1)  # Bellman equation
        
        loss = self.loss_func(q_eval, q_target)  # Compute loss
        self.optimizer.zero_grad()  # Zero the gradients
        loss.backward()  # Backpropagation
        self.optimizer.step()  # Update the network
        return loss

