import torch
import torch.nn as nn
from dqn import DQN
from experience_replay import Replay_Buffer
import random
import numpy as np

# Define a Deep Q-Network (DQN) agent
class Agent():
    def __init__(self, env, arg):
        self.arg = arg
        self.Buffer = Replay_Buffer(arg)  # Initialize replay buffer
        self.Net = DQN(arg.Frames, arg.action_dim).to(self.arg.cuda)  # Main Q-network
        self.targetNets = [DQN(arg.Frames, arg.action_dim).to(self.arg.cuda) for _ in range(3)]  # Multiple target Q-networks
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
        action, obs, next_obs, reward = data['action'], data['obs'], data['next_obs'], data['reward']
        
        # Periodically update all target networks in Maxmin Q-learning
        if self.learnstep % self.arg.Q_NETWORK_ITERATION == 0:
            for target_net in self.targetNets:
                target_net.load_state_dict(self.Net.state_dict())
        self.learnstep += 1
        
        # Convert data to PyTorch tensors and move to the appropriate device
        obs = torch.tensor(obs, device=self.arg.cuda, dtype=torch.float32)
        next_obs = torch.tensor(next_obs, device=self.arg.cuda, dtype=torch.float32)
        action = torch.tensor(action, device=self.arg.cuda, dtype=torch.long)
        reward = torch.tensor(reward, device=self.arg.cuda, dtype=torch.float32)  # Ensure reward is a float tensor
        
        # Evaluate Q-values for the current state-action pairs
        net_output = self.Net(obs)  # Get network output

        if action.ndim == 1:  # If action is 1D, unsqueeze it to match net_output
            action = action.unsqueeze(1)

        q_eval = net_output.gather(1, action)  # Gather Q-values for the actions taken

        
        # Maxmin Q-learning logic
        # Select action using online network
        next_actions = self.Net(next_obs).detach().max(1)[1].unsqueeze(1)  # Get next actions
        # Use the minimum Q-value among target networks for the selected action
        q_next_values = [target_net(next_obs).detach().gather(1, next_actions) for target_net in self.targetNets]
        q_next_min = torch.min(torch.stack(q_next_values), dim=0)[0]
        
        # Compute the target Q-values
        q_target = reward + self.arg.gamma * q_next_min
        q_target = q_target.view(-1, 1)  # Ensure target size matches q_eval
        
        # Compute loss and update network
        loss = self.loss_func(q_eval, q_target)  # Compute MSE loss
        self.optimizer.zero_grad()  # Zero the gradients
        loss.backward()  # Backpropagation
        self.optimizer.step()  # Update the network
        
        return loss


