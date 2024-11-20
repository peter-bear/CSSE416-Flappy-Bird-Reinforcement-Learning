import torch
import torch.nn as nn

# Define the Q-network architecture
class DQN(nn.Module):
    def __init__(self, Dim_in, act_dim):
        super().__init__()
        self.conv1 = nn.Conv2d(in_channels=Dim_in, out_channels=32, kernel_size=(8, 8), stride=(4, 4))  # Convolutional layer 1
        self.maxpool1 = nn.MaxPool2d(2, stride=2)  # Max pooling layer 1
        self.conv2 = nn.Conv2d(in_channels=32, out_channels=64, kernel_size=(4, 4), stride=(2, 2))  # Convolutional layer 2
        self.maxpool2 = nn.MaxPool2d(2, stride=2, padding=1)  # Max pooling layer 2
        self.conv3 = nn.Conv2d(in_channels=64, out_channels=64, kernel_size=(3, 3), stride=(1, 1))  # Convolutional layer 3
        self.fc1 = nn.Linear(in_features=256, out_features=256)  # Fully connected layer 1
        self.fc2 = nn.Linear(in_features=256, out_features=act_dim)  # Fully connected layer 2
        self.Relu = nn.ReLU()  # Activation function

    def forward(self, x):  # Forward pass
        x = self.Relu(self.conv1(x))  # Apply ReLU to conv1
        x = self.Relu(self.conv2(x))  # Apply ReLU to conv2
        x = self.maxpool1(x)  # Apply max pooling
        x = self.Relu(self.conv3(x))  # Apply ReLU to conv3
        x = self.maxpool2(x)  # Apply max pooling
        x = x.view(x.size(0), -1)  # Flatten the output for the fully connected layer
        x = self.Relu(self.fc1(x))  # Apply ReLU to fc1
        x = self.fc2(x)  # Output layer
        return x