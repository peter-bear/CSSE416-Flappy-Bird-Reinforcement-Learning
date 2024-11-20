# script_model.py

import flappy_bird_gym
import torch
from agent import Agent
from arguments import arguments

# Initialize arguments and load the agent with environment setup
arg = arguments()
env = flappy_bird_gym.make("FlappyBird-rgb-v0")

agent = Agent(env, arg)

# Load your trained model
model = agent.Net  # Adjust this if your agent has a specific model instantiation

# Load the trained model weights
model.load_state_dict(torch.load("model/model0.pkl", map_location=arg.cuda))  # Adjust path as necessary
model.eval()  # Set the model to evaluation mode

# Convert to TorchScript
scripted_model = torch.jit.script(model)

# Save the TorchScript model
scripted_model.save("models/model0.pt")

print("Model has been successfully scripted and saved to models/model_scripted.pt")
