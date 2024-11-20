# Flappy Bird RL Agent

This project implements a Reinforcement Learning (RL) agent to play the game Flappy Bird. The agent is trained using Deep Q-Learning techniques and can be deployed to observe the agent's performance in a simulated environment.

## Project Structure

### Folders

- **`model/`**: Contains saved model checkpoints during training, including policy and target network states.

  

### Python Files

- **`agent.py`**: Defines the main RL agent class, implementing the logic for interacting with the environment and updating policies.

- **`arguments.py`**: Defines hyperparameters and other configurable settings for the RL model training.

- **`demo.py`**: A script to demonstrate or test the agent's performance, possibly by loading a trained model and running it in the environment.

- **`dqn.py`**: Likely implements the Deep Q-Network (DQN) algorithm used for training the RL agent.

- **`experience_replay.py`**: Implements the experience replay buffer, a crucial component for DQN that stores past experiences for training.

- **`main.py`**: The main entry point for training the agent, handling initialization and the training loop.

  

### Data and Results

- **`epsilon_history.csv`**: Logs the history of the epsilon parameter (exploration rate) over training episodes.
- **`loss_history.csv`**: Records the loss values during training, useful for monitoring model convergence.
- **`q_value_data.csv`**: Tracks the Q-values over time, providing insights into how the model’s predictions evolve during training.
- **`reward_graph_data.csv`**: Logs the reward values obtained by the agent across training episodes.
- **`test_average_rewards.csv`**: Stores average rewards obtained during test episodes to evaluate model performance.



### Visualization and Graphs

- **`epsilon.png`**: Graph showing the epsilon (exploration rate) decay over time.
- **`loss.png`**: Visualization of the loss function over time, showing how well the model is learning.
- **`q_value.png`**: Graph depicting the Q-values during training, indicating the stability of the agent’s value estimates.
- **`train_reward.png`**: Plot showing the rewards obtained during training.
- **`test_reward.png`**: Plot of the rewards obtained during testing episodes.





### Additional Files

- **`requirements.txt`**: Lists all dependencies required for this project, allowing easy installation with `pip install -r requirements.txt`.



## Getting Started

1. **Environment Setup**:
   - Make sure you have Conda or virtualenv installed.
   - Install dependencies with:
     ```bash
     pip install -r requirements.txt
     ```

2. **Training the Model**:
   - Run `main.py` to start training the model:
     ```bash
     python main.py
     ```

3. **Running the Demo**:
   - Use `demo.py` to see the trained agent in action:
     ```bash
     python demo.py
     ```

## Explanation of Key Components

- **Deep Q-Learning (DQN)**: This RL algorithm is used to train the agent to play Flappy Bird by learning Q-values, which estimate the expected reward for taking specific actions in given states.
- **Experience Replay**: A mechanism to store and reuse past experiences, helping to break the correlation between consecutive actions and improve training stability.



## Visualizing Results

The CSV files (`epsilon_history.csv`, `loss_history.csv`, etc.) and the corresponding `.png` graphs provide insights into how the model trains over time. By analyzing these files, you can understand the agent's learning progress and performance.



Training Reward

![Train Reward](./train_reward.png)



Test Reward

![Test Reward](./test_reward.png)

