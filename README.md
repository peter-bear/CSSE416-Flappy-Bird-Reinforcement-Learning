# Reinforcement Learning Framework

This project contains implementations of various reinforcement learning algorithms using Python, focused on solving environments via deep Q-learning variants.

## Table of Contents

1. [Overview](#overview)
2. [Project Structure](#project-structure)
3. [Algorithms Implemented](#algorithms-implemented)
4. [Installation](#installation)
5. [How to Run](#how-to-run)
6. [Evaluation Results](#evaluation-results)
7. [References](#references)

---

## Overview

This repository provides an implementation of reinforcement learning algorithms for training agents to solve specific environments. It includes basic Deep Q-Network (DQN), Double DQN, Dueling DQN, and Maxmin Q-learning methods.

The project is designed to:
- Train agents using different DQN approaches.
- Evaluate the performance of these agents.
- Compare results visually through graphs and metrics.

---

## Project Structure

### Main Directories
- **Demo/**: Contains the main scripts for showcasing the agents' performance, including a lightweight application and model inference.
  - `agent.py`: Defines the agent and its interaction with the environment.
  - `app.py`: A simple application for demonstrating the trained agent.
  - `dqn.py`: Core implementation of the DQN algorithm.
  - `experience_replay.py`: Handles the replay buffer for training.
  - `main.py`: Script to train, test, or visualize agent performance.
- **Evaluations/**: Stores performance results and evaluation metrics such as loss curves, reward graphs, and Q-value data for each algorithm.
  - Files include `.csv` data and visualizations (e.g., `.png`) for loss, rewards, and Q-value progression.
- **TrainBasicDQN/**, **TrainDoubleDQN/**, **TrainDuelingDQN/**, **TrainMaxminQLearning/**: Individual modules for training agents with respective algorithms.
  - Each module includes agent definition, environment setup, and training scripts.

---

## Algorithms Implemented

1. **Basic DQN**: Implements the foundational Deep Q-Learning algorithm.
2. **Double DQN**: Mitigates overestimation bias by separating target and evaluation networks.
3. **Dueling DQN**: Adds advantage estimation for improved policy evaluation.
4. **Maxmin Q-Learning**: Combines multiple Q-functions to enhance learning stability.

---

## Installation

1. Clone the repository:
   ```bash
   git clone <repository_url>
   cd <repository_name>
   ```

2. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```

3. Ensure Python 3.9+ is installed.

---

## How to Run

### Training an Agent
Navigate to the respective training directory (e.g., `TrainBasicDQN`) and execute:
```bash
python main.py
```

### Evaluating an Agent
Use the `Demo` directory to visualize results:

Take a look at the README file in DEMO directory.



### Viewing Evaluation Results
Evaluation graphs and metrics are stored in the `Evaluations` directory for each algorithm.

---

## Evaluation Results

The `Evaluations/` folder includes:
- **Loss History**: Tracks the loss over training epochs.
- **Reward Graphs**: Shows the progression of cumulative rewards.
- **Q-Value Analysis**: Illustrates Q-value changes during training.

Each algorithm has separate `.csv` data and visualizations.

---

## References

This project uses concepts from the following sources:
1. Mnih, V., et al. "Playing Atari with Deep Reinforcement Learning." (2013)
2. Van Hasselt, H., et al. "Deep Reinforcement Learning with Double Q-Learning." (2016)
3. Wang, Z., et al. "Dueling Network Architectures for Deep Reinforcement Learning." (2016)
