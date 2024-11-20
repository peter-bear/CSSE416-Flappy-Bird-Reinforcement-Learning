# Flappy Bird RL Agent

This project implements a Reinforcement Learning (RL) agent to play the game Flappy Bird. The agent is trained using Deep Q-Learning techniques and can be deployed to observe the agent's performance in a simulated environment.

## Project Structure

### Folders

- **`models/`**: Directory for model files, potentially with different training versions or architectures.
- **`static/`**: Stores static files used by the project, such as images and assets for web deployment.



### Python Files

- **`agent.py`**: Defines the main RL agent class, implementing the logic for interacting with the environment and updating policies.
- **`app.py`**: Contains the FastAPI application for deploying the model to a web interface, including WebSocket communication for real-time gameplay.
- **`arguments.py`**: Defines hyperparameters and other configurable settings for the RL model training.
- **`dqn.py`**: Likely implements the Deep Q-Network (DQN) algorithm used for training the RL agent.
- **`experience_replay.py`**: Implements the experience replay buffer, a crucial component for DQN that stores past experiences for training.
- **`script_model.py`**: Contains code to export or serialize the model into a scriptable format (e.g., TorchScript) for deployment.



### Data and Results

- **`epsilon_history.csv`**: Logs the history of the epsilon parameter (exploration rate) over training episodes.
- **`loss_history.csv`**: Records the loss values during training, useful for monitoring model convergence.
- **`q_value_data.csv`**: Tracks the Q-values over time, providing insights into how the model’s predictions evolve during training.
- **`reward_graph_data.csv`**: Logs the reward values obtained by the agent across training episodes.
- **`test_average_rewards.csv`**: Stores average rewards obtained during test episodes to evaluate model performance.



### Visualization and Graphs

- **`epsilon.png`**: Graph showing the epsilon (exploration rate) decay over time.
- **`graph_end_46850.jpg`**: A saved graph or screenshot, possibly showing the final state of training or test results.
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

4. **Running the Web Application**
   
   To view the trained agent in action within a web application, follow these steps:
   
   1. **Start the FastAPI Server**:
      - Run the following command from the root directory to start the FastAPI backend using Uvicorn:
        ```bash
        uvicorn app:app --host 0.0.0.0 --port 8000
        ```
      - This will start the backend server, which handles WebSocket connections and communicates with the RL agent.
   
   2. **Start the Static File Server**:
      - Navigate to the `static` folder:
        ```bash
        cd static
        ```
      - Start a simple HTTP server to serve the frontend files:
        ```bash
        python -m http.server 8080
        ```
      - This serves the HTML, JavaScript, and CSS files for the frontend at `localhost:8080`.
   
   3. **Access the Web Application**:
      - Open a web browser and go to [http://localhost:8080](http://localhost:8080) to view and interact with the RL agent.
   
   The application will now be running, with the backend server on `localhost:8000` and the frontend available at `localhost:8080`.
   



## Explanation of Key Components

- **Deep Q-Learning (DQN)**: This RL algorithm is used to train the agent to play Flappy Bird by learning Q-values, which estimate the expected reward for taking specific actions in given states.
- **Experience Replay**: A mechanism to store and reuse past experiences, helping to break the correlation between consecutive actions and improve training stability.



## Visualizing Results

The CSV files (`epsilon_history.csv`, `loss_history.csv`, etc.) and the corresponding `.png` graphs provide insights into how the model trains over time. By analyzing these files, you can understand the agent's learning progress and performance.

