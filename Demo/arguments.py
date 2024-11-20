# Define hyperparameters for training
class arguments():
    def __init__(self):
        self.gamma = 0.99  # Discount factor
        self.action_dim = 2  # Number of possible actions
        self.obs_dim = (4, 84, 84)  # Observation dimensions
        # self.capacity = 50000  # Replay buffer capacity
        self.capacity = 50  # Replay buffer capacity
        self.cuda = 'cuda:0'  # Device to use for training
        self.Frames = 4  # Number of frames to stack
        self.episodes = int(1e8)  # Number of episodes to train
        self.updatebatch = 512  # Batch size for updates
        self.test_episodes = 10  # Number of episodes for testing
        self.epsilon = 0.1  # Epsilon for epsilon-greedy policy
        self.Q_NETWORK_ITERATION = 50  # Iteration frequency to update target network
        self.learning_rate = 0.001  # Learning rate for optimizer
        
        
        self.loss_graph = './loss.png'
        self.q_value_graph = './q_value.png'
        self.save_policy_net_path = './models/policy/'
        self.save_target_net_path = './models/target/'
        self.train_reward_graph = './train_reward.png'
        self.test_reward_graph = './test_reward.png'
        self.epsilon_graph = './epsilon.png'