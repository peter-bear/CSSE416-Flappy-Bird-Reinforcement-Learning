import cv2
import math
import random
import time
import flappy_bird_gym
import cv2
import torch
import numpy as np
from tqdm import tqdm
from agent import Agent
from arguments import arguments
import matplotlib.pyplot as plt

def process_img(image):
    image= cv2.resize(image, (84, 84)) # resize
    image= cv2.cvtColor(image, cv2.COLOR_BGR2GRAY) # to gray
    image=cv2.threshold(image, 199, 1, cv2.THRESH_BINARY_INV) # to binary
    return image[1]


env = flappy_bird_gym.make("FlappyBird-rgb-v0")
arg = arguments()
agent=Agent(env,arg)


def save_plot(reward_graph_data, test_average_rewards, epsilon_history):
    plt.figure(figsize=(10,5))
    plt.plot(reward_graph_data)
    plt.xlabel('Epochs')
    plt.ylabel('Reward')
    plt.savefig(arg.train_reward_graph)
    plt.close()

    plt.figure(figsize=(10,5))
    plt.plot(range(0, len(test_average_rewards)*50, 50), test_average_rewards)
    plt.xlabel('Epochs')
    plt.ylabel('Reward')
    plt.savefig(arg.test_reward_graph)
    plt.close()

    plt.figure(figsize=(10,5))
    plt.plot(epsilon_history)
    plt.xlabel('Epochs')
    plt.ylabel('Epsilon')
    plt.savefig(arg.epsilon_graph)
    plt.close()

def save_q_value_plot(q_value_data):
    plt.figure(figsize=(10, 5))
    plt.plot(q_value_data)
    plt.xlabel('Epochs')
    plt.ylabel('Average Q-value')
    plt.title('Q-value over Training')
    plt.savefig(arg.q_value_graph)
    plt.close()

def save_loss_plot(loss_history):
    plt.figure(figsize=(10, 5))
    plt.plot(loss_history)
    plt.xlabel('Training Steps')
    plt.ylabel('Loss')
    plt.title('Loss Curve')
    plt.savefig(arg.loss_graph)
    plt.close()

def test_performance(): # test the performance of the model
    reward_list = []
    for i in range(arg.test_episodes):
        obs = process_img(env.reset())
        obs = np.expand_dims(obs, axis=0)
        obs = np.repeat(obs, 4, axis=0)
        done = False
        reward = 0
        while not done:
            action = agent.greedy_action(obs)
            transition = env.step(action)
            next_obs = np.expand_dims(process_img(transition[0]),axis=0)
            obs = np.concatenate((next_obs,obs[:3,:,:]),axis=0)
            reward += transition[1]
            done = transition[2]
        reward_list.append(reward)
    return (sum(reward_list)/arg.test_episodes)-101

def load_state(model_id):
    modelpath = f'model/model{model_id}.pkl'
    state_file = torch.load(modelpath)
    agent.Net.load_state_dict(state_file)
    agent.targetNet.load_state_dict(state_file)

def training():
    test_average_rewards = []
    epsilon_history = []
    q_value_data = []
    loss_history = []  
    
    reward_graph_data=[]
    
    for i in tqdm(range(arg.episodes), desc="Training Epochs"):
        obs = process_img(env.reset())
        obs = np.expand_dims(obs, axis=0)
        obs = np.repeat(obs, 4, axis=0)
        done = False
        transition_store = {
            'obs': [],
            'next_obs': [],
            'action': [],
            'reward': [],
            'done': []
        }
        
        total_reward = 0
        q_values = []  # List to store Q-values per episode
        
        if i%10000==0 and i>100:  #epsilon decay per 10000 steps
            arg.epsilon/=math.sqrt(10)
        while not done:
            action = agent.get_action(obs)
            if i < 300:
                action = random.randint(0,1)
            transition = env.step(action)
            next_obs = np.expand_dims(process_img(transition[0]),axis=0)
            next_obs = np.concatenate((next_obs,obs[:3,:,:]),axis=0)
            reward = transition[1]
            done = transition[2]
            if done:
                reward -= 101 #penalty
            
                        # Store Q-values from the policy network
            state_tensor = torch.tensor(obs, device=arg.cuda, dtype=torch.float32).unsqueeze(0)
            q_value = agent.Net(state_tensor).cpu().detach().numpy().squeeze()
            q_values.append(q_value)
            
            transition_store['obs'].append(obs)
            transition_store['next_obs'].append(next_obs)
            transition_store['reward'].append(reward)
            transition_store['done'].append(done)
            transition_store['action'].append(action)
            obs = next_obs
            
            total_reward += reward
            
        # store in the graph
        q_value_data.append(np.mean(q_values))
        epsilon_history.append(arg.epsilon)
        reward_graph_data.append(total_reward)
            
        agent.Buffer.store_data(transition_store, len(transition_store['obs']))
        
        if agent.Buffer.ptr > 500:  #  Make sure have enough data to update
            loss = agent.update(agent.Buffer.sample(arg.updatebatch))
            loss_history.append(loss)
        
        if i % 50 == 0: #Test model every 50 episodes
            if i % 300 == 0:  # Save model every 300 episodes
                torch.save(agent.Net.state_dict(),'model/model'+str(i)+'.pkl')
            average_r = test_performance()
            print('iteration episodes: '+str(i)+' test average reward: '+str(average_r))
            test_average_rewards.append(average_r)


        save_q_value_plot(q_value_data)
        save_plot(reward_graph_data, test_average_rewards, epsilon_history)
        save_loss_plot(loss_history)
        
        
        # Save all graph data as CSV files after training
        np.savetxt('reward_graph_data.csv', np.array(reward_graph_data), delimiter=',', header='Reward', comments='')
        np.savetxt('test_average_rewards.csv', np.array(test_average_rewards), delimiter=',', header='Average Reward', comments='')
        np.savetxt('epsilon_history.csv', np.array(epsilon_history), delimiter=',', header='Epsilon', comments='')
        np.savetxt('loss_history.csv', np.array(loss_history), delimiter=',', header='Loss', comments='')
        np.savetxt('q_value_data.csv', np.array(q_value_data), delimiter=',', header='Q-Value', comments='')
        
        
if __name__ == '__main__':
    # load_state(24000)
    training()