import time
import cv2
import flappy_bird_gym
import numpy as np
import torch

from agent import Agent
from arguments import arguments

env = flappy_bird_gym.make("FlappyBird-rgb-v0")
arg = arguments()
agent=Agent(env,arg)

def process_img(image):
    image= cv2.resize(image, (84, 84))
    image= cv2.cvtColor(image, cv2.COLOR_BGR2GRAY) 
    image=cv2.threshold(image, 199, 1, cv2.THRESH_BINARY_INV) 
    return image[1]

def load_state(modelId):
    modelpath = f'model/model{modelId}.pkl'
    # modelpath = f'double_dqn_model/model{modelId}.pkl'
    # modelpath = f'maxmin_q_learning/model{modelId}.pkl'
    state_file = torch.load(modelpath, weights_only=True)
    agent.Net.load_state_dict(state_file)
    agent.targetNet.load_state_dict(state_file)


def demo():
    
    env = flappy_bird_gym.make("FlappyBird-rgb-v0")
    start = 20100
    end = 46500
    # reward_list = []
    # for id in range(start, end+1, 300):
    #     # print(id)
    #     load_state(id)
    #     obs = env.reset()
    #     done = False
    #     total_reward = 0
        
    #     obs = process_img(obs)
    #     obs = np.expand_dims(obs, axis=0)
    #     obs = np.repeat(obs, 4, axis=0)
            
    #     while not done:
    #         action = agent.greedy_action(obs)
    #         next_obs, reward, done, info = env.step(action)
    #         next_obs = np.expand_dims(process_img(next_obs),axis=0)
    #         next_obs = np.concatenate((next_obs,obs[:3,:,:]),axis=0)
            
    #         total_reward += reward
    #         obs = next_obs
    #         # env.render()
    #         # time.sleep(1/30)
        
    #     reward_list.append(total_reward)
    #     # env.close()
    
    # highest_reward = max(reward_list)
    # highest_reward_id = start+reward_list.index(highest_reward)*300
    # print(f"highest_reward: {highest_reward} highest_reward_id: {highest_reward_id}")
    
    
    # load_state(highest_reward_id)
    load_state(40200)
    obs = env.reset()
    done = False
    total_reward = 0
    obs = process_img(obs)
    obs = np.expand_dims(obs, axis=0)
    obs = np.repeat(obs, 4, axis=0)
    while not done:
        action = agent.greedy_action(obs)
        next_obs, reward, done, info = env.step(action)
        next_obs = np.expand_dims(process_img(next_obs),axis=0)
        next_obs = np.concatenate((next_obs,obs[:3,:,:]),axis=0)
        
        total_reward += reward
        obs = next_obs
        env.render()
        time.sleep(1/30)

    
if __name__ == '__main__':
    demo()
