import gym
import numpy as np
from agent import DQNAgent
from utils.plotter import plot_rewards
from utils.config import *

def train():
    env = gym.make('CartPole-v1')
    agent = DQNAgent(state_size=4, action_space=env.action_space.n)
    rewards = []

    for episode in range(MAX_EPISODES):
        state = env.reset()
        state = np.reshape(state, [1, 4])
        total_reward = 0

        for time in range(500):
            action = agent.act(state)
            next_state, reward, done, _, _ = env.step(action)
            next_state = np.reshape(next_state, [1, 4])
            agent.memory.store(state, action, reward, next_state, done)
            state = next_state
            total_reward += reward

            agent.train()

            if time % TARGET_UPDATE_FREQ == 0:
                agent.update_target_network()

            if done:
                break
        
        rewards.append(total_reward)
        print(f"Episode: {episode+1}/{MAX_EPISODES}, Total Reward: {total_reward}")
    
    plot_rewards(rewards)
    env.close()

if __name__ == '__main__':
    train()
