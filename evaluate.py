import gym
import numpy as np
from agent import DQNAgent

def evaluate():
    env = gym.make('CartPole-v1')
    agent = DQNAgent(state_size=4, action_space=env.action_space.n)
    agent.model.load_weights('dqn_cartpole_model.h5')

    state = env.reset()
    state = np.reshape(state, [1, 4])
    total_reward = 0

    for time in range(500):
        action = agent.act(state)
        next_state, reward, done, _, _ = env.step(action)
        next_state = np.reshape(next_state, [1, 4])
        state = next_state
        total_reward += reward

        if done:
            break
    
    print(f"Evaluation completed. Total Reward: {total_reward}")
    env.close()

if __name__ == '__main__':
    evaluate()
