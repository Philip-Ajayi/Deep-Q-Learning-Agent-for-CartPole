import numpy as np
import tensorflow as tf
from models.dqn_model import DQNModel
from utils.experience_replay import ExperienceReplay
from utils.epsilon_scheduler import EpsilonScheduler
from utils.config import *

class DQNAgent:
    def __init__(self, state_size, action_space):
        self.state_size = state_size
        self.action_space = action_space
        
        self.model = DQNModel(action_space)
        self.target_model = DQNModel(action_space)
        self.target_model.set_weights(self.model.get_weights())  # Initialize target model
        
        self.memory = ExperienceReplay(MAX_MEMORY)
        self.epsilon_scheduler = EpsilonScheduler(EPSILON_START, EPSILON_END, EPSILON_DECAY)
        self.optimizer = tf.keras.optimizers.Adam(learning_rate=LEARNING_RATE)
        
    def act(self, state):
        if np.random.rand() <= self.epsilon_scheduler.get_epsilon():
            return np.random.choice(self.action_space)  # Exploration
        q_values = self.model(state)
        return np.argmax(q_values[0])  # Exploitation
    
    def train(self):
        if self.memory.size() < BATCH_SIZE:
            return
        
        batch = self.memory.sample(BATCH_SIZE)
        states, actions, rewards, next_states, dones = zip(*batch)
        
        states = np.array(states)
        next_states = np.array(next_states)
        
        with tf.GradientTape() as tape:
            q_values = self.model(states)
            next_q_values = self.target_model(next_states)
            
            targets = q_values.numpy()
            for i in range(BATCH_SIZE):
                target = rewards[i]
                if not dones[i]:
                    target += GAMMA * np.max(next_q_values[i])
                targets[i, actions[i]] = target
            
            loss = tf.reduce_mean(tf.square(targets - q_values))
        
        grads = tape.gradient(loss, self.model.trainable_variables)
        self.optimizer.apply_gradients(zip(grads, self.model.trainable_variables))
        
        if self.epsilon_scheduler.get_epsilon() > EPSILON_END:
            self.epsilon_scheduler.update_epsilon()
        
        return loss.numpy()
    
    def update_target_network(self):
        self.target_model.set_weights(self.model.get_weights())
