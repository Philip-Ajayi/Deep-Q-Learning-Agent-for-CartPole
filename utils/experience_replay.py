import random
import numpy as np

class ExperienceReplay:
    def __init__(self, max_size):
        self.max_size = max_size
        self.memory = []
        
    def store(self, state, action, reward, next_state, done):
        if len(self.memory) >= self.max_size:
            self.memory.pop(0)  # Remove oldest experience
        self.memory.append((state, action, reward, next_state, done))
    
    def sample(self, batch_size):
        return random.sample(self.memory, batch_size)
    
    def size(self):
        return len(self.memory)
        
