class EpsilonScheduler:
    def __init__(self, epsilon_start, epsilon_end, epsilon_decay):
        self.epsilon_start = epsilon_start
        self.epsilon_end = epsilon_end
        self.epsilon_decay = epsilon_decay
        self.epsilon = epsilon_start
        
    def get_epsilon(self):
        return self.epsilon
    
    def update_epsilon(self):
        if self.epsilon > self.epsilon_end:
            self.epsilon *= self.epsilon_decay
