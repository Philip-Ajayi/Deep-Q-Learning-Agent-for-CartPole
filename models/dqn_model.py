import tensorflow as tf
from tensorflow.keras import layers, models

class DQNModel(tf.keras.Model):
    def __init__(self, action_space):
        super(DQNModel, self).__init__()
        self.action_space = action_space
        
        self.hidden_layer1 = layers.Dense(128, activation='relu', input_shape=(4,))
        self.hidden_layer2 = layers.Dense(128, activation='relu')
        self.output_layer = layers.Dense(self.action_space, activation='linear')
        
    def call(self, state):
        x = self.hidden_layer1(state)
        x = self.hidden_layer2(x)
        q_values = self.output_layer(x)
        return q_values
