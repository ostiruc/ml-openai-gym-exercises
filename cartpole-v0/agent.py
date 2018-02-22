import random
import numpy as np
from collections import deque

#from tensorflowmodel import Model
from kerasmodel import Model

# Deep Q-learning Agent
class DQNAgent:
    def __init__(self, state_size, action_size):
        self.state_size = state_size
        self.action_size = action_size
        self.memory = deque(maxlen=2000)
        self.gamma = 0.95    # discount rate
        self.epsilon = 1.0  # exploration rate
        self.epsilon_min = 0.01
        self.epsilon_decay = 0.995
        
        self.model = Model(self.state_size, self.action_size)

    def remember(self, state, action, reward, next_state, done):
        self.memory.append((state, action, reward, next_state, done))

    def act(self, state, force_exploitation=False):
        if np.random.rand() <= self.epsilon and not force_exploitation:
            return random.randrange(self.action_size)

        act_values = self.model.predict(state)

        return np.argmax(act_values[0])  # returns action

    def replay(self, batch_size):
        if len(self.memory) < batch_size:
            return

        minibatch = np.array(random.sample(self.memory, batch_size))
        
        states = np.array([x[0].tolist() for x in minibatch[:,0]])
        actions = minibatch[:,1]
        rewards = minibatch[:,2]
        next_states = np.array([x[0].tolist() for x in minibatch[:,3]])
        gamma_values = [0.0 if done else self.gamma for done in minibatch[:,4]]

        targets = rewards + gamma_values * np.amax(self.model.predict(next_states), axis=1)
        target_fs = self.model.predict(states)

        for target_f, target, action in zip(target_fs, targets, actions):            
            target_f[action] = target
        
        self.model.fit(states, target_fs, epochs=1, verbose=0)

        if self.epsilon > self.epsilon_min:
            self.epsilon *= self.epsilon_decay