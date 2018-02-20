import random
import numpy as np
from collections import deque

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

        minibatch = random.sample(self.memory, batch_size)

        states = None
        target_fs = None
        for state, action, reward, next_state, done in minibatch:
            target = reward

            if not done:
              target = reward + self.gamma * np.amax(self.model.predict(next_state)[0])

            target_f = self.model.predict(state)
            target_f[0][action] = target

            # Concatenante our sample set
            if states is not None:
                states = np.concatenate((states, state), axis=0)
            else:
                states = state

            if target_fs is not None:
                target_fs = np.concatenate((target_fs, target_f), axis=0)
            else:
                target_fs = target_f

        self.model.fit(states, target_fs, epochs=1, verbose=0)

        if self.epsilon > self.epsilon_min:
            self.epsilon *= self.epsilon_decay