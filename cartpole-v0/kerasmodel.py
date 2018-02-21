from keras.models import Sequential
from keras.layers import Dense
from keras.optimizers import Adam
from keras import backend as K

class Model:
    def __init__(self, state_size, action_size):
        self.learning_rate = 0.001
        self.model = self._build_model(state_size, action_size)        

    def _build_model(self, state_size, action_size):
        model = Sequential()
        model.add(Dense(24, input_dim=state_size, activation='relu'))
        model.add(Dense(24, activation='relu'))
        model.add(Dense(action_size, activation='linear'))
        model.compile(loss='mse',
                      optimizer=Adam(lr=self.learning_rate))
        return model
    
    def predict(self, state):
        return self.model.predict(state)

    def fit(self, state, target_f, epochs, verbose):
        self.model.fit(state, target_f, epochs=epochs, verbose=verbose)
    