class KerasModel:
    def __init__(self, model):
        self.model = model
    
    def predict(self, state):
        return self.model.predict(state)

    def fit(self, state, target_f, epochs, verbose):
        self.model.fit(state, target_f, epochs=epochs, verbose=verbose)
    