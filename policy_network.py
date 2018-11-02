import numpy as np
from keras.models import Sequential
from keras.layers import Activation, LSTM, Dense, BatchNormalization
from keras.optimizers import sgd
from keras.models import model_from_json


class PolicyNetwork:
    def __init__(self, input_dim, output_dim, lr=0.001):
        self.input_dim = input_dim
        self.lr = lr
        self.loaded_model = None

        self.model = Sequential()
        self.model.add(LSTM(256, input_shape=(1, input_dim),return_sequences=True, stateful=False, dropout=0.5))
        self.model.add(BatchNormalization())
        self.model.add(LSTM(256, return_sequences=True, stateful=False, dropout=0.5))
        self.model.add(BatchNormalization())
        self.model.add(LSTM(256, return_sequences=False, stateful=False, dropout=0.5))
        self.model.add(BatchNormalization())
        self.model.add(Dense(output_dim))
        self.model.add(Activation('sigmoid'))

        self.model.compile(optimizer=sgd(lr=lr), loss='mse')
        self.prob = None

    def reset(self):
        self.prob = None

    def predict(self, sample): # input_dim = 17 (15+2)
#        print(np.array(sample))
        self.prob = self.model.predict(np.array(sample).reshape((1, -1, self.input_dim)))[0]
#        print(self.prob)
        return self.prob

    def train_on_batch(self, x, y):
        return self.model.train_on_batch(x, y)

    def evaluate(self,x,y,batch_size):
        return model.evaluate(x, y, batch_size=batch_size)

    def save_model(self, model_path):
        model_json = self.model.to_json()
        with open("model.json", "w") as json_file:
            json_file.write(model_json)

        if model_path is not None and self.model is not None:
            self.model.save_weights(model_path, overwrite=True)

    def load_model(self, model_path):
        json_file = open('model.json', 'r')
        loaded_model_json = json_file.read()
        json_file.close()
        self.model = model_from_json(loaded_model_json)

        if model_path is not None:
            self.model.load_weights(model_path)
