import keras.optimizers as Optimizers
import pandas as pd
from keras.models import model_from_json

from keras.layers import Dense
from keras.models import Sequential
import os
import matplotlib.pyplot as plt


class Nn:

    def __init__(self):
        self.epochs = 10000
        self.learning_rate = 0.00005
        self.batch_size = 20
        self.model_dir = "models"
        self.model = None
        self.history = None
        self.input_dim = 0
        self.output_dim = 0

        if not os.path.exists(self.model_dir):
            os.mkdir(self.model_dir)

    def create_model(self, input_dim=11, output_dim=2):
        """
        Returns a compiled untrained model.
        :param input_dim:
        :param output_dim:
        :return:
        """
        self.output_dim = output_dim
        self.input_dim = input_dim

        model = Sequential()
        model.add(
            Dense(input_dim, input_dim=input_dim, kernel_initializer="uniform", activation='tanh', name='layer_0'))
        model.add(Dense(128, activation="tanh", use_bias=False, kernel_initializer="uniform", name='layer_1'))
        model.add(Dense(256, activation="tanh", use_bias=False, kernel_initializer="uniform", name='layer_2'))
        model.add(Dense(256, activation="tanh", use_bias=False, kernel_initializer="uniform", name='layer_3'))
        model.add(Dense(128, activation="tanh", use_bias=False, kernel_initializer="uniform", name='layer_4'))
        model.add(Dense(output_dim, activation="tanh", use_bias=False, kernel_initializer="uniform", name='layer_5'))

        opt = Optimizers.RMSprop(lr=self.learning_rate, rho=0.9, epsilon=1e-06, decay=0.0)
        model.compile(optimizer=opt, loss='mse')

        model.summary()
        self.model = model

    def train(self, x, y, validation_data):
        """
        Trains the model accordingly to to class variables.
        :return:
        """
        if self.model is None:
            raise ModelNotCreatedException("The model has not been created yet. call create_model() first.")

        if not isinstance(self.model, Sequential):
            raise ModelTypeMissmatchException("Expected object of Sequential. Given " + str(type(self.model)) + ".")

        history = self.model.fit(x=x, y=y, epochs=self.epochs, batch_size=self.batch_size,
                                 validation_data=validation_data)

        self.history = history
        df = pd.DataFrame(list(zip(history.history["val_loss"], history.history["loss"])),
                          columns=["Val_loss", "Training_loss"])
        plt.close('all')
        plt.figure()
        df.plot()
        plt.savefig(os.path.join(self.model_dir, "history.png"), bbox_inches='tight')
        return history

    def persist_model(self):
        """
        Persist model to file.
        :return:
        """
        with open(os.path.join(self.model_dir, "model.json"), 'w') as outfile:
            outfile.write(self.model.to_json(sort_keys=True, indent=4, separators=(',', ': ')))
            outfile.close()

        self.model.save(os.path.join(self.model_dir, "model.h5"))

    def load_model_from_json(self):
        """
        Load a pre-trained model from file.
        :return:
        """
        filename_full = os.path.join(self.model_dir, "model")

        # Load model structure
        with open(filename_full + '.json', 'r') as file:
            model = model_from_json(file.read())
            file.close()
            model.load_weights(filename_full + '.h5')

            opt = Optimizers.RMSprop(lr=self.learning_rate, rho=0.9, epsilon=1e-06, decay=0.0)
            model.compile(optimizer=opt, loss='mse')
            self.model = model

    def evaluate(self, x, y, batch_size=10):
        score = self.model.evaluate(x=x, y=y, batch_size=batch_size, verbose=1, sample_weight=None, steps=None)
        return score


class ModelTypeMissmatchException(Exception):
    def __init__(self, *args, **kwargs):
        Exception.__init__(self, *args, **kwargs)


class ModelNotCreatedException(Exception):
    def __init__(self, *args, **kwargs):
        Exception.__init__(self, *args, **kwargs)
