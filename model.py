import tensorflow as tf
from tensorflow import keras as keras
from keras.backend import relu, softmax
from keras.layers import Input, Flatten, Dense 
from keras.models import Model
from keras.optimizers import Adam


class Model(tf.keras.Model): 
    def model(self, lr:int=0.001):
        super(Model, self).__init__()

        inputs = Input(shape=(5*52))
        
        # flatten = Flatten()(inputs)

        dense1 = Dense(256, activation=relu)(inputs)
        dense2 = Dense(512, activation=relu)(dense1)
        dense3 = Dense(512, activation=relu)(dense2)
        dense4 = Dense(256, activation=relu)(dense3)
        output = Dense(52, activation=softmax)(dense4)

        model = Model(inputs=inputs, outputs=output, name='model')
        model.compile(optimizer=Adam(lr), 
                      loss='binary_crossentropy',
                      metrics=['accuracy'],)
        return model


if __name__ == '__main__':
    m = Model()
    model = m.model()
    model.summary()
    model.save('Model')
