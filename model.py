import tensorflow as tf
from tensorflow import keras as keras
from keras.backend import relu, softmax
from keras.layers import Input, Flatten, Dense, Conv1D, MaxPool1D
from keras.models import Model
from keras.optimizers import Adam


class Model(tf.keras.Model): 
    def model(self, lr:int=0.0001):
        super(Model, self).__init__()
        
        inputs = Input(shape=(93))
    
        # cnv1 = Conv1D(64, kernel_size=(2,), padding='SAME', activation=relu)(inputs)
        # cnv2 = Conv1D(64, kernel_size=(2,), padding='SAME', activation=relu)(cnv1)
        # maxpooling1 = MaxPool1D(pool_size=(2,), padding='SAME')(cnv2)

        # cnv2 = Conv1D(104, kernel_size=(3,), padding='SAME', activation=relu)(maxpooling1)
        # maxpooling2 = MaxPool1D(pool_size=(2,), padding='SAME')(cnv2)

        flatten = Flatten()(inputs)
        dense1 = Dense(256, activation=relu)(flatten)
        dense2 = Dense(256, activation=relu)(dense1)
        output = Dense(26, activation=softmax)(dense2)

        # inputs = Input(shape=(3*52)) # 5*52
        
        # # flatten = Flatten()(inputs)

        # dense1 = Dense(256, activation=relu)(inputs)
        # dense2 = Dense(512, activation=relu)(dense1)
        # dense3 = Dense(512, activation=relu)(dense2)
        # dense4 = Dense(256, activation=relu)(dense3)
        # output = Dense(52, activation=softmax)(dense4)

        model = Model(inputs=inputs, outputs=output, name='model')
        model.compile(optimizer=Adam(lr), 
                      loss='categorical_crossentropy',
                    #   loss = 'mae',
                      metrics=['accuracy'],)
        return model


if __name__ == '__main__':
    m = Model()
    model = m.model()
    model.summary()
    model.save('Model')
