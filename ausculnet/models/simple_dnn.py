from keras.layers import Dense, Embedding, regularizers
from keras.models import Sequential


def construct_model(num_classes):
    """
    Constructs a simple keras Deep Neural Network model

    Returns:
        keras model of a Deep Neural Network

    """
    model = Sequential()

    model.add(Dense(200, activation='tanh'))

    model.add(Dense(300, activation='sigmoid', kernel_regularizer=regularizers.l2(0.01)))

    model.add(Dense(num_classes, activation='softmax'))

    return model
