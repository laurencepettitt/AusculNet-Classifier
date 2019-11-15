from keras.layers import Dense
from keras.models import Sequential
from keras import regularizers


def construct_model(num_classes):
    """
    Constructs a simple Deep Neural Network model in Keras

    Returns:
        The constructed Deep Neural Network Keras model

    """
    model = Sequential()

    model.add(Dense(200, activation='tanh'))

    model.add(Dense(300, activation='sigmoid', kernel_regularizer=regularizers.l2(0.01)))

    model.add(Dense(num_classes, activation='softmax'))

    return model
