from keras.layers import Dense, Dropout, Conv1D, MaxPooling1D, GlobalAveragePooling1D, Reshape
from keras.models import Sequential


def construct_model(num_classes, height):
    model = Sequential()
    # model.add(Reshape((height, 1), input_shape=(height, 1)))
    model.add(Conv1D(100, 10, activation='relu', input_shape=(height, 1)))
    model.add(Conv1D(100, 10, activation='relu'))
    model.add(MaxPooling1D(3))
    model.add(Conv1D(160, 10, activation='relu'))
    model.add(Conv1D(160, 10, activation='relu'))
    model.add(GlobalAveragePooling1D())
    model.add(Dropout(0.5))
    model.add(Dense(num_classes, activation='softmax'))
    return model
